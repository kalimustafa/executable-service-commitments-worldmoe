"""Differentiable, shared-shield training rollout for R4-V7.

The formal :mod:`dcvpp.e2e_moe_r4_v7.runtime` and this training path deliberately
share ``torch_shield_step``.  The neural policy is queried only at selected
rolling origins, but every simulated interval is projected by the exact EDF,
SOC and power-envelope shield.  This gives the curriculum horizons a useful
closed-loop surrogate without introducing a second set of feasibility
equations.

The day-ahead call receives an explicit issue-time allowlist.  Realized targets,
distribution targets and revealed prefixes cannot reach ``forward_issue``.
Distribution samples are exogenous rollout worlds and are always detached;
decision gradients can therefore reach only the Policy and its decision
adapter through ``forward_issue``/``forward_origin``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import inspect
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .runtime import (
    HORIZON,
    SHIELD_ALGEBRA_VERSION,
    SITES,
    ShieldConfig,
    ShieldLimits,
    reject_removed_actions,
    torch_shield_step,
)


CHANNELS = 21
TRAINING_HORIZONS = (8, 24, 48, 96)
ROLLOUT_SCHEMA_VERSION = "dcvpp.r4.v7.training-rollout.v1"


class V7RolloutError(RuntimeError):
    """Raised when a batch cannot satisfy the V7 training-rollout contract."""


def _call_supported(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(callback)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = (
        kwargs
        if accepts_kwargs
        else {key: value for key, value in kwargs.items() if key in signature.parameters}
    )
    return callback(*args, **selected)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V7RolloutError(f"{label} must be a mapping")
    return value


def _detach(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    return value


def _first(mapping: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
    for name in names:
        if mapping.get(name) is not None:
            return mapping[name]
    raise V7RolloutError(f"missing {label}; accepted keys={list(names)}")


def _tensor(value: Any, reference: Tensor) -> Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _distribution_view(distribution: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = distribution.get("distribution")
    return nested if isinstance(nested, Mapping) else distribution


def _world_samples(distribution: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    view = _distribution_view(distribution)
    samples = _first(
        view,
        ("samples", "samples_zonal", "particles_zonal", "worlds"),
        "detached Distribution samples",
    )
    if not isinstance(samples, Tensor):
        raise V7RolloutError("Distribution samples must be a tensor")
    samples = samples.detach()
    if samples.ndim != 4 or samples.shape[2:] != (HORIZON, CHANNELS):
        raise V7RolloutError("Distribution samples must be [day,world,96,21]")
    if samples.shape[1] < 2:
        raise V7RolloutError("policy risk training requires at least two worlds")
    if not bool(torch.isfinite(samples).all()):
        raise V7RolloutError("Distribution samples contain non-finite values")
    weights = view.get("weights", view.get("world_weights"))
    if weights is None:
        weights_tensor = samples.new_full(
            samples.shape[:2], 1.0 / float(samples.shape[1])
        )
    else:
        weights_tensor = torch.as_tensor(
            weights, device=samples.device, dtype=samples.dtype
        ).detach()
        if weights_tensor.ndim == 1 and weights_tensor.shape[0] == samples.shape[1]:
            weights_tensor = weights_tensor[None].expand(samples.shape[0], -1)
        if weights_tensor.shape != samples.shape[:2]:
            raise V7RolloutError("Distribution weights must be [day,world]")
        if bool((weights_tensor < 0.0).any()) or bool(
            (weights_tensor.sum(1) <= 0.0).any()
        ):
            raise V7RolloutError("Distribution weights must be nonnegative with positive rows")
        weights_tensor = weights_tensor / weights_tensor.sum(1, keepdim=True)
    return samples, weights_tensor


def _initial_snapshot(value: Any) -> Any:
    if not isinstance(value, Tensor):
        return value
    if value.ndim == 3 and value.shape[1] == HORIZON:
        return value[:, 0]
    return value


_ISSUE_KEYS = frozenset(
    {
        "market",
        "market_features",
        "market_inputs",
        "market_raw",
        "historical_market",
        "historical_market_available",
        "workload",
        "workload_features",
        "workload_inputs",
        "issue_workload_raw",
        "issue_pv_raw",
        "physical_static",
        "physical_dynamic",
        "physical",
        "physical_features",
        "forecast_anchors",
        "forecast_residual_caps",
        "anchor_center_zonal",
        "day_of_week",
        "month",
        "holiday",
        "dst",
        "workload_dt",
        "state_dt",
        "limits",
        "physical_limits",
    }
)


def issue_time_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only fields allowed to reach day-ahead issuance.

    ``rollout_targets``, ``targets``, current-day distribution targets and all
    revealed-prefix aliases are excluded by construction rather than by a
    brittle denylist.  A state sequence is reduced to its issue-time snapshot.
    """

    result = {key: value for key, value in batch.items() if key in _ISSUE_KEYS}
    state = next(
        (
            batch[key]
            for key in ("initial_state", "state_sequence", "state", "state_features")
            if batch.get(key) is not None
        ),
        None,
    )
    if state is not None:
        result["initial_state"] = _initial_snapshot(state)
    result["origin"] = 0
    return result


def _site_matrix(value: Any, reference: Tensor, batch: int, label: str) -> Tensor:
    result = _tensor(value, reference)
    if result.ndim == 0:
        result = result.reshape(1, 1).expand(batch, SITES)
    elif result.shape == (SITES,):
        result = result[None].expand(batch, -1)
    elif result.shape == (batch, 1):
        result = result.expand(-1, SITES)
    if result.shape != (batch, SITES):
        raise V7RolloutError(f"{label} must resolve to [day,4]")
    if not bool(torch.isfinite(result).all()):
        raise V7RolloutError(f"{label} contains non-finite values")
    return result


def _common_site_vector(value: Tensor, label: str) -> np.ndarray:
    if value.ndim != 2 or value.shape[1] != SITES:
        raise V7RolloutError(f"{label} must be [day,4]")
    first = value[0]
    if not bool(torch.allclose(value, first[None].expand_as(value), atol=0.0, rtol=0.0)):
        raise V7RolloutError(
            f"{label} differs within a training batch; batch days with common physical limits"
        )
    return first.detach().to(device="cpu", dtype=torch.float64).numpy()


def _limits(
    targets: Mapping[str, Any], reference: Tensor, batch: int
) -> tuple[ShieldLimits, dict[str, Tensor]]:
    grid = _site_matrix(
        _first(targets, ("grid_limit_mw", "grid_mw"), "grid limits"),
        reference,
        batch,
        "grid_limit_mw",
    )
    battery_power = _site_matrix(
        _first(targets, ("battery_power_mw", "battery_mw"), "battery power"),
        reference,
        batch,
        "battery_power_mw",
    )
    battery_energy = _site_matrix(
        _first(targets, ("battery_energy_mwh", "battery_mwh"), "battery energy"),
        reference,
        batch,
        "battery_energy_mwh",
    )
    compute = _site_matrix(
        _first(targets, ("capacity_work_units", "compute_work"), "compute capacity"),
        reference,
        batch,
        "capacity_work_units",
    )
    idle = _site_matrix(_first(targets, ("idle_mw",), "idle power"), reference, batch, "idle_mw")
    coefficient = _site_matrix(
        _first(targets, ("work_mw_per_unit",), "work power coefficient"),
        reference,
        batch,
        "work_mw_per_unit",
    )
    frozen = ShieldLimits(
        grid_mw=_common_site_vector(grid, "grid_limit_mw"),
        battery_mw=_common_site_vector(battery_power, "battery_power_mw"),
        battery_mwh=_common_site_vector(battery_energy, "battery_energy_mwh"),
        compute_work=_common_site_vector(compute, "capacity_work_units"),
        idle_mw=_common_site_vector(idle, "idle_mw"),
        work_mw_per_unit=_common_site_vector(coefficient, "work_mw_per_unit"),
    )
    return frozen, {
        "grid_mw": grid,
        "battery_mw": battery_power,
        "battery_mwh": battery_energy,
        "compute_work": compute,
        "idle_mw": idle,
        "work_mw_per_unit": coefficient,
    }


def _repeat_worlds(value: Tensor, worlds: int) -> Tensor:
    return (
        value[:, None]
        .expand(value.shape[0], worlds, *value.shape[1:])
        .reshape(value.shape[0] * worlds, *value.shape[1:])
    )


def _repeat_context(value: Any, batch: int, worlds: int) -> Any:
    if isinstance(value, Tensor):
        if value.ndim > 0 and value.shape[0] == batch:
            return _repeat_worlds(value.detach(), worlds)
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _repeat_context(item, batch, worlds) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_repeat_context(item, batch, worlds) for item in value)
    if isinstance(value, list):
        return [_repeat_context(item, batch, worlds) for item in value]
    return value


def _time_site(value: Any, reference: Tensor, batch: int, label: str) -> Tensor:
    result = _tensor(value, reference)
    if result.shape == (batch, SITES, HORIZON):
        result = result.transpose(1, 2)
    elif result.shape == (SITES, HORIZON) and batch == 1:
        result = result.transpose(0, 1)[None]
    elif result.shape == (HORIZON, SITES) and batch == 1:
        result = result[None]
    if result.shape != (batch, HORIZON, SITES):
        raise V7RolloutError(f"{label} must resolve to [day,96,4]")
    return result


def _day_plan(value: Any, reference: Tensor, batch: int, label: str) -> Tensor:
    result = _time_site(value, reference, batch, label)
    if not bool(torch.isfinite(result).all()):
        raise V7RolloutError(f"{label} contains non-finite values")
    return result


def _world_origins(horizon: int, budget: int | None) -> tuple[int, ...]:
    if budget is None or int(budget) >= horizon:
        return tuple(range(horizon))
    count = max(1, int(budget))
    values = torch.linspace(0, horizon - 1, count).round().long().tolist()
    return tuple(sorted(set([0, *map(int, values), horizon - 1])))


def _selected_origins(
    horizon: int,
    sampled_origins: Sequence[int] | None,
    origin_budget: int | None,
) -> tuple[int, ...]:
    if sampled_origins is None:
        return _world_origins(horizon, origin_budget)
    result = tuple(sorted(set(int(value) for value in sampled_origins)))
    if not result or result[0] != 0:
        raise V7RolloutError("sampled_origins must contain origin zero")
    if result[-1] >= horizon or result[0] < 0:
        raise V7RolloutError("sampled_origins must lie inside the curriculum horizon")
    return result


def _policy_section(output: Mapping[str, Any]) -> Mapping[str, Any]:
    value = output.get("policy")
    return value if isinstance(value, Mapping) else output


def _current_action(output: Mapping[str, Any], origin: int) -> tuple[Tensor, Tensor]:
    policy = _policy_section(output)
    actions = output.get("actions", policy.get("free_actions", policy.get("actions")))
    actions = _mapping(actions, "policy free actions")
    reject_removed_actions(actions)

    def current(name: str) -> Tensor:
        value = actions.get(name, policy.get(name))
        if not isinstance(value, Tensor):
            raise V7RolloutError(f"policy output is missing tensor action {name}")
        return value[:, origin] if value.ndim == 3 else value

    return current("battery"), current("compute")


def _plans(output: Mapping[str, Any]) -> tuple[Tensor | None, Tensor | None]:
    policy = _policy_section(output)
    battery = policy.get("battery_plan")
    compute = policy.get("compute_plan")
    return (
        battery if isinstance(battery, Tensor) else None,
        compute if isinstance(compute, Tensor) else None,
    )


def _critic(output: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = output.get("critic", _policy_section(output).get("critic"))
    return value if isinstance(value, Mapping) else None


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum(1)


def _weighted_upper_tail(values: Tensor, weights: Tensor, alpha: float) -> Tensor:
    tail_mass = 1.0 - float(alpha)
    if not 0.0 < tail_mass <= 1.0:
        raise V7RolloutError("cvar_alpha must lie in [0,1)")
    order = values.argsort(dim=1, descending=True)
    sorted_values = values.gather(1, order)
    sorted_weights = weights.gather(1, order)
    previous = sorted_weights.cumsum(1) - sorted_weights
    taken = torch.minimum(
        sorted_weights,
        (tail_mass - previous).clamp_min(0.0),
    )
    return (sorted_values * taken).sum(1) / tail_mass


def _fixed_baseline_costs(
    targets: Mapping[str, Any],
    *,
    reference: Tensor,
    batch: int,
    worlds: int,
    horizon: int,
    da_price: Tensor,
    rt_price: Tensor,
    arrival: Tensor,
    pv: Tensor,
    multiplier: Tensor,
    initial_soc: Tensor,
    limits: ShieldLimits,
    config: ShieldConfig,
) -> Tensor | None:
    raw = targets.get("baseline_actions")
    if not isinstance(raw, Mapping):
        return None
    if not all(raw.get(name) is not None for name in ("day_ahead", "battery", "compute")):
        return None
    reject_removed_actions(raw)
    day_ahead = _repeat_worlds(
        _day_plan(raw["day_ahead"], reference, batch, "baseline day_ahead"), worlds
    )
    battery = _repeat_worlds(
        _day_plan(raw["battery"], reference, batch, "baseline battery"), worlds
    )
    compute = _repeat_worlds(
        _day_plan(raw["compute"], reference, batch, "baseline compute"), worlds
    )
    soc = initial_soc.detach().clone()
    queue = reference.new_zeros((batch * worlds, config.queue_size))
    total = reference.new_zeros(batch * worlds)
    for origin in range(horizon):
        step = torch_shield_step(
            {"battery": battery[:, origin], "compute": compute[:, origin]},
            {"soc_mwh": soc, "queue": queue},
            {
                "origin": origin,
                "arrival": arrival[:, origin],
                "pv_mw": pv[:, origin],
                "physical_multiplier": multiplier[:, origin],
            },
            limits,
            config,
        )
        da_now = day_ahead[:, origin]
        total = total + config.interval_hours * (
            (da_now * da_price[:, origin]).sum(1)
            + ((step["grid_mw"] - da_now) * rt_price[:, origin]).sum(1)
        )
        total = total + (
            config.interval_hours
            * config.battery_degradation_per_mwh
            * (step["charge"] + step["discharge"]).sum(1)
            + config.sla_penalty_per_work * step["sla_violation"]
        )
        soc = step["soc_next_mwh"]
        queue = step["queue_next"]
    return total.reshape(batch, worlds).detach()


def differentiable_training_rollout(
    model: nn.Module,
    batch: Mapping[str, Any],
    distribution: Mapping[str, Any],
    *,
    horizon: int,
    phase: str | None = None,
    shield_config: ShieldConfig | None = None,
    sampled_origins: Sequence[int] | None = None,
    origin_budget: int | None = 8,
    cvar_alpha: float = 0.95,
) -> dict[str, Any]:
    """Run a differentiable V7 policy rollout on detached world paths.

    The full physical horizon is always executed.  ``sampled_origins`` controls
    only how often the expensive neural policy is reconditioned; between those
    origins the latest suffix plan is used.  Thus K=8/24/48/96 all retain exact
    stepwise shield dynamics and terminal-state accounting.
    """

    steps = int(horizon)
    if steps not in TRAINING_HORIZONS:
        raise V7RolloutError(f"horizon must be one of {TRAINING_HORIZONS}")
    if not isinstance(batch, Mapping):
        raise V7RolloutError("batch must be a mapping")
    targets = _mapping(batch.get("rollout_targets"), "batch.rollout_targets")
    samples, weights = _world_samples(_mapping(distribution, "distribution"))
    batch_size, world_count = samples.shape[:2]
    selected = _selected_origins(steps, sampled_origins, origin_budget)
    selected_set = set(selected)
    config = shield_config or ShieldConfig()
    reference = samples
    frozen_limits, batch_limits = _limits(targets, reference, batch_size)

    # Only issue-time fields reach the DA call.  In particular, ``targets`` and
    # ``rollout_targets`` are not visible to model.forward_issue.
    issue_batch = issue_time_batch(batch)
    model_limits = {
        "grid_mw": batch_limits["grid_mw"],
        "battery_mw": batch_limits["battery_mw"],
        "compute_work": batch_limits["compute_work"],
    }
    forward_issue = getattr(model, "forward_issue", None)
    if not callable(forward_issue):
        raise V7RolloutError("model must expose forward_issue")
    issue_output = _mapping(
        _call_supported(forward_issue, issue_batch, limits=model_limits),
        "forward_issue output",
    )
    reject_removed_actions(issue_output)
    issued_da = issue_output.get(
        "issued_day_ahead", _policy_section(issue_output).get("day_ahead")
    )
    if not isinstance(issued_da, Tensor):
        raise V7RolloutError("forward_issue must return issued_day_ahead")
    issued_da = _day_plan(issued_da, samples, batch_size, "issued_day_ahead")
    raw_context = _mapping(issue_output.get("issue_context"), "forward_issue issue_context")
    # Context tensors are detached so decision loss cannot traverse the truth
    # or Representation encoders.  The issued schedule is reattached as the
    # one immutable action that legitimately carries Policy gradient.
    context = _repeat_context(raw_context, batch_size, world_count)
    issued_flat = _repeat_worlds(issued_da, world_count)
    context = dict(context)
    context["issued_day_ahead"] = issued_flat

    samples_flat = samples.reshape(batch_size * world_count, HORIZON, CHANNELS)
    da_price = samples_flat[..., 0:4]
    rt_price = samples_flat[..., 4:8]
    arrival = samples_flat[..., 12].clamp_min(0.0)
    pv = samples_flat[..., 13:17].clamp_min(0.0)
    multiplier_day = _site_matrix(
        _first(targets, ("physical_multiplier",), "physical multiplier"),
        samples,
        batch_size,
        "physical_multiplier",
    )
    multiplier = _repeat_worlds(
        multiplier_day[:, None].expand(-1, HORIZON, -1), world_count
    )
    initial_soc_day = _site_matrix(
        _first(targets, ("initial_soc", "initial_soc_mwh"), "initial SOC"),
        samples,
        batch_size,
        "initial_soc",
    )
    initial_soc_day = torch.minimum(
        initial_soc_day.clamp_min(0.0), batch_limits["battery_mwh"]
    )
    initial_soc = _repeat_worlds(initial_soc_day, world_count)
    flat_limits = {
        "grid_mw": _repeat_worlds(batch_limits["grid_mw"], world_count),
        "battery_mw": _repeat_worlds(batch_limits["battery_mw"], world_count),
        "compute_work": _repeat_worlds(batch_limits["compute_work"], world_count),
    }

    def issue_reference(name: str, width: int, offset: int | None = None) -> Tensor:
        raw = batch.get(name)
        if raw is None and name == "market_raw":
            raw = batch.get("market")
        if raw is None:
            return samples.new_zeros((batch_size * world_count, HORIZON, width))
        value = _tensor(raw, samples)
        if value.ndim == 2 and width == 1:
            value = value[..., None]
        if offset is not None:
            value = value[..., offset : offset + width]
        if value.shape != (batch_size, HORIZON, width):
            raise V7RolloutError(f"{name} cannot provide [day,96,{width}] issue values")
        return _repeat_worlds(value, world_count)

    issue_market = issue_reference("market_raw", 4, 4)
    issue_workload = issue_reference("issue_workload_raw", 1)
    issue_pv = issue_reference("issue_pv_raw", 4)

    forward_origin = getattr(model, "forward_origin", None)
    if not callable(forward_origin):
        raise V7RolloutError("model must expose forward_origin")
    soc = initial_soc
    queue = samples.new_zeros((batch_size * world_count, config.queue_size))
    previous_grid = samples.new_zeros((batch_size * world_count, SITES))
    previous_battery = torch.zeros_like(previous_grid)
    previous_compute = torch.zeros_like(previous_grid)
    total_cost = samples.new_zeros(batch_size * world_count)
    total_sla = samples.new_zeros(batch_size * world_count)
    total_intervention = samples.new_zeros(batch_size * world_count)
    all_feasible = torch.ones(
        batch_size * world_count, device=samples.device, dtype=torch.bool
    )
    raw_battery_path: list[Tensor] = []
    raw_compute_path: list[Tensor] = []
    battery_path: list[Tensor] = []
    compute_path: list[Tensor] = []
    grid_path: list[Tensor] = []
    pv_used_path: list[Tensor] = []
    soc_path: list[Tensor] = [soc]
    critic_records: dict[str, list[Tensor]] = {}
    cached_battery_plan: Tensor | None = None
    cached_compute_plan: Tensor | None = None
    cached_battery: Tensor | None = None
    cached_compute: Tensor | None = None
    revealed_mask = torch.zeros(
        (batch_size * world_count, HORIZON),
        device=samples.device,
        dtype=torch.bool,
    )

    for origin in range(steps):
        revealed_mask[:, origin] = True
        backlog_total = queue.sum(1, keepdim=True)
        backlog = backlog_total.expand(-1, SITES) / float(SITES)
        pressure = torch.where(
            backlog_total > 0.0,
            queue[:, :1] / backlog_total.clamp_min(1.0e-12),
            torch.zeros_like(backlog_total),
        ).expand(-1, SITES)
        rolling_state = {
            "soc_mwh": soc,
            "current_soc": soc,
            "queue": queue,
            "backlog": backlog,
            "deadline_pressure": pressure,
            "previous_grid": previous_grid,
            "previous_battery": previous_battery,
            "previous_compute": previous_compute,
            "rt_innovation": rt_price[:, origin] - issue_market[:, origin],
            "workload_innovation": (
                arrival[:, origin, None] - issue_workload[:, origin]
            ),
            "pv_innovation": pv[:, origin] - issue_pv[:, origin],
            "revealed_mask": revealed_mask,
        }
        if origin in selected_set:
            output = _mapping(
                _call_supported(
                    forward_origin,
                    issue_context=context,
                    rolling_state=rolling_state,
                    state=rolling_state,
                    origin=origin,
                    limits=flat_limits,
                    issued_day_ahead=issued_flat,
                    revealed_zonal=samples_flat,
                    revealed_mask=revealed_mask,
                    detach_distribution_core=True,
                ),
                "forward_origin output",
            )
            reject_removed_actions(output)
            cached_battery, cached_compute = _current_action(output, origin)
            cached_battery_plan, cached_compute_plan = _plans(output)
            critic = _critic(output)
            if critic is not None:
                for name, value in critic.items():
                    if isinstance(value, Tensor) and value.ndim >= 1:
                        critic_records.setdefault(str(name), []).append(
                            value.reshape(batch_size, world_count, *value.shape[1:]).mean(1)
                        )
        else:
            if cached_battery_plan is not None:
                cached_battery = cached_battery_plan[:, origin]
            if cached_compute_plan is not None:
                cached_compute = cached_compute_plan[:, origin]
        if cached_battery is None or cached_compute is None:
            raise V7RolloutError("origin zero did not produce battery and compute actions")
        raw_battery = cached_battery
        raw_compute = cached_compute
        step = torch_shield_step(
            {"battery": raw_battery, "compute": raw_compute},
            {"soc_mwh": soc, "queue": queue},
            {
                "origin": origin,
                "arrival": arrival[:, origin],
                "pv_mw": pv[:, origin],
                "physical_multiplier": multiplier[:, origin],
            },
            frozen_limits,
            config,
        )
        da_now = issued_flat[:, origin]
        energy_cost = config.interval_hours * (
            (da_now * da_price[:, origin]).sum(1)
            + ((step["grid_mw"] - da_now) * rt_price[:, origin]).sum(1)
        )
        total_cost = total_cost + energy_cost + (
            config.interval_hours
            * config.battery_degradation_per_mwh
            * (step["charge"] + step["discharge"]).sum(1)
            + config.sla_penalty_per_work * step["sla_violation"]
        )
        total_sla = total_sla + step["sla_violation"]
        total_intervention = total_intervention + step["shield_intervention"]
        all_feasible = all_feasible & step["feasible"]
        raw_battery_path.append(raw_battery)
        raw_compute_path.append(raw_compute)
        battery_path.append(step["signed_battery"])
        compute_path.append(step["compute"])
        grid_path.append(step["grid_mw"])
        pv_used_path.append(step["pv_used_mw"])
        soc = step["soc_next_mwh"]
        queue = step["queue_next"]
        soc_path.append(soc)
        previous_grid = step["grid_mw"]
        previous_battery = step["signed_battery"]
        previous_compute = step["compute"]

    world_costs = total_cost.reshape(batch_size, world_count)
    arrival_total = arrival[:, :steps].sum(1).clamp_min(1.0e-8)
    normalized_sla = (total_sla / arrival_total).reshape(batch_size, world_count)
    intervention = (total_intervention / float(steps)).reshape(
        batch_size, world_count
    )
    reserve = (
        float(config.terminal_soc_fraction)
        * frozen_limits.torch(device=samples.device, dtype=samples.dtype)["battery_mwh"]
    )
    terminal_soc_gap = (reserve[None] - soc).clamp_min(0.0).sum(1).reshape(
        batch_size, world_count
    )
    terminal_backlog = queue.sum(1).reshape(batch_size, world_count)
    feasible = all_feasible.reshape(batch_size, world_count) & (
        terminal_soc_gap <= config.tolerance
    )

    baseline_world_costs = _fixed_baseline_costs(
        targets,
        reference=samples,
        batch=batch_size,
        worlds=world_count,
        horizon=steps,
        da_price=da_price,
        rt_price=rt_price,
        arrival=arrival,
        pv=pv,
        multiplier=multiplier,
        initial_soc=initial_soc,
        limits=frozen_limits,
        config=config,
    )
    if baseline_world_costs is None:
        raw_baseline = targets.get("baseline_realized_cost")
        baseline_cost = (
            None
            if raw_baseline is None
            else _tensor(raw_baseline, samples).reshape(batch_size)
            * (float(steps) / float(HORIZON))
        )
        baseline_cvar = baseline_cost
    else:
        baseline_cost = _weighted_mean(baseline_world_costs, weights)
        baseline_cvar = _weighted_upper_tail(
            baseline_world_costs, weights, cvar_alpha
        )

    def path(items: list[Tensor]) -> Tensor:
        return torch.stack(items, dim=1).reshape(
            batch_size, world_count, steps, SITES
        )

    origin_critic = {
        name: torch.stack(values, dim=1) for name, values in critic_records.items()
    }
    result: dict[str, Any] = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "shield_algebra_version": SHIELD_ALGEBRA_VERSION,
        "phase": None if phase is None else str(phase),
        "horizon": steps,
        "world_count": world_count,
        "weights": weights,
        "world_weights": weights,
        # joint_policy_loss_terms intentionally consumes this [day,world] tensor.
        "realized_cost": world_costs,
        "world_cost": world_costs,
        "world_costs": world_costs,
        "post_shield_cost": world_costs,
        "normalized_sla": normalized_sla,
        "sla_violation": normalized_sla,
        "sla_work": total_sla.reshape(batch_size, world_count),
        "terminal_soc_gap": terminal_soc_gap,
        "terminal_backlog": terminal_backlog,
        "shield_intervention": intervention,
        "feasible": feasible,
        "issued_day_ahead": issued_da,
        "raw_actions": {
            "day_ahead": issued_da,
            "battery": path(raw_battery_path),
            "compute": path(raw_compute_path),
        },
        "current_actions": {
            "battery": path(battery_path),
            "compute": path(compute_path),
            "grid_mw": path(grid_path),
            "pv_used_mw": path(pv_used_path),
        },
        "soc_path": torch.stack(soc_path, dim=1).reshape(
            batch_size, world_count, steps + 1, SITES
        ),
        "sampled_origins": selected,
        "reconditioned_origin_count": len(selected),
        "origin_critic": origin_critic,
        "distribution_detached": True,
        "day_ahead_issued_before_realized_prefix": True,
        "day_ahead_frozen": True,
        "current_actions_executed": steps,
        "suffix_actions_executed": 0,
        "free_action_schema": ("day_ahead", "battery", "compute"),
    }
    if baseline_cost is not None:
        result["baseline_cost"] = baseline_cost.detach()
    if baseline_cvar is not None:
        result["baseline_cvar"] = baseline_cvar.detach()
    if baseline_world_costs is not None:
        result["baseline_world_costs"] = baseline_world_costs
    reject_removed_actions(result)
    return result


def make_training_rollout(
    model: nn.Module,
    *,
    phase: str | None = None,
    horizon: int | None = None,
    shield_config: ShieldConfig | None = None,
    sampled_origins: Sequence[int] | None = None,
    origin_budget: int | None = 8,
    cvar_alpha: float = 0.95,
) -> Callable[..., Mapping[str, Any]]:
    """Build the callback consumed by :class:`V7JointTrainer`.

    ``horizon=None`` lets the trainer choose its 8/24/48/96 curriculum length
    through ``decision_horizon``.  Supplying ``horizon`` pins the callback for
    an isolated phase or test.
    """

    if horizon is not None and int(horizon) not in TRAINING_HORIZONS:
        raise V7RolloutError(f"horizon must be one of {TRAINING_HORIZONS}")

    def callback(
        *,
        batch: Mapping[str, Any],
        distribution: Mapping[str, Any],
        decision_horizon: int | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        resolved = horizon if horizon is not None else decision_horizon
        if resolved is None:
            raise V7RolloutError("training rollout requires a decision horizon")
        return differentiable_training_rollout(
            model,
            batch,
            _detach(distribution),
            horizon=int(resolved),
            phase=phase,
            shield_config=shield_config,
            sampled_origins=sampled_origins,
            origin_budget=origin_budget,
            cvar_alpha=cvar_alpha,
        )

    callback.__name__ = "r4_v7_shared_shield_training_rollout"
    return callback


__all__ = [
    "ROLLOUT_SCHEMA_VERSION",
    "TRAINING_HORIZONS",
    "V7RolloutError",
    "differentiable_training_rollout",
    "issue_time_batch",
    "make_training_rollout",
]
