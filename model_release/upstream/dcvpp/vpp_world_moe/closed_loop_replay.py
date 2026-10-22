"""Policy-specific state feedback and first-step physical execution.

The model sees prepared causal exogenous inputs plus its own controlled state
and action history. An execute call accepts only one current realization,
after a decision exists. Legacy cumulative-deficit expiry and irreversible
cohort on-time completion are deliberately different recorded quantities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Mapping

import numpy as np
import torch

from .packets import WorldInputPacket
from .r_ecmoe import RECMoEOutput, RiskConditionPacket
from .r_ecmoe_evaluation import HARD_RESIDUAL_EXCLUSIONS, _safe_raw
from .r_ecmoe_replay import _split_raw_structured_actions


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).copy()


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if is_dataclass(value): return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    return value


def _clone(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if is_dataclass(value): return replace(value, **{f.name: _clone(getattr(value, f.name)) for f in fields(value)})
    return value


@dataclass
class Cohort:
    released_at: int
    due_at: int
    original: np.ndarray
    remaining: np.ndarray
    ontime_completed: np.ndarray
    initial_backlog: bool = False


@dataclass
class EpisodeState:
    policy_id: str
    operating_date: str
    start_origin: int
    next_origin: int
    soc: np.ndarray
    queue: np.ndarray
    previous_grid: np.ndarray
    previous_previous_grid: np.ndarray
    initial_queue: np.ndarray
    cumulative_arrived_work: float = 0.0
    cumulative_served_work: float = 0.0
    cumulative_proxy_expired_work: float = 0.0
    cumulative_late_service_work: float = 0.0
    cohorts: list[Cohort] = field(default_factory=list)

    def to_dict(self):
        return _plain(self)


class ClosedLoopEpisode:
    """One policy's isolated episode; no training or full-day truth arguments."""

    def __init__(self, model, statistics, policy_id, operating_date, initial_packet,
                 *, historical_state_sequence=None, previous_grid=None,
                 previous_previous_grid=None):
        if initial_packet.fast_values.shape[0] != 1:
            raise ValueError("Closed-loop episodes require batch one")
        self.model, self.statistics = model, statistics
        self.deadline = int(model.config.deadline_intervals)
        origin = int(initial_packet.origin_index[0].detach().cpu())
        c = initial_packet.constraints
        if historical_state_sequence is not None:
            if origin < 2 or historical_state_sequence.shape[-1] < 16:
                raise ValueError("Two completed grid states are required at the episode boundary")
            # state[t] stores the action/outcome from completed interval t-1.
            previous_grid = _numpy(historical_state_sequence[origin, 12:16])
            previous_previous_grid = _numpy(historical_state_sequence[origin - 1, 12:16])
            boundary = "causal_passive_edf_state_sequence_completed_prefix"
        elif previous_grid is not None and previous_previous_grid is not None:
            boundary = "explicit_shared_boundary_grid_values"
        else:
            raise ValueError("Explicit common previous two grids are required; no silent zero initialization")
        soc, queue = _numpy(c.initial_soc[0]), _numpy(c.initial_queue[0])
        self.state = EpisodeState(str(policy_id), str(operating_date), origin, origin,
            soc, queue, _numpy(previous_grid), _numpy(previous_previous_grid), queue.copy())
        if any(value.shape != (4,) or not np.isfinite(value).all() for value in
               (soc, queue, self.state.previous_grid, self.state.previous_previous_grid)):
            raise ValueError("Invalid four-node initial state")
        if (queue < 0).any(): raise ValueError("Initial queue must be nonnegative")
        if queue.sum() > 0:
            self.state.cohorts.append(Cohort(origin, origin + self.deadline - 1,
                queue.copy(), queue.copy(), np.zeros(4), True))
        self.rows: list[dict[str, Any]] = []
        self._snapshots = {origin: self._snapshot()}
        self._grid_range = _numpy(c.grid_import_limit[0] + c.grid_export_limit[0])
        self._import_limit = _numpy(c.grid_import_limit[0])
        self._capacity = _numpy(c.compute_capacity[0])
        self._energy = _numpy(c.battery_energy_capacity[0])
        self.metadata = {
            "boundary_grid_source": boundary,
            "initial_backlog_age_assumption": "one_initial_cohort_due_at_start_plus_deadline_minus_one; historical_age_unavailable",
            "exogenous_feedback": "prepared_causal_exogenous_features_preserved; no current_or_future_truth_in_packet",
            "medium_slow_feedback": "7_completed_days_hourly_and_56_completed_days_daily_exclude_current_episode_day; preserved",
            "controlled_medium_history_updates": 0, "controlled_slow_history_updates": 0,
            "power_units": "MW", "soc_units": "MWh", "dt_units": "hours",
            "work_units": "registered_physical_work_units_not_task_counts",
            "cost_units": "raw_composite; market_prices_times_MW_times_hours; service_coefficients_require_separate_units",
            "true_v_rebound_definition": "positive_import_V_amplitude_over_sum_import_limit; includes_known_previous_two_interval_boundary",
        }

    def _deadline_fraction(self, origin):
        due = sum((cohort.remaining for cohort in self.state.cohorts if cohort.due_at <= origin), np.zeros(4))
        return np.divide(due, self.state.queue, out=np.zeros(4), where=self.state.queue > 1e-12)

    def _snapshot(self):
        return {"soc": self.state.soc.copy(), "queue": self.state.queue.copy(),
                "deadline_fraction": self._deadline_fraction(self.state.next_origin)}

    def _set_feature(self, packet, slot, feature, indices, raw):
        scale = self.statistics.fast_scale
        center = getattr(self.statistics, "fast_center", None)
        center = torch.zeros_like(scale) if center is None else center
        mean = center[slot, feature] if center.ndim == 2 else center[feature]
        spread = scale[slot, feature] if scale.ndim == 2 else scale[feature]
        values = torch.as_tensor(raw, dtype=packet.fast_values.dtype, device=packet.fast_values.device)
        values = (values - mean.to(values)) / spread.to(values).clamp_min(1e-6)
        old = packet.fast_values[0, indices, slot, feature]
        packet.fast_values[0, indices, slot, feature] = torch.where(packet.fast_mask[0, indices, slot], values, old)

    def prepare_packet(self, prepared_packet: WorldInputPacket) -> WorldInputPacket:
        origin = self.state.next_origin
        if int(prepared_packet.origin_index[0].detach().cpu()) != origin:
            raise ValueError("Prepared packet origin differs from episode state")
        packet = _clone(prepared_packet)
        device = packet.fast_values.device
        packet = replace(packet, constraints=replace(packet.constraints,
            initial_soc=torch.as_tensor(self.state.soc, dtype=torch.float32, device=device)[None],
            initial_queue=torch.as_tensor(self.state.queue, dtype=torch.float32, device=device)[None]))
        timeline = packet.fast_values.shape[1]
        start = self.state.start_origin
        for t in range(start, timeline):
            snapshot = self._snapshots[min(t, origin)]
            for z in range(4):
                slot = 8 * z
                fraction = snapshot["soc"][z] / max(self._energy[z], 1e-6)
                for feature, value in ((0, fraction), (1, snapshot["soc"][z]), (4, 1 - fraction)):
                    self._set_feature(packet, slot + 1, feature, t, value)
                self._set_feature(packet, slot + 5, 0, t, snapshot["queue"][z])
                self._set_feature(packet, slot + 5, 1, t, snapshot["deadline_fraction"][z])
                packet.constraint_slacks[0, t, z, 0] = float(np.clip(fraction, 0, 1))
                packet.constraint_slacks[0, t, z, 3] = float(np.clip(1 - snapshot["deadline_fraction"][z], 0, 1))
                packet.constraint_slacks[0, t, z, 5] = packet.constraint_slacks[0, t, z, 3]
        # Suppress stale passive-policy actions at unexecuted episode times.
        # Only the registered known channels 1/2/5 are rewritten; channels
        # 3/4 have no executed-history definition in the released corpus.
        for channel in (1, 2, 5):
            packet.previous_actions[0, start:, :, channel] = 0
            packet.action_mask[0, start:, :, channel] = False
        for row in self.rows:
            t = row["origin"]
            for channel, key in ((1, "actual_battery"), (2, "actual_compute"), (5, "actual_pv_curtailed")):
                packet.previous_actions[0, t, :, channel] = torch.as_tensor(row[key], dtype=torch.float32, device=device)
                packet.action_mask[0, t, :, channel] = True
            for z in range(4):
                slot, compute, grid = 8 * z, row["actual_compute"][z], row["actual_grid"][z]
                for feature, value in ((0, compute), (3, compute / max(self._capacity[z], 1e-6)),
                                       (4, max(self._capacity[z] - compute, 0))):
                    self._set_feature(packet, slot + 6, feature, t, value)
                for feature, value in ((0, grid), (2, max(self._import_limit[z] - grid, 0)),
                                       (3, grid / max(self._import_limit[z], 1e-6))):
                    self._set_feature(packet, slot + 7, feature, t, value)
            self._set_feature(packet, 36, 0, t, np.maximum(self._import_limit - row["actual_grid"], 0).mean())
            self._set_feature(packet, 36, 1, t, np.maximum(self._capacity - row["actual_compute"], 0).mean())
        latest_compute = np.asarray(self.rows[-1]["actual_compute"]) if self.rows else None
        for t in range(start, timeline):
            if t < origin:
                row = self.rows[t - start]
                grid, compute = np.asarray(row["actual_grid"]), np.asarray(row["actual_compute"])
            elif latest_compute is not None:
                grid, compute = self.state.previous_grid, latest_compute
            else:
                continue
            packet.constraint_slacks[0, t, :, 1] = torch.as_tensor(
                np.maximum(self._import_limit - grid, 0) / np.maximum(self._import_limit, 1e-6), device=device)
            packet.constraint_slacks[0, t, :, 2] = torch.as_tensor(
                np.maximum(self._capacity - compute, 0) / np.maximum(self._capacity, 1e-6), device=device)
        self._set_feature(packet, 37, 6, slice(None), self.state.queue.sum() / max(self._capacity.sum(), 1e-6))
        for feature, channel in enumerate((0, 1, 2)):
            self._set_feature(packet, 38, feature, slice(None), packet.previous_actions[0, :, :, channel].mean(-1))
        if packet.dynamic_graph_edge_features is not None:
            normal = packet.graph_edge_type != 5  # data.EDGE_MIGRATION_UNAVAILABLE
            source, target = packet.graph_edge_index[:, normal]
            source_value = packet.fast_values[:, :, source, 0]
            target_value = packet.fast_values[:, :, target, 0]
            enabled = packet.dynamic_graph_edge_mask[:, :, normal]
            packet.dynamic_graph_edge_features[:, :, normal, 4] = torch.where(enabled, source_value, 0)
            packet.dynamic_graph_edge_features[:, :, normal, 5] = torch.where(enabled, target_value - source_value, 0)
        # Medium/slow are prior COMPLETE days. None of this intraday episode is
        # represented there; replacing them with present state would corrupt
        # their chronology. Their values and availability masks stay identical.
        return packet

    def mpc_history(self):
        profile = np.zeros(96)
        for cohort in self.state.cohorts:
            if cohort.due_at < 96:
                profile[max(cohort.due_at, 0):] += cohort.original.sum()
        return {"known_cumulative_due_profile": profile,
                "served_before": self.state.cumulative_served_work,
                "expired_before": self.state.cumulative_proxy_expired_work,
                "previous_grid": self.state.previous_grid.copy(),
                "previous_previous_grid": self.state.previous_previous_grid.copy()}

    def _actual_reference(self, expert_packet, actual_current):
        current = torch.as_tensor(actual_current, dtype=torch.float32,
                                  device=expert_packet.state_latent.device).detach()
        if current.shape == (1, 21): current = current[0]
        if current.shape != (21,) or not bool(torch.isfinite(current).all()):
            raise ValueError("Execute accepts only the one current finite 21-channel realization")
        if bool((current[13:17] < 0).any()) or bool((current[17:21] < 1).any()):
            raise ValueError("Current PV/PUE violates physical support")
        origin = self.state.next_origin
        reference = expert_packet.reference
        values = {name: getattr(reference, name).detach().clone() for name in
                  ("da", "rt", "dart", "arrivals", "pv", "pue", "public_21")}
        capacity = expert_packet.constraints.compute_capacity[0].float()
        arrivals = current[12].clamp_min(0) * capacity / capacity.sum().clamp_min(1e-8)
        for name, value in (("da", current[:4]), ("rt", current[4:8]),
                            ("dart", current[:4] - current[4:8]), ("arrivals", arrivals),
                            ("pv", current[13:17]), ("pue", current[17:21])):
            values[name][0, origin] = value
        canonical = current.clone(); canonical[8:12] = current[:4] - current[4:8]
        values["public_21"][0, origin] = canonical
        return replace(expert_packet, reference=replace(reference, **values)), current, arrivals

    def execute_neural(self, output: RECMoEOutput, risk: RiskConditionPacket, actual_current, extra=None):
        return self._execute(output.expert_packet, risk, output.raw_structured_actions,
                             actual_current, int(risk.horizon[0].detach().cpu()), "neural_full_plan_hybrid", extra)

    def execute_mpc(self, expert_packet, risk, first_action, actual_current, extra=None):
        raw = _safe_raw(expert_packet)
        t = self.state.next_origin
        c = expert_packet.constraints
        scale = float(risk.action_range[0].detach().cpu())
        if scale <= 0: raise ValueError("A positive inference action range is required")
        device = expert_packet.state_latent.device
        value = lambda key: torch.as_tensor(first_action[key], dtype=torch.float32, device=device)
        logit = lambda x: torch.logit(x.clamp(1e-6, 1 - 1e-6))
        raw["battery"][0, t] = torch.atanh((value("battery") / c.battery_power_limit[0].clamp_min(1e-8)).clamp(-1 + 1e-6, 1 - 1e-6)) / scale
        raw["compute"][0, t] = logit(value("compute") / c.compute_capacity[0].clamp_min(1e-8)) / scale
        raw["pv"][0, t] = (logit(value("pv_fraction")) - 2) / scale
        raw["target_soc"][0, t] = logit(value("target_soc") / c.battery_energy_capacity[0].clamp_min(1e-8)) / scale
        envelope = float(self.model.config.action_residual_fraction) * c.grid_import_limit[0]
        delta = value("day_ahead") - expert_packet.previous_actions[0, t, :, 0]
        raw["day_ahead"][0, t] = torch.atanh((delta / envelope.clamp_min(1e-8)).clamp(-1 + 1e-6, 1 - 1e-6)) / scale
        one = replace(risk, horizon=torch.ones_like(risk.horizon))
        return self._execute(expert_packet, one, raw["joined"], actual_current,
                             int(risk.horizon[0].detach().cpu()), "mpc_fixed_first_action", extra)

    def _execute(self, expert_packet, risk, joined, actual_current, prediction_horizon, execution_rule, extra):
        t = self.state.next_origin
        if expert_packet.state_latent.shape[0] != 1:
            raise ValueError("Execute requires one policy and one sample")
        if int(expert_packet.origin_index[0].detach().cpu()) != t:
            raise ValueError("Decision origin is stale")
        for name, expected in (("initial_soc", self.state.soc), ("initial_queue", self.state.queue)):
            if not np.allclose(_numpy(getattr(expert_packet.constraints, name)[0]), expected, atol=1e-6, rtol=1e-6):
                raise ValueError(f"Decision did not receive its own current {name}")
        packet, current, arrivals = self._actual_reference(expert_packet, actual_current)
        assembled = self.model.assembler(_split_raw_structured_actions(joined), packet, risk)
        if float(assembled.actions.migration[0, t].abs().max().detach().cpu()) > 1e-8:
            raise ValueError("This closed-loop ledger requires the registered zero-migration experiment")
        action = assembled.actions
        battery, compute, grid, pv_used, pv_curtailed, unserved = (
            _numpy(getattr(action, name)[0, t]) for name in
            ("battery", "compute", "grid", "pv_used", "pv_curtailed", "unserved_power"))
        q_da = _numpy(assembled.day_ahead_commitment[0, t])
        soc, queue = _numpy(action.soc[0, t + 1]), _numpy(action.queue[0, t + 1])
        arrivals = _numpy(arrivals)
        before_grid, before2 = self.state.previous_grid.copy(), self.state.previous_previous_grid.copy()
        self.state.cohorts.append(Cohort(t, t + self.deadline - 1, arrivals.copy(), arrivals.copy(), np.zeros(4)))
        service = compute.copy()
        for cohort in self.state.cohorts:
            allocated = np.minimum(cohort.remaining, np.maximum(service, 0))
            cohort.remaining -= allocated; service -= allocated
            if t <= cohort.due_at: cohort.ontime_completed += allocated
            else: self.state.cumulative_late_service_work += float(allocated.sum())
        ledger_queue = sum((cohort.remaining for cohort in self.state.cohorts), np.zeros(4))
        ledger_error = float(np.max(np.abs(ledger_queue - queue)))
        if not np.allclose(ledger_queue, queue, atol=2e-5, rtol=2e-5):
            raise ValueError("FIFO retained backlog does not match physical queue conservation")
        self.state.cumulative_arrived_work += float(arrivals.sum())
        self.state.cumulative_served_work += float(compute.sum())
        due = sum(float(cohort.original.sum()) for cohort in self.state.cohorts if cohort.due_at <= t)
        proxy = max(self.state.cumulative_proxy_expired_work, due - self.state.cumulative_served_work, 0)
        delta_expiry = proxy - self.state.cumulative_proxy_expired_work
        self.state.cumulative_proxy_expired_work = proxy
        dt = float(packet.constraints.dt_hours[0].detach().cpu())
        cfg = self.model.world_model.config
        market = float(((_numpy(current[:4]) * q_da + _numpy(current[4:8]) * (grid - q_da)) * dt).sum())
        degradation = float(cfg.battery_degradation_weight) * float(np.abs(battery).sum()) * dt
        curtailed_energy = float(pv_curtailed.sum()) * dt
        unserved_energy = float(unserved.sum()) * dt
        cumulative_unserved_power = float(unserved.sum()) + sum(float(np.asarray(row["actual_unserved_power"]).sum()) for row in self.rows)
        expiry_fraction = proxy / max(float(self.state.initial_queue.sum()) + self.state.cumulative_arrived_work, 1e-6)
        unserved_fraction = cumulative_unserved_power / max((len(self.rows) + 1) * self._grid_range.sum(), 1e-6)
        energy_service = curtailed_energy + float(cfg.unserved_power_weight) * unserved_energy
        expiry_cost = float(cfg.unserved_power_weight) * delta_expiry
        grid_range = np.maximum(self._grid_range, 1e-6)
        local_ramp = float((np.abs(grid - before_grid) / grid_range).mean())
        curvature = max(float(grid.sum() - 2 * before_grid.sum() + before2.sum()), 0) / max(self._grid_range.sum(), 1e-6)
        positive = lambda x: float(np.maximum(x, 0).sum())
        v_rebound = min(max(positive(before2) - positive(before_grid), 0), max(positive(grid) - positive(before_grid), 0)) / max(self._import_limit.sum(), 1e-6)
        residuals = {}
        for name, residual in assembled.residuals.items():
            value = residual[0]
            if value.ndim and value.shape[0] == joined.shape[1] + 1: value = value[t + 1]
            elif value.ndim and value.shape[0] == joined.shape[1]: value = value[t]
            residuals[name] = float(value.detach().float().abs().max().cpu())
        hard = max((value for name, value in residuals.items() if name not in HARD_RESIDUAL_EXCLUSIONS), default=0)
        self.state.soc, self.state.queue = soc, queue
        self.state.previous_previous_grid, self.state.previous_grid = before_grid, grid
        self.state.next_origin = t + 1
        self._snapshots[t + 1] = self._snapshot()
        cohort = self._cohort_metrics(t)
        row = {"policy_id": self.state.policy_id, "operating_date": self.state.operating_date, "origin": t,
            "prediction_horizon": prediction_horizon, "execution_rule": execution_rule,
            "actual_arrivals": arrivals, "actual_compute": compute, "actual_battery": battery,
            "actual_grid": grid, "actual_pv_used": pv_used, "actual_pv_curtailed": pv_curtailed,
            "actual_unserved_power": unserved, "actual_day_ahead": q_da,
            "soc_after": soc, "queue_after": queue, "previous_grid": before_grid, "previous_previous_grid": before2,
            "arrived_work": float(arrivals.sum()), "served_work": float(compute.sum()),
            "expiry_proxy_increment": delta_expiry, "expiry_proxy_cumulative": proxy,
            "legacy_expiry_work_fraction": expiry_fraction,
            "legacy_compound_sla_proxy": 0.5 * (expiry_fraction + unserved_fraction),
            "unserved_power_fraction": unserved_fraction,
            "compound_sla_aggregation": "full_executed_episode; expiry_fraction_plus_mean_normalized_unserved_power",
            "market_cost": market, "battery_degradation_cost": degradation,
            "energy_service_cost": energy_service, "expiry_proxy_cost": expiry_cost,
            "raw_cost": market + degradation + energy_service + expiry_cost,
            "unserved_energy": unserved_energy, "pv_curtailed_energy": curtailed_energy,
            "local_ramp": local_ramp, "recovery_curvature": curvature, "true_v_rebound": v_rebound,
            "hard_residual_max": hard, "hard_constraint_violation": hard > 1e-4,
            "constraint_residuals": residuals, "dt_hours": dt,
            "cohort_queue_conservation_max_error": ledger_error,
            "future_realizations_used": False, **cohort, **self.metadata}
        row = _plain(row)
        if extra: row.update(extra)
        self.rows.append(row)
        return row

    def _cohort_metrics(self, completed_origin):
        matured = [c for c in self.state.cohorts if c.due_at <= completed_origin]
        censored = [c for c in self.state.cohorts if c.due_at > completed_origin]
        denominator = sum(float(c.original.sum()) for c in matured)
        numerator = sum(float(c.ontime_completed.sum()) for c in matured)
        return {"cohort_matured_work": denominator, "cohort_ontime_completed_work": numerator,
            "cohort_ontime_work_completion_ratio": numerator / denominator if denominator > 0 else None,
            "cohort_censored_work": sum(float(c.original.sum()) for c in censored),
            "cohort_censored_remaining_work": sum(float(c.remaining.sum()) for c in censored),
            "cohort_matured_late_work": denominator - numerator,
            "cohort_late_remaining_work": sum(float(c.remaining.sum()) for c in matured),
            "cohort_cumulative_late_service_work": self.state.cumulative_late_service_work}

    def summary(self):
        totals = {name: sum(float(row[name]) for row in self.rows) for name in
            ("raw_cost", "market_cost", "battery_degradation_cost", "energy_service_cost", "expiry_proxy_cost",
             "arrived_work", "served_work", "expiry_proxy_increment", "unserved_energy", "pv_curtailed_energy")}
        result = {"policy_id": self.state.policy_id, "operating_date": self.state.operating_date,
            "steps": len(self.rows), "start_origin": self.state.start_origin,
            "stop_origin_exclusive": self.state.next_origin, **totals,
            "final_soc": self.state.soc, "final_queue": self.state.queue,
            "expiry_proxy_cumulative": self.state.cumulative_proxy_expired_work,
            "legacy_compound_sla_proxy": self.rows[-1]["legacy_compound_sla_proxy"] if self.rows else 0.0,
            "legacy_expiry_work_fraction": self.rows[-1]["legacy_expiry_work_fraction"] if self.rows else 0.0,
            "max_hard_residual": max((row["hard_residual_max"] for row in self.rows), default=0),
            "hard_violation_count": sum(row["hard_constraint_violation"] for row in self.rows),
            **self._cohort_metrics(self.state.next_origin - 1), **self.metadata}
        for name in ("local_ramp", "recovery_curvature", "true_v_rebound"):
            result[f"mean_{name}"] = sum(float(row[name]) for row in self.rows) / len(self.rows) if self.rows else None
        result["max_true_v_rebound"] = max((float(row["true_v_rebound"]) for row in self.rows), default=0.0)
        return _plain(result)


__all__ = ["ClosedLoopEpisode", "EpisodeState", "Cohort"]
