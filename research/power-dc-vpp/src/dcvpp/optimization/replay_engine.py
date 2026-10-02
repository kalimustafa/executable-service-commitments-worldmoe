from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from dcvpp.core import atomic_json, sha256_json
from dcvpp.optimization.stochastic_milp import (
    FrozenDayAheadPlan,
    SolvePolicy,
    array_sha256,
    solve_recourse_with_frozen_plan,
)
from dcvpp.research_branch.contracts import (
    FrozenPhysicalArtifact,
    NYISO_ZONES,
    ScenarioBundle,
)
from dcvpp.research_branch.optimization import solve_policy
from dcvpp.research_branch.rolling import ROLLING_INTERVAL_HOURS, _first_interval_dispatch


REPLAY_SCHEMA_VERSION = "dart_external_replay_v1"
NYISO_SOURCE_RELATIONSHIP = "third_party_referenced_public_data"


def _frozen_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _source_hashes(value: Mapping[str, str], *, name: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, raw_digest in sorted(value.items()):
        digest = str(raw_digest).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name}[{key!r}] is not a SHA-256 digest")
        output[str(key)] = digest
    if not output:
        raise ValueError(f"{name} cannot be empty")
    return output


def canonical_dart_da_minus_rt(da_lbmp: Any, rt_lbmp: Any) -> np.ndarray:
    """Canonical paper sign: DART = DA - RT."""

    da = np.asarray(da_lbmp, dtype=float)
    rt = np.asarray(rt_lbmp, dtype=float)
    if da.shape != rt.shape:
        raise ValueError("DA and RT arrays must have identical shapes")
    return da - rt


def inverse_rt_spread_rt_minus_da(da_lbmp: Any, rt_lbmp: Any) -> np.ndarray:
    """Legacy diagnostic sign retained explicitly: RT spread = RT - DA."""

    da = np.asarray(da_lbmp, dtype=float)
    rt = np.asarray(rt_lbmp, dtype=float)
    if da.shape != rt.shape:
        raise ValueError("DA and RT arrays must have identical shapes")
    return rt - da


def validate_dart_inverse_gate(
    da_lbmp: Any,
    rt_lbmp: Any,
    *,
    dart_da_minus_rt: Any | None = None,
    rt_spread_rt_minus_da: Any | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fail if the canonical DART and inverse spread conventions diverge."""

    canonical = canonical_dart_da_minus_rt(da_lbmp, rt_lbmp)
    inverse = inverse_rt_spread_rt_minus_da(da_lbmp, rt_lbmp)
    supplied_dart = canonical if dart_da_minus_rt is None else np.asarray(dart_da_minus_rt, dtype=float)
    supplied_spread = (
        inverse
        if rt_spread_rt_minus_da is None
        else np.asarray(rt_spread_rt_minus_da, dtype=float)
    )
    if supplied_dart.shape != canonical.shape or supplied_spread.shape != canonical.shape:
        raise ValueError("supplied DART/spread arrays have incompatible shapes")
    residual = float(
        max(
            np.max(np.abs(supplied_dart - canonical), initial=0.0),
            np.max(np.abs(supplied_spread - inverse), initial=0.0),
            np.max(np.abs(supplied_dart + supplied_spread), initial=0.0),
        )
    )
    if residual > tolerance:
        raise ValueError(
            "DART sign gate failed: require dart_da_minus_rt=DA-RT and "
            "rt_spread_rt_minus_da=RT-DA"
        )
    return {
        "passed": True,
        "canonical_definition": "dart_da_minus_rt = da_lbmp - rt_lbmp",
        "inverse_definition": "rt_spread_rt_minus_da = rt_lbmp - da_lbmp",
        "max_abs_residual": residual,
    }


def rt_from_fixed_da_and_dart(fixed_da_lbmp: Any, dart_da_minus_rt: Any) -> np.ndarray:
    """A1 identity: RT = fixed_DA - DART."""

    fixed_da = np.asarray(fixed_da_lbmp, dtype=float)
    dart = np.asarray(dart_da_minus_rt, dtype=float)
    try:
        rt = fixed_da - dart
    except ValueError as exc:
        raise ValueError("fixed DA and DART arrays are not broadcast-compatible") from exc
    validate_dart_inverse_gate(np.broadcast_to(fixed_da, rt.shape), rt, dart_da_minus_rt=dart)
    return rt


@dataclass(frozen=True)
class ReferencedNYISOReplay:
    """External historical NYISO settlement path; never a forecast input."""

    operating_date: str
    timestamps_utc: np.ndarray
    da_lbmp: np.ndarray
    rt_lbmp: np.ndarray
    source_hashes: Mapping[str, str]
    zones: tuple[str, ...] = NYISO_ZONES
    source_relationship: str = NYISO_SOURCE_RELATIONSHIP
    evidence_role: str = "observed"
    measurement_relationship: str = "source_level_historical_record_not_investigator_measurement"

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_utc)
        if timestamps.ndim != 1 or timestamps.size == 0 or len(np.unique(timestamps)) != len(timestamps):
            raise ValueError("timestamps_utc must be nonempty and unique")
        if tuple(self.zones) != NYISO_ZONES:
            raise ValueError(f"zones must be exactly {NYISO_ZONES}")
        shape = (len(self.zones), len(timestamps))
        da = _frozen_array(self.da_lbmp, name="da_lbmp", shape=shape)
        rt = _frozen_array(self.rt_lbmp, name="rt_lbmp", shape=shape)
        if self.source_relationship != NYISO_SOURCE_RELATIONSHIP or self.evidence_role != "observed":
            raise ValueError("historical NYISO replay must retain referenced-public observed semantics")
        timestamps = np.array(timestamps, copy=True)
        timestamps.setflags(write=False)
        object.__setattr__(self, "timestamps_utc", timestamps)
        object.__setattr__(self, "da_lbmp", da)
        object.__setattr__(self, "rt_lbmp", rt)
        object.__setattr__(self, "source_hashes", _source_hashes(self.source_hashes, name="source_hashes"))
        validate_dart_inverse_gate(da, rt)

    @property
    def dart_da_minus_rt(self) -> np.ndarray:
        output = canonical_dart_da_minus_rt(self.da_lbmp, self.rt_lbmp)
        output.setflags(write=False)
        return output

    @property
    def input_sha256(self) -> str:
        return sha256_json(
            {
                "operating_date": self.operating_date,
                "timestamps": array_sha256(self.timestamps_utc),
                "da_lbmp": array_sha256(self.da_lbmp),
                "rt_lbmp": array_sha256(self.rt_lbmp),
                "source_hashes": dict(self.source_hashes),
                "source_relationship": self.source_relationship,
                "evidence_role": self.evidence_role,
            }
        )


@dataclass(frozen=True)
class FrozenReplayForcing:
    """Candidate-independent realized workload/PV/physical replay forcing."""

    operating_date: str
    timestamps_utc: np.ndarray
    workload: np.ndarray
    pv: np.ndarray
    physical_multiplier: np.ndarray
    source_hashes: Mapping[str, str]
    zones: tuple[str, ...] = NYISO_ZONES
    evidence_roles: Mapping[str, str] = field(
        default_factory=lambda: {
            "workload": "replayed",
            "pv": "reconstructed",
            "physical_multiplier": "counterfactual",
        }
    )
    frozen: bool = True

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_utc)
        if timestamps.ndim != 1 or timestamps.size == 0 or len(np.unique(timestamps)) != len(timestamps):
            raise ValueError("timestamps_utc must be nonempty and unique")
        if tuple(self.zones) != NYISO_ZONES:
            raise ValueError(f"zones must be exactly {NYISO_ZONES}")
        if not self.frozen:
            raise ValueError("replay forcing must be frozen before candidate evaluation")
        horizon = len(timestamps)
        workload = _frozen_array(self.workload, name="workload", shape=(horizon,))
        pv = _frozen_array(self.pv, name="pv", shape=(len(self.zones), horizon))
        multiplier = _frozen_array(
            self.physical_multiplier,
            name="physical_multiplier",
            shape=(len(self.zones),),
        )
        if (workload < 0).any() or (pv < 0).any() or (multiplier <= 0).any():
            raise ValueError("workload/PV must be nonnegative and physical multipliers positive")
        timestamps = np.array(timestamps, copy=True)
        timestamps.setflags(write=False)
        object.__setattr__(self, "timestamps_utc", timestamps)
        object.__setattr__(self, "workload", workload)
        object.__setattr__(self, "pv", pv)
        object.__setattr__(self, "physical_multiplier", multiplier)
        object.__setattr__(self, "source_hashes", _source_hashes(self.source_hashes, name="source_hashes"))

    @property
    def input_sha256(self) -> str:
        return sha256_json(
            {
                "operating_date": self.operating_date,
                "timestamps": array_sha256(self.timestamps_utc),
                "workload": array_sha256(self.workload),
                "pv": array_sha256(self.pv),
                "physical_multiplier": array_sha256(self.physical_multiplier),
                "source_hashes": dict(self.source_hashes),
                "evidence_roles": dict(self.evidence_roles),
                "frozen": self.frozen,
            }
        )


def validate_replay_alignment(
    issue_bundle: ScenarioBundle,
    settlement: ReferencedNYISOReplay,
    forcing: FrozenReplayForcing,
) -> None:
    issue_bundle.validate()
    if not (issue_bundle.operating_date == settlement.operating_date == forcing.operating_date):
        raise ValueError("issue-time, settlement, and forcing operating dates differ")
    if tuple(issue_bundle.zones) != tuple(settlement.zones) or tuple(issue_bundle.zones) != tuple(
        forcing.zones
    ):
        raise ValueError("issue-time, settlement, and forcing zones differ")
    issue_times = np.asarray(issue_bundle.timestamps_utc).astype("datetime64[ns]")
    settlement_times = np.asarray(settlement.timestamps_utc).astype("datetime64[ns]")
    forcing_times = np.asarray(forcing.timestamps_utc).astype("datetime64[ns]")
    if not np.array_equal(issue_times, settlement_times) or not np.array_equal(
        issue_times, forcing_times
    ):
        raise ValueError("issue-time, settlement, and forcing timestamps differ")


def build_a1_dart_bundle(
    issue_bundle: ScenarioBundle,
    *,
    fixed_da_lbmp: Any,
    dart_scenarios_da_minus_rt: Any,
) -> ScenarioBundle:
    """Replace only DART/derived RT; all non-price blocks are byte-identical."""

    issue_bundle.validate()
    fixed_da = np.asarray(fixed_da_lbmp, dtype=float)
    if fixed_da.shape != issue_bundle.da_price.shape:
        raise ValueError("fixed DA shape differs from the issue-time bundle")
    dart = np.asarray(dart_scenarios_da_minus_rt, dtype=float)
    if dart.shape != issue_bundle.rt_price.shape:
        raise ValueError("DART scenario shape differs from the issue-time RT scenario shape")
    rt = rt_from_fixed_da_and_dart(fixed_da[None, :, :], dart)
    output = replace(
        issue_bundle,
        da_price=np.array(fixed_da, copy=True),
        rt_price=np.array(rt, copy=True),
    )
    output.validate()
    for name in ("workload", "pv", "physical_multiplier", "probabilities"):
        if array_sha256(getattr(output, name)) != array_sha256(getattr(issue_bundle, name)):
            raise AssertionError(f"A1 changed forbidden block {name}")
    return output


def build_causal_recourse_bundle(
    issue_bundle: ScenarioBundle,
    settlement: ReferencedNYISOReplay,
    forcing: FrozenReplayForcing,
    *,
    interval: int,
) -> tuple[ScenarioBundle, dict[str, Any]]:
    """Reveal exactly the current outcome; retain issue-time future ensembles."""

    validate_replay_alignment(issue_bundle, settlement, forcing)
    horizon = issue_bundle.rt_price.shape[2]
    if not 0 <= interval < horizon:
        raise ValueError("interval is outside the replay horizon")
    rt = np.array(issue_bundle.rt_price[:, :, interval:], copy=True)
    workload = np.array(issue_bundle.workload[:, interval:], copy=True)
    pv = np.array(issue_bundle.pv[:, :, interval:], copy=True)
    rt[:, :, 0] = settlement.rt_lbmp[:, interval]
    workload[:, 0] = forcing.workload[interval]
    pv[:, :, 0] = forcing.pv[:, interval]
    suffix = replace(
        issue_bundle,
        timestamps_utc=np.array(issue_bundle.timestamps_utc[interval:], copy=True),
        da_price=np.array(issue_bundle.da_price[:, interval:], copy=True),
        rt_price=rt,
        workload=workload,
        pv=pv,
        physical_multiplier=np.array(issue_bundle.physical_multiplier, copy=True),
    )
    suffix.validate()
    future_hashes = {
        "rt_price": array_sha256(issue_bundle.rt_price[:, :, interval + 1 :]),
        "workload": array_sha256(issue_bundle.workload[:, interval + 1 :]),
        "pv": array_sha256(issue_bundle.pv[:, :, interval + 1 :]),
        "physical_multiplier": array_sha256(issue_bundle.physical_multiplier),
    }
    replay_future_hashes = {
        "rt_price": array_sha256(suffix.rt_price[:, :, 1:]),
        "workload": array_sha256(suffix.workload[:, 1:]),
        "pv": array_sha256(suffix.pv[:, :, 1:]),
        "physical_multiplier": array_sha256(suffix.physical_multiplier),
    }
    if replay_future_hashes != future_hashes:
        raise AssertionError("causal replay changed future issue-time ensembles")
    audit = {
        "revealed_through_interval": interval,
        "future_realized_targets_used": False,
        "future_issue_time_hashes": future_hashes,
        "replay_future_hashes": replay_future_hashes,
        "external_da_used_for_settlement_only": True,
        "current_reveal_sha256": sha256_json(
            {
                "rt": settlement.rt_lbmp[:, interval].tolist(),
                "workload": float(forcing.workload[interval]),
                "pv": forcing.pv[:, interval].tolist(),
            }
        ),
    }
    return suffix, audit


def settle_realized_components(
    q_da_mw: Any,
    grid_import_mw: Any,
    da_lbmp: Any,
    rt_lbmp: Any,
    *,
    interval_hours: float = ROLLING_INTERVAL_HOURS,
    rt_sell_haircut_fraction: float = 0.0,
    component_costs_usd: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Recompute referenced DA/RT settlement plus declared operating costs."""

    q_da = np.asarray(q_da_mw, dtype=float)
    grid = np.asarray(grid_import_mw, dtype=float)
    da = np.asarray(da_lbmp, dtype=float)
    rt = np.asarray(rt_lbmp, dtype=float)
    if not (q_da.shape == grid.shape == da.shape == rt.shape):
        raise ValueError("q_DA, grid, DA, and RT arrays must have identical shapes")
    if not all(np.isfinite(value).all() for value in (q_da, grid, da, rt)):
        raise ValueError("settlement inputs must be finite")
    if (q_da < 0).any() or (grid < 0).any() or interval_hours <= 0:
        raise ValueError("power must be nonnegative and interval_hours positive")
    if not 0.0 <= rt_sell_haircut_fraction <= 1.0:
        raise ValueError("rt_sell_haircut_fraction must lie in [0, 1]")
    deviation = grid - q_da
    purchase = np.maximum(deviation, 0.0)
    sale = np.maximum(-deviation, 0.0)
    sell_price = rt - rt_sell_haircut_fraction * np.abs(rt)
    da_cost = interval_hours * float(np.sum(q_da * da))
    rt_cost = interval_hours * float(np.sum(purchase * rt - sale * sell_price))
    reserved = {
        "day_ahead_cost_usd",
        "rt_imbalance_cost_usd",
        "simulated_total_cost_usd",
    }
    components: dict[str, float] = {}
    for key, raw_value in sorted((component_costs_usd or {}).items()):
        if key in reserved:
            raise ValueError(f"component hook cannot overwrite reserved field {key}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"component cost {key} is not finite")
        components[str(key)] = value
    result = {
        "day_ahead_cost_usd": da_cost,
        "rt_imbalance_cost_usd": rt_cost,
        **components,
    }
    result["simulated_total_cost_usd"] = float(sum(result.values()))
    result["settlement_identity_residual_mw"] = float(
        np.max(np.abs(deviation - (purchase - sale)), initial=0.0)
    )
    return result


ComponentCostHook = Callable[[int, Mapping[str, Any]], Mapping[str, float]]


def _external_replay_fingerprint(
    issue_bundle: ScenarioBundle,
    plan: FrozenDayAheadPlan,
    settlement: ReferencedNYISOReplay,
    forcing: FrozenReplayForcing,
    config: Mapping[str, Any],
    recourse_solver: SolvePolicy,
    component_cost_hook: ComponentCostHook | None,
) -> str:
    def callable_identity(value: Any) -> dict[str, str] | None:
        if value is None:
            return None
        return {
            "module": str(getattr(value, "__module__", "unknown")),
            "qualified_name": str(
                getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
            ),
        }

    return sha256_json(
        {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "settlement_sha256": settlement.input_sha256,
            "forcing_sha256": forcing.input_sha256,
            "issue_bundle": {
                "rt": array_sha256(issue_bundle.rt_price),
                "workload": array_sha256(issue_bundle.workload),
                "pv": array_sha256(issue_bundle.pv),
                "physical": array_sha256(issue_bundle.physical_multiplier),
                "probabilities": array_sha256(issue_bundle.probabilities),
            },
            "config": dict(config),
            "recourse_solver": callable_identity(recourse_solver),
            "component_cost_hook": callable_identity(component_cost_hook),
        }
    )


def run_external_replay_day(
    issue_bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    plan: FrozenDayAheadPlan,
    settlement: ReferencedNYISOReplay,
    forcing: FrozenReplayForcing,
    *,
    config: Mapping[str, Any] | None = None,
    checkpoint_path: Path | None = None,
    max_intervals: int | None = None,
    recourse_solver: SolvePolicy = solve_policy,
    component_cost_hook: ComponentCostHook | None = None,
) -> dict[str, Any]:
    """Run/resume a causal historical replay with a pre-frozen first stage."""

    validate_replay_alignment(issue_bundle, settlement, forcing)
    if plan.operating_date != issue_bundle.operating_date:
        raise ValueError("frozen plan operating date differs from the replay day")
    if tuple(plan.zones) != tuple(issue_bundle.zones) or tuple(plan.timestamps_utc) != tuple(
        str(value) for value in issue_bundle.timestamps_utc
    ):
        raise ValueError("frozen plan grid differs from the replay day")
    if max_intervals is not None and max_intervals < 0:
        raise ValueError("max_intervals must be nonnegative or null")
    settings = dict(config or {})
    fingerprint = _external_replay_fingerprint(
        issue_bundle,
        plan,
        settlement,
        forcing,
        settings,
        recourse_solver,
        component_cost_hook,
    )
    checkpoint: dict[str, Any]
    resumed = False
    if checkpoint_path is not None and checkpoint_path.is_file():
        import json

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ValueError("external replay checkpoint schema does not match")
        if checkpoint.get("input_sha256") != fingerprint:
            raise ValueError("external replay checkpoint input fingerprint does not match")
        resumed = True
    else:
        initial_soc = np.asarray(
            settings.get("initial_soc_mwh", 0.5 * np.asarray(physical.battery_energy_mwh)),
            dtype=float,
        )
        if initial_soc.shape != (len(issue_bundle.zones),):
            raise ValueError("initial_soc_mwh must contain one value per zone")
        checkpoint = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "status": "IN_PROGRESS",
            "operating_date": issue_bundle.operating_date,
            "candidate_id": plan.candidate_id,
            "experiment_id": plan.experiment_id,
            "seed": plan.seed,
            "input_sha256": fingerprint,
            "plan_sha256": plan.plan_sha256,
            "settlement_sha256": settlement.input_sha256,
            "forcing_sha256": forcing.input_sha256,
            "evidence_hashes": {
                "first_stage": plan.plan_sha256,
                "nyiso_replay": settlement.input_sha256,
                "replay_forcing": forcing.input_sha256,
                **dict(plan.evidence_hashes),
            },
            "future_realized_targets_used": False,
            "state": {
                "next_interval": 0,
                "soc_mwh": initial_soc.tolist(),
                "previous_grid_import_mw": None,
            },
            "intervals": [],
        }
        if checkpoint_path is not None:
            atomic_json(checkpoint_path, checkpoint)
    if checkpoint.get("status") == "COMPLETE":
        output = dict(checkpoint)
        output["resumed"] = True
        return output

    processed = 0
    horizon = len(issue_bundle.timestamps_utc)
    while int(checkpoint["state"]["next_interval"]) < horizon:
        if max_intervals is not None and processed >= max_intervals:
            break
        interval = int(checkpoint["state"]["next_interval"])
        recourse_bundle, causal_audit = build_causal_recourse_bundle(
            issue_bundle, settlement, forcing, interval=interval
        )
        recourse_config = dict(settings)
        recourse_config["initial_soc_mwh"] = checkpoint["state"]["soc_mwh"]
        if checkpoint["state"]["previous_grid_import_mw"] is not None:
            recourse_config["previous_grid_import_mw"] = checkpoint["state"][
                "previous_grid_import_mw"
            ]
        result = solve_recourse_with_frozen_plan(
            recourse_bundle,
            physical,
            plan,
            start_interval=interval,
            config=recourse_config,
            solver=recourse_solver,
        )
        dispatch = _first_interval_dispatch(result)
        q_da = plan.as_array()[:, interval : interval + 1]
        grid = np.asarray(dispatch["grid_import_mw"], dtype=float)[:, None]
        hook_costs = (
            {}
            if component_cost_hook is None
            else dict(component_cost_hook(interval, dispatch))
        )
        costs = settle_realized_components(
            q_da,
            grid,
            settlement.da_lbmp[:, interval : interval + 1],
            settlement.rt_lbmp[:, interval : interval + 1],
            interval_hours=ROLLING_INTERVAL_HOURS,
            rt_sell_haircut_fraction=float(settings.get("rt_sell_back_haircut", 0.0)),
            component_costs_usd=hook_costs,
        )
        record = {
            "interval": interval,
            "timestamp_utc": str(issue_bundle.timestamps_utc[interval]),
            "revealed_through_interval": interval,
            "future_realized_targets_used": False,
            "causal_audit": causal_audit,
            "day_ahead_schedule_mw": q_da[:, 0].tolist(),
            "grid_import_mw": grid[:, 0].tolist(),
            "costs": costs,
            "solver": {
                "termination_condition": result.get("termination_condition"),
                "mip_gap": result.get("mip_gap"),
                "solve_time_seconds": result.get("solve_time_seconds"),
            },
        }
        checkpoint["intervals"].append(record)
        checkpoint["state"]["next_interval"] = interval + 1
        checkpoint["state"]["soc_mwh"] = np.asarray(
            dispatch["soc_end_mwh"], dtype=float
        ).tolist()
        checkpoint["state"]["previous_grid_import_mw"] = grid[:, 0].tolist()
        processed += 1
        if checkpoint_path is not None:
            atomic_json(checkpoint_path, checkpoint)

    checkpoint["completed_interval_count"] = len(checkpoint["intervals"])
    checkpoint["status"] = (
        "COMPLETE"
        if int(checkpoint["state"]["next_interval"]) == horizon
        else "IN_PROGRESS"
    )
    component_names = sorted(
        {
            key
            for record in checkpoint["intervals"]
            for key in record["costs"]
            if key != "settlement_identity_residual_mw"
        }
    )
    checkpoint["component_costs_usd"] = {
        key: float(sum(float(record["costs"].get(key, 0.0)) for record in checkpoint["intervals"]))
        for key in component_names
    }
    checkpoint["simulated_total_cost_usd"] = float(
        checkpoint["component_costs_usd"].get("simulated_total_cost_usd", 0.0)
    )
    checkpoint["maximum_settlement_identity_residual_mw"] = float(
        max(
            (
                float(record["costs"]["settlement_identity_residual_mw"])
                for record in checkpoint["intervals"]
            ),
            default=0.0,
        )
    )
    checkpoint["replay_rows_sha256"] = sha256_json(checkpoint["intervals"])
    if checkpoint_path is not None:
        atomic_json(checkpoint_path, checkpoint)
    output = dict(checkpoint)
    output["resumed"] = resumed
    return output


__all__ = [
    "FrozenReplayForcing",
    "NYISO_SOURCE_RELATIONSHIP",
    "REPLAY_SCHEMA_VERSION",
    "ReferencedNYISOReplay",
    "build_a1_dart_bundle",
    "build_causal_recourse_bundle",
    "canonical_dart_da_minus_rt",
    "inverse_rt_spread_rt_minus_da",
    "rt_from_fixed_da_and_dart",
    "run_external_replay_day",
    "settle_realized_components",
    "validate_dart_inverse_gate",
    "validate_replay_alignment",
]
