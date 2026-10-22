from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from dcvpp.core import sha256_json
from dcvpp.research_branch.contracts import FrozenPhysicalArtifact, ScenarioBundle
from dcvpp.research_branch.optimization import solve_policy


FIRST_STAGE_SCHEMA_VERSION = "dart_frozen_first_stage_v1"

# Wall-clock observations describe an execution, not the mathematical plan.
# Keeping them outside the identity payload makes an interrupted replay able to
# reconstruct the same frozen first stage without invalidating its checkpoint.
NON_SEMANTIC_SOLVER_EVIDENCE_KEYS = frozenset({"solve_time_seconds"})

SolvePolicy = Callable[
    [ScenarioBundle, FrozenPhysicalArtifact, str, Mapping[str, Any] | None],
    dict[str, Any],
]


def array_sha256(value: Any) -> str:
    """Hash an array together with its dtype and shape."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        array = np.asarray(value, dtype="U")
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_sha256_map(value: Mapping[str, str], *, name: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, digest in sorted(value.items()):
        candidate = str(digest).lower()
        if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
            raise ValueError(f"{name}[{key!r}] must be a lowercase-compatible SHA-256 digest")
        normalized[str(key)] = candidate
    if not normalized:
        raise ValueError(f"{name} must contain at least one evidence hash")
    return normalized


def _assert_issue_time_only(value: Any, *, path: str = "config") -> None:
    """Reject an accidental realized/future target passed into a day-ahead solve."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            lowered = key.lower()
            forbidden = (
                "realized" in lowered
                or "future_actual" in lowered
                or "observed_target" in lowered
                or "locked_test_target" in lowered
            )
            empty = nested is None or nested is False
            if isinstance(nested, (str, bytes, list, tuple, dict, set)):
                empty = empty or len(nested) == 0
            if forbidden and not empty:
                raise ValueError(f"day-ahead solve received forbidden future information at {path}.{key}")
            _assert_issue_time_only(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_issue_time_only(nested, path=f"{path}[{index}]")


@dataclass(frozen=True)
class FrozenDayAheadPlan:
    """Immutable, evidence-addressed first-stage procurement decision."""

    operating_date: str
    candidate_id: str
    experiment_id: str
    seed: int
    policy_id: str
    zones: tuple[str, ...]
    timestamps_utc: tuple[str, ...]
    q_da_mw: tuple[tuple[float, ...], ...]
    input_sha256: str
    evidence_hashes: Mapping[str, str]
    solver_evidence: Mapping[str, Any]
    plan_sha256: str
    schema_version: str = FIRST_STAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FIRST_STAGE_SCHEMA_VERSION:
            raise ValueError("unsupported frozen first-stage schema")
        if not self.candidate_id or self.experiment_id not in {"A1", "A2", "BASELINE"}:
            raise ValueError("candidate_id and a registered experiment_id are required")
        schedule = np.asarray(self.q_da_mw, dtype=float)
        expected_shape = (len(self.zones), len(self.timestamps_utc))
        if schedule.shape != expected_shape:
            raise ValueError(f"q_da_mw must have shape {expected_shape}, got {schedule.shape}")
        if not np.isfinite(schedule).all() or (schedule < 0).any():
            raise ValueError("q_da_mw must be finite and nonnegative")
        hashes = _validate_sha256_map(self.evidence_hashes, name="evidence_hashes")
        object.__setattr__(self, "evidence_hashes", hashes)
        expected = sha256_json(self._hash_payload())
        if self.plan_sha256 != expected:
            raise ValueError("frozen first-stage plan SHA-256 does not match its payload")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operating_date": self.operating_date,
            "candidate_id": self.candidate_id,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "policy_id": self.policy_id,
            "zones": list(self.zones),
            "timestamps_utc": list(self.timestamps_utc),
            "q_da_mw": [list(row) for row in self.q_da_mw],
            "input_sha256": self.input_sha256,
            "evidence_hashes": dict(self.evidence_hashes),
            "solver_evidence": {
                str(key): value
                for key, value in sorted(self.solver_evidence.items())
                if key not in NON_SEMANTIC_SOLVER_EVIDENCE_KEYS
            },
        }

    def as_array(self) -> np.ndarray:
        schedule = np.asarray(self.q_da_mw, dtype=float)
        schedule.setflags(write=False)
        return schedule

    def to_mapping(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["plan_sha256"] = self.plan_sha256
        return payload


def _first_stage_input_sha256(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    policy_id: str,
    config: Mapping[str, Any],
) -> str:
    return sha256_json(
        {
            "operating_date": bundle.operating_date,
            "regime": bundle.regime,
            "kappa": bundle.kappa,
            "scenario_seed": bundle.seed,
            "policy_id": policy_id,
            "config": dict(config),
            "calibration_id": physical.calibration_id,
            "twin_ids": list(physical.twin_ids),
            "arrays": {
                "timestamps_utc": array_sha256(bundle.timestamps_utc),
                "da_price": array_sha256(bundle.da_price),
                "rt_price": array_sha256(bundle.rt_price),
                "workload": array_sha256(bundle.workload),
                "pv": array_sha256(bundle.pv),
                "physical_multiplier": array_sha256(bundle.physical_multiplier),
                "probabilities": array_sha256(bundle.probabilities),
                "idle_mw": array_sha256(physical.idle_mw),
                "work_mw_per_unit": array_sha256(physical.work_mw_per_unit),
                "capacity_work_units": array_sha256(physical.capacity_work_units),
            },
        }
    )


def freeze_stochastic_first_stage(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    *,
    candidate_id: str,
    experiment_id: str,
    seed: int,
    policy_id: str = "B6",
    config: Mapping[str, Any] | None = None,
    evidence_hashes: Mapping[str, str],
    solver: SolvePolicy = solve_policy,
) -> tuple[FrozenDayAheadPlan, dict[str, Any]]:
    """Solve once from issue-time scenarios and freeze non-anticipative ``q_DA``.

    Realized NYISO prices and realized workload/PV forcing are deliberately not
    arguments to this function.  They enter only the replay adapter after this
    plan has been hashed and frozen.
    """

    if policy_id == "B8":
        raise ValueError("B8 is a perfect-information bound, not a deployable first stage")
    settings = dict(config or {})
    _assert_issue_time_only(settings)
    bundle.validate()
    input_sha256 = _first_stage_input_sha256(bundle, physical, policy_id, settings)
    result = solver(bundle, physical, policy_id, settings)
    raw_schedule = np.asarray(result["day_ahead_schedule_mw"], dtype=float)
    zones = len(bundle.zones)
    horizon = len(bundle.timestamps_utc)
    if raw_schedule.shape == (zones, horizon):
        schedule = raw_schedule
    elif raw_schedule.shape == (bundle.reduced_count, zones, horizon):
        if not np.allclose(raw_schedule, raw_schedule[0], rtol=0.0, atol=1e-8):
            raise ValueError("stochastic first-stage q_DA is scenario-dependent")
        schedule = raw_schedule[0]
    else:
        raise ValueError(
            "solver day_ahead_schedule_mw must have shape "
            f"{(zones, horizon)} or {(bundle.reduced_count, zones, horizon)}"
        )
    if not np.isfinite(schedule).all() or (schedule < 0).any():
        raise ValueError("solver returned an invalid day-ahead schedule")
    solver_evidence = {
        "solver": result.get("solver"),
        "solver_status": result.get("solver_status"),
        "termination_condition": result.get("termination_condition"),
        "mip_gap": result.get("mip_gap"),
        "solve_time_seconds": result.get("solve_time_seconds"),
        "counterfactual": result.get("counterfactual", True),
        "future_realized_targets_used": False,
        "nonanticipativity_verified": True,
    }
    normalized_hashes = _validate_sha256_map(evidence_hashes, name="evidence_hashes")
    provisional = {
        "schema_version": FIRST_STAGE_SCHEMA_VERSION,
        "operating_date": bundle.operating_date,
        "candidate_id": candidate_id,
        "experiment_id": experiment_id,
        "seed": int(seed),
        "policy_id": policy_id,
        "zones": list(bundle.zones),
        "timestamps_utc": [str(value) for value in bundle.timestamps_utc],
        "q_da_mw": schedule.tolist(),
        "input_sha256": input_sha256,
        "evidence_hashes": normalized_hashes,
        "solver_evidence": {
            key: value
            for key, value in sorted(solver_evidence.items())
            if key not in NON_SEMANTIC_SOLVER_EVIDENCE_KEYS
        },
    }
    plan = FrozenDayAheadPlan(
        operating_date=bundle.operating_date,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
        seed=int(seed),
        policy_id=policy_id,
        zones=tuple(bundle.zones),
        timestamps_utc=tuple(str(value) for value in bundle.timestamps_utc),
        q_da_mw=tuple(tuple(float(value) for value in row) for row in schedule),
        input_sha256=input_sha256,
        evidence_hashes=normalized_hashes,
        solver_evidence=solver_evidence,
        plan_sha256=sha256_json(provisional),
    )
    return plan, result


def solve_recourse_with_frozen_plan(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    plan: FrozenDayAheadPlan,
    *,
    start_interval: int,
    config: Mapping[str, Any] | None = None,
    solver: SolvePolicy = solve_policy,
) -> dict[str, Any]:
    """Delegate recourse to the research solver while keeping ``q_DA`` fixed."""

    settings = dict(config or {})
    if "fixed_day_ahead_schedule_mw" in settings:
        raise ValueError("the adapter owns fixed_day_ahead_schedule_mw")
    if not 0 <= start_interval < len(plan.timestamps_utc):
        raise ValueError("start_interval is outside the frozen plan")
    expected_horizon = len(plan.timestamps_utc) - start_interval
    if bundle.rt_price.shape[2] != expected_horizon:
        raise ValueError("recourse bundle horizon does not match the frozen plan suffix")
    schedule = plan.as_array()[:, start_interval:]
    settings.update(
        {
            "fixed_day_ahead_schedule_mw": schedule,
            "recourse_nonanticipative_prefix_steps": 1,
            "return_dispatch_details": True,
        }
    )
    result = solver(bundle, physical, plan.policy_id, settings)
    solved = np.asarray(result["day_ahead_schedule_mw"], dtype=float)
    if solved.ndim == 3:
        solved = solved[0]
    if solved.shape != schedule.shape or not np.allclose(solved, schedule, rtol=0.0, atol=1e-8):
        raise ValueError("recourse solve changed the frozen day-ahead schedule")
    return result


def solve_perfect_information_bound(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    *,
    config: Mapping[str, Any] | None = None,
    solver: SolvePolicy = solve_policy,
) -> dict[str, Any]:
    """Return the per-day PI cost lower bound / operational-value upper bound."""

    result = solver(bundle, physical, "B8", dict(config or {}))
    return {
        "operating_date": bundle.operating_date,
        "policy_id": "B8",
        "simulated_cost_lower_bound_usd": float(result["expected_total_cost_usd"]),
        "operational_value_upper_bound": True,
        "deployable": False,
        "result": result,
    }


__all__ = [
    "FIRST_STAGE_SCHEMA_VERSION",
    "FrozenDayAheadPlan",
    "NON_SEMANTIC_SOLVER_EVIDENCE_KEYS",
    "array_sha256",
    "freeze_stochastic_first_stage",
    "solve_perfect_information_bound",
    "solve_recourse_with_frozen_plan",
]
