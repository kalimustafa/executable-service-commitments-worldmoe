"""Exact post-shield replay of sampled complete future trajectories.

The differentiable policy rollout is not an operational CVaR measurement: its
actions are not re-executed after every particle-specific observation.  This
module closes that seam by reusing :class:`R4Runtime` once per particle.  Each
run therefore uses the same 96-origin causal session contract, the same
``AnalyticConstraintShield``, the independent residual validator, and the
official two-settlement accounting as formal deterministic evaluation.

No optimizer or external mathematical solver is used here.  Ordinary replay is
restricted to train/validation.  The formal V6 Distribution replay may enter a
claimed sealed-selection run only through the explicit inference-mode runtime
flag; locked 2025 always fails before a sample is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Mapping

import numpy as np
import torch
import sys

from .constrained_runtime import PhysicalLimits, R4Runtime
from .data import INTERVALS, NYISO_ZONES, R4DataError, R4SelectionDay, SCENARIOS
from .hashing import sha256_bytes, sha256_json


ZONAL_VARIABLE_COUNT = 21
REPLAY_SCHEMA_VERSION = "dcvpp.r4.post_shield_particle_replay.v1"
DISTRIBUTION_REPLAY_SCHEMA_VERSION = (
    "dcvpp.r4.v6.post_shield_distribution_replay.v1"
)
DISTRIBUTION_ABLATION_REPLAY_SCHEMA_VERSION = (
    "dcvpp.r4.v6.post_shield_distribution_ablation_replay.v1"
)
FORMAL_EMPIRICAL_RISK_SAMPLES = 512
DISTRIBUTION_SAMPLING_BUDGETS = (20, 64, 128, 256, 512)
_ZONE_COUNT = len(NYISO_ZONES)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FORBIDDEN_SOLVER_MODULES = ("pyomo", "gurobipy")


class PostShieldParticleReplayError(R4DataError):
    """A post-shield particle replay contract failed closed."""


def _assert_solver_free_active_imports() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == root or name.startswith(root + ".")
            for root in _FORBIDDEN_SOLVER_MODULES
        )
    )
    if loaded:
        raise PostShieldParticleReplayError(
            "post-shield replay loaded an external mathematical solver: "
            + ", ".join(loaded[:8])
        )


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(item not in _SHA256_CHARACTERS for item in digest):
        raise PostShieldParticleReplayError(f"{name} must be a lowercase SHA-256")
    return digest


def _finite_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if value is None:
        raise PostShieldParticleReplayError(f"{name} is required")
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    result = np.asarray(current, dtype=np.float64)
    if result.shape != shape:
        raise PostShieldParticleReplayError(
            f"{name} must have shape {shape}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise PostShieldParticleReplayError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result)


def _frozen_day_ahead(value: Any) -> np.ndarray:
    if value is None:
        raise PostShieldParticleReplayError("frozen_day_ahead is required")
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    result = np.asarray(current, dtype=np.float64)
    if result.shape == (1, _ZONE_COUNT, INTERVALS):
        result = result[0]
    if result.shape == (INTERVALS, _ZONE_COUNT):
        result = result.T
    return _finite_array(
        result,
        name="frozen_day_ahead",
        shape=(_ZONE_COUNT, INTERVALS),
    )


def _array_sha256(
    value: np.ndarray, *, role: str, schema_version: str = REPLAY_SCHEMA_VERSION
) -> str:
    canonical = np.ascontiguousarray(value, dtype="<f8")
    payload = bytearray()
    payload.extend(str(schema_version).encode("ascii"))
    payload.extend(role.encode("utf-8"))
    payload.extend(str(canonical.shape).encode("ascii"))
    payload.extend(b"<f8")
    payload.extend(canonical.tobytes(order="C"))
    return sha256_bytes(bytes(payload))


def _validated_scenario(
    particles_zonal: Any,
    weights: Any,
    *,
    identity_tolerance: float,
    expected_sample_count: int = SCENARIOS,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(expected_sample_count)
    if isinstance(expected_sample_count, bool) or count != expected_sample_count or count < 2:
        raise PostShieldParticleReplayError(
            "expected_sample_count must be an integer of at least two"
        )
    particles = _finite_array(
        particles_zonal,
        name="particles_zonal",
        shape=(count, INTERVALS, ZONAL_VARIABLE_COUNT),
    )
    probabilities = _finite_array(
        weights,
        name="weights",
        shape=(count,),
    )
    if (probabilities < 0.0).any():
        raise PostShieldParticleReplayError("weights must be nonnegative")
    weight_sum = float(probabilities.sum())
    if weight_sum <= 0.0 or abs(weight_sum - 1.0) > 1.0e-6:
        raise PostShieldParticleReplayError(
            "weights must already sum to one within 1e-6; replay never repairs weights"
        )
    if (particles[..., 12] < 0.0).any():
        raise PostShieldParticleReplayError("particle workload must be nonnegative")
    if (particles[..., 13:17] < 0.0).any():
        raise PostShieldParticleReplayError("particle PV must be nonnegative")
    if (particles[..., 17:21] <= 0.0).any():
        raise PostShieldParticleReplayError("particle PUE must be strictly positive")
    identity = particles[..., 0:4] - particles[..., 4:8] - particles[..., 8:12]
    identity_max = float(np.max(np.abs(identity), initial=0.0))
    if identity_max > float(identity_tolerance):
        raise PostShieldParticleReplayError(
            "particle DA-RT-DART identity exceeds replay tolerance: "
            f"{identity_max:.9g} > {float(identity_tolerance):.9g}"
        )
    # Retain the caller's exact canonical weights in the content hash, then use
    # an exact unit-sum copy for arithmetic.  The correction is only round-off
    # (bounded above by the validated 1e-6 contract), not silent calibration.
    probabilities = probabilities / weight_sum
    return particles, probabilities


def particle_scenario_hashes(
    particles_zonal: Any,
    weights: Any,
    *,
    identity_tolerance: float = 1.0e-6,
    expected_sample_count: int = SCENARIOS,
    schema_version: str = REPLAY_SCHEMA_VERSION,
) -> dict[str, str]:
    """Return canonical hashes for one immutable complete-trajectory sample."""

    particles, probabilities = _validated_scenario(
        particles_zonal,
        weights,
        identity_tolerance=identity_tolerance,
        expected_sample_count=expected_sample_count,
    )
    particles_sha256 = _array_sha256(
        particles, role="particles_zonal", schema_version=schema_version
    )
    weights_sha256 = _array_sha256(
        probabilities, role="scenario_weights", schema_version=schema_version
    )
    scenario_sha256 = sha256_json(
        {
            "schema_version": schema_version,
            "particle_count": int(expected_sample_count),
            "interval_count": INTERVALS,
            "zonal_variable_count": ZONAL_VARIABLE_COUNT,
            "particles_sha256": particles_sha256,
            "weights_sha256": weights_sha256,
        }
    )
    return {
        "particles_sha256": particles_sha256,
        "weights_sha256": weights_sha256,
        "scenario_sha256": scenario_sha256,
    }


def _physical_contract_sha256(physical: PhysicalLimits, plant: Any) -> str:
    payload = bytearray(REPLAY_SCHEMA_VERSION.encode("ascii"))
    for name in (
        "idle_mw",
        "work_mw_per_unit",
        "capacity_work_units",
        "grid_import_limit_mw",
        "battery_power_mw",
        "battery_energy_mwh",
    ):
        value = np.ascontiguousarray(getattr(physical, name), dtype="<f8")
        payload.extend(name.encode("ascii"))
        payload.extend(str(value.shape).encode("ascii"))
        payload.extend(value.tobytes(order="C"))
    payload.extend(
        sha256_json(
            {
                "zones": tuple(physical.zones),
                "calibration_id": physical.calibration_id,
                "source_hashes": dict(physical.source_hashes),
                "frozen": bool(physical.frozen),
                "plant_contract": dict(plant.contract),
            }
        ).encode("ascii")
    )
    return sha256_bytes(bytes(payload))


def _day_contract_sha256(day: R4SelectionDay) -> str:
    payload = bytearray(REPLAY_SCHEMA_VERSION.encode("ascii"))
    for name in (
        "timestamps_utc",
        "market",
        "workload",
        "physical_static",
        "physical_dynamic",
        "initial_soc_mwh",
        "da_price",
        "rt_price_pool",
        "workload_pool",
        "pv_pool",
        "physical_multiplier",
    ):
        value = np.asarray(getattr(day, name))
        payload.extend(name.encode("ascii"))
        payload.extend(str(value.shape).encode("ascii"))
        if value.dtype.kind in "OUSM":
            encoded = "\n".join(map(str, value.reshape(-1))).encode("utf-8")
            payload.extend(b"text")
            payload.extend(encoded)
        else:
            canonical = np.ascontiguousarray(value, dtype="<f8")
            payload.extend(b"<f8")
            payload.extend(canonical.tobytes(order="C"))
    payload.extend(
        sha256_json(
            {
                "operating_date": day.operating_date,
                "issue_time_utc": day.issue_time_utc,
                "history_dates": tuple(day.history_dates),
                "feature_provenance": {
                    key: dict(value) for key, value in day.feature_provenance.items()
                },
                "source_hashes": dict(day.source_hashes),
                "physical_twin_metadata": dict(day.physical_twin_metadata),
            }
        ).encode("ascii")
    )
    return sha256_bytes(bytes(payload))


@dataclass(frozen=True)
class PostShieldParticleReplayBinding:
    """Immutable pre-execution binding for one particle-policy replay."""

    schema_version: str
    operating_date: str
    particles_sha256: str
    weights_sha256: str
    scenario_sha256: str
    scenario_contract_sha256: str
    policy_contract_sha256: str
    day_contract_sha256: str
    frozen_day_ahead_sha256: str
    physical_contract_sha256: str
    runtime_config_sha256: str
    replay_contract_sha256: str
    sample_count: int = SCENARIOS

    def __post_init__(self) -> None:
        if self.schema_version not in {
            REPLAY_SCHEMA_VERSION,
            DISTRIBUTION_REPLAY_SCHEMA_VERSION,
            DISTRIBUTION_ABLATION_REPLAY_SCHEMA_VERSION,
        }:
            raise PostShieldParticleReplayError("unsupported replay binding schema")
        count = int(self.sample_count)
        if count < 2:
            raise PostShieldParticleReplayError("replay binding sample_count must be >=2")
        if self.schema_version == REPLAY_SCHEMA_VERSION and count != SCENARIOS:
            raise PostShieldParticleReplayError(
                "legacy V5 replay schema is restricted to exact-20 history"
            )
        if (
            self.schema_version == DISTRIBUTION_REPLAY_SCHEMA_VERSION
            and count != FORMAL_EMPIRICAL_RISK_SAMPLES
        ):
            raise PostShieldParticleReplayError(
                "formal V6 empirical-risk replay requires N=512"
            )
        if (
            self.schema_version == DISTRIBUTION_ABLATION_REPLAY_SCHEMA_VERSION
            and count not in DISTRIBUTION_SAMPLING_BUDGETS
        ):
            raise PostShieldParticleReplayError(
                "V6 sampling-budget replay uses an unregistered sample count"
            )
        object.__setattr__(self, "sample_count", count)
        for item in fields(self):
            if item.name.endswith("_sha256"):
                object.__setattr__(
                    self,
                    item.name,
                    _require_sha256(getattr(self, item.name), name=item.name),
                )

    def as_dict(self) -> dict[str, str | int]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_mapping(
        cls, value: "PostShieldParticleReplayBinding | Mapping[str, Any]"
    ) -> "PostShieldParticleReplayBinding":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise PostShieldParticleReplayError("replay binding is required")
        try:
            payload = {
                item.name: (
                    value[item.name]
                    if item.name in value
                    else SCENARIOS
                    if item.name == "sample_count"
                    else (_ for _ in ()).throw(KeyError(item.name))
                )
                for item in fields(cls)
            }
            return cls(**payload)
        except KeyError as exc:
            raise PostShieldParticleReplayError(
                f"replay binding lacks {exc.args[0]}"
            ) from exc


class _FrozenDayAheadCallback(torch.nn.Module):
    """Delegate every neural choice while replacing only the issued DA value."""

    def __init__(self, callback: Any, frozen_day_ahead: np.ndarray) -> None:
        super().__init__()
        # Register a deployment nn.Module as a child.  R4Runtime can then find
        # its parameter device/dtype through this wrapper and move every
        # numeric session input before calling a CUDA-resident model.  A plain
        # benchmark callback remains a normal Python attribute.
        self.callback = callback
        self.frozen_day_ahead = np.array(frozen_day_ahead, copy=True)

    def prepare_issue_context(self, value: Mapping[str, Any]) -> Any:
        target = getattr(self.callback, "prepare_issue_context", None)
        return value if target is None else target(value)

    def forward_session(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        target = getattr(self.callback, "forward_session", self.callback)
        output = target(value)
        if not isinstance(output, Mapping):
            raise PostShieldParticleReplayError(
                "policy callback must return a mapping during particle replay"
            )
        copied = dict(output)
        replaced = False
        for output_key in ("actions", "policy"):
            if not isinstance(copied.get(output_key), Mapping):
                continue
            action_mapping = dict(copied[output_key])
            if isinstance(action_mapping.get("free_actions"), Mapping):
                free_actions = dict(action_mapping["free_actions"])
                free_actions["day_ahead"] = np.array(
                    self.frozen_day_ahead, copy=True
                )
                action_mapping["free_actions"] = free_actions
            else:
                action_mapping["day_ahead"] = np.array(
                    self.frozen_day_ahead, copy=True
                )
            copied[output_key] = action_mapping
            replaced = True
        if not replaced:
            copied["day_ahead"] = np.array(self.frozen_day_ahead, copy=True)
        return copied


PolicyCallbackFactory = Callable[[int, R4SelectionDay], Any]


class PostShieldParticleReplay:
    """Replay every formal particle through the production shield/runtime."""

    def __init__(
        self,
        physical: PhysicalLimits,
        runtime_config: Mapping[str, Any] | None = None,
        *,
        physical_twin_core: Any | None = None,
        sample_count: int = SCENARIOS,
        policy_uses_distribution_samples: bool = True,
        schema_version: str = REPLAY_SCHEMA_VERSION,
    ) -> None:
        _assert_solver_free_active_imports()
        self.runtime_config = dict(runtime_config or {})
        requested_sealed = bool(
            self.runtime_config.get("allow_sealed_selection", False)
        )
        self.allow_sealed_selection = bool(
            requested_sealed
            and schema_version == DISTRIBUTION_REPLAY_SCHEMA_VERSION
            and self.runtime_config.get("inference_mode") is True
        )
        if requested_sealed and not self.allow_sealed_selection:
            raise PostShieldParticleReplayError(
                "only formal inference-mode V6 Distribution replay may use sealed selection"
            )
        self.identity_tolerance = float(
            self.runtime_config.get("scenario_identity_tolerance", 1.0e-6)
        )
        if not np.isfinite(self.identity_tolerance) or self.identity_tolerance < 0.0:
            raise PostShieldParticleReplayError(
                "scenario_identity_tolerance must be finite and nonnegative"
            )
        self.sample_count = int(sample_count)
        self.policy_uses_distribution_samples = bool(policy_uses_distribution_samples)
        self.schema_version = str(schema_version)
        if self.sample_count < 2:
            raise PostShieldParticleReplayError("replay sample_count must be at least two")
        if self.schema_version == REPLAY_SCHEMA_VERSION and self.sample_count != SCENARIOS:
            raise PostShieldParticleReplayError(
                "legacy PostShieldParticleReplay remains exact-20 compatibility only"
            )
        if self.schema_version == DISTRIBUTION_REPLAY_SCHEMA_VERSION:
            if self.sample_count != FORMAL_EMPIRICAL_RISK_SAMPLES:
                raise PostShieldParticleReplayError(
                    "formal Distribution replay requires N=512"
                )
            if self.policy_uses_distribution_samples:
                raise PostShieldParticleReplayError(
                    "V6 Policy must not consume Monte Carlo distribution samples"
                )
        self.runtime = R4Runtime(
            physical,
            self.runtime_config,
            physical_twin_core=physical_twin_core,
        )
        self.physical_contract_sha256 = _physical_contract_sha256(
            physical, self.runtime.shield.plant
        )
        self.runtime_config_sha256 = sha256_json(
            {
                "schema_version": self.schema_version,
                "runtime_config": self.runtime_config,
                "constraint_shield_class": (
                    "dcvpp.e2e_moe_r4.constrained_runtime.AnalyticConstraintShield"
                ),
                "runtime_class": "dcvpp.e2e_moe_r4.constrained_runtime.R4Runtime",
                "sample_count": self.sample_count,
                "policy_uses_distribution_samples": self.policy_uses_distribution_samples,
                "interval_hours": 0.25,
                "official_settlement_formula": (
                    "0.25*sum_z(DA_schedule*DA_price"
                    "+(grid_import-DA_schedule)*RT_price)"
                ),
            }
        )

    def _guard_day(self, day: R4SelectionDay) -> None:
        if not isinstance(day, R4SelectionDay):
            raise PostShieldParticleReplayError("day must be an R4SelectionDay")
        allowed_splits = {"train", "validation"}
        if self.allow_sealed_selection:
            allowed_splits.add("sealed_selection")
        if day.split not in allowed_splits:
            raise PostShieldParticleReplayError(
                "post-shield replay split is unauthorized; locked 2025 remains closed"
            )

    def bind(
        self,
        *,
        day: R4SelectionDay,
        particles_zonal: Any,
        weights: Any,
        frozen_day_ahead: Any,
        policy_contract_sha256: str,
        scenario_contract_sha256: str,
    ) -> PostShieldParticleReplayBinding:
        """Freeze all replay inputs before any particle policy is executed."""

        self._guard_day(day)
        particles, probabilities = _validated_scenario(
            particles_zonal,
            weights,
            identity_tolerance=self.identity_tolerance,
            expected_sample_count=self.sample_count,
        )
        scenario_hashes = particle_scenario_hashes(
            particles,
            probabilities,
            identity_tolerance=self.identity_tolerance,
            expected_sample_count=self.sample_count,
            schema_version=self.schema_version,
        )
        frozen_da = _frozen_day_ahead(frozen_day_ahead)
        values = {
            "schema_version": self.schema_version,
            "operating_date": day.operating_date,
            **scenario_hashes,
            "scenario_contract_sha256": _require_sha256(
                scenario_contract_sha256, name="scenario_contract_sha256"
            ),
            "policy_contract_sha256": _require_sha256(
                policy_contract_sha256, name="policy_contract_sha256"
            ),
            "day_contract_sha256": _day_contract_sha256(day),
            "frozen_day_ahead_sha256": _array_sha256(
                frozen_da,
                role="frozen_day_ahead",
                schema_version=self.schema_version,
            ),
            "physical_contract_sha256": self.physical_contract_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
        }
        replay_contract_sha256 = sha256_json(values)
        return PostShieldParticleReplayBinding(
            **values,
            replay_contract_sha256=replay_contract_sha256,
            sample_count=self.sample_count,
        )

    def run(
        self,
        *,
        day: R4SelectionDay,
        particles_zonal: Any,
        weights: Any,
        frozen_day_ahead: Any,
        policy_callback_factory: PolicyCallbackFactory,
        binding: PostShieldParticleReplayBinding | Mapping[str, Any],
        _binding_frozen_in_current_call: bool = False,
    ) -> dict[str, Any]:
        """Execute every bound future or fail without emitting a CVaR.

        Public callers that supply a persisted binding retain the historical
        tamper check.  The functional entry points freeze and consume a binding
        in one call and set the private flag, avoiding a second full content
        hash over the same N x 96 x 21 tensor.
        """

        supplied_binding = PostShieldParticleReplayBinding.from_mapping(binding)
        if _binding_frozen_in_current_call:
            current_binding = supplied_binding
            if (
                current_binding.schema_version != self.schema_version
                or current_binding.sample_count != self.sample_count
            ):
                raise PostShieldParticleReplayError(
                    "fresh replay binding schema/sample count differs from runtime"
                )
        else:
            current_binding = self.bind(
                day=day,
                particles_zonal=particles_zonal,
                weights=weights,
                frozen_day_ahead=frozen_day_ahead,
                policy_contract_sha256=supplied_binding.policy_contract_sha256,
                scenario_contract_sha256=supplied_binding.scenario_contract_sha256,
            )
            if supplied_binding != current_binding:
                changed = [
                    item.name
                    for item in fields(PostShieldParticleReplayBinding)
                    if getattr(supplied_binding, item.name) != getattr(current_binding, item.name)
                ]
                raise PostShieldParticleReplayError(
                    "post-shield replay input differs from frozen binding: "
                    + ", ".join(changed)
                )
        if not callable(policy_callback_factory):
            raise PostShieldParticleReplayError("policy_callback_factory must be callable")

        particles, probabilities = _validated_scenario(
            particles_zonal,
            weights,
            identity_tolerance=self.identity_tolerance,
            expected_sample_count=self.sample_count,
        )
        frozen_da = _frozen_day_ahead(frozen_day_ahead)
        callbacks: list[Any] = []
        particle_rows: list[dict[str, Any]] = []

        for particle_index in range(self.sample_count):
            particle = particles[particle_index]
            particle_day = replace(
                day,
                realized_da_price=particle[:, 0:4].T,
                realized_rt_price=particle[:, 4:8].T,
                realized_workload=particle[:, 12],
                realized_pv=particle[:, 13:17].T,
                realized_pue=particle[:, 17:21],
            )
            callback = policy_callback_factory(particle_index, particle_day)
            if callback is None:
                raise PostShieldParticleReplayError(
                    f"policy callback factory returned None for particle {particle_index}"
                )
            if any(callback is previous for previous in callbacks):
                raise PostShieldParticleReplayError(
                    "policy callback factory must return a fresh session object "
                    "for every particle"
                )
            if bool(getattr(callback, "training", False)):
                raise PostShieldParticleReplayError(
                    "particle replay policy callback must be in evaluation mode"
                )
            callbacks.append(callback)
            wrapped_callback = _FrozenDayAheadCallback(callback, frozen_da).eval()
            result = self.runtime.run_day(
                particle_day,
                wrapped_callback,
                joint_scenarios=(
                    particles if self.policy_uses_distribution_samples else None
                ),
                scenario_weights=(
                    probabilities if self.policy_uses_distribution_samples else None
                ),
            )
            records = tuple(result.get("records", ()))
            completed = int(result.get("completed_origins", len(records)))
            independently_validated = sum(
                bool(record.get("passed", False)) for record in records
            )
            day_ahead_frozen = bool(
                result.get("day_ahead_frozen_after_origin_zero", False)
            )
            feasible = bool(
                result.get("status") == "COMPLETE"
                and completed == INTERVALS
                and len(records) == INTERVALS
                and independently_validated == INTERVALS
                and result.get("all_actions_independently_validated", False)
                and day_ahead_frozen
            )
            if not feasible:
                raise PostShieldParticleReplayError(
                    "particle replay failed the 96-origin post-shield contract at "
                    f"particle {particle_index}: status={result.get('status')}, "
                    f"completed={completed}, validated={independently_validated}, "
                    f"day_ahead_frozen={day_ahead_frozen}"
                )

            da_cost = float(sum(float(row["day_ahead_cost_usd"]) for row in records))
            rt_cost = float(sum(float(row["rt_imbalance_cost_usd"]) for row in records))
            day_cost = float(sum(float(row["total_cost_usd"]) for row in records))
            sla = float(
                sum(float(row["sla_shortfall_work_units"]) for row in records)
            )
            workload = float(particle[:, 12].sum())
            corrections = np.asarray(
                [
                    float(row.get("corrective_intervention_magnitude", 0.0))
                    for row in records
                ],
                dtype=float,
            )
            finite = np.asarray(
                [da_cost, rt_cost, day_cost, sla, workload, *corrections], dtype=float
            )
            if not np.isfinite(finite).all():
                raise PostShieldParticleReplayError(
                    f"particle {particle_index} emitted a non-finite cost/SLA diagnostic"
                )
            particle_rows.append(
                {
                    "particle_index": particle_index,
                    "weight": float(probabilities[particle_index]),
                    "status": "COMPLETE",
                    "completed_origins": completed,
                    "independently_validated_origins": independently_validated,
                    "all_actions_independently_validated": True,
                    "day_ahead_frozen_after_origin_zero": day_ahead_frozen,
                    "day_ahead_settlement_cost_usd": da_cost,
                    "rt_imbalance_settlement_cost_usd": rt_cost,
                    "day_cost_usd": day_cost,
                    "sla_shortfall_raw_work_units": sla,
                    "normalized_sla_shortfall": sla / max(workload, 1.0e-8),
                    "corrective_intervention_rate": float(
                        np.mean(
                            [
                                bool(row.get("corrective_intervention", False))
                                for row in records
                            ]
                        )
                    ),
                    "corrective_intervention_mean": float(corrections.mean()),
                    "corrective_intervention_p95": float(
                        np.quantile(corrections, 0.95)
                    ),
                }
            )

        costs = np.asarray([row["day_cost_usd"] for row in particle_rows], dtype=float)
        sla_values = np.asarray(
            [row["sla_shortfall_raw_work_units"] for row in particle_rows], dtype=float
        )
        normalized_sla = np.asarray(
            [row["normalized_sla_shortfall"] for row in particle_rows], dtype=float
        )
        order = np.argsort(costs, kind="stable")[::-1]
        ordered_costs = costs[order]
        ordered_weights = probabilities[order]
        tail_mass = 0.05
        mass_before = np.cumsum(ordered_weights) - ordered_weights
        included = np.minimum(
            ordered_weights, np.maximum(tail_mass - mass_before, 0.0)
        )
        cvar = float(np.dot(ordered_costs, included) / tail_mass)
        expected_cost = float(np.dot(probabilities, costs))
        expected_sla = float(np.dot(probabilities, sla_values))
        if not np.isfinite([cvar, expected_cost, expected_sla]).all():
            raise PostShieldParticleReplayError(
                "weighted particle cost/SLA output is non-finite"
            )

        result = {
            "schema_version": self.schema_version,
            "status": "COMPLETE",
            "operating_date": day.operating_date,
            "particle_count": self.sample_count,
            "formal_empirical_risk_sample_count": self.sample_count,
            "policy_uses_distribution_samples": self.policy_uses_distribution_samples,
            "origin_count_per_particle": INTERVALS,
            "official_two_settlement_contract": (
                "0.25*sum_z(DA_schedule*DA_price"
                "+(grid_import-DA_schedule)*RT_price)"
            ),
            "sla_penalty_included_in_day_cost": False,
            "weights": probabilities.tolist(),
            "particle_day_costs_usd": costs.tolist(),
            "weighted_expected_day_cost_usd": expected_cost,
            "weighted_cvar95_usd": cvar,
            "particle_sla_shortfall_raw_work_units": sla_values.tolist(),
            "particle_normalized_sla_shortfall": normalized_sla.tolist(),
            "weighted_expected_sla_shortfall_raw_work_units": expected_sla,
            "maximum_particle_normalized_sla_shortfall": float(
                normalized_sla.max(initial=0.0)
            ),
            "particle_results": particle_rows,
            "all_particles_finite": bool(np.isfinite(costs).all()),
            "all_particles_feasible": True,
            "all_particles_completed_96_origins": True,
            "all_particles_independently_validated": True,
            "all_particles_day_ahead_frozen": True,
            "solver_free": True,
            "input_content_hash_passes": 1 if _binding_frozen_in_current_call else 2,
            "contract_passed": True,
            **current_binding.as_dict(),
        }
        evidence_payload = dict(result)
        result["evidence_sha256"] = sha256_json(evidence_payload)
        return result


class PostShieldDistributionReplay(PostShieldParticleReplay):
    """Formal V6 N=512 replay with samples excluded from the Policy input."""

    def __init__(
        self,
        physical: PhysicalLimits,
        runtime_config: Mapping[str, Any] | None = None,
        *,
        physical_twin_core: Any | None = None,
    ) -> None:
        super().__init__(
            physical,
            runtime_config,
            physical_twin_core=physical_twin_core,
            sample_count=FORMAL_EMPIRICAL_RISK_SAMPLES,
            policy_uses_distribution_samples=False,
            schema_version=DISTRIBUTION_REPLAY_SCHEMA_VERSION,
        )


class PostShieldDistributionAblationReplay(PostShieldParticleReplay):
    """Variable-budget V6 replay used only for preregistered N sensitivity."""

    def __init__(
        self,
        physical: PhysicalLimits,
        sample_count: int,
        runtime_config: Mapping[str, Any] | None = None,
        *,
        physical_twin_core: Any | None = None,
    ) -> None:
        count = int(sample_count)
        if count not in DISTRIBUTION_SAMPLING_BUDGETS:
            raise PostShieldParticleReplayError(
                "Distribution sampling-budget ablation requires one of "
                f"{DISTRIBUTION_SAMPLING_BUDGETS}"
            )
        super().__init__(
            physical,
            runtime_config,
            physical_twin_core=physical_twin_core,
            sample_count=count,
            policy_uses_distribution_samples=False,
            schema_version=DISTRIBUTION_ABLATION_REPLAY_SCHEMA_VERSION,
        )


def replay_post_shield_particle_risk(
    *,
    day: R4SelectionDay,
    particles_zonal: Any,
    weights: Any,
    frozen_day_ahead: Any,
    policy_callback_factory: PolicyCallbackFactory,
    physical: PhysicalLimits,
    policy_contract_sha256: str,
    scenario_contract_sha256: str,
    runtime_config: Mapping[str, Any] | None = None,
    physical_twin_core: Any | None = None,
    binding: PostShieldParticleReplayBinding | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Functional entry point with an optional externally persisted binding.

    When ``binding`` is omitted the function freezes one immediately before
    replay.  Formal multi-stage materialization should persist ``bind(...)``
    first and pass it back here so any intervening mutation fails closed.
    """

    replay = PostShieldParticleReplay(
        physical,
        runtime_config,
        physical_twin_core=physical_twin_core,
    )
    binding_created_now = binding is None
    frozen_binding = (
        replay.bind(
            day=day,
            particles_zonal=particles_zonal,
            weights=weights,
            frozen_day_ahead=frozen_day_ahead,
            policy_contract_sha256=policy_contract_sha256,
            scenario_contract_sha256=scenario_contract_sha256,
        )
        if binding is None
        else PostShieldParticleReplayBinding.from_mapping(binding)
    )
    if frozen_binding.policy_contract_sha256 != _require_sha256(
        policy_contract_sha256, name="policy_contract_sha256"
    ):
        raise PostShieldParticleReplayError(
            "policy_contract_sha256 differs from the supplied replay binding"
        )
    if frozen_binding.scenario_contract_sha256 != _require_sha256(
        scenario_contract_sha256, name="scenario_contract_sha256"
    ):
        raise PostShieldParticleReplayError(
            "scenario_contract_sha256 differs from the supplied replay binding"
        )
    return replay.run(
        day=day,
        particles_zonal=particles_zonal,
        weights=weights,
        frozen_day_ahead=frozen_day_ahead,
        policy_callback_factory=policy_callback_factory,
        binding=frozen_binding,
        _binding_frozen_in_current_call=binding_created_now,
    )


def replay_post_shield_distribution_risk(
    *,
    day: R4SelectionDay,
    samples_zonal: Any,
    weights: Any,
    frozen_day_ahead: Any,
    policy_callback_factory: PolicyCallbackFactory,
    physical: PhysicalLimits,
    policy_contract_sha256: str,
    distribution_contract_sha256: str,
    runtime_config: Mapping[str, Any] | None = None,
    physical_twin_core: Any | None = None,
    binding: PostShieldParticleReplayBinding | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay all N=512 sampled futures before reporting empirical VaR/CVaR.

    The deterministic Policy never receives these samples.  Every trajectory
    instead becomes the realized future of a fresh 96-origin causal session,
    and only post-shield official two-settlement costs enter the empirical
    distribution.
    """

    replay = PostShieldDistributionReplay(
        physical,
        runtime_config,
        physical_twin_core=physical_twin_core,
    )
    binding_created_now = binding is None
    frozen_binding = (
        replay.bind(
            day=day,
            particles_zonal=samples_zonal,
            weights=weights,
            frozen_day_ahead=frozen_day_ahead,
            policy_contract_sha256=policy_contract_sha256,
            scenario_contract_sha256=distribution_contract_sha256,
        )
        if binding is None
        else PostShieldParticleReplayBinding.from_mapping(binding)
    )
    if frozen_binding.schema_version != DISTRIBUTION_REPLAY_SCHEMA_VERSION:
        raise PostShieldParticleReplayError(
            "formal Distribution risk requires a V6 replay binding"
        )
    if frozen_binding.sample_count != FORMAL_EMPIRICAL_RISK_SAMPLES:
        raise PostShieldParticleReplayError(
            "formal Distribution risk binding is not N=512"
        )
    if frozen_binding.policy_contract_sha256 != _require_sha256(
        policy_contract_sha256, name="policy_contract_sha256"
    ):
        raise PostShieldParticleReplayError(
            "policy_contract_sha256 differs from the supplied replay binding"
        )
    if frozen_binding.scenario_contract_sha256 != _require_sha256(
        distribution_contract_sha256, name="distribution_contract_sha256"
    ):
        raise PostShieldParticleReplayError(
            "distribution_contract_sha256 differs from the supplied replay binding"
        )
    result = replay.run(
        day=day,
        particles_zonal=samples_zonal,
        weights=weights,
        frozen_day_ahead=frozen_day_ahead,
        policy_callback_factory=policy_callback_factory,
        binding=frozen_binding,
        _binding_frozen_in_current_call=binding_created_now,
    )
    result["risk_measurement"] = "empirical_post_shield_complete_future_replay"
    result["neural_risk_head_used_as_formal_measurement"] = False
    result["evidence_sha256"] = sha256_json(
        {key: value for key, value in result.items() if key != "evidence_sha256"}
    )
    return result


def replay_post_shield_distribution_ablation(
    *,
    day: R4SelectionDay,
    samples_zonal: Any,
    weights: Any,
    frozen_day_ahead: Any,
    policy_callback_factory: PolicyCallbackFactory,
    physical: PhysicalLimits,
    policy_contract_sha256: str,
    distribution_contract_sha256: str,
    sample_count: int,
    runtime_config: Mapping[str, Any] | None = None,
    physical_twin_core: Any | None = None,
    binding: PostShieldParticleReplayBinding | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one preregistered N without ever passing samples to Policy.

    This is the decision-side interface for the N=20/64/128/256/512
    sensitivity experiment.  Probability scores and sampling latency are
    materialized by the Distribution evaluator; this function contributes the
    post-shield realized-cost and empirical-risk side under the same N.
    """

    replay = PostShieldDistributionAblationReplay(
        physical,
        sample_count,
        runtime_config,
        physical_twin_core=physical_twin_core,
    )
    binding_created_now = binding is None
    frozen_binding = (
        replay.bind(
            day=day,
            particles_zonal=samples_zonal,
            weights=weights,
            frozen_day_ahead=frozen_day_ahead,
            policy_contract_sha256=policy_contract_sha256,
            scenario_contract_sha256=distribution_contract_sha256,
        )
        if binding is None
        else PostShieldParticleReplayBinding.from_mapping(binding)
    )
    if frozen_binding.schema_version != DISTRIBUTION_ABLATION_REPLAY_SCHEMA_VERSION:
        raise PostShieldParticleReplayError(
            "sampling-budget replay requires a V6 ablation binding"
        )
    if frozen_binding.sample_count != int(sample_count):
        raise PostShieldParticleReplayError(
            "sampling-budget replay binding N differs from requested N"
        )
    result = replay.run(
        day=day,
        particles_zonal=samples_zonal,
        weights=weights,
        frozen_day_ahead=frozen_day_ahead,
        policy_callback_factory=policy_callback_factory,
        binding=frozen_binding,
        _binding_frozen_in_current_call=binding_created_now,
    )
    result.update(
        {
            "evaluation_role": "sampling_budget_ablation",
            "sampling_budget": int(sample_count),
            "policy_output_structurally_n_invariant": True,
            "risk_measurement": "empirical_post_shield_complete_future_replay",
            "neural_risk_head_used_as_formal_measurement": False,
        }
    )
    result["evidence_sha256"] = sha256_json(
        {key: value for key, value in result.items() if key != "evidence_sha256"}
    )
    return result


__all__ = [
    "DISTRIBUTION_ABLATION_REPLAY_SCHEMA_VERSION",
    "DISTRIBUTION_REPLAY_SCHEMA_VERSION",
    "DISTRIBUTION_SAMPLING_BUDGETS",
    "FORMAL_EMPIRICAL_RISK_SAMPLES",
    "PostShieldDistributionReplay",
    "PostShieldDistributionAblationReplay",
    "PostShieldParticleReplay",
    "PostShieldParticleReplayBinding",
    "PostShieldParticleReplayError",
    "REPLAY_SCHEMA_VERSION",
    "particle_scenario_hashes",
    "replay_post_shield_distribution_risk",
    "replay_post_shield_distribution_ablation",
    "replay_post_shield_particle_risk",
]
