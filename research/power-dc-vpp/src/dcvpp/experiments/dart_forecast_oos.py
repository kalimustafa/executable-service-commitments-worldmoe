"""Gate-aligned rolling-OOS DART forecast orchestration.

This module is deliberately an orchestration boundary, not a training script.
Model implementations plug in through :class:`OOSForecastRunner`; importing the
module never starts fitting.  The producer owns the chronology, common random
numbers, immutable manifests, serialization gates, and the once-only 2025
contract.

The paper contract is intentionally narrow:

* DA and RT-at-the-DA-gate are separate tasks built from one pre-gate
  information set;
* models are refit once at each annual expanding origin, never every day;
* DART is produced draw by draw as ``DA - RT``;
* 2025 cannot be entered through the ordinary unit runner and can be resumed
  only under the exact fingerprint that opened the locked test.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now


SCHEMA_VERSION = "dart_forecast_oos_v2"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
STUDY_SEEDS = (20260809, 20260819, 20260829)
COMMON_UNIFORM_SEED = 20260809
DEFAULT_DRAW_COUNT = 100


class DARTForecastContractError(RuntimeError):
    """Base error for a forecast-contract violation."""


class FutureDataError(DARTForecastContractError):
    """Raised when a result consumed information unavailable at the DA gate."""


class ResumeContractError(DARTForecastContractError):
    """Raised when a checkpoint cannot be resumed under the current request."""


class LockedTestContractError(DARTForecastContractError):
    """Raised when the once-only locked-test contract would be violated."""


@dataclass(frozen=True)
class SeedSlot:
    """A declared reproducibility slot, including algorithms that ignore RNGs."""

    name: str
    value: int
    role: Literal["neural_study_seed", "statistical_fixed", "deterministic"]


NEURAL_SEED_SLOTS = tuple(
    SeedSlot(f"study_seed_{seed}", seed, "neural_study_seed") for seed in STUDY_SEEDS
)
DETERMINISTIC_SEED_SLOT = SeedSlot("deterministic_seed_0", 0, "deterministic")
STATISTICAL_SEED_SLOT = SeedSlot(
    "statistical_seed_20260809", STUDY_SEEDS[0], "statistical_fixed"
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: Literal["deterministic", "statistical", "neural"]
    seed_slots: tuple[SeedSlot, ...]
    deployable: bool = True


CANDIDATE_POOL = (
    CandidateSpec("seasonal_naive", "deterministic", (DETERMINISTIC_SEED_SLOT,)),
    CandidateSpec("persistence", "deterministic", (DETERMINISTIC_SEED_SLOT,)),
    CandidateSpec("lightgbm_quantile", "statistical", (STATISTICAL_SEED_SLOT,)),
    CandidateSpec("lstm", "neural", NEURAL_SEED_SLOTS),
    CandidateSpec("dlinear", "neural", NEURAL_SEED_SLOTS),
    CandidateSpec("patchtst", "neural", NEURAL_SEED_SLOTS),
    CandidateSpec("itransformer", "neural", NEURAL_SEED_SLOTS),
    CandidateSpec("static_graph_patch_transformer", "neural", NEURAL_SEED_SLOTS),
    CandidateSpec("dgpt_e", "neural", NEURAL_SEED_SLOTS),
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target_field: str
    issue_clock: str = "day_ahead_market_gate"
    information_policy: str = "common_pre_da_gate_only"


TARGETS = (
    TargetSpec("da_lbmp", "da_lbmp"),
    TargetSpec("rt_lbmp_at_da_gate", "rt_lbmp"),
)


@dataclass(frozen=True)
class AnnualOrigin:
    name: str
    fit_through_year: int
    forecast_year: int
    role: Literal["selection_oos", "locked_test"]

    @property
    def training_start_utc(self) -> str:
        return "2021-01-01T00:00:00Z"

    @property
    def training_cutoff_exclusive_utc(self) -> str:
        return f"{self.fit_through_year + 1:04d}-01-01T00:00:00Z"

    @property
    def forecast_start_utc(self) -> str:
        return f"{self.forecast_year:04d}-01-01T00:00:00Z"

    @property
    def forecast_end_exclusive_utc(self) -> str:
        return f"{self.forecast_year + 1:04d}-01-01T00:00:00Z"


ORIGINS = (
    AnnualOrigin("through_2021_to_2022_oos", 2021, 2022, "selection_oos"),
    AnnualOrigin("through_2022_to_2023_oos", 2022, 2023, "selection_oos"),
    AnnualOrigin("through_2023_to_2024_oos", 2023, 2024, "selection_oos"),
    AnnualOrigin("through_2024_refit_for_2025", 2024, 2025, "locked_test"),
)


@dataclass(frozen=True)
class WorkUnit:
    origin: AnnualOrigin
    target: TargetSpec
    candidate: CandidateSpec
    seed_slot: SeedSlot

    @property
    def unit_id(self) -> str:
        return "__".join(
            (
                self.origin.name,
                self.target.name,
                self.candidate.name,
                self.seed_slot.name,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "origin": asdict(self.origin),
            "target": asdict(self.target),
            "candidate": {
                "name": self.candidate.name,
                "family": self.candidate.family,
                "deployable": self.candidate.deployable,
            },
            "seed_slot": asdict(self.seed_slot),
        }


@dataclass(frozen=True)
class DARTOOSPlan:
    candidates: tuple[CandidateSpec, ...] = CANDIDATE_POOL
    targets: tuple[TargetSpec, ...] = TARGETS
    origins: tuple[AnnualOrigin, ...] = ORIGINS
    quantiles: tuple[float, ...] = QUANTILES
    draw_count: int = DEFAULT_DRAW_COUNT
    common_uniform_seed: int = COMMON_UNIFORM_SEED
    contract_profile: Literal["paper", "test_fixture"] = "paper"

    def __post_init__(self) -> None:
        if len(self.targets) != 2 or {item.name for item in self.targets} != {
            "da_lbmp",
            "rt_lbmp_at_da_gate",
        }:
            raise ValueError(
                "the producer requires separate DA and RT-at-DA-gate tasks"
            )
        if tuple(self.quantiles) != QUANTILES:
            raise ValueError(f"quantiles must be exactly {QUANTILES}")
        if self.draw_count < 2:
            raise ValueError("at least two common draws are required")
        if not self.candidates or not all(item.deployable for item in self.candidates):
            raise ValueError("the candidate pool may contain deployable models only")
        if len({item.name for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate names must be unique")
        if len({item.name for item in self.origins}) != len(self.origins):
            raise ValueError("origin names must be unique")
        locked = [item for item in self.origins if item.role == "locked_test"]
        if len(locked) != 1 or locked[0].forecast_year != 2025:
            raise ValueError("exactly one 2025 locked-test origin is required")
        for origin in self.origins:
            if origin.forecast_year != origin.fit_through_year + 1:
                raise ValueError("each annual OOS origin must forecast the next year")
        for candidate in self.candidates:
            if candidate.family == "neural":
                values = tuple(slot.value for slot in candidate.seed_slots)
                roles = {slot.role for slot in candidate.seed_slots}
                if values != STUDY_SEEDS or roles != {"neural_study_seed"}:
                    raise ValueError(
                        f"neural candidate {candidate.name} must use the exact three study seeds"
                    )
            elif len(candidate.seed_slots) != 1:
                raise ValueError(
                    f"non-neural candidate {candidate.name} needs one declared seed slot"
                )
        if self.contract_profile == "paper":
            if self.candidates != CANDIDATE_POOL:
                raise ValueError(
                    "paper mode requires the fixed nine-model candidate pool"
                )
            if self.origins != ORIGINS:
                raise ValueError(
                    "paper mode requires the fixed 2022-2025 annual origins"
                )
            if self.draw_count < DEFAULT_DRAW_COUNT:
                raise ValueError("paper mode requires at least 100 common draws")

    def work_units(self, *, role: str | None = None) -> tuple[WorkUnit, ...]:
        units: list[WorkUnit] = []
        for origin in self.origins:
            if role is not None and origin.role != role:
                continue
            for target in self.targets:
                for round_index, study_seed in enumerate(STUDY_SEEDS):
                    for candidate in self.candidates:
                        slots = tuple(
                            slot
                            for slot in candidate.seed_slots
                            if slot.value == study_seed
                            or (candidate.family != "neural" and round_index == 0)
                        )
                        for seed_slot in slots:
                            units.append(WorkUnit(origin, target, candidate, seed_slot))
        return tuple(units)

    def to_dict(self) -> dict[str, Any]:
        units = self.work_units()
        return {
            "schema": SCHEMA_VERSION,
            "contract_profile": self.contract_profile,
            "candidate_pool": [
                {
                    "name": item.name,
                    "family": item.family,
                    "deployable": item.deployable,
                    "seed_slots": [asdict(slot) for slot in item.seed_slots],
                }
                for item in self.candidates
            ],
            "targets": [asdict(item) for item in self.targets],
            "origins": [
                {
                    **asdict(item),
                    "training_start_utc": item.training_start_utc,
                    "training_cutoff_exclusive_utc": (
                        item.training_cutoff_exclusive_utc
                    ),
                    "forecast_start_utc": item.forecast_start_utc,
                    "forecast_end_exclusive_utc": (item.forecast_end_exclusive_utc),
                }
                for item in self.origins
            ],
            "quantiles": list(self.quantiles),
            "draw_count": self.draw_count,
            "common_uniforms": {
                "base_seed": self.common_uniform_seed,
                "generator": "numpy.PCG64",
                "latent_channel_count": 2,
                "dimensions": ["event", "draw", "latent_channel"],
                "frozen_once_per_annual_origin": True,
                "shared_across_targets_candidates_and_seed_slots": True,
                "not_itself_a_da_rt_coupling": True,
            },
            "execution_schedule": {
                "order": "seed_major_all_candidates_once_per_round",
                "round_seeds": list(STUDY_SEEDS),
                "non_neural_one_fit": "executed_in_round_1_then_reused_without_refit",
                "between_round_diagnostic_scope": "training_and_validation_only",
                "locked_test_feedback_forbidden": True,
            },
            "fit_policy": {
                "window": "expanding",
                "refit_frequency": "annual",
                "daily_retraining": False,
                "selection_oos_years": [
                    item.forecast_year
                    for item in self.origins
                    if item.role == "selection_oos"
                ],
                "locked_test_year": 2025,
            },
            "information_contract": {
                "issue_clock": "day_ahead_market_gate",
                "common_pre_da_gate_information": True,
                "separate_target_tables_scalers_checkpoints": True,
                "future_realized_inputs_forbidden": True,
                "locked_test_feedback_forbidden": True,
            },
            "dart_contract": {
                "canonical_field": "dart_da_minus_rt",
                "canonical_formula": "da_lbmp - rt_lbmp",
                "inverse_field": "rt_spread_rt_minus_da",
                "inverse_formula": "rt_lbmp - da_lbmp",
                "draw_wise_construction": True,
                "sign_gate_after_generation_and_serialization": True,
                "ambiguous_rt_spread_forbidden": True,
                "paper_joint_draw_provider_required": True,
                "joint_coupling_fit_source": "causal_validation_residuals_only",
            },
            "checkpoint_granularity": (
                "annual_origin_x_target_x_candidate_x_seed_slot"
            ),
            "unit_count": len(units),
            "work_units": [item.to_dict() for item in units],
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


def build_dart_oos_plan(
    *,
    draw_count: int = DEFAULT_DRAW_COUNT,
    candidates: Sequence[CandidateSpec] = CANDIDATE_POOL,
    targets: Sequence[TargetSpec] = TARGETS,
    origins: Sequence[AnnualOrigin] = ORIGINS,
    contract_profile: Literal["paper", "test_fixture"] = "paper",
) -> DARTOOSPlan:
    """Build the immutable plan without fitting or opening 2025 data."""

    return DARTOOSPlan(
        candidates=tuple(candidates),
        targets=tuple(targets),
        origins=tuple(origins),
        draw_count=int(draw_count),
        contract_profile=contract_profile,
    )


@dataclass(frozen=True)
class OOSForecastRequest:
    """One annual-fit request passed to an implementation adapter."""

    unit_id: str
    request_fingerprint: str
    plan_fingerprint: str
    origin_name: str
    fit_through_year: int
    forecast_year: int
    target_name: str
    target_field: str
    candidate_name: str
    candidate_family: str
    seed_slot_name: str
    seed: int
    quantiles: tuple[float, ...]
    training_start_utc: str
    training_cutoff_exclusive_utc: str
    forecast_start_utc: str
    forecast_end_exclusive_utc: str
    checkpoint_directory: Path
    resume_checkpoint: Mapping[str, Any] | None
    expected_information_set_fingerprint: str | None
    fit_policy: str = "expanding_annual_refit"
    daily_retraining: bool = False
    issue_clock: str = "day_ahead_market_gate"
    information_policy: str = "common_pre_da_gate_only"


@dataclass(frozen=True)
class OOSForecastResult:
    """Quantile output and mandatory leakage audit returned by a runner.

    ``event_keys`` identifies flattened target cells (for example
    date/zone/horizon tuples).  ``quantile_values`` therefore has shape
    ``(len(event_keys), 5)``.  The information-set fingerprint must describe
    the common source view before candidate-specific transforms, not a
    model-specific feature subset.
    """

    event_keys: Sequence[str]
    quantile_values: np.ndarray
    target_times_utc: Sequence[str]
    da_gate_times_utc: Sequence[str]
    max_feature_availability_times_utc: Sequence[str]
    training_target_max_utc: str
    information_set_fingerprint: str
    realized_targets_used_as_features: bool = False
    future_information_used: bool = False
    model_checkpoint_path: str | None = None
    model_checkpoint_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class OOSForecastRunner(Protocol):
    """Stable adapter interface for existing or future model implementations."""

    def run(self, request: OOSForecastRequest) -> OOSForecastResult:
        """Fit/resume one annual unit and return gate-auditable quantiles."""


@dataclass(frozen=True)
class JointDrawRequest:
    """Causal request for transforming shared base random numbers.

    ``common_base_uniforms`` has two independent latent channels.  It is shared
    across candidates so candidate comparisons use common random numbers, but
    it does not prescribe the DA/RT dependence.  A provider must fit that
    dependence only from residuals available before ``fit_cutoff_exclusive_utc``.
    """

    origin_name: str
    fit_through_year: int
    forecast_year: int
    fit_cutoff_exclusive_utc: str
    forecast_start_utc: str
    candidate_name: str
    seed_slot_name: str
    event_keys: tuple[str, ...]
    common_base_uniforms: np.ndarray
    common_base_random_numbers_sha256: str
    contract_profile: Literal["paper", "test_fixture"]


@dataclass(frozen=True)
class JointDrawResult:
    """Paired marginal uniforms plus immutable coupling provenance."""

    da_uniforms: np.ndarray
    rt_uniforms: np.ndarray
    origin_name: str
    candidate_name: str
    seed_slot_name: str
    fit_cutoff_exclusive_utc: str
    validation_residual_target_max_utc: str
    common_base_random_numbers_sha256: str
    coupling_manifest: Mapping[str, Any]
    coupling_manifest_sha256: str
    locked_targets_used: bool = False


@runtime_checkable
class JointDrawCouplingProvider(Protocol):
    """Adapter for a validation-residual-fitted DA/RT joint draw model."""

    def draw_joint_uniforms(self, request: JointDrawRequest) -> JointDrawResult:
        """Transform shared base randomness without reading locked targets."""


@dataclass(frozen=True)
class UnitExecution:
    unit_id: str
    status: Literal["COMPLETED", "RESUMED"]
    prediction_path: Path
    prediction_sha256: str
    checkpoint_path: Path
    dart_path: Path | None


def _parse_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FutureDataError(f"{label} is not a valid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise FutureDataError(f"{label} must be timezone-aware: {value}")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: str, *, label: str) -> str:
    return _parse_utc(value, label=label).isoformat().replace("+00:00", "Z")


def _valid_sha256(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-fA-F]{64}", str(value)))


def _validate_fingerprints(
    values: Mapping[str, str], *, required: Sequence[str] = ()
) -> dict[str, str]:
    normalized = {str(key): str(value).lower() for key, value in values.items()}
    missing = sorted(set(required) - set(normalized))
    if missing:
        raise DARTForecastContractError(
            f"missing required fingerprints: {', '.join(missing)}"
        )
    invalid = sorted(
        key for key, value in normalized.items() if not _valid_sha256(value)
    )
    if invalid:
        raise DARTForecastContractError(
            f"fingerprints must be SHA-256 hex strings: {', '.join(invalid)}"
        )
    return dict(sorted(normalized.items()))


def _sha256_array(value: np.ndarray) -> str:
    """Hash an array with its dtype and shape, independent of NPZ metadata."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def assert_dart_sign_gate(
    dart_da_minus_rt: np.ndarray,
    rt_spread_rt_minus_da: np.ndarray,
    *,
    da_lbmp: np.ndarray | None = None,
    rt_lbmp: np.ndarray | None = None,
) -> None:
    """Apply the canonical sign gate, optionally including the source draws."""

    dart = np.asarray(dart_da_minus_rt, dtype="float64")
    inverse = np.asarray(rt_spread_rt_minus_da, dtype="float64")
    if dart.shape != inverse.shape:
        raise DARTForecastContractError("DART and inverse-spread shapes differ")
    if not np.allclose(dart, -inverse, equal_nan=True):
        raise DARTForecastContractError(
            "DART sign gate failed: dart_da_minus_rt != -rt_spread_rt_minus_da"
        )
    if (da_lbmp is None) != (rt_lbmp is None):
        raise DARTForecastContractError("DA and RT draws must be supplied together")
    if da_lbmp is not None and rt_lbmp is not None:
        da = np.asarray(da_lbmp, dtype="float64")
        rt = np.asarray(rt_lbmp, dtype="float64")
        if da.shape != dart.shape or rt.shape != dart.shape:
            raise DARTForecastContractError("DA/RT/DART draw shapes differ")
        if not np.allclose(dart, da - rt, equal_nan=True):
            raise DARTForecastContractError(
                "canonical DART does not equal draw-wise DA minus RT"
            )


def quantile_draws_from_uniforms(
    quantile_values: np.ndarray,
    uniforms: np.ndarray,
    *,
    quantiles: Sequence[float] = QUANTILES,
) -> np.ndarray:
    """Interpolate marginal quantiles with common frozen uniforms.

    Uniforms outside the supported 0.05--0.95 range use the boundary quantile;
    no synthetic tail model is silently introduced by this orchestration layer.
    """

    values = np.asarray(quantile_values, dtype="float64")
    common = np.asarray(uniforms, dtype="float64")
    levels = np.asarray(tuple(quantiles), dtype="float64")
    if values.ndim != 2 or values.shape[1] != len(levels):
        raise ValueError("quantile values must have shape (event, quantile)")
    if common.ndim != 2 or common.shape[0] != values.shape[0]:
        raise ValueError("uniforms must have shape (event, draw)")
    if not np.isfinite(values).all() or not np.isfinite(common).all():
        raise ValueError("quantiles and uniforms must be finite")
    if (common < 0.0).any() or (common > 1.0).any():
        raise ValueError("uniforms must lie in [0, 1]")
    if (np.diff(values, axis=1) < 0.0).any():
        raise ValueError("quantile predictions must be monotone")

    upper = np.searchsorted(levels, common, side="right")
    upper = np.clip(upper, 1, len(levels) - 1)
    lower = upper - 1
    lower_values = np.take_along_axis(values, lower, axis=1)
    upper_values = np.take_along_axis(values, upper, axis=1)
    lower_levels = levels[lower]
    upper_levels = levels[upper]
    fraction = (common - lower_levels) / (upper_levels - lower_levels)
    draws = lower_values + fraction * (upper_values - lower_values)
    draws = np.where(common <= levels[0], values[:, :1], draws)
    return np.where(common >= levels[-1], values[:, -1:], draws)


class IndependenceFixtureJointDrawProvider:
    """Explicit test-only coupling using the two independent base channels.

    This adapter is never eligible in paper mode.  Its manifest makes the lack
    of empirical residual fitting machine-readable instead of silently
    imposing identical DA and RT uniforms.
    """

    def draw_joint_uniforms(self, request: JointDrawRequest) -> JointDrawResult:
        if request.contract_profile != "test_fixture":
            raise DARTForecastContractError(
                "the independence fixture coupling is forbidden in paper mode"
            )
        base = np.asarray(request.common_base_uniforms, dtype="float64")
        if base.ndim != 3 or base.shape[2] != 2:
            raise DARTForecastContractError(
                "fixture coupling requires two-channel base random numbers"
            )
        residual_max = f"{request.fit_through_year:04d}-12-31T23:59:59Z"
        manifest = {
            "provider_name": "independence_fixture_joint_draw_provider",
            "provider_version": 1,
            "method": "independent_two_channel_uniform_fixture",
            "fit_role": "test_fixture_only_not_empirically_fitted",
            "fixture": True,
            "origin_name": request.origin_name,
            "candidate_name": request.candidate_name,
            "seed_slot_name": request.seed_slot_name,
            "fit_cutoff_exclusive_utc": request.fit_cutoff_exclusive_utc,
            "validation_residual_target_max_utc": residual_max,
            "source_residuals_sha256": sha256_json(
                {
                    "fixture": "independence",
                    "origin": request.origin_name,
                    "candidate": request.candidate_name,
                }
            ),
            "common_base_random_numbers_sha256": (
                request.common_base_random_numbers_sha256
            ),
            "locked_targets_used": False,
            "locked_target_years_used": [],
        }
        return JointDrawResult(
            da_uniforms=base[:, :, 0],
            rt_uniforms=base[:, :, 1],
            origin_name=request.origin_name,
            candidate_name=request.candidate_name,
            seed_slot_name=request.seed_slot_name,
            fit_cutoff_exclusive_utc=request.fit_cutoff_exclusive_utc,
            validation_residual_target_max_utc=residual_max,
            common_base_random_numbers_sha256=(
                request.common_base_random_numbers_sha256
            ),
            coupling_manifest=manifest,
            coupling_manifest_sha256=sha256_json(manifest),
            locked_targets_used=False,
        )


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DARTForecastContractError(f"JSON root must be a mapping: {path}")
    return value


FREEZE_FINGERPRINT_KEYS = (
    "data_through_2024",
    "feature_contract",
    "model_hyperparameters",
    "calibration_rule",
    "selection_rule",
    "scenario_optimizer_user_contract",
    "code",
)


class DARTForecastOOSProducer:
    """Plan, validate, serialize, resume, freeze, and lock DART forecasts."""

    def __init__(
        self,
        output_root: Path,
        *,
        plan: DARTOOSPlan | None = None,
        input_fingerprints: Mapping[str, str] | None = None,
        joint_draw_provider: JointDrawCouplingProvider | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.plan = plan or build_dart_oos_plan()
        self.input_fingerprints = _validate_fingerprints(input_fingerprints or {})
        self.joint_draw_provider = joint_draw_provider
        if self.plan.contract_profile == "test_fixture" and joint_draw_provider is None:
            self.joint_draw_provider = IndependenceFixtureJointDrawProvider()

    @property
    def plan_manifest_path(self) -> Path:
        return self.output_root / "dart_oos_plan_manifest.json"

    @property
    def freeze_manifest_path(self) -> Path:
        return self.output_root / "selection_freeze_manifest.json"

    @property
    def locked_test_ledger_path(self) -> Path:
        return self.output_root / "locked_test_once.json"

    def initialize(self) -> dict[str, Any]:
        core = {
            "schema": SCHEMA_VERSION,
            "status": "PLANNED",
            "plan": self.plan.to_dict(),
            "plan_fingerprint": self.plan.fingerprint,
            "input_fingerprints": self.input_fingerprints,
            "initialize_starts_formal_training": False,
        }
        manifest_fingerprint = sha256_json(core)
        if self.plan_manifest_path.is_file():
            existing = _json_load(self.plan_manifest_path)
            if existing.get("manifest_fingerprint") != manifest_fingerprint:
                raise ResumeContractError(
                    "existing OOS plan manifest has a different contract"
                )
            return existing
        manifest = {
            **core,
            "manifest_fingerprint": manifest_fingerprint,
            "created_at_utc": utc_now(),
        }
        atomic_json(self.plan_manifest_path, manifest)
        return manifest

    def get_work_unit(
        self,
        origin_name: str,
        target_name: str,
        candidate_name: str,
        seed_slot_name: str,
    ) -> WorkUnit:
        for unit in self.plan.work_units():
            if (
                unit.origin.name,
                unit.target.name,
                unit.candidate.name,
                unit.seed_slot.name,
            ) == (origin_name, target_name, candidate_name, seed_slot_name):
                return unit
        raise KeyError(
            f"unknown work unit: {origin_name}/{target_name}/"
            f"{candidate_name}/{seed_slot_name}"
        )

    def _unit_directory(self, unit: WorkUnit) -> Path:
        return (
            self.output_root
            / "units"
            / unit.origin.name
            / unit.target.name
            / unit.candidate.name
            / unit.seed_slot.name
        )

    def _prediction_path(self, unit: WorkUnit) -> Path:
        return self._unit_directory(unit) / "quantile_predictions.npz"

    def _checkpoint_path(self, unit: WorkUnit) -> Path:
        return self._unit_directory(unit) / "unit_checkpoint.json"

    def _dart_path(self, unit: WorkUnit) -> Path:
        return (
            self.output_root
            / "dart"
            / unit.origin.name
            / unit.candidate.name
            / unit.seed_slot.name
            / "dart_draws.npz"
        )

    def _request_fingerprint(self, unit: WorkUnit) -> str:
        return sha256_json(
            {
                "schema": SCHEMA_VERSION,
                "plan_fingerprint": self.plan.fingerprint,
                "input_fingerprints": self.input_fingerprints,
                "unit": unit.to_dict(),
                "quantiles": list(self.plan.quantiles),
                "fit_policy": "expanding_annual_refit",
                "daily_retraining": False,
                "information_policy": "common_pre_da_gate_only",
            }
        )

    def _information_manifest_path(self, origin: AnnualOrigin) -> Path:
        return self.output_root / "common_information" / f"{origin.name}.json"

    def _uniform_paths(self, origin: AnnualOrigin) -> tuple[Path, Path]:
        root = self.output_root / "common_uniforms"
        return root / f"{origin.name}.npz", root / f"{origin.name}.json"

    def _joint_uniform_paths(self, unit: WorkUnit) -> tuple[Path, Path]:
        root = (
            self.output_root
            / "joint_uniforms"
            / unit.origin.name
            / unit.candidate.name
            / unit.seed_slot.name
        )
        return root / "paired_uniforms.npz", root / "coupling_manifest.json"

    @staticmethod
    def _normalized_result(unit: WorkUnit, result: OOSForecastResult) -> dict[str, Any]:
        keys = tuple(str(value) for value in result.event_keys)
        if not keys or len(set(keys)) != len(keys):
            raise DARTForecastContractError("event keys must be nonempty and unique")
        values = np.asarray(result.quantile_values, dtype="float64")
        if values.shape != (len(keys), len(QUANTILES)):
            raise DARTForecastContractError(
                f"quantile result must have shape ({len(keys)}, {len(QUANTILES)})"
            )
        if not np.isfinite(values).all():
            raise DARTForecastContractError("forecast quantiles must be finite")
        if (np.diff(values, axis=1) < 0.0).any():
            raise DARTForecastContractError("forecast quantiles must be monotone")
        if result.realized_targets_used_as_features or result.future_information_used:
            raise FutureDataError(
                "runner declared future or realized-target feature use"
            )
        if not _valid_sha256(result.information_set_fingerprint):
            raise DARTForecastContractError(
                "information_set_fingerprint must be a SHA-256 hex string"
            )
        lengths = {
            len(result.target_times_utc),
            len(result.da_gate_times_utc),
            len(result.max_feature_availability_times_utc),
        }
        if lengths != {len(keys)}:
            raise DARTForecastContractError(
                "target, DA-gate, and feature-availability audits must align to event keys"
            )

        targets = tuple(
            _utc_text(value, label="target_times_utc")
            for value in result.target_times_utc
        )
        gates = tuple(
            _utc_text(value, label="da_gate_times_utc")
            for value in result.da_gate_times_utc
        )
        availability = tuple(
            _utc_text(value, label="max_feature_availability_times_utc")
            for value in result.max_feature_availability_times_utc
        )
        forecast_start = _parse_utc(
            unit.origin.forecast_start_utc, label="forecast_start_utc"
        )
        forecast_end = _parse_utc(
            unit.origin.forecast_end_exclusive_utc,
            label="forecast_end_exclusive_utc",
        )
        for index, (target_text, gate_text, available_text) in enumerate(
            zip(targets, gates, availability, strict=True)
        ):
            target = _parse_utc(target_text, label=f"target[{index}]")
            gate = _parse_utc(gate_text, label=f"DA gate[{index}]")
            available = _parse_utc(
                available_text, label=f"feature availability[{index}]"
            )
            if not forecast_start <= target < forecast_end:
                raise FutureDataError(
                    f"target {index} is outside OOS year {unit.origin.forecast_year}"
                )
            if gate >= target:
                raise FutureDataError(f"DA gate {index} is not before its target")
            if available > gate:
                raise FutureDataError(
                    f"feature {index} became available after the DA gate"
                )

        training_max_text = _utc_text(
            result.training_target_max_utc, label="training_target_max_utc"
        )
        training_max = _parse_utc(training_max_text, label="training_target_max_utc")
        cutoff = _parse_utc(
            unit.origin.training_cutoff_exclusive_utc,
            label="training_cutoff_exclusive_utc",
        )
        if training_max >= cutoff or training_max >= forecast_start:
            raise FutureDataError(
                "training targets cross the annual expanding-window cutoff"
            )
        metadata = dict(result.metadata)
        if metadata.get("daily_retraining") is True:
            raise DARTForecastContractError("daily retraining is forbidden")
        if metadata.get("realized_targets_used_to_generate_forecast") not in (
            None,
            False,
        ):
            raise FutureDataError(
                "metadata declares realized-target forecast generation"
            )

        checkpoint_path = result.model_checkpoint_path
        checkpoint_sha = result.model_checkpoint_sha256
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise DARTForecastContractError(
                    f"runner model checkpoint is absent: {checkpoint_path}"
                )
            actual = sha256_file(path)
            if checkpoint_sha is not None and actual != str(checkpoint_sha).lower():
                raise DARTForecastContractError("runner model checkpoint hash mismatch")
            checkpoint_sha = actual
        elif checkpoint_sha is not None:
            raise DARTForecastContractError(
                "model checkpoint hash was supplied without a checkpoint path"
            )

        return {
            "event_keys": keys,
            "quantile_values": values,
            "target_times_utc": targets,
            "da_gate_times_utc": gates,
            "max_feature_availability_times_utc": availability,
            "training_target_max_utc": training_max_text,
            "information_set_fingerprint": str(
                result.information_set_fingerprint
            ).lower(),
            "model_checkpoint_path": checkpoint_path,
            "model_checkpoint_sha256": checkpoint_sha,
            "metadata": metadata,
        }

    def _register_common_information(
        self, unit: WorkUnit, normalized: Mapping[str, Any]
    ) -> dict[str, Any]:
        path = self._information_manifest_path(unit.origin)
        contract = {
            "origin": unit.origin.name,
            "forecast_year": unit.origin.forecast_year,
            "information_set_fingerprint": normalized["information_set_fingerprint"],
            "event_keys_sha256": sha256_json(list(normalized["event_keys"])),
            "target_times_sha256": sha256_json(list(normalized["target_times_utc"])),
            "da_gate_times_sha256": sha256_json(list(normalized["da_gate_times_utc"])),
            "max_feature_availability_times_sha256": sha256_json(
                list(normalized["max_feature_availability_times_utc"])
            ),
            "policy": "common_pre_da_gate_only",
            "future_realized_inputs_forbidden": True,
        }
        fingerprint = sha256_json(contract)
        if path.is_file():
            existing = _json_load(path)
            if existing.get("contract_fingerprint") != fingerprint:
                raise FutureDataError(
                    "candidate/target results do not share the frozen pre-DA-gate information set"
                )
            return existing
        manifest = {
            "schema": SCHEMA_VERSION,
            "status": "FROZEN",
            "contract": contract,
            "contract_fingerprint": fingerprint,
            "created_at_utc": utc_now(),
        }
        atomic_json(path, manifest)
        return manifest

    def _ensure_uniforms(
        self, unit: WorkUnit, normalized: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        artifact_path, manifest_path = self._uniform_paths(unit.origin)
        event_keys = tuple(normalized["event_keys"])
        event_keys_sha = sha256_json(list(event_keys))
        derived_seed = int.from_bytes(
            hashlib.sha256(
                f"{self.plan.common_uniform_seed}|{unit.origin.name}".encode()
            ).digest()[:8],
            "big",
        )
        contract = {
            "origin": unit.origin.name,
            "event_keys_sha256": event_keys_sha,
            "event_count": len(event_keys),
            "draw_count": self.plan.draw_count,
            "latent_channel_count": 2,
            "base_seed": self.plan.common_uniform_seed,
            "derived_seed": derived_seed,
            "generator": "numpy.PCG64",
            "shared_across_targets_candidates_and_seed_slots": True,
            "not_itself_a_da_rt_coupling": True,
        }
        contract_fingerprint = sha256_json(contract)
        if artifact_path.is_file() or manifest_path.is_file():
            if not artifact_path.is_file() or not manifest_path.is_file():
                raise ResumeContractError("common-uniform artifact is incomplete")
            manifest = _json_load(manifest_path)
            if manifest.get("contract_fingerprint") != contract_fingerprint:
                raise ResumeContractError("common-uniform contract changed")
            if sha256_file(artifact_path) != manifest.get("artifact_sha256"):
                raise ResumeContractError("frozen common-uniform bytes changed")
            with np.load(artifact_path, allow_pickle=False) as data:
                if "base_uniforms" not in data.files or "rt_spread" in data.files:
                    raise ResumeContractError("invalid common-uniform serialization")
                uniforms = np.asarray(data["base_uniforms"], dtype="float64")
                stored_keys = tuple(map(str, data["event_keys"].tolist()))
            if stored_keys != event_keys or uniforms.shape != (
                len(event_keys),
                self.plan.draw_count,
                2,
            ):
                raise ResumeContractError("frozen common-uniform shape/key mismatch")
            return uniforms, manifest

        rng = np.random.Generator(np.random.PCG64(derived_seed))
        uniforms = rng.random((len(event_keys), self.plan.draw_count, 2))
        _atomic_npz(
            artifact_path,
            base_uniforms=uniforms,
            event_keys=np.asarray(event_keys),
            derived_seed=np.asarray(derived_seed, dtype="uint64"),
        )
        manifest = {
            "schema": SCHEMA_VERSION,
            "status": "FROZEN",
            "contract": contract,
            "contract_fingerprint": contract_fingerprint,
            "artifact_path": artifact_path.as_posix(),
            "artifact_sha256": sha256_file(artifact_path),
            "created_at_utc": utc_now(),
        }
        atomic_json(manifest_path, manifest)
        return uniforms, manifest

    def _ensure_joint_uniforms(
        self,
        unit: WorkUnit,
        *,
        event_keys: tuple[str, ...],
        common_base_uniforms: np.ndarray,
        common_base_manifest: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Materialize provider-derived paired uniforms with causal provenance."""

        provider = self.joint_draw_provider
        if provider is None:
            raise DARTForecastContractError(
                "paper DART assembly requires an explicit validation-residual-fitted "
                "JointDrawCouplingProvider; identical marginal uniforms are forbidden"
            )
        base_sha = str(common_base_manifest.get("artifact_sha256", "")).lower()
        if not _valid_sha256(base_sha):
            raise DARTForecastContractError(
                "common base-random-number artifact hash is absent or invalid"
            )
        request = JointDrawRequest(
            origin_name=unit.origin.name,
            fit_through_year=unit.origin.fit_through_year,
            forecast_year=unit.origin.forecast_year,
            fit_cutoff_exclusive_utc=unit.origin.training_cutoff_exclusive_utc,
            forecast_start_utc=unit.origin.forecast_start_utc,
            candidate_name=unit.candidate.name,
            seed_slot_name=unit.seed_slot.name,
            event_keys=event_keys,
            common_base_uniforms=np.asarray(
                common_base_uniforms, dtype="float64"
            ).copy(),
            common_base_random_numbers_sha256=base_sha,
            contract_profile=self.plan.contract_profile,
        )
        result = provider.draw_joint_uniforms(request)
        if not isinstance(result, JointDrawResult):
            raise DARTForecastContractError(
                "joint-draw provider must return JointDrawResult"
            )
        identity_fields = {
            "origin_name": unit.origin.name,
            "candidate_name": unit.candidate.name,
            "seed_slot_name": unit.seed_slot.name,
            "fit_cutoff_exclusive_utc": unit.origin.training_cutoff_exclusive_utc,
        }
        for field_name, expected in identity_fields.items():
            if getattr(result, field_name) != expected:
                raise DARTForecastContractError(
                    f"joint-draw provider returned the wrong {field_name}"
                )
        if result.locked_targets_used:
            raise FutureDataError("joint coupling used locked targets")
        if result.common_base_random_numbers_sha256.lower() != base_sha:
            raise DARTForecastContractError(
                "joint coupling did not preserve the common base-random-number hash"
            )

        residual_max_text = _utc_text(
            result.validation_residual_target_max_utc,
            label="validation_residual_target_max_utc",
        )
        residual_max = _parse_utc(
            residual_max_text, label="validation_residual_target_max_utc"
        )
        fit_cutoff = _parse_utc(
            unit.origin.training_cutoff_exclusive_utc,
            label="coupling fit cutoff",
        )
        if residual_max >= fit_cutoff:
            raise FutureDataError(
                "joint coupling residuals reach or cross the annual fit cutoff"
            )

        coupling_manifest = dict(result.coupling_manifest)
        coupling_sha = str(result.coupling_manifest_sha256).lower()
        if not _valid_sha256(coupling_sha) or coupling_sha != sha256_json(
            coupling_manifest
        ):
            raise DARTForecastContractError(
                "joint coupling manifest hash is absent or does not match its content"
            )
        required_manifest_values: dict[str, Any] = {
            "origin_name": unit.origin.name,
            "candidate_name": unit.candidate.name,
            "seed_slot_name": unit.seed_slot.name,
            "fit_cutoff_exclusive_utc": unit.origin.training_cutoff_exclusive_utc,
            "validation_residual_target_max_utc": residual_max_text,
            "common_base_random_numbers_sha256": base_sha,
            "locked_targets_used": False,
            "locked_target_years_used": [],
        }
        for field_name, expected in required_manifest_values.items():
            if coupling_manifest.get(field_name) != expected:
                raise DARTForecastContractError(
                    f"joint coupling manifest has invalid {field_name}"
                )
        if not coupling_manifest.get("provider_name") or not coupling_manifest.get(
            "method"
        ):
            raise DARTForecastContractError(
                "joint coupling manifest must name its provider and method"
            )
        if not _valid_sha256(str(coupling_manifest.get("source_residuals_sha256"))):
            raise DARTForecastContractError(
                "joint coupling manifest requires a validation-residual source hash"
            )
        if self.plan.contract_profile == "paper":
            if coupling_manifest.get("fixture") is not False:
                raise DARTForecastContractError(
                    "paper joint coupling cannot be a fixture"
                )
            if coupling_manifest.get("fit_role") != "validation_residuals_only":
                raise DARTForecastContractError(
                    "paper joint coupling must be fitted from validation residuals only"
                )

        da_uniforms = np.asarray(result.da_uniforms, dtype="float64")
        rt_uniforms = np.asarray(result.rt_uniforms, dtype="float64")
        expected_shape = (len(event_keys), self.plan.draw_count)
        if da_uniforms.shape != expected_shape or rt_uniforms.shape != expected_shape:
            raise DARTForecastContractError(
                f"joint uniforms must both have shape {expected_shape}"
            )
        if not np.isfinite(da_uniforms).all() or not np.isfinite(rt_uniforms).all():
            raise DARTForecastContractError("joint uniforms must be finite")
        if (
            (da_uniforms < 0.0).any()
            or (da_uniforms > 1.0).any()
            or (rt_uniforms < 0.0).any()
            or (rt_uniforms > 1.0).any()
        ):
            raise DARTForecastContractError("joint uniforms must lie in [0, 1]")
        if (
            self.plan.contract_profile == "paper"
            and np.array_equal(da_uniforms, rt_uniforms)
            and coupling_manifest.get(
                "perfect_comonotonicity_fitted_and_verified"
            )
            is not True
        ):
            raise DARTForecastContractError(
                "identical DA/RT uniforms are an implicit comonotonic coupling; "
                "paper mode requires explicit residual evidence"
            )

        contract = {
            "origin": unit.origin.name,
            "candidate": unit.candidate.name,
            "seed_slot": unit.seed_slot.name,
            "event_keys_sha256": sha256_json(list(event_keys)),
            "event_count": len(event_keys),
            "draw_count": self.plan.draw_count,
            "fit_cutoff_exclusive_utc": unit.origin.training_cutoff_exclusive_utc,
            "validation_residual_target_max_utc": residual_max_text,
            "common_base_random_numbers_sha256": base_sha,
            "coupling_manifest_sha256": coupling_sha,
            "da_uniforms_sha256": _sha256_array(da_uniforms),
            "rt_uniforms_sha256": _sha256_array(rt_uniforms),
            "locked_targets_used": False,
            "contract_profile": self.plan.contract_profile,
        }
        contract_fingerprint = sha256_json(contract)
        artifact_path, manifest_path = self._joint_uniform_paths(unit)
        if artifact_path.is_file() or manifest_path.is_file():
            if not artifact_path.is_file() or not manifest_path.is_file():
                raise ResumeContractError("joint-uniform artifact is incomplete")
            manifest = _json_load(manifest_path)
            if manifest.get("contract_fingerprint") != contract_fingerprint:
                raise ResumeContractError("joint-coupling contract changed")
            if manifest.get("coupling_manifest_sha256") != coupling_sha:
                raise ResumeContractError("joint-coupling manifest hash changed")
            if manifest.get("coupling_manifest") != coupling_manifest:
                raise ResumeContractError("joint-coupling manifest content changed")
            if sha256_file(artifact_path) != manifest.get("artifact_sha256"):
                raise ResumeContractError("frozen joint-uniform bytes changed")
            with np.load(artifact_path, allow_pickle=False) as data:
                required = {
                    "event_keys",
                    "da_uniforms",
                    "rt_uniforms",
                    "common_base_random_numbers_sha256",
                    "coupling_manifest_sha256",
                }
                if not required.issubset(data.files):
                    raise ResumeContractError(
                        "joint-uniform serialization is incomplete"
                    )
                stored_keys = tuple(map(str, data["event_keys"].tolist()))
                stored_da = np.asarray(data["da_uniforms"], dtype="float64")
                stored_rt = np.asarray(data["rt_uniforms"], dtype="float64")
                stored_base_sha = str(
                    data["common_base_random_numbers_sha256"].item()
                )
                stored_coupling_sha = str(data["coupling_manifest_sha256"].item())
            if (
                stored_keys != event_keys
                or _sha256_array(stored_da) != contract["da_uniforms_sha256"]
                or _sha256_array(stored_rt) != contract["rt_uniforms_sha256"]
                or stored_base_sha != base_sha
                or stored_coupling_sha != coupling_sha
            ):
                raise ResumeContractError(
                    "joint-uniform serialization/provenance changed"
                )
            return stored_da, stored_rt, manifest

        _atomic_npz(
            artifact_path,
            event_keys=np.asarray(event_keys),
            da_uniforms=da_uniforms,
            rt_uniforms=rt_uniforms,
            common_base_random_numbers_sha256=np.asarray(base_sha),
            coupling_manifest_sha256=np.asarray(coupling_sha),
        )
        manifest = {
            "schema": SCHEMA_VERSION,
            "status": "FROZEN",
            "contract": contract,
            "contract_fingerprint": contract_fingerprint,
            "coupling_manifest": coupling_manifest,
            "coupling_manifest_sha256": coupling_sha,
            "artifact_path": artifact_path.as_posix(),
            "artifact_sha256": sha256_file(artifact_path),
            "created_at_utc": utc_now(),
        }
        atomic_json(manifest_path, manifest)
        return da_uniforms, rt_uniforms, manifest

    def _write_prediction(
        self,
        unit: WorkUnit,
        normalized: Mapping[str, Any],
        request_fingerprint: str,
    ) -> tuple[Path, str]:
        path = self._prediction_path(unit)
        metadata = {
            "schema": SCHEMA_VERSION,
            "unit": unit.to_dict(),
            "request_fingerprint": request_fingerprint,
            "information_set_fingerprint": normalized["information_set_fingerprint"],
            "training_target_max_utc": normalized["training_target_max_utc"],
            "model_checkpoint_path": normalized["model_checkpoint_path"],
            "model_checkpoint_sha256": normalized["model_checkpoint_sha256"],
            "runner_metadata": normalized["metadata"],
            "realized_targets_used_as_features": False,
            "future_information_used": False,
            "daily_retraining": False,
            "evidence_role": "derived",
            "source_relationship": "third_party_referenced_public_data",
        }
        _atomic_npz(
            path,
            event_keys=np.asarray(normalized["event_keys"]),
            quantile_values=np.asarray(normalized["quantile_values"], dtype="float64"),
            quantile_levels=np.asarray(self.plan.quantiles, dtype="float64"),
            target_times_utc=np.asarray(normalized["target_times_utc"]),
            da_gate_times_utc=np.asarray(normalized["da_gate_times_utc"]),
            max_feature_availability_times_utc=np.asarray(
                normalized["max_feature_availability_times_utc"]
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
        )
        loaded = self._load_prediction(path, expected_request=request_fingerprint)
        if not np.array_equal(loaded["quantile_values"], normalized["quantile_values"]):
            raise DARTForecastContractError(
                "quantile serialization round-trip changed predictions"
            )
        return path, sha256_file(path)

    def _load_prediction(
        self, path: Path, *, expected_request: str | None = None
    ) -> dict[str, Any]:
        if not path.is_file():
            raise ResumeContractError(f"prediction artifact is absent: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "rt_spread" in data.files:
                raise DARTForecastContractError(
                    "ambiguous rt_spread field is forbidden in forecast artifacts"
                )
            required = {
                "event_keys",
                "quantile_values",
                "quantile_levels",
                "target_times_utc",
                "da_gate_times_utc",
                "max_feature_availability_times_utc",
                "metadata_json",
            }
            if not required.issubset(data.files):
                raise ResumeContractError("prediction artifact schema is incomplete")
            metadata = json.loads(str(data["metadata_json"].item()))
            result = {
                "event_keys": tuple(map(str, data["event_keys"].tolist())),
                "quantile_values": np.asarray(data["quantile_values"], dtype="float64"),
                "quantile_levels": tuple(map(float, data["quantile_levels"].tolist())),
                "target_times_utc": tuple(map(str, data["target_times_utc"].tolist())),
                "da_gate_times_utc": tuple(
                    map(str, data["da_gate_times_utc"].tolist())
                ),
                "max_feature_availability_times_utc": tuple(
                    map(
                        str,
                        data["max_feature_availability_times_utc"].tolist(),
                    )
                ),
                "metadata": metadata,
            }
        if result["quantile_levels"] != self.plan.quantiles:
            raise ResumeContractError("serialized quantile levels changed")
        if (
            expected_request is not None
            and metadata.get("request_fingerprint") != expected_request
        ):
            raise ResumeContractError("prediction request fingerprint changed")
        values = result["quantile_values"]
        if values.shape != (len(result["event_keys"]), len(self.plan.quantiles)):
            raise ResumeContractError("serialized prediction shape is invalid")
        if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0.0).any():
            raise ResumeContractError("serialized quantiles are invalid")
        return result

    def _load_completed_unit(
        self, unit: WorkUnit, *, request_fingerprint: str
    ) -> dict[str, Any] | None:
        checkpoint_path = self._checkpoint_path(unit)
        if not checkpoint_path.is_file():
            return None
        checkpoint = _json_load(checkpoint_path)
        if checkpoint.get("request_fingerprint") != request_fingerprint:
            raise ResumeContractError(f"unit request changed: {unit.unit_id}")
        if checkpoint.get("status") != "COMPLETE":
            return None
        prediction_path = Path(checkpoint["prediction_artifact"]["path"])
        if sha256_file(prediction_path) != checkpoint["prediction_artifact"]["sha256"]:
            raise ResumeContractError(f"unit prediction bytes changed: {unit.unit_id}")
        prediction = self._load_prediction(
            prediction_path, expected_request=request_fingerprint
        )
        return {"checkpoint": checkpoint, "prediction": prediction}

    def _paired_unit(self, unit: WorkUnit) -> WorkUnit:
        other = "rt_lbmp_at_da_gate" if unit.target.name == "da_lbmp" else "da_lbmp"
        return self.get_work_unit(
            unit.origin.name,
            other,
            unit.candidate.name,
            unit.seed_slot.name,
        )

    def _ensure_dart_artifact(self, unit: WorkUnit) -> Path | None:
        pair = self._paired_unit(unit)
        unit_loaded = self._load_completed_unit(
            unit, request_fingerprint=self._request_fingerprint(unit)
        )
        pair_loaded = self._load_completed_unit(
            pair, request_fingerprint=self._request_fingerprint(pair)
        )
        if unit_loaded is None or pair_loaded is None:
            return None
        by_target = {
            unit.target.name: unit_loaded["prediction"],
            pair.target.name: pair_loaded["prediction"],
        }
        da = by_target["da_lbmp"]
        rt = by_target["rt_lbmp_at_da_gate"]
        for field_name in (
            "event_keys",
            "target_times_utc",
            "da_gate_times_utc",
            "max_feature_availability_times_utc",
        ):
            if da[field_name] != rt[field_name]:
                raise FutureDataError(
                    f"DA/RT-at-DA-gate pairing differs for {field_name}"
                )
        if (
            da["metadata"]["information_set_fingerprint"]
            != rt["metadata"]["information_set_fingerprint"]
        ):
            raise FutureDataError("DA and RT used different pre-gate information")

        normalized = {
            "event_keys": da["event_keys"],
            "quantile_values": da["quantile_values"],
        }
        base_uniforms, uniform_manifest = self._ensure_uniforms(unit, normalized)
        da_uniforms, rt_uniforms, joint_manifest = self._ensure_joint_uniforms(
            unit,
            event_keys=tuple(da["event_keys"]),
            common_base_uniforms=base_uniforms,
            common_base_manifest=uniform_manifest,
        )
        joint_artifact_path = Path(joint_manifest["artifact_path"])
        _, joint_manifest_path = self._joint_uniform_paths(unit)
        dart_path = self._dart_path(unit)
        dart_manifest_path = dart_path.with_suffix(".json")
        source_hashes = {
            "da_prediction": unit_loaded["checkpoint"]["prediction_artifact"]["sha256"]
            if unit.target.name == "da_lbmp"
            else pair_loaded["checkpoint"]["prediction_artifact"]["sha256"],
            "rt_prediction": unit_loaded["checkpoint"]["prediction_artifact"]["sha256"]
            if unit.target.name == "rt_lbmp_at_da_gate"
            else pair_loaded["checkpoint"]["prediction_artifact"]["sha256"],
            "common_base_random_numbers": uniform_manifest["artifact_sha256"],
            "joint_uniforms": joint_manifest["artifact_sha256"],
            "joint_coupling_manifest_file": sha256_file(joint_manifest_path),
        }
        contract = {
            "origin": unit.origin.name,
            "candidate": unit.candidate.name,
            "seed_slot": unit.seed_slot.name,
            "source_hashes": source_hashes,
            "formula": "draw-wise da_lbmp - rt_lbmp",
            "inverse_formula": "rt_lbmp - da_lbmp",
            "quantiles": list(self.plan.quantiles),
            "coupling_manifest_sha256": joint_manifest[
                "coupling_manifest_sha256"
            ],
            "coupling_fit_cutoff_exclusive_utc": joint_manifest["contract"][
                "fit_cutoff_exclusive_utc"
            ],
            "coupling_validation_residual_target_max_utc": joint_manifest[
                "contract"
            ]["validation_residual_target_max_utc"],
            "locked_targets_used_for_coupling": False,
        }
        contract_fingerprint = sha256_json(contract)
        if dart_path.is_file() or dart_manifest_path.is_file():
            if not dart_path.is_file() or not dart_manifest_path.is_file():
                raise ResumeContractError("DART artifact is incomplete")
            manifest = _json_load(dart_manifest_path)
            if manifest.get("contract_fingerprint") != contract_fingerprint:
                raise ResumeContractError("DART source contract changed")
            if sha256_file(dart_path) != manifest.get("artifact_sha256"):
                raise ResumeContractError("serialized DART bytes changed")
            self._verify_dart_serialization(dart_path)
            return dart_path

        da_draws = quantile_draws_from_uniforms(
            da["quantile_values"], da_uniforms, quantiles=self.plan.quantiles
        )
        rt_draws = quantile_draws_from_uniforms(
            rt["quantile_values"], rt_uniforms, quantiles=self.plan.quantiles
        )
        dart = da_draws - rt_draws
        inverse = rt_draws - da_draws
        assert_dart_sign_gate(dart, inverse, da_lbmp=da_draws, rt_lbmp=rt_draws)
        dart_quantiles = np.quantile(
            dart, np.asarray(self.plan.quantiles), axis=1, method="linear"
        ).T
        _atomic_npz(
            dart_path,
            event_keys=np.asarray(da["event_keys"]),
            quantile_levels=np.asarray(self.plan.quantiles),
            da_lbmp_draws=da_draws,
            rt_lbmp_draws=rt_draws,
            dart_da_minus_rt=dart,
            rt_spread_rt_minus_da=inverse,
            dart_quantile_values=dart_quantiles,
            common_base_random_numbers_sha256=np.asarray(
                uniform_manifest["artifact_sha256"]
            ),
            joint_uniforms_sha256=np.asarray(sha256_file(joint_artifact_path)),
            coupling_manifest_sha256=np.asarray(
                joint_manifest["coupling_manifest_sha256"]
            ),
        )
        self._verify_dart_serialization(dart_path)
        manifest = {
            "schema": SCHEMA_VERSION,
            "status": "COMPLETE",
            "contract": contract,
            "contract_fingerprint": contract_fingerprint,
            "artifact_path": dart_path.as_posix(),
            "artifact_sha256": sha256_file(dart_path),
            "sign_gate_after_generation": "PASSED",
            "sign_gate_after_serialization": "PASSED",
            "ambiguous_rt_spread_present": False,
            "joint_coupling_provenance_verified": True,
            "created_at_utc": utc_now(),
        }
        atomic_json(dart_manifest_path, manifest)
        return dart_path

    @staticmethod
    def _verify_dart_serialization(path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            if "rt_spread" in data.files:
                raise DARTForecastContractError(
                    "ambiguous rt_spread field survived serialization"
                )
            required = {
                "da_lbmp_draws",
                "rt_lbmp_draws",
                "dart_da_minus_rt",
                "rt_spread_rt_minus_da",
                "common_base_random_numbers_sha256",
                "joint_uniforms_sha256",
                "coupling_manifest_sha256",
            }
            if not required.issubset(data.files):
                raise DARTForecastContractError("serialized DART schema is incomplete")
            assert_dart_sign_gate(
                data["dart_da_minus_rt"],
                data["rt_spread_rt_minus_da"],
                da_lbmp=data["da_lbmp_draws"],
                rt_lbmp=data["rt_lbmp_draws"],
            )
            for field_name in (
                "common_base_random_numbers_sha256",
                "joint_uniforms_sha256",
                "coupling_manifest_sha256",
            ):
                if not _valid_sha256(str(data[field_name].item())):
                    raise DARTForecastContractError(
                        f"serialized DART has invalid {field_name}"
                    )

    def execute_unit(
        self,
        unit: WorkUnit,
        runner: OOSForecastRunner,
        *,
        resume: bool = True,
        _allow_locked_test: bool = False,
    ) -> UnitExecution:
        """Execute one annual unit; locked 2025 units require ``run_locked_test``."""

        self.initialize()
        if unit.origin.role == "locked_test" and not _allow_locked_test:
            raise LockedTestContractError(
                "2025 can be opened only through the once-only locked-test method"
            )
        request_fingerprint = self._request_fingerprint(unit)
        completed = self._load_completed_unit(
            unit, request_fingerprint=request_fingerprint
        )
        if completed is not None:
            if not resume:
                raise ResumeContractError(
                    f"completed unit cannot be overwritten: {unit.unit_id}"
                )
            dart_path = self._ensure_dart_artifact(unit)
            artifact = completed["checkpoint"]["prediction_artifact"]
            return UnitExecution(
                unit.unit_id,
                "RESUMED",
                Path(artifact["path"]),
                artifact["sha256"],
                self._checkpoint_path(unit),
                dart_path,
            )

        checkpoint_path = self._checkpoint_path(unit)
        prior_checkpoint: dict[str, Any] | None = None
        if checkpoint_path.is_file():
            prior_checkpoint = _json_load(checkpoint_path)
            if prior_checkpoint.get("request_fingerprint") != request_fingerprint:
                raise ResumeContractError(f"unit request changed: {unit.unit_id}")
            if not resume:
                raise ResumeContractError(
                    f"incomplete unit requires same-fingerprint resume: {unit.unit_id}"
                )
        attempt_count = int((prior_checkpoint or {}).get("attempt_count", 0)) + 1
        started_checkpoint = {
            "schema": SCHEMA_VERSION,
            "status": "STARTED",
            "checkpoint_granularity": (
                "annual_origin_x_target_x_candidate_x_seed_slot"
            ),
            "unit": unit.to_dict(),
            "request_fingerprint": request_fingerprint,
            "runner_checkpoint_directory": (
                self._unit_directory(unit) / "runner_state"
            ).as_posix(),
            "attempt_count": attempt_count,
            "first_started_at_utc": (prior_checkpoint or {}).get(
                "first_started_at_utc", utc_now()
            ),
            "last_started_at_utc": utc_now(),
        }
        atomic_json(checkpoint_path, started_checkpoint)

        information_path = self._information_manifest_path(unit.origin)
        expected_information = None
        if information_path.is_file():
            expected_information = _json_load(information_path)["contract"][
                "information_set_fingerprint"
            ]
        request = OOSForecastRequest(
            unit_id=unit.unit_id,
            request_fingerprint=request_fingerprint,
            plan_fingerprint=self.plan.fingerprint,
            origin_name=unit.origin.name,
            fit_through_year=unit.origin.fit_through_year,
            forecast_year=unit.origin.forecast_year,
            target_name=unit.target.name,
            target_field=unit.target.target_field,
            candidate_name=unit.candidate.name,
            candidate_family=unit.candidate.family,
            seed_slot_name=unit.seed_slot.name,
            seed=unit.seed_slot.value,
            quantiles=self.plan.quantiles,
            training_start_utc=unit.origin.training_start_utc,
            training_cutoff_exclusive_utc=(unit.origin.training_cutoff_exclusive_utc),
            forecast_start_utc=unit.origin.forecast_start_utc,
            forecast_end_exclusive_utc=unit.origin.forecast_end_exclusive_utc,
            checkpoint_directory=self._unit_directory(unit) / "runner_state",
            resume_checkpoint=prior_checkpoint,
            expected_information_set_fingerprint=expected_information,
        )
        try:
            result = runner.run(request)
            normalized = self._normalized_result(unit, result)
            if (
                expected_information is not None
                and normalized["information_set_fingerprint"] != expected_information
            ):
                raise FutureDataError(
                    "runner changed the frozen pre-gate information set"
                )
            self._register_common_information(unit, normalized)
            self._ensure_uniforms(unit, normalized)
            prediction_path, prediction_sha = self._write_prediction(
                unit, normalized, request_fingerprint
            )
        except Exception as exc:
            incomplete = {
                **started_checkpoint,
                "status": "INCOMPLETE",
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "last_error_at_utc": utc_now(),
                "resume_same_fingerprint_allowed": True,
            }
            atomic_json(checkpoint_path, incomplete)
            raise
        checkpoint = {
            "schema": SCHEMA_VERSION,
            "status": "COMPLETE",
            "checkpoint_granularity": (
                "annual_origin_x_target_x_candidate_x_seed_slot"
            ),
            "unit": unit.to_dict(),
            "request_fingerprint": request_fingerprint,
            "prediction_artifact": {
                "path": prediction_path.as_posix(),
                "sha256": prediction_sha,
            },
            "runner_model_checkpoint": {
                "path": normalized["model_checkpoint_path"],
                "sha256": normalized["model_checkpoint_sha256"],
            },
            "training_target_max_utc": normalized["training_target_max_utc"],
            "information_set_fingerprint": normalized["information_set_fingerprint"],
            "daily_retraining": False,
            "attempt_count": attempt_count,
            "resumed_from_incomplete_checkpoint": prior_checkpoint is not None,
            "first_started_at_utc": started_checkpoint["first_started_at_utc"],
            "completed_at_utc": utc_now(),
        }
        atomic_json(self._checkpoint_path(unit), checkpoint)
        dart_path = self._ensure_dart_artifact(unit)
        return UnitExecution(
            unit.unit_id,
            "COMPLETED",
            prediction_path,
            prediction_sha,
            self._checkpoint_path(unit),
            dart_path,
        )

    def execute_selection(
        self, runner: OOSForecastRunner, *, resume: bool = True
    ) -> list[UnitExecution]:
        """Execute only 2022--2024 annual OOS units; never opens 2025."""

        return [
            self.execute_unit(unit, runner, resume=resume)
            for unit in self.plan.work_units(role="selection_oos")
        ]

    def _collect_evidence(self, *, role: str) -> dict[str, Any]:
        prediction_records: list[dict[str, str]] = []
        dart_records: dict[str, dict[str, str]] = {}
        information_records: dict[str, str] = {}
        uniform_records: dict[str, str] = {}
        coupling_records: dict[str, str] = {}
        for unit in self.plan.work_units(role=role):
            request_fingerprint = self._request_fingerprint(unit)
            completed = self._load_completed_unit(
                unit, request_fingerprint=request_fingerprint
            )
            if completed is None:
                raise DARTForecastContractError(
                    f"required unit is incomplete: {unit.unit_id}"
                )
            prediction_records.append(
                {
                    "unit_id": unit.unit_id,
                    "sha256": completed["checkpoint"]["prediction_artifact"]["sha256"],
                }
            )
            dart_path = self._ensure_dart_artifact(unit)
            if dart_path is None:
                raise DARTForecastContractError(
                    f"paired DART artifact is incomplete: {unit.unit_id}"
                )
            dart_records[dart_path.as_posix()] = {
                "path": dart_path.as_posix(),
                "sha256": sha256_file(dart_path),
            }
            info_path = self._information_manifest_path(unit.origin)
            _, uniform_manifest_path = self._uniform_paths(unit.origin)
            _, coupling_manifest_path = self._joint_uniform_paths(unit)
            information_records[unit.origin.name] = sha256_file(info_path)
            uniform_records[unit.origin.name] = sha256_file(uniform_manifest_path)
            coupling_records[
                f"{unit.origin.name}/{unit.candidate.name}/{unit.seed_slot.name}"
            ] = sha256_file(coupling_manifest_path)
        return {
            "prediction_artifacts": sorted(
                prediction_records, key=lambda item: item["unit_id"]
            ),
            "dart_artifacts": sorted(
                dart_records.values(), key=lambda item: item["path"]
            ),
            "common_information_manifest_sha256": dict(
                sorted(information_records.items())
            ),
            "common_uniform_manifest_sha256": dict(sorted(uniform_records.items())),
            "joint_coupling_manifest_sha256": dict(
                sorted(coupling_records.items())
            ),
        }

    def _locked_output_exists(self) -> bool:
        locked = next(item for item in self.plan.origins if item.role == "locked_test")
        paths = (
            self.output_root / "units" / locked.name,
            self.output_root / "dart" / locked.name,
            self._information_manifest_path(locked),
            self._uniform_paths(locked)[0],
            self._uniform_paths(locked)[1],
            self.output_root / "joint_uniforms" / locked.name,
        )
        return any(path.exists() for path in paths)

    def freeze_selection(
        self, dependency_fingerprints: Mapping[str, str]
    ) -> dict[str, Any]:
        """Freeze 2022--2024 evidence and all rules before 2025 is opened."""

        self.initialize()
        dependencies = _validate_fingerprints(
            dependency_fingerprints, required=FREEZE_FINGERPRINT_KEYS
        )
        if self._locked_output_exists() or self.locked_test_ledger_path.exists():
            raise LockedTestContractError(
                "selection cannot be frozen or changed after any 2025 output exists"
            )
        evidence = self._collect_evidence(role="selection_oos")
        frozen_contract = {
            "schema": SCHEMA_VERSION,
            "plan_fingerprint": self.plan.fingerprint,
            "input_fingerprints": self.input_fingerprints,
            "dependency_fingerprints": dependencies,
            "selection_evidence": evidence,
            "selection_evidence_fingerprint": sha256_json(evidence),
            "selection_years": [
                item.forecast_year
                for item in self.plan.origins
                if item.role == "selection_oos"
            ],
            "final_refit_through_year": 2024,
            "locked_test_year": 2025,
            "locked_test_runs_allowed": 1,
            "locked_test_results_may_modify_selection": False,
            "common_pre_da_gate_information": True,
            "daily_retraining": False,
        }
        fingerprint = sha256_json(frozen_contract)
        if self.freeze_manifest_path.is_file():
            existing = self._load_freeze_manifest()
            if existing["freeze_fingerprint"] != fingerprint:
                raise LockedTestContractError(
                    "immutable selection freeze already exists with different inputs"
                )
            return existing
        manifest = {
            "status": "FROZEN_BEFORE_2025",
            "frozen_contract": frozen_contract,
            "freeze_fingerprint": fingerprint,
            "frozen_at_utc": utc_now(),
        }
        atomic_json(self.freeze_manifest_path, manifest)
        return manifest

    def _load_freeze_manifest(self) -> dict[str, Any]:
        if not self.freeze_manifest_path.is_file():
            raise LockedTestContractError("selection freeze manifest is absent")
        manifest = _json_load(self.freeze_manifest_path)
        contract = manifest.get("frozen_contract")
        if not isinstance(contract, dict):
            raise LockedTestContractError("selection freeze contract is invalid")
        if manifest.get("freeze_fingerprint") != sha256_json(contract):
            raise LockedTestContractError("selection freeze manifest was modified")
        if contract.get("plan_fingerprint") != self.plan.fingerprint:
            raise LockedTestContractError("selection freeze pins a different plan")
        return manifest

    def run_locked_test(
        self,
        runner: OOSForecastRunner,
        *,
        dependency_fingerprints: Mapping[str, str],
        resume: bool = True,
    ) -> dict[str, Any]:
        """Run or resume the once-only 2025 refit/evaluation fingerprint."""

        self.initialize()
        dependencies = _validate_fingerprints(
            dependency_fingerprints, required=FREEZE_FINGERPRINT_KEYS
        )
        freeze = self._load_freeze_manifest()
        frozen_dependencies = freeze["frozen_contract"]["dependency_fingerprints"]
        if dependencies != frozen_dependencies:
            raise LockedTestContractError(
                "2025 dependencies differ from the pre-2025 selection freeze"
            )
        freeze_file_sha = sha256_file(self.freeze_manifest_path)
        locked_origin = next(
            item for item in self.plan.origins if item.role == "locked_test"
        )
        locked_contract = {
            "schema": SCHEMA_VERSION,
            "plan_fingerprint": self.plan.fingerprint,
            "freeze_fingerprint": freeze["freeze_fingerprint"],
            "freeze_manifest_sha256": freeze_file_sha,
            "dependency_fingerprints": dependencies,
            "origin": asdict(locked_origin),
            "locked_test_runs_allowed": 1,
            "resume_same_fingerprint_allowed": True,
            "locked_test_feedback_forbidden": True,
        }
        locked_fingerprint = sha256_json(locked_contract)
        if self.locked_test_ledger_path.is_file():
            ledger = _json_load(self.locked_test_ledger_path)
            if ledger.get("locked_test_fingerprint") != locked_fingerprint:
                raise LockedTestContractError(
                    "a different fingerprint already opened the 2025 locked test"
                )
            if ledger.get("freeze_manifest_sha256") != freeze_file_sha:
                raise LockedTestContractError(
                    "selection freeze bytes changed after opening 2025"
                )
            if ledger.get("status") == "COMPLETE":
                if not resume:
                    raise LockedTestContractError(
                        "the once-only 2025 test is already complete"
                    )
                evidence = self._collect_evidence(role="locked_test")
                if sha256_json(evidence) != ledger.get("evidence_fingerprint"):
                    raise LockedTestContractError("locked-test evidence changed")
                return ledger
            if not resume:
                raise LockedTestContractError(
                    "the once-only 2025 test is already open; use resume"
                )
        else:
            if self._locked_output_exists():
                raise LockedTestContractError(
                    "2025 outputs exist without the once-only opening ledger"
                )
            ledger = {
                "schema": SCHEMA_VERSION,
                "status": "STARTED",
                "locked_test_fingerprint": locked_fingerprint,
                "locked_contract": locked_contract,
                "freeze_manifest_sha256": freeze_file_sha,
                "run_count": 1,
                "started_at_utc": utc_now(),
                "locked_test_results_may_modify_selection": False,
            }
            atomic_json(self.locked_test_ledger_path, ledger)

        try:
            executions = [
                self.execute_unit(
                    unit,
                    runner,
                    resume=resume,
                    _allow_locked_test=True,
                )
                for unit in self.plan.work_units(role="locked_test")
            ]
            evidence = self._collect_evidence(role="locked_test")
            complete = {
                **ledger,
                "status": "COMPLETE",
                "completed_at_utc": utc_now(),
                "execution_count": len(executions),
                "evidence": evidence,
                "evidence_fingerprint": sha256_json(evidence),
                "run_count": 1,
                "selection_updated_from_2025": False,
            }
            atomic_json(self.locked_test_ledger_path, complete)
            return complete
        except Exception as exc:
            failed = {
                **ledger,
                "status": "STARTED",
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "last_error_at_utc": utc_now(),
                "run_count": 1,
            }
            atomic_json(self.locked_test_ledger_path, failed)
            raise


__all__ = [
    "AnnualOrigin",
    "CANDIDATE_POOL",
    "CandidateSpec",
    "COMMON_UNIFORM_SEED",
    "DEFAULT_DRAW_COUNT",
    "DARTForecastContractError",
    "DARTForecastOOSProducer",
    "DARTOOSPlan",
    "FREEZE_FINGERPRINT_KEYS",
    "FutureDataError",
    "IndependenceFixtureJointDrawProvider",
    "JointDrawCouplingProvider",
    "JointDrawRequest",
    "JointDrawResult",
    "LockedTestContractError",
    "OOSForecastRequest",
    "OOSForecastResult",
    "OOSForecastRunner",
    "ORIGINS",
    "QUANTILES",
    "ResumeContractError",
    "STUDY_SEEDS",
    "SeedSlot",
    "TARGETS",
    "TargetSpec",
    "UnitExecution",
    "WorkUnit",
    "assert_dart_sign_gate",
    "build_dart_oos_plan",
    "quantile_draws_from_uniforms",
]
