"""Real, fail-closed evidence bridge for the DART value-selection experiment.

This module connects completed annual OOS DART artifacts to the historical
replay contract.  It intentionally does not train models and it never creates
synthetic replacements for missing evidence.  Candidate forecasts are read
from the DART OOS producer, NYISO settlement is an external referenced record,
and workload/PV replay forcing must be supplied as a frozen, candidate-
independent input.  Realized paths are opened only after the day-ahead plan is
frozen.

The public entry point is :func:`run_real_dart_phase_evidence`.  A successful
call returns a mapping containing a :class:`DARTPhaseEvidence`; a fail-closed
preflight or replay error returns ``status=PARTIAL`` and an actionable manifest
without an evidence object.  No exception is converted into a completed cell.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.evaluation.action_weighted_crps import (
    action_weighted_crps,
    economic_action_labels,
    economic_action_metrics,
    economic_action_probabilities,
    economic_net_gain_signal,
    feasibility_conditioned_threshold_weighted_crps,
    feasible_opportunity_ranking_loss,
    validate_gate_time_flexibility,
    validate_normalized_feasibility_weight,
)
from dcvpp.evaluation.dart_metrics import (
    crps_from_quantiles,
    emphasized_tail_threshold_weighted_crps,
    energy_score,
    exceedance_calibration,
    interval_calibration,
    pinball_loss,
    pit_from_ensemble,
    threshold_weighted_crps,
    variogram_score,
)
from dcvpp.experiments.dart_forecast_oos import (
    CANDIDATE_POOL,
    FREEZE_FINGERPRINT_KEYS,
    ORIGINS,
    QUANTILES,
)
from dcvpp.experiments.dart_partial_candidate import (
    candidate_specs_for_contract,
    validate_partial_candidate_pool_contract,
)
from dcvpp.experiments.dart_replay import (
    A1_ALLOWED_TO_DIFFER,
    A2_ALLOWED_TO_DIFFER,
    CANDIDATE_INDEPENDENT_HASHES,
    DART_VALUE_SELECT_SEEDS,
    CandidateEvidence,
    ReplayCell,
    assert_experiment_identity,
    build_common_replay_panel,
    run_dart_replay_panel,
)
from dcvpp.experiments.dart_training import ReferencedNYISOPanelSource
from dcvpp.experiments.dart_value_select import DARTPhaseEvidence
from dcvpp.optimization.replay_engine import (
    FrozenReplayForcing,
    ReferencedNYISOReplay,
    build_a1_dart_bundle,
    run_external_replay_day,
)
from dcvpp.optimization.stochastic_milp import (
    freeze_stochastic_first_stage,
    solve_perfect_information_bound,
)
from dcvpp.research_branch.contracts import (
    FrozenForecastArtifact,
    FrozenPhysicalArtifact,
    NYISO_ZONES,
    ScenarioBundle,
)
from dcvpp.research_branch.optimization import solve_policy
from dcvpp.research_branch.rolling import ROLLING_INTERVAL_HOURS, _first_interval_dispatch
from dcvpp.research_branch.runner import FrozenForecastStore, _forecast_days
from dcvpp.research_branch.scenarios import DEFAULT_REGIMES, build_scenario_bundle


SCHEMA_VERSION = "dart_real_phase_evidence_v3"
MAXIMUM_ACCEPTED_MIP_GAP = 0.05
INPUT_BLOCK = "dart_real_evidence_inputs"
TAIL_THRESHOLD_SCHEMA = "dcvpp.per_zone_rolling_q01_q05_q95_q99_thresholds.v1"
_REAL_EVIDENCE_COMMON_HASHES = {
    "issue_time_panel_sha256",
    "calibration_panel_sha256",
    "tail_threshold_contract_sha256",
}
REQUIRED_GATE_KEYS = (
    "external_historical_nyiso_replay",
    "first_stage_frozen_before_replay",
    "causal_recourse_no_future_reveal",
    "counterfactual_labels",
    "a1_identity_exact",
    "a2_allowed_changes_exact",
    "three_layers_complete",
    "no_future_or_locked_leakage",
    "external_replay_causal_provenance_complete",
    "bounded_gap_solver_feasibility_rule_applied",
    "ratio_of_sums_metric_aggregation_complete",
    "layer1_layer2_diagnostics_emitted",
)
_CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATE_POOL}


class DARTRealEvidenceError(RuntimeError):
    """Raised for a real-evidence contract violation."""


class MissingRealEvidenceError(DARTRealEvidenceError):
    """Raised when a required real input is absent; no fallback is allowed."""

    def __init__(self, message: str, *, pending_count: int = 1) -> None:
        super().__init__(message)
        self.pending_count = max(int(pending_count), 1)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _valid_sha256(value: Any, *, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DARTRealEvidenceError(f"{label} is not a SHA-256 digest")
    return digest


def _resolve_path(raw: str | Path, *, root: Path, sibling: Path | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    root_candidate = (root / path).resolve()
    if root_candidate.exists() or sibling is None:
        return root_candidate
    return (sibling / path).resolve()


def _fingerprint_value(value: Any) -> Any:
    """JSON-safe value fingerprint that never string-truncates numpy arrays."""

    if isinstance(value, np.ndarray):
        return {
            "array_sha256": _array_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, FrozenPhysicalArtifact):
        return _physical_fingerprint(value)
    if isinstance(value, FrozenForecastArtifact):
        return _forecast_fingerprint(value)
    if isinstance(value, Mapping):
        return {str(key): _fingerprint_value(nested) for key, nested in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(nested) for nested in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _physical_fingerprint(physical: FrozenPhysicalArtifact) -> dict[str, Any]:
    return {
        "calibration_id": physical.calibration_id,
        "twin_ids": list(physical.twin_ids),
        "transfer_regime": physical.transfer_regime,
        "zones": list(physical.zones),
        "arrays": {
            name: _array_sha256(getattr(physical, name))
            for name in (
                "idle_mw",
                "work_mw_per_unit",
                "capacity_work_units",
                "grid_import_limit_mw",
                "battery_power_mw",
                "battery_energy_mwh",
                "physical_multiplier_pool",
            )
        },
        "source_hashes": dict(sorted(physical.source_hashes.items())),
        "frozen": physical.frozen,
    }


def _forecast_fingerprint(forecast: FrozenForecastArtifact) -> dict[str, Any]:
    return {
        "issue_time_utc": forecast.issue_time_utc,
        "operating_date": forecast.operating_date,
        "forecast_id": forecast.forecast_id,
        "zones": list(forecast.zones),
        "arrays": {
            name: _array_sha256(getattr(forecast, name))
            for name in (
                "timestamps_utc",
                "da_price",
                "rt_price_pool",
                "workload_pool",
                "pv_pool",
            )
        },
        "source_hashes": dict(sorted(forecast.source_hashes.items())),
        "locked_test": forecast.locked_test,
    }


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DARTRealEvidenceError(f"JSON root must be a mapping: {path}")
    return value


def _phase_origins(phase: str) -> tuple[Any, ...]:
    if phase == "selection":
        return tuple(origin for origin in ORIGINS if origin.role == "selection_oos")
    if phase == "locked_test":
        return tuple(origin for origin in ORIGINS if origin.role == "locked_test")
    raise ValueError("phase must be selection or locked_test")


def _slot_for_seed(candidate_name: str, seed: int) -> str:
    candidate = _CANDIDATE_BY_NAME[candidate_name]
    if candidate.family == "neural":
        return f"study_seed_{seed}"
    if candidate.family == "statistical":
        return "statistical_seed_20260809"
    return "deterministic_seed_0"


def _parse_event_key(value: str) -> tuple[pd.Timestamp, str]:
    if "|" not in value:
        raise DARTRealEvidenceError(f"DART event key lacks zone separator: {value}")
    timestamp_text, zone = value.rsplit("|", 1)
    if zone not in NYISO_ZONES:
        raise DARTRealEvidenceError(f"DART event key has an unknown zone: {value}")
    timestamp = pd.Timestamp(timestamp_text)
    if timestamp.tzinfo is None:
        raise DARTRealEvidenceError(f"DART event key timestamp is timezone-naive: {value}")
    return timestamp.tz_convert("UTC"), zone


def _event_layout(event_keys: Sequence[str], *, forecast_year: int) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[str, ...]]]:
    by_day: dict[str, dict[str, list[tuple[pd.Timestamp, int]]]] = {}
    for index, key in enumerate(event_keys):
        timestamp, zone = _parse_event_key(str(key))
        if timestamp.year != forecast_year:
            raise DARTRealEvidenceError(
                f"event {key} is outside forecast year {forecast_year}"
            )
        day = timestamp.date().isoformat()
        by_day.setdefault(day, {name: [] for name in NYISO_ZONES})[zone].append((timestamp, index))
    indices: dict[str, tuple[int, ...]] = {}
    timestamps: dict[str, tuple[str, ...]] = {}
    for day, zones in sorted(by_day.items()):
        ordered_times: tuple[pd.Timestamp, ...] | None = None
        day_indices: list[int] = []
        for zone in NYISO_ZONES:
            rows = sorted(zones[zone], key=lambda item: item[0])
            if not rows:
                raise DARTRealEvidenceError(f"{day} lacks DART events for {zone}")
            zone_times = tuple(item[0] for item in rows)
            if len(set(zone_times)) != len(zone_times):
                raise DARTRealEvidenceError(f"{day}/{zone} has duplicate target times")
            if ordered_times is None:
                ordered_times = zone_times
            elif zone_times != ordered_times:
                raise DARTRealEvidenceError(f"{day} has an unaligned four-zone time grid")
            day_indices.extend(item[1] for item in rows)
        assert ordered_times is not None
        indices[day] = tuple(day_indices)
        timestamps[day] = tuple(item.isoformat().replace("+00:00", "Z") for item in ordered_times)
    if not indices:
        raise DARTRealEvidenceError("DART artifact contains no events")
    return indices, timestamps


@dataclass(frozen=True)
class OOSArtifactRecord:
    origin_name: str
    forecast_year: int
    candidate_name: str
    seed_slot_name: str
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path
    manifest_sha256: str
    event_keys_sha256: str
    dates: tuple[str, ...]
    timestamps_by_date: Mapping[str, tuple[str, ...]]
    model_sha256: str
    model_byte_count: int
    da_prediction_sha256: str
    rt_prediction_sha256: str
    calibration_sha256: str
    calibration_panel_sha256: str
    dependence_sha256: str
    information_set_sha256: str
    issue_time_panel_sha256: str
    leakage_gate_passed: bool

    @property
    def key(self) -> tuple[int, str, str]:
        return self.forecast_year, self.candidate_name, self.seed_slot_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_name": self.origin_name,
            "forecast_year": self.forecast_year,
            "candidate_name": self.candidate_name,
            "seed_slot_name": self.seed_slot_name,
            "artifact_path": self.artifact_path.as_posix(),
            "artifact_sha256": self.artifact_sha256,
            "manifest_path": self.manifest_path.as_posix(),
            "manifest_sha256": self.manifest_sha256,
            "event_keys_sha256": self.event_keys_sha256,
            "dates": list(self.dates),
            "timestamps_by_date_sha256": sha256_json(self.timestamps_by_date),
            "model_sha256": self.model_sha256,
            "model_byte_count": self.model_byte_count,
            "da_prediction_sha256": self.da_prediction_sha256,
            "rt_prediction_sha256": self.rt_prediction_sha256,
            "calibration_sha256": self.calibration_sha256,
            "calibration_panel_sha256": self.calibration_panel_sha256,
            "dependence_sha256": self.dependence_sha256,
            "information_set_sha256": self.information_set_sha256,
            "issue_time_panel_sha256": self.issue_time_panel_sha256,
            "leakage_gate_passed": self.leakage_gate_passed,
        }


@dataclass(frozen=True)
class OOSInventory:
    records: tuple[OOSArtifactRecord, ...]
    records_by_key: Mapping[tuple[int, str, str], OOSArtifactRecord]
    dates: tuple[str, ...]
    timestamps_by_date: Mapping[str, tuple[str, ...]]
    fingerprint: str
    all_leakage_gates_passed: bool


def _load_unit_contract(
    stage_root: Path,
    *,
    repository_root: Path,
    origin_name: str,
    target_name: str,
    candidate_name: str,
    seed_slot_name: str,
) -> dict[str, Any]:
    unit_root = stage_root / "units" / origin_name / target_name / candidate_name / seed_slot_name
    checkpoint_path = unit_root / "unit_checkpoint.json"
    prediction_path = unit_root / "quantile_predictions.npz"
    if not checkpoint_path.is_file() or not prediction_path.is_file():
        raise MissingRealEvidenceError(
            f"required OOS unit evidence is absent: {origin_name}/{target_name}/{candidate_name}/{seed_slot_name}"
        )
    checkpoint = _read_json(checkpoint_path)
    if checkpoint.get("status") != "COMPLETE":
        raise MissingRealEvidenceError(f"OOS unit is not COMPLETE: {checkpoint_path}")
    prediction_sha = sha256_file(prediction_path)
    if checkpoint.get("prediction_artifact", {}).get("sha256") != prediction_sha:
        raise DARTRealEvidenceError(f"prediction hash mismatch: {prediction_path}")
    model = checkpoint.get("runner_model_checkpoint", {})
    model_path_raw = model.get("path")
    model_sha = _valid_sha256(model.get("sha256"), label=f"model checkpoint {checkpoint_path}")
    if not model_path_raw:
        raise MissingRealEvidenceError(f"unit lacks a real model checkpoint path: {checkpoint_path}")
    model_path = _resolve_path(model_path_raw, root=repository_root, sibling=unit_root)
    if not model_path.is_file() or sha256_file(model_path) != model_sha:
        raise DARTRealEvidenceError(f"model checkpoint bytes changed: {model_path}")
    with np.load(prediction_path, allow_pickle=False) as prediction:
        prediction_required = {
            "event_keys",
            "quantile_values",
            "quantile_levels",
            "target_times_utc",
            "da_gate_times_utc",
            "max_feature_availability_times_utc",
            "metadata_json",
        }
        prediction_missing = prediction_required.difference(prediction.files)
        if prediction_missing:
            raise DARTRealEvidenceError(
                f"prediction schema is incomplete ({sorted(prediction_missing)}): {prediction_path}"
            )
        event_keys = tuple(map(str, prediction["event_keys"].tolist()))
        quantile_values = np.asarray(prediction["quantile_values"], dtype=float)
        quantile_levels = tuple(map(float, prediction["quantile_levels"].tolist()))
        target_times = pd.to_datetime(
            prediction["target_times_utc"].tolist(), utc=True, errors="raise"
        )
        gate_times = pd.to_datetime(
            prediction["da_gate_times_utc"].tolist(), utc=True, errors="raise"
        )
        availability_times = pd.to_datetime(
            prediction["max_feature_availability_times_utc"].tolist(),
            utc=True,
            errors="raise",
        )
        metadata = json.loads(str(prediction["metadata_json"].item()))
    if (
        quantile_levels != QUANTILES
        or quantile_values.shape != (len(event_keys), len(QUANTILES))
        or not np.isfinite(quantile_values).all()
        or np.any(np.diff(quantile_values, axis=1) < 0.0)
    ):
        raise DARTRealEvidenceError(f"prediction quantile contract failed: {prediction_path}")
    if not (
        len(target_times) == len(gate_times) == len(availability_times) == len(event_keys)
        and np.all(availability_times < gate_times)
        and np.all(gate_times < target_times)
    ):
        raise DARTRealEvidenceError(f"prediction DA-gate chronology failed: {prediction_path}")
    if (
        metadata.get("information_set_fingerprint")
        != checkpoint.get("information_set_fingerprint")
        or metadata.get("model_checkpoint_sha256") != model_sha
        or metadata.get("future_information_used") is not False
        or metadata.get("realized_targets_used_as_features") is not False
        or metadata.get("daily_retraining") is not False
    ):
        raise DARTRealEvidenceError(f"prediction leakage/model metadata failed: {prediction_path}")
    runner_metadata = metadata.get("runner_metadata", {})
    if not isinstance(runner_metadata, Mapping):
        raise DARTRealEvidenceError(
            f"prediction runner metadata is not a mapping: {prediction_path}"
        )
    calibration_panel_sha = runner_metadata.get("calibration_panel_sha256")
    calibration_panel_sha = _valid_sha256(
        calibration_panel_sha,
        label=f"calibration panel {prediction_path}",
    )
    issue_time_panel_sha = sha256_json(
        {
            "event_keys": list(event_keys),
            "target_times_utc": [value.isoformat() for value in target_times],
            "da_gate_times_utc": [value.isoformat() for value in gate_times],
        }
    )
    training_max = pd.Timestamp(checkpoint["training_target_max_utc"])
    if training_max.tzinfo is None:
        raise DARTRealEvidenceError(f"training cutoff is timezone-naive: {checkpoint_path}")
    return {
        "prediction_sha256": prediction_sha,
        "model_sha256": model_sha,
        "model_byte_count": model_path.stat().st_size,
        "information_set_sha256": _valid_sha256(
            checkpoint.get("information_set_fingerprint"),
            label=f"information set {checkpoint_path}",
        ),
        "issue_time_panel_sha256": issue_time_panel_sha,
        "calibration_panel_sha256": calibration_panel_sha,
        "training_target_max_utc": training_max.tz_convert("UTC"),
        "future_information_used": metadata["future_information_used"],
        "realized_targets_used_as_features": metadata[
            "realized_targets_used_as_features"
        ],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _verify_joint_coupling(
    stage_root: Path,
    *,
    origin: Any,
    candidate_name: str,
    seed_slot_name: str,
    event_keys_sha256: str,
    event_count: int,
    coupling_sha256: str,
    dart_source_hashes: Mapping[str, Any],
) -> str:
    coupling_root = (
        stage_root
        / "joint_uniforms"
        / origin.name
        / candidate_name
        / seed_slot_name
    )
    artifact_path = coupling_root / "paired_uniforms.npz"
    manifest_path = coupling_root / "coupling_manifest.json"
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise MissingRealEvidenceError(
            f"paired residual-coupling evidence is absent: {coupling_root}"
        )
    manifest = _read_json(manifest_path)
    artifact_sha = sha256_file(artifact_path)
    manifest_sha = sha256_file(manifest_path)
    if (
        manifest.get("status") != "FROZEN"
        or manifest.get("artifact_sha256") != artifact_sha
        or manifest.get("coupling_manifest_sha256") != coupling_sha256
        or dart_source_hashes.get("joint_uniforms") != artifact_sha
        or dart_source_hashes.get("joint_coupling_manifest_file") != manifest_sha
    ):
        raise DARTRealEvidenceError(
            f"paired residual-coupling hashes/status changed: {coupling_root}"
        )
    contract = manifest.get("contract")
    provenance = manifest.get("coupling_manifest")
    if not isinstance(contract, Mapping) or not isinstance(provenance, Mapping):
        raise DARTRealEvidenceError(f"coupling contract/provenance is absent: {manifest_path}")
    if sha256_json(provenance) != coupling_sha256:
        raise DARTRealEvidenceError(
            f"coupling provenance content hash changed: {manifest_path}"
        )
    cutoff = pd.Timestamp(origin.training_cutoff_exclusive_utc)
    residual_max = pd.Timestamp(contract.get("validation_residual_target_max_utc"))
    if cutoff.tzinfo is None or residual_max.tzinfo is None or residual_max.tz_convert("UTC") >= cutoff:
        raise DARTRealEvidenceError(f"coupling residuals cross the fit cutoff: {manifest_path}")
    expected_contract = {
        "origin": origin.name,
        "candidate": candidate_name,
        "seed_slot": seed_slot_name,
        "event_keys_sha256": event_keys_sha256,
        "event_count": event_count,
        "draw_count": 100,
        "fit_cutoff_exclusive_utc": origin.training_cutoff_exclusive_utc,
        "coupling_manifest_sha256": coupling_sha256,
        "locked_targets_used": False,
        "contract_profile": "paper",
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise DARTRealEvidenceError(
                f"coupling contract field {key} changed: {manifest_path}"
            )
    provenance_required = {
        "provider_name",
        "method",
        "fit_role",
        "fixture",
        "origin_name",
        "candidate_name",
        "seed_slot_name",
        "fit_cutoff_exclusive_utc",
        "validation_residual_target_max_utc",
        "source_residuals_sha256",
        "common_base_random_numbers_sha256",
        "locked_targets_used",
        "locked_target_years_used",
    }
    if not provenance_required.issubset(provenance):
        raise DARTRealEvidenceError(
            f"coupling provenance fields are incomplete: {manifest_path}"
        )
    if (
        provenance["fit_role"] != "validation_residuals_only"
        or provenance["fixture"] is not False
        or provenance["origin_name"] != origin.name
        or provenance["candidate_name"] != candidate_name
        or provenance["seed_slot_name"] != seed_slot_name
        or provenance["fit_cutoff_exclusive_utc"] != origin.training_cutoff_exclusive_utc
        or provenance["validation_residual_target_max_utc"]
        != contract["validation_residual_target_max_utc"]
        or provenance["locked_targets_used"] is not False
        or list(provenance["locked_target_years_used"]) != []
    ):
        raise DARTRealEvidenceError(f"coupling provenance/leakage gate failed: {manifest_path}")
    residual_source_sha = _valid_sha256(
        provenance["source_residuals_sha256"],
        label=f"coupling residual source {manifest_path}",
    )
    base_sha = _valid_sha256(
        provenance["common_base_random_numbers_sha256"],
        label=f"coupling common base {manifest_path}",
    )
    if contract.get("common_base_random_numbers_sha256") != base_sha:
        raise DARTRealEvidenceError(f"coupling common-base hash changed: {manifest_path}")
    with np.load(artifact_path, allow_pickle=False) as paired:
        required = {
            "event_keys",
            "da_uniforms",
            "rt_uniforms",
            "common_base_random_numbers_sha256",
            "coupling_manifest_sha256",
        }
        if not required.issubset(paired.files):
            raise DARTRealEvidenceError(f"paired-uniform schema is incomplete: {artifact_path}")
        stored_event_sha = sha256_json(list(map(str, paired["event_keys"].tolist())))
        da_uniforms = np.asarray(paired["da_uniforms"], dtype=float)
        rt_uniforms = np.asarray(paired["rt_uniforms"], dtype=float)
        stored_base_sha = str(paired["common_base_random_numbers_sha256"].item())
        stored_coupling_sha = str(paired["coupling_manifest_sha256"].item())
    if (
        stored_event_sha != event_keys_sha256
        or da_uniforms.shape != (event_count, 100)
        or rt_uniforms.shape != (event_count, 100)
        or not np.isfinite(da_uniforms).all()
        or not np.isfinite(rt_uniforms).all()
        or np.any((da_uniforms < 0.0) | (da_uniforms > 1.0))
        or np.any((rt_uniforms < 0.0) | (rt_uniforms > 1.0))
        or np.array_equal(da_uniforms, rt_uniforms)
        or stored_base_sha != base_sha
        or stored_coupling_sha != coupling_sha256
    ):
        raise DARTRealEvidenceError(
            f"paired uniforms are invalid or implicitly comonotonic: {artifact_path}"
        )
    return sha256_json(
        {
            "artifact_sha256": artifact_sha,
            "manifest_sha256": manifest_sha,
            "coupling_manifest_sha256": coupling_sha256,
            "source_residuals_sha256": residual_source_sha,
        }
    )


def _discover_oos_inventory(
    *,
    repository_root: Path,
    stage_root: Path,
    phase: str,
    real_contract: Mapping[str, Any],
    candidate_specs: Sequence[Any] = CANDIDATE_POOL,
) -> OOSInventory:
    records: list[OOSArtifactRecord] = []
    common_dates: tuple[str, ...] | None = None
    common_timestamps: dict[str, tuple[str, ...]] | None = None
    expected_intervals = int(real_contract.get("expected_intervals_per_day", 96))
    allow_sparse = bool(real_contract.get("allow_sparse_oos_for_focused_tests", False))
    for origin in _phase_origins(phase):
        origin_dates: tuple[str, ...] | None = None
        origin_event_sha: str | None = None
        origin_timestamps: dict[str, tuple[str, ...]] | None = None
        origin_issue_time_panel_sha: str | None = None
        origin_calibration_panel_sha: str | None = None
        for candidate in candidate_specs:
            for seed_slot in candidate.seed_slots:
                artifact_path = (
                    stage_root
                    / "dart"
                    / origin.name
                    / candidate.name
                    / seed_slot.name
                    / "dart_draws.npz"
                )
                manifest_path = artifact_path.with_suffix(".json")
                if not artifact_path.is_file() or not manifest_path.is_file():
                    raise MissingRealEvidenceError(
                        f"required DART artifact is absent: {artifact_path}"
                    )
                artifact_sha = sha256_file(artifact_path)
                manifest = _read_json(manifest_path)
                if manifest.get("status") != "COMPLETE":
                    raise MissingRealEvidenceError(f"DART artifact is not COMPLETE: {artifact_path}")
                if manifest.get("artifact_sha256") != artifact_sha:
                    raise DARTRealEvidenceError(f"DART manifest hash mismatch: {artifact_path}")
                contract = manifest.get("contract")
                if not isinstance(contract, Mapping):
                    raise DARTRealEvidenceError(f"DART manifest lacks its contract: {manifest_path}")
                if (
                    contract.get("origin") != origin.name
                    or contract.get("candidate") != candidate.name
                    or contract.get("seed_slot") != seed_slot.name
                ):
                    raise DARTRealEvidenceError(f"DART path and contract disagree: {manifest_path}")
                da_unit = _load_unit_contract(
                    stage_root,
                    repository_root=repository_root,
                    origin_name=origin.name,
                    target_name="da_lbmp",
                    candidate_name=candidate.name,
                    seed_slot_name=seed_slot.name,
                )
                rt_unit = _load_unit_contract(
                    stage_root,
                    repository_root=repository_root,
                    origin_name=origin.name,
                    target_name="rt_lbmp_at_da_gate",
                    candidate_name=candidate.name,
                    seed_slot_name=seed_slot.name,
                )
                if da_unit["information_set_sha256"] != rt_unit["information_set_sha256"]:
                    raise DARTRealEvidenceError("DA and RT OOS units used different information sets")
                if da_unit["issue_time_panel_sha256"] != rt_unit["issue_time_panel_sha256"]:
                    raise DARTRealEvidenceError(
                        "DA and RT OOS units used different target/gate-time panels"
                    )
                if (
                    da_unit["calibration_panel_sha256"]
                    != rt_unit["calibration_panel_sha256"]
                ):
                    raise DARTRealEvidenceError(
                        "DA and RT OOS units used different calibration panels"
                    )
                cutoff = pd.Timestamp(f"{origin.forecast_year:04d}-01-01T00:00:00Z")
                if da_unit["training_target_max_utc"] >= cutoff or rt_unit["training_target_max_utc"] >= cutoff:
                    raise DARTRealEvidenceError("OOS training targets cross the annual cutoff")
                source_hashes = contract.get("source_hashes", {})
                if source_hashes.get("da_prediction") != da_unit["prediction_sha256"]:
                    raise DARTRealEvidenceError("DART DA source hash differs from its unit")
                if source_hashes.get("rt_prediction") != rt_unit["prediction_sha256"]:
                    raise DARTRealEvidenceError("DART RT source hash differs from its unit")
                leakage_passed = bool(
                    manifest.get("joint_coupling_provenance_verified")
                    and manifest.get("sign_gate_after_generation") == "PASSED"
                    and manifest.get("sign_gate_after_serialization") == "PASSED"
                    and contract.get("locked_targets_used_for_coupling") is False
                    and not da_unit["future_information_used"]
                    and not rt_unit["future_information_used"]
                    and not da_unit["realized_targets_used_as_features"]
                    and not rt_unit["realized_targets_used_as_features"]
                )
                if not leakage_passed:
                    raise DARTRealEvidenceError(f"future/locked leakage gate failed: {manifest_path}")
                with np.load(artifact_path, allow_pickle=False) as data:
                    required = {
                        "event_keys",
                        "quantile_levels",
                        "da_lbmp_draws",
                        "rt_lbmp_draws",
                        "dart_da_minus_rt",
                        "rt_spread_rt_minus_da",
                        "dart_quantile_values",
                        "common_base_random_numbers_sha256",
                        "joint_uniforms_sha256",
                        "coupling_manifest_sha256",
                    }
                    missing = required.difference(data.files)
                    if missing:
                        raise DARTRealEvidenceError(
                            f"DART artifact schema is incomplete ({sorted(missing)}): {artifact_path}"
                        )
                    keys = tuple(map(str, data["event_keys"].tolist()))
                    if tuple(map(float, data["quantile_levels"].tolist())) != QUANTILES:
                        raise DARTRealEvidenceError(f"DART quantiles changed: {artifact_path}")
                    da = np.asarray(data["da_lbmp_draws"], dtype=float)
                    rt = np.asarray(data["rt_lbmp_draws"], dtype=float)
                    dart = np.asarray(data["dart_da_minus_rt"], dtype=float)
                    inverse = np.asarray(data["rt_spread_rt_minus_da"], dtype=float)
                    qvalues = np.asarray(data["dart_quantile_values"], dtype=float)
                    if not (
                        da.shape == rt.shape == dart.shape == inverse.shape
                        and da.ndim == 2
                        and da.shape[0] == len(keys)
                        and da.shape[1] == 100
                        and qvalues.shape == (len(keys), len(QUANTILES))
                    ):
                        raise DARTRealEvidenceError(f"DART array shapes are invalid: {artifact_path}")
                    if not all(np.isfinite(value).all() for value in (da, rt, dart, inverse, qvalues)):
                        raise DARTRealEvidenceError(f"DART artifact contains non-finite values: {artifact_path}")
                    if not np.allclose(dart, da - rt) or not np.allclose(dart, -inverse):
                        raise DARTRealEvidenceError(f"DART sign/identity gate failed: {artifact_path}")
                    recomputed_quantiles = np.quantile(
                        dart, np.asarray(QUANTILES), axis=1, method="linear"
                    ).T
                    if not np.allclose(qvalues, recomputed_quantiles) or np.any(
                        np.diff(qvalues, axis=1) < 0.0
                    ):
                        raise DARTRealEvidenceError(
                            f"DART quantile serialization gate failed: {artifact_path}"
                        )
                    day_indices, timestamps_by_date = _event_layout(
                        keys, forecast_year=origin.forecast_year
                    )
                    dates = tuple(sorted(day_indices))
                    if any(len(timestamps_by_date[day]) != expected_intervals for day in dates):
                        raise DARTRealEvidenceError(
                            f"DART daily horizon differs from {expected_intervals}: {artifact_path}"
                        )
                    event_sha = sha256_json(list(keys))
                    coupling_sha = _valid_sha256(
                        str(data["coupling_manifest_sha256"].item()),
                        label=f"coupling manifest {artifact_path}",
                    )
                    serialized_base_sha = _valid_sha256(
                        str(data["common_base_random_numbers_sha256"].item()),
                        label=f"common base random numbers {artifact_path}",
                    )
                    serialized_joint_sha = _valid_sha256(
                        str(data["joint_uniforms_sha256"].item()),
                        label=f"joint uniforms {artifact_path}",
                    )
                    if (
                        serialized_base_sha
                        != source_hashes.get("common_base_random_numbers")
                        or serialized_joint_sha != source_hashes.get("joint_uniforms")
                    ):
                        raise DARTRealEvidenceError(
                            f"DART serialized random-source hashes changed: {artifact_path}"
                        )
                dependence_sha = _verify_joint_coupling(
                    stage_root,
                    origin=origin,
                    candidate_name=candidate.name,
                    seed_slot_name=seed_slot.name,
                    event_keys_sha256=event_sha,
                    event_count=len(keys),
                    coupling_sha256=coupling_sha,
                    dart_source_hashes=source_hashes,
                )
                if not allow_sparse:
                    expected_dates = []
                    current = date(origin.forecast_year, 1, 1)
                    while current.year == origin.forecast_year:
                        expected_dates.append(current.isoformat())
                        current += timedelta(days=1)
                    if dates != tuple(expected_dates):
                        raise DARTRealEvidenceError(
                            f"{artifact_path} does not cover every date in {origin.forecast_year}"
                        )
                if origin_dates is None:
                    origin_dates = dates
                    origin_event_sha = event_sha
                    origin_timestamps = dict(timestamps_by_date)
                    origin_issue_time_panel_sha = str(
                        da_unit["issue_time_panel_sha256"]
                    )
                    origin_calibration_panel_sha = str(
                        da_unit["calibration_panel_sha256"]
                    )
                elif dates != origin_dates or event_sha != origin_event_sha or dict(timestamps_by_date) != origin_timestamps:
                    raise DARTRealEvidenceError(
                        f"candidate/seed event contract changed within {origin.name}"
                    )
                elif (
                    da_unit["issue_time_panel_sha256"]
                    != origin_issue_time_panel_sha
                ):
                    raise DARTRealEvidenceError(
                        f"candidate/seed issue-time panel changed within {origin.name}"
                    )
                elif (
                    da_unit["calibration_panel_sha256"]
                    != origin_calibration_panel_sha
                ):
                    raise DARTRealEvidenceError(
                        f"candidate/seed calibration panel changed within {origin.name}"
                    )
                calibration_sha = sha256_json(
                    {
                        "da_unit": da_unit["checkpoint_sha256"],
                        "rt_unit": rt_unit["checkpoint_sha256"],
                    }
                )
                model_sha = sha256_json(
                    {
                        "da": da_unit["model_sha256"],
                        "rt": rt_unit["model_sha256"],
                    }
                )
                records.append(
                    OOSArtifactRecord(
                        origin_name=origin.name,
                        forecast_year=origin.forecast_year,
                        candidate_name=candidate.name,
                        seed_slot_name=seed_slot.name,
                        artifact_path=artifact_path,
                        artifact_sha256=artifact_sha,
                        manifest_path=manifest_path,
                        manifest_sha256=sha256_file(manifest_path),
                        event_keys_sha256=event_sha,
                        dates=dates,
                        timestamps_by_date=dict(timestamps_by_date),
                        model_sha256=model_sha,
                        model_byte_count=int(da_unit["model_byte_count"] + rt_unit["model_byte_count"]),
                        da_prediction_sha256=da_unit["prediction_sha256"],
                        rt_prediction_sha256=rt_unit["prediction_sha256"],
                        calibration_sha256=calibration_sha,
                        calibration_panel_sha256=str(
                            da_unit["calibration_panel_sha256"]
                        ),
                        dependence_sha256=dependence_sha,
                        information_set_sha256=da_unit["information_set_sha256"],
                        issue_time_panel_sha256=str(
                            da_unit["issue_time_panel_sha256"]
                        ),
                        leakage_gate_passed=leakage_passed,
                    )
                )
        assert origin_dates is not None and origin_timestamps is not None
        if common_dates is None:
            common_dates = origin_dates
            common_timestamps = origin_timestamps
        else:
            common_dates = tuple(sorted(set(common_dates).union(origin_dates)))
            assert common_timestamps is not None
            common_timestamps.update(origin_timestamps)
    assert common_dates is not None and common_timestamps is not None
    record_map = {record.key: record for record in records}
    if len(record_map) != len(records):
        raise DARTRealEvidenceError("duplicate OOS artifact key")
    expected_record_count = sum(
        len(candidate.seed_slots) for candidate in candidate_specs
    ) * len(_phase_origins(phase))
    if len(records) != expected_record_count:
        raise DARTRealEvidenceError(
            f"OOS inventory has {len(records)} records; {expected_record_count} required"
        )
    serialized = [record.to_dict() for record in sorted(records, key=lambda item: item.key)]
    return OOSInventory(
        records=tuple(records),
        records_by_key=record_map,
        dates=common_dates,
        timestamps_by_date=common_timestamps,
        fingerprint=sha256_json(serialized),
        all_leakage_gates_passed=all(record.leakage_gate_passed for record in records),
    )


def _selected_dates(
    inventory: OOSInventory,
    *,
    phase: str,
    real_contract: Mapping[str, Any],
) -> tuple[str, ...]:
    configured = real_contract.get("operating_dates_by_phase", {}).get(phase)
    if configured is None:
        return inventory.dates
    dates = tuple(sorted(map(str, configured)))
    if not dates or len(set(dates)) != len(dates):
        raise DARTRealEvidenceError("configured operating dates must be nonempty and unique")
    missing = sorted(set(dates).difference(inventory.dates))
    if missing:
        raise MissingRealEvidenceError(
            f"configured operating dates are absent from OOS artifacts: {missing[:5]}"
        )
    expected_years = {2022, 2023, 2024} if phase == "selection" else {2025}
    if {date.fromisoformat(day).year for day in dates} != expected_years:
        raise DARTRealEvidenceError(
            f"{phase} operating-date selection does not cover exact years {sorted(expected_years)}"
        )
    return dates


def _expand_day_records(raw: Any, *, root: Path, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    index_path: Path | None = None
    if isinstance(raw, (str, Path)):
        index_path = _resolve_path(raw, root=root)
        raw = _read_json(index_path)
    elif isinstance(raw, Mapping) and "index_path" in raw:
        index_path = _resolve_path(raw["index_path"], root=root)
        raw = _read_json(index_path)
    if not isinstance(raw, Mapping):
        raise DARTRealEvidenceError(f"{label} must be a mapping or JSON index")
    if raw.get("status") is not None and str(raw["status"]).upper() != "COMPLETE":
        raise MissingRealEvidenceError(f"{label} index is not COMPLETE")
    days = raw.get("days", raw)
    entries: dict[str, Any] = {}
    if isinstance(days, Sequence) and not isinstance(days, (str, bytes)):
        for record in days:
            if not isinstance(record, Mapping) or "operating_date" not in record:
                raise DARTRealEvidenceError(f"{label} list record lacks operating_date")
            entries[str(record["operating_date"])] = dict(record)
    elif isinstance(days, Mapping):
        entries = {str(day): value for day, value in days.items()}
    else:
        raise DARTRealEvidenceError(f"{label}.days must be a mapping or list")
    if index_path is not None:
        for day, value in list(entries.items()):
            if isinstance(value, Mapping) and "path" in value:
                normalized = dict(value)
                normalized["path"] = str(
                    _resolve_path(value["path"], root=root, sibling=index_path.parent)
                )
                entries[day] = normalized
    return entries


def _load_npz_record(record: Mapping[str, Any], *, root: Path, label: str) -> dict[str, Any]:
    path = _resolve_path(record["path"], root=root)
    if not path.is_file():
        raise MissingRealEvidenceError(f"{label} NPZ is absent: {path}")
    actual_sha = sha256_file(path)
    if record.get("sha256") is not None and record["sha256"] != actual_sha:
        raise DARTRealEvidenceError(f"{label} NPZ hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files if not name.endswith("_json")}
        if "source_hashes_json" in data.files:
            payload["source_hashes"] = json.loads(str(data["source_hashes_json"].item()))
        if "evidence_roles_json" in data.files:
            payload["evidence_roles"] = json.loads(str(data["evidence_roles_json"].item()))
    payload["artifact_sha256"] = actual_sha
    return payload


class _ExternalInputStore:
    def __init__(
        self,
        *,
        repository_root: Path,
        phase: str,
        records: Mapping[str, Any],
        nyiso_data_root: Path | None,
    ) -> None:
        self.root = repository_root
        self.phase = phase
        self.records = dict(records)
        self.nyiso_data_root = nyiso_data_root
        self._panel = None
        self._matrix: dict[str, pd.DataFrame] | None = None

    def _direct(self, day: str) -> ReferencedNYISOReplay | None:
        raw = self.records.get(day)
        if raw is None:
            return None
        if isinstance(raw, ReferencedNYISOReplay):
            return raw
        if isinstance(raw, Mapping) and "path" in raw:
            raw = _load_npz_record(raw, root=self.root, label=f"NYISO replay {day}")
        if not isinstance(raw, Mapping):
            raise DARTRealEvidenceError(f"NYISO replay {day} must be a mapping")
        payload = raw.get("settlement", raw)
        return ReferencedNYISOReplay(
            operating_date=day,
            timestamps_utc=payload["timestamps_utc"],
            da_lbmp=payload["da_lbmp"],
            rt_lbmp=payload["rt_lbmp"],
            source_hashes=dict(payload["source_hashes"]),
        )

    def _ensure_panel(self) -> None:
        if self._matrix is not None:
            return
        maximum_year = 2024 if self.phase == "selection" else 2025
        source = ReferencedNYISOPanelSource(
            repository_root=self.root,
            data_root=self.nyiso_data_root,
        )
        panel = source.load_through_year(maximum_year)
        matrices: dict[str, pd.DataFrame] = {}
        for field_name, frame in (("da_lbmp", panel.da_15m), ("rt_lbmp", panel.rt_15m)):
            rows = frame[["timestamp_utc", "zone", field_name]].copy()
            rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
            matrices[field_name] = rows.pivot(
                index="timestamp_utc", columns="zone", values=field_name
            ).reindex(columns=NYISO_ZONES)
        self._panel = panel
        self._matrix = matrices

    def settlement(self, day: str, timestamps: Sequence[str]) -> ReferencedNYISOReplay:
        direct = self._direct(day)
        expected = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
        if direct is not None:
            actual = pd.DatetimeIndex(pd.to_datetime(direct.timestamps_utc, utc=True))
            if not actual.equals(expected):
                raise DARTRealEvidenceError(f"NYISO replay timestamps do not align for {day}")
            return direct
        self._ensure_panel()
        assert self._matrix is not None and self._panel is not None
        values: dict[str, np.ndarray] = {}
        for field_name, matrix in self._matrix.items():
            subset = matrix.reindex(expected)
            if subset.isna().any().any():
                raise MissingRealEvidenceError(
                    f"referenced NYISO {field_name} is incomplete for {day}"
                )
            values[field_name] = subset.to_numpy(dtype=float).T
        source_hash = _valid_sha256(
            self._panel.source_manifest["partitions_sha256"],
            label="referenced NYISO partitions",
        )
        return ReferencedNYISOReplay(
            operating_date=day,
            timestamps_utc=np.asarray(expected.to_numpy(dtype="datetime64[ns]")),
            da_lbmp=values["da_lbmp"],
            rt_lbmp=values["rt_lbmp"],
            source_hashes={"referenced_nyiso_partitions": source_hash},
        )

    def thresholds(self, year: int, supplied: Mapping[str, Any] | None) -> dict[str, Any]:
        if supplied is not None:
            required = {
                "zone_order",
                "dart_lower_q01_by_zone",
                "dart_lower_q05_by_zone",
                "dart_upper_q95_by_zone",
                "dart_upper_q99_by_zone",
                "congestion_upper",
                "fit_cutoff_exclusive_utc",
                "source_sha256",
                "locked_targets_used",
                "future_targets_used",
            }
            if not required.issubset(supplied):
                raise MissingRealEvidenceError(
                    f"training thresholds for {year} lack {sorted(required - set(supplied))}"
                )
            cutoff = pd.Timestamp(supplied["fit_cutoff_exclusive_utc"])
            expected_cutoff = pd.Timestamp(f"{year:04d}-01-01T00:00:00Z")
            if cutoff.tzinfo is None or cutoff.tz_convert("UTC") != expected_cutoff:
                raise DARTRealEvidenceError(
                    f"training thresholds for {year} use the wrong causal cutoff"
                )
            if (
                supplied["locked_targets_used"] is not False
                or supplied["future_targets_used"] is not False
            ):
                raise DARTRealEvidenceError(
                    f"training thresholds for {year} used future/locked targets"
                )
            source_sha = _valid_sha256(
                supplied["source_sha256"], label=f"training thresholds {year}"
            )
            zone_order = tuple(map(str, supplied["zone_order"]))
            if zone_order != tuple(NYISO_ZONES):
                raise DARTRealEvidenceError(
                    f"training thresholds for {year} use the wrong zone order"
                )
            lower_extreme = np.asarray(
                supplied["dart_lower_q01_by_zone"], dtype=float
            )
            lower_tail = np.asarray(
                supplied["dart_lower_q05_by_zone"], dtype=float
            )
            upper_tail = np.asarray(
                supplied["dart_upper_q95_by_zone"], dtype=float
            )
            upper_extreme = np.asarray(
                supplied["dart_upper_q99_by_zone"], dtype=float
            )
            if (
                lower_extreme.shape != (len(NYISO_ZONES),)
                or lower_tail.shape != lower_extreme.shape
                or upper_tail.shape != lower_extreme.shape
                or upper_extreme.shape != lower_extreme.shape
                or not np.all(np.isfinite(lower_extreme))
                or not np.all(np.isfinite(lower_tail))
                or not np.all(np.isfinite(upper_tail))
                or not np.all(np.isfinite(upper_extreme))
            ):
                raise DARTRealEvidenceError(
                    f"training thresholds for {year} have invalid shape/non-finite values"
                )
            if np.any(
                (lower_extreme > lower_tail)
                | (lower_tail >= upper_tail)
                | (upper_tail > upper_extreme)
            ):
                raise DARTRealEvidenceError(
                    f"training thresholds for {year} must satisfy q01 <= q05 < q95 <= q99"
                )
            congestion_upper = float(supplied["congestion_upper"])
            if not np.isfinite(congestion_upper):
                raise DARTRealEvidenceError(
                    f"training congestion threshold for {year} is non-finite"
                )
            core = {
                "schema_version": TAIL_THRESHOLD_SCHEMA,
                "forecast_year": int(year),
                "zone_order": list(zone_order),
                "dart_lower_q01_by_zone": lower_extreme.tolist(),
                "dart_lower_q05_by_zone": lower_tail.tolist(),
                "dart_upper_q95_by_zone": upper_tail.tolist(),
                "dart_upper_q99_by_zone": upper_extreme.tolist(),
                "lower_quantile": 0.01,
                "upper_quantile": 0.99,
                "tail_quantiles": [0.01, 0.05, 0.95, 0.99],
                "combined_tail_event": "DART_below_zone_q01_or_above_zone_q99",
                "congestion_upper": congestion_upper,
                "fit_cutoff_exclusive_utc": expected_cutoff.isoformat(),
                "source_sha256": source_sha,
                "locked_targets_used": False,
                "future_targets_used": False,
            }
            contract_sha = sha256_json(core)
            supplied_contract_sha = supplied.get("threshold_contract_sha256")
            if supplied_contract_sha is not None and _valid_sha256(
                supplied_contract_sha,
                label=f"training threshold contract {year}",
            ) != contract_sha:
                raise DARTRealEvidenceError(
                    f"training threshold contract hash changed for {year}"
                )
            return {**core, "threshold_contract_sha256": contract_sha}
        self._ensure_panel()
        assert self._matrix is not None
        cutoff = pd.Timestamp(f"{year:04d}-01-01T00:00:00Z")
        da = self._matrix["da_lbmp"].loc[lambda frame: frame.index < cutoff]
        rt = self._matrix["rt_lbmp"].loc[lambda frame: frame.index < cutoff]
        common = da.index.intersection(rt.index)
        da_values = da.reindex(common).to_numpy(dtype=float)
        rt_values = rt.reindex(common).to_numpy(dtype=float)
        valid = np.isfinite(da_values).all(axis=1) & np.isfinite(rt_values).all(axis=1)
        if np.count_nonzero(valid) < 500:
            raise MissingRealEvidenceError(
                f"fewer than 500 causal NYISO rows exist before {year}"
            )
        dart = (da_values - rt_values)[valid]
        congestion = np.ptp(rt_values[valid], axis=1)
        lower_extreme, lower_tail, upper_tail, upper_extreme = np.quantile(
            dart, [0.01, 0.05, 0.95, 0.99], axis=0
        )
        source_sha = sha256_json(
            {
                "year": year,
                "cutoff_exclusive_utc": cutoff.isoformat(),
                "zone_order": list(NYISO_ZONES),
                "da_history": _array_sha256(da_values[valid]),
                "rt_history": _array_sha256(rt_values[valid]),
            }
        )
        core = {
            "schema_version": TAIL_THRESHOLD_SCHEMA,
            "forecast_year": int(year),
            "zone_order": list(NYISO_ZONES),
            "dart_lower_q01_by_zone": lower_extreme.tolist(),
            "dart_lower_q05_by_zone": lower_tail.tolist(),
            "dart_upper_q95_by_zone": upper_tail.tolist(),
            "dart_upper_q99_by_zone": upper_extreme.tolist(),
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "tail_quantiles": [0.01, 0.05, 0.95, 0.99],
            "combined_tail_event": "DART_below_zone_q01_or_above_zone_q99",
            "congestion_upper": float(np.quantile(congestion, 0.95)),
            "fit_cutoff_exclusive_utc": cutoff.isoformat(),
            "source_sha256": source_sha,
            "locked_targets_used": False,
            "future_targets_used": False,
        }
        return {**core, "threshold_contract_sha256": sha256_json(core)}


class _ForcingStore:
    def __init__(self, *, repository_root: Path, records: Mapping[str, Any]) -> None:
        self.root = repository_root
        self.records = dict(records)

    def forcing(self, day: str, timestamps: Sequence[str]) -> FrozenReplayForcing:
        raw = self.records.get(day)
        if raw is None:
            raise MissingRealEvidenceError(
                f"frozen workload/PV/physical replay forcing is absent for {day}"
            )
        if isinstance(raw, FrozenReplayForcing):
            forcing = raw
        else:
            if isinstance(raw, Mapping) and "path" in raw:
                raw = _load_npz_record(raw, root=self.root, label=f"replay forcing {day}")
            if not isinstance(raw, Mapping):
                raise DARTRealEvidenceError(f"replay forcing {day} must be a mapping")
            payload = raw.get("forcing", raw)
            if payload.get("candidate_independent") is not True:
                raise MissingRealEvidenceError(
                    f"replay forcing {day} lacks candidate_independent=true provenance"
                )
            if payload.get("selected_scenario_as_realized") is not False:
                raise MissingRealEvidenceError(
                    f"replay forcing {day} must declare selected_scenario_as_realized=false"
                )
            forcing = FrozenReplayForcing(
                operating_date=day,
                timestamps_utc=payload["timestamps_utc"],
                workload=payload["workload"],
                pv=payload["pv"],
                physical_multiplier=payload["physical_multiplier"],
                source_hashes=dict(payload["source_hashes"]),
                evidence_roles=dict(
                    payload.get(
                        "evidence_roles",
                        {
                            "workload": "replayed",
                            "pv": "reconstructed",
                            "physical_multiplier": "counterfactual",
                        },
                    )
                ),
                frozen=bool(payload.get("frozen", True)),
            )
        expected = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
        actual = pd.DatetimeIndex(pd.to_datetime(forcing.timestamps_utc, utc=True))
        if not actual.equals(expected):
            raise DARTRealEvidenceError(f"replay forcing timestamps do not align for {day}")
        return forcing


@dataclass(frozen=True)
class ActionabilityDay:
    feasible_by_event_mwh: np.ndarray
    normalized_feasibility_weight: np.ndarray
    total_feasible_mwh: float
    lower_no_action_bound: np.ndarray
    upper_no_action_bound: np.ndarray
    source_fields: tuple[str, ...]
    available_at_gate: np.ndarray
    source_sha256: str


class _ActionabilityStore:
    def __init__(self, *, repository_root: Path, records: Mapping[str, Any]) -> None:
        self.root = repository_root
        self.records = dict(records)

    @staticmethod
    def _event_array(value: Any, *, shape: tuple[int, int], label: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.size == 1:
            return np.full(shape, float(array.reshape(-1)[0]), dtype=float)
        if array.shape == shape:
            return array
        if array.shape == (shape[0] * shape[1],):
            return array.reshape(shape)
        raise DARTRealEvidenceError(f"{label} must be scalar, {shape}, or flattened event shape")

    def load(self, day: str, *, horizon: int) -> ActionabilityDay:
        raw = self.records.get(day)
        if raw is None:
            raise MissingRealEvidenceError(f"gate-time actionability evidence is absent for {day}")
        if isinstance(raw, Mapping) and "path" in raw:
            raw = _load_npz_record(raw, root=self.root, label=f"actionability {day}")
        if not isinstance(raw, Mapping):
            raise DARTRealEvidenceError(f"actionability {day} must be a mapping")
        shape = (len(NYISO_ZONES), horizon)
        if "feasible_flexible_energy_mwh_by_event" in raw:
            feasible = self._event_array(
                raw["feasible_flexible_energy_mwh_by_event"],
                shape=shape,
                label="feasible_flexible_energy_mwh_by_event",
            )
            total = float(raw.get("gate_time_feasible_flexible_energy_mwh", np.sum(feasible)))
        elif "gate_time_feasible_flexible_energy_mwh" in raw:
            total = float(raw["gate_time_feasible_flexible_energy_mwh"])
            feasible = np.full(shape, total / (shape[0] * shape[1]), dtype=float)
        else:
            raise MissingRealEvidenceError(
                f"actionability {day} lacks gate-time feasible flexible energy"
            )
        if "normalized_feasibility_weight_by_event" not in raw:
            raise MissingRealEvidenceError(
                f"actionability {day} lacks normalized candidate-independent F"
            )
        normalized_weight = self._event_array(
            raw["normalized_feasibility_weight_by_event"],
            shape=shape,
            label="normalized_feasibility_weight_by_event",
        )
        lower = self._event_array(
            raw["lower_no_action_bound"], shape=shape, label="lower_no_action_bound"
        )
        upper = self._event_array(
            raw["upper_no_action_bound"], shape=shape, label="upper_no_action_bound"
        )
        source_fields = tuple(map(str, raw.get("source_fields", ())))
        available = np.asarray(raw.get("available_at_gate", True), dtype=bool)
        available = np.broadcast_to(available, shape).copy()
        validate_gate_time_flexibility(
            feasible,
            source_fields=source_fields,
            available_at_gate=available,
        )
        normalized_weight = validate_normalized_feasibility_weight(
            normalized_weight,
            source_fields=source_fields,
            available_at_gate=available,
        )
        if np.any((feasible <= 0.0) & (normalized_weight > 0.0)):
            raise DARTRealEvidenceError(
                f"actionability {day} assigns positive F to zero physical flexibility"
            )
        if not np.isfinite(total) or total < 0.0 or not np.isclose(total, np.sum(feasible)):
            raise DARTRealEvidenceError(
                f"actionability {day} total differs from its event-level values"
            )
        source_hashes = raw.get("source_hashes")
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise MissingRealEvidenceError(f"actionability {day} lacks source hashes")
        normalized = {
            str(key): _valid_sha256(value, label=f"actionability {day}/{key}")
            for key, value in source_hashes.items()
        }
        source_contract = {
            "source_hashes": normalized,
            "feasible_by_event_mwh_sha256": _array_sha256(feasible),
            "normalized_feasibility_weight_sha256": _array_sha256(
                normalized_weight
            ),
            "normalization_contract": "candidate_independent_F_in_closed_unit_interval",
            "lower_no_action_bound_sha256": _array_sha256(lower),
            "upper_no_action_bound_sha256": _array_sha256(upper),
            "source_fields": list(source_fields),
            "available_at_gate_sha256": _array_sha256(available.astype(np.uint8)),
        }
        return ActionabilityDay(
            feasible_by_event_mwh=feasible,
            normalized_feasibility_weight=normalized_weight,
            total_feasible_mwh=total,
            lower_no_action_bound=lower,
            upper_no_action_bound=upper,
            source_fields=source_fields,
            available_at_gate=available,
            source_sha256=sha256_json(source_contract),
        )


def _input_block(config: Mapping[str, Any], forecast_result: Mapping[str, Any]) -> dict[str, Any]:
    config_block = config.get(INPUT_BLOCK, {})
    result_block = forecast_result.get(INPUT_BLOCK, {})
    if config_block and result_block and sha256_json(_fingerprint_value(config_block)) != sha256_json(
        _fingerprint_value(result_block)
    ):
        raise DARTRealEvidenceError(
            f"config and forecast_result provide different {INPUT_BLOCK} blocks"
        )
    selected = result_block or config_block
    if not isinstance(selected, Mapping):
        raise DARTRealEvidenceError(f"{INPUT_BLOCK} must be a mapping")
    return dict(selected)


def _issue_store(
    *,
    repository_root: Path,
    forecast_result: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> FrozenForecastStore:
    direct = inputs.get("issue_time_ensembles_by_date")
    if direct is not None:
        entries = _expand_day_records(
            direct, root=repository_root, label="issue_time_ensembles_by_date"
        )
        return FrozenForecastStore(entries, repository_root)
    # Only the frozen day index/artifacts are consumed.  Existing v1 metric
    # tables, winner labels, and 2025 scores in forecast_result are ignored.
    return _forecast_days(forecast_result, repository_root)


def _preflight_inputs(
    dates: Sequence[str],
    *,
    inventory: OOSInventory,
    issue_store: FrozenForecastStore,
    settlement_store: _ExternalInputStore,
    forcing_store: _ForcingStore,
    action_store: _ActionabilityStore,
    threshold_records: Mapping[str, Any],
) -> tuple[
    dict[str, FrozenForecastArtifact],
    dict[str, ReferencedNYISOReplay],
    dict[str, FrozenReplayForcing],
    dict[str, ActionabilityDay],
    dict[int, dict[str, Any]],
]:
    issue: dict[str, FrozenForecastArtifact] = {}
    settlement: dict[str, ReferencedNYISOReplay] = {}
    forcing: dict[str, FrozenReplayForcing] = {}
    actionability: dict[str, ActionabilityDay] = {}
    thresholds: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    for day in dates:
        timestamps = inventory.timestamps_by_date[day]
        try:
            if day not in issue_store:
                raise MissingRealEvidenceError(
                    f"frozen issue-time workload/PV ensemble is absent for {day}"
                )
            forecast = issue_store[day]
            actual_times = pd.DatetimeIndex(pd.to_datetime(forecast.timestamps_utc, utc=True))
            expected_times = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
            if not actual_times.equals(expected_times):
                raise DARTRealEvidenceError(f"issue-time forecast timestamps do not align for {day}")
            issue[day] = forecast
            settlement[day] = settlement_store.settlement(day, timestamps)
            forcing[day] = forcing_store.forcing(day, timestamps)
            actionability[day] = action_store.load(day, horizon=len(timestamps))
            year = date.fromisoformat(day).year
            if year not in thresholds:
                supplied = threshold_records.get(str(year), threshold_records.get(year))
                thresholds[year] = settlement_store.thresholds(year, supplied)
        except Exception as exc:
            failures.append(f"{day}: {type(exc).__name__}: {exc}")
    if failures:
        preview = "; ".join(failures[:12])
        raise MissingRealEvidenceError(
            f"real-evidence preflight failed for {len(failures)} operating dates; {preview}",
            pending_count=len(failures),
        )
    return issue, settlement, forcing, actionability, thresholds


def _write_daily_shards(
    inventory: OOSInventory,
    *,
    dates: Sequence[str],
    output_root: Path,
    candidate_specs: Sequence[Any] = CANDIDATE_POOL,
) -> dict[tuple[int, str, str, str], Path]:
    selected = set(dates)
    outputs: dict[tuple[int, str, str, str], Path] = {}
    for record in inventory.records:
        record_dates = tuple(day for day in record.dates if day in selected)
        if not record_dates:
            continue
        cache_root = output_root / "oos_daily_cache" / record.artifact_sha256
        manifest_path = cache_root / "manifest.json"
        contract = {
            "schema_version": SCHEMA_VERSION,
            "source_artifact_sha256": record.artifact_sha256,
            "event_keys_sha256": record.event_keys_sha256,
            "dates": list(record_dates),
        }
        fingerprint = sha256_json(contract)
        cached_ok = False
        if manifest_path.is_file():
            cached = _read_json(manifest_path)
            cached_ok = cached.get("input_sha256") == fingerprint and cached.get("status") == "COMPLETE"
            if cached_ok:
                for day_record in cached.get("days", []):
                    path = Path(day_record["path"])
                    if not path.is_file() or sha256_file(path) != day_record["sha256"]:
                        cached_ok = False
                        break
        if not cached_ok:
            with np.load(record.artifact_path, allow_pickle=False) as data:
                keys = tuple(map(str, data["event_keys"].tolist()))
                indices_by_day, timestamps_by_day = _event_layout(
                    keys, forecast_year=record.forecast_year
                )
                da = np.asarray(data["da_lbmp_draws"], dtype=float)
                rt = np.asarray(data["rt_lbmp_draws"], dtype=float)
                dart = np.asarray(data["dart_da_minus_rt"], dtype=float)
                inverse = np.asarray(data["rt_spread_rt_minus_da"], dtype=float)
                quantiles = np.asarray(data["dart_quantile_values"], dtype=float)
                day_records: list[dict[str, Any]] = []
                for day in record_dates:
                    index = np.asarray(indices_by_day[day], dtype=int)
                    horizon = len(timestamps_by_day[day])
                    path = cache_root / f"{day}.npz"
                    _atomic_npz(
                        path,
                        operating_date=np.asarray(day),
                        timestamps_utc=np.asarray(timestamps_by_day[day]),
                        da_lbmp_draws=da[index].reshape(4, horizon, -1).transpose(2, 0, 1),
                        rt_lbmp_draws=rt[index].reshape(4, horizon, -1).transpose(2, 0, 1),
                        dart_da_minus_rt=dart[index].reshape(4, horizon, -1).transpose(2, 0, 1),
                        rt_spread_rt_minus_da=inverse[index].reshape(4, horizon, -1).transpose(2, 0, 1),
                        dart_quantile_values=quantiles[index].reshape(4, horizon, len(QUANTILES)),
                        source_artifact_sha256=np.asarray(record.artifact_sha256),
                    )
                    day_records.append(
                        {"operating_date": day, "path": path.as_posix(), "sha256": sha256_file(path)}
                    )
            atomic_json(
                manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "COMPLETE",
                    "input_sha256": fingerprint,
                    "contract": contract,
                    "days": day_records,
                    "days_sha256": sha256_json(day_records),
                },
            )
        manifest = _read_json(manifest_path)
        by_day = {str(item["operating_date"]): item for item in manifest["days"]}
        for day in record_dates:
            outputs[(record.forecast_year, record.candidate_name, record.seed_slot_name, day)] = Path(
                by_day[day]["path"]
            )
    expected = len(dates) * sum(len(candidate.seed_slots) for candidate in candidate_specs)
    if len(outputs) != expected:
        raise MissingRealEvidenceError(
            f"daily DART cache has {len(outputs)} cells; {expected} required"
        )
    return outputs


@dataclass(frozen=True)
class DailyDART:
    da_draws: np.ndarray
    rt_draws: np.ndarray
    dart_draws: np.ndarray
    dart_quantiles: np.ndarray
    timestamps_utc: np.ndarray
    source_sha256: str
    load_seconds: float


def _load_daily_dart(path: Path) -> DailyDART:
    started = perf_counter()
    with np.load(path, allow_pickle=False) as data:
        da = np.asarray(data["da_lbmp_draws"], dtype=float)
        rt = np.asarray(data["rt_lbmp_draws"], dtype=float)
        dart = np.asarray(data["dart_da_minus_rt"], dtype=float)
        inverse = np.asarray(data["rt_spread_rt_minus_da"], dtype=float)
        qvalues = np.asarray(data["dart_quantile_values"], dtype=float)
        timestamps = np.asarray(data["timestamps_utc"])
        source = _valid_sha256(
            str(data["source_artifact_sha256"].item()), label=f"daily DART source {path}"
        )
    if not (da.shape == rt.shape == dart.shape == inverse.shape):
        raise DARTRealEvidenceError(f"daily DART shapes differ: {path}")
    if not np.allclose(dart, da - rt) or not np.allclose(dart, -inverse):
        raise DARTRealEvidenceError(f"daily DART sign gate failed: {path}")
    if qvalues.shape != (*dart.shape[1:], len(QUANTILES)):
        raise DARTRealEvidenceError(f"daily DART quantile shape is invalid: {path}")
    return DailyDART(
        da_draws=da,
        rt_draws=rt,
        dart_draws=dart,
        dart_quantiles=qvalues,
        timestamps_utc=timestamps,
        source_sha256=source,
        load_seconds=max(perf_counter() - started, 0.0),
    )


def _base_issue_bundle(
    physical: FrozenPhysicalArtifact,
    forecast: FrozenForecastArtifact,
    forcing: FrozenReplayForcing,
    *,
    seed: int,
    reduced_count: int,
) -> ScenarioBundle:
    bundle = build_scenario_bundle(
        physical,
        forecast,
        regime=DEFAULT_REGIMES[0],
        reduced_count=reduced_count,
        study_seed=seed,
    )
    # The physical multiplier is an explicitly frozen engineering
    # counterfactual, not a scenario selected after observing an outcome.
    # It is therefore common and known before every candidate first stage.
    multiplier = np.broadcast_to(
        forcing.physical_multiplier,
        (bundle.reduced_count, len(bundle.zones)),
    ).copy()
    output = replace(bundle, physical_multiplier=multiplier)
    output.validate()
    return output


def _candidate_issue_bundle(
    base: ScenarioBundle,
    daily: DailyDART,
    *,
    experiment_id: str,
) -> ScenarioBundle:
    parents = np.asarray(base.selected_parent_ids, dtype=int)
    if parents.max(initial=-1) >= daily.dart_draws.shape[0]:
        raise DARTRealEvidenceError("scenario reduction selected a missing DART draw")
    if experiment_id == "A1":
        return build_a1_dart_bundle(
            base,
            fixed_da_lbmp=base.da_price,
            dart_scenarios_da_minus_rt=daily.dart_draws[parents],
        )
    if experiment_id != "A2":
        raise DARTRealEvidenceError(f"unknown DART experiment {experiment_id}")
    da_median = np.quantile(daily.da_draws, 0.50, axis=0, method="linear")
    output = replace(
        base,
        da_price=np.asarray(da_median, dtype=float),
        rt_price=np.asarray(daily.rt_draws[parents], dtype=float),
    )
    output.validate()
    return output


def _component_costs(dispatch: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, float]:
    migrated = 0.5 * float(np.asarray(dispatch.get("reassignment_work_units", 0.0), dtype=float).sum())
    migration_energy = float(settings.get("transfer_energy_mwh_per_work_unit", 0.0)) * migrated
    migration = float(settings.get("migration_penalty_usd_per_work_unit", 0.05)) * (
        1.0 + float(settings.get("checkpoint_overhead_fraction", 0.0))
    ) * migrated
    migration_energy_cost = float(
        settings.get("migration_energy_price_usd_per_mwh", 0.0)
    ) * migration_energy
    throughput = ROLLING_INTERVAL_HOURS * float(
        np.asarray(dispatch.get("battery_charge_mw", 0.0), dtype=float).sum()
        + np.asarray(dispatch.get("battery_discharge_mw", 0.0), dtype=float).sum()
    )
    battery = float(settings.get("battery_degradation_usd_per_mwh", 1.0)) * throughput
    curtailed = ROLLING_INTERVAL_HOURS * float(
        np.asarray(dispatch.get("pv_curtailment_mw", 0.0), dtype=float).sum()
    )
    curtailment = float(settings.get("pv_curtailment_usd_per_mwh", 0.01)) * curtailed
    shortfall = float(np.asarray(dispatch.get("shortfall_work_units", 0.0), dtype=float).sum())
    sla = float(settings.get("sla_penalty_usd_per_work_unit", 10000.0)) * shortfall
    slack = float(settings.get("constraint_slack_penalty_usd", 100000.0)) * (
        float(dispatch.get("ramp_slack_mw", 0.0))
        + float(np.asarray(dispatch.get("concentration_slack_mw", 0.0), dtype=float).sum())
    )
    return {
        "migration_cost_usd": migration,
        "migration_energy_cost_usd": migration_energy_cost,
        "battery_degradation_cost_usd": battery,
        "curtailment_cost_usd": curtailment,
        "sla_penalty_cost_usd": sla,
        "constraint_slack_cost_usd": slack,
    }


def _binding_constraint_labels(result: Mapping[str, Any]) -> tuple[str, ...]:
    labels: set[str] = set()
    for key in (
        "binding_constraints",
        "active_constraints",
        "constraint_activity_labels",
        "binding_constraint",
    ):
        raw = result.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, Mapping):
            values = [name for name, active in raw.items() if bool(active)]
        elif (
            isinstance(raw, (Sequence, np.ndarray))
            and not isinstance(raw, (bytes, bytearray))
        ):
            values = list(raw)
        else:
            raise DARTRealEvidenceError(
                f"solver field {key} has an unsupported binding-constraint form"
            )
        labels.update(str(value).strip() for value in values if str(value).strip())
    return tuple(sorted(labels))


def _solver_audit(result: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    """Normalize one solve under the incumbent-required five-percent gap rule."""

    raw_gap = result.get("relative_mip_gap", result.get("mip_gap"))
    gap = None if raw_gap is None else float(raw_gap)
    if gap is not None and (not np.isfinite(gap) or gap < 0.0):
        raise DARTRealEvidenceError(f"{stage} returned an invalid MIP gap")
    termination = str(result.get("termination_condition", "unknown"))
    normalized = termination.replace("_", "").replace(" ", "").lower()
    solver_outcome = str(result.get("solver_outcome", ""))
    normalized_outcome = solver_outcome.replace("_", "").replace(" ", "").lower()
    acceptable = {
        "optimal",
        "feasible",
        "locallyoptimal",
        "globallyoptimal",
        "convergencecriteriasatisfied",
    }
    timeout = bool(
        result.get("time_limit_reached", False)
        or normalized in {"maxtimelimit", "timelimit"}
        or normalized_outcome == "feasibletimeout"
    )
    raw_has_incumbent = result.get(
        "has_incumbent", result.get("solver_has_incumbent")
    )
    has_incumbent_reported = isinstance(raw_has_incumbent, (bool, np.bool_))
    has_incumbent = bool(raw_has_incumbent) if has_incumbent_reported else bool(
        normalized in acceptable or normalized_outcome == "optimal"
    )
    nonzero_gap = bool(gap is not None and gap > 0.0)
    gap_over_cap = bool(gap is not None and gap > MAXIMUM_ACCEPTED_MIP_GAP)
    missing_gap = raw_gap is None
    termination_infeasible = bool(
        normalized not in acceptable
        and normalized_outcome not in {"optimal", "feasiblelimit", "feasibletimeout"}
        and not (timeout and has_incumbent)
    )
    rejected = bool(
        not has_incumbent or missing_gap or gap_over_cap or termination_infeasible
    )
    lower_bound = result.get(
        "solver_objective_lower_bound", result.get("objective_lower_bound")
    )
    upper_bound = result.get(
        "solver_objective_upper_bound", result.get("objective_upper_bound")
    )
    try:
        lower_bound = None if lower_bound is None else float(lower_bound)
        upper_bound = None if upper_bound is None else float(upper_bound)
    except (TypeError, ValueError) as exc:
        raise DARTRealEvidenceError(f"{stage} returned invalid objective bounds") from exc
    if lower_bound is not None and not np.isfinite(lower_bound):
        lower_bound = None
    if upper_bound is not None and not np.isfinite(upper_bound):
        upper_bound = None
    binding_labels = _binding_constraint_labels(result)
    return {
        "stage": stage,
        "termination_condition": termination,
        "mip_gap": gap,
        "mip_gap_reported": not missing_gap,
        "missing_mip_gap": missing_gap,
        "time_limit_reached": timeout,
        "nonzero_gap": nonzero_gap,
        "gap_over_cap": gap_over_cap,
        "maximum_accepted_relative_mip_gap": MAXIMUM_ACCEPTED_MIP_GAP,
        "has_incumbent": has_incumbent,
        "has_incumbent_reported": has_incumbent_reported,
        "termination_infeasible": termination_infeasible,
        "rejected_under_bounded_gap_rule": rejected,
        # Compatibility alias; its semantics are now explicitly the bounded-
        # gap rejection rule rather than a zero-gap rule.
        "infeasible_under_frozen_rule": rejected,
        "solver_objective_lower_bound": lower_bound,
        "solver_objective_upper_bound": upper_bound,
        "solve_time_seconds": float(result.get("solve_time_seconds", 0.0)),
        "number_of_variables": int(result.get("number_of_variables", 0)),
        "number_of_constraints": int(result.get("number_of_constraints", 0)),
        "binding_constraint_labels": list(binding_labels) if binding_labels else None,
        "binding_constraint_status": (
            "SOLVER_REPORTED" if binding_labels else "UNAVAILABLE"
        ),
        "binding_constraint_undefined_reason": (
            None
            if binding_labels
            else "SOLVER_DID_NOT_REPORT_DUAL_OR_ACTIVITY_DIAGNOSTICS"
        ),
    }


def _causal_provenance_complete(result: Mapping[str, Any]) -> bool:
    provenance = result.get("causal_provenance")
    if not isinstance(provenance, Mapping):
        return False
    required_true = (
        "first_stage_inputs_known_at_gate",
        "realized_settlement_revealed_after_first_stage_freeze",
        "rolling_recourse_current_interval_only",
        "future_issue_time_hashes_match_replay",
        "candidate_independent_forcing",
    )
    required_hashes = (
        "plan_sha256",
        "settlement_sha256",
        "forcing_sha256",
        "issue_bundle_sha256",
        "evidence_hashes_sha256",
        "external_replay_rows_sha256",
        "operational_sidecar_records_sha256",
    )
    try:
        hashes_valid = all(
            bool(_valid_sha256(provenance.get(key), label=f"causal provenance {key}"))
            for key in required_hashes
        )
    except DARTRealEvidenceError:
        return False
    return bool(
        hashes_valid
        and all(provenance.get(key) is True for key in required_true)
        and provenance.get("selected_scenario_as_realized") is False
        and provenance.get("future_realized_targets_used") is False
    )


def _bounded_gap_rule_applied(result: Mapping[str, Any]) -> bool:
    first_stage = result.get("first_stage_solver_audit")
    recourse = result.get("recourse_solver_audit")
    if not isinstance(first_stage, Mapping) or not isinstance(recourse, Mapping):
        return False
    counts = {
        field_name: recourse.get(field_name)
        for field_name in (
            "timeout_count",
            "nonzero_gap_count",
            "gap_over_cap_count",
            "no_incumbent_count",
            "infeasible_under_frozen_rule_count",
            "mip_gap_missing_count",
        )
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        return False
    first_missing = first_stage.get("mip_gap_reported") is not True
    expected_timeout = bool(
        first_stage.get("time_limit_reached") is True
        or int(counts["timeout_count"]) > 0
    )
    expected_nonzero = bool(
        first_stage.get("nonzero_gap") is True
        or int(counts["nonzero_gap_count"]) > 0
    )
    expected_infeasible = bool(
        first_stage.get("rejected_under_bounded_gap_rule") is True
        or first_missing
        or int(counts["gap_over_cap_count"]) > 0
        or int(counts["no_incumbent_count"]) > 0
        or int(counts["infeasible_under_frozen_rule_count"]) > 0
        or int(counts["mip_gap_missing_count"]) > 0
    )
    return bool(
        result.get("solver_feasibility_rule")
        == "incumbent_required_and_finite_relative_mip_gap_le_0_05"
        and first_stage.get("missing_mip_gap") is first_missing
        and result.get("solver_any_time_limit_reached") is expected_timeout
        and result.get("solver_any_nonzero_gap") is expected_nonzero
        and result.get("solver_any_gap_over_cap")
        is bool(
            first_stage.get("gap_over_cap") is True
            or int(counts["gap_over_cap_count"]) > 0
        )
        and result.get("solver_any_no_incumbent")
        is bool(
            first_stage.get("has_incumbent") is not True
            or int(counts["no_incumbent_count"]) > 0
        )
        and result.get("solver_any_infeasible_under_frozen_rule") is expected_infeasible
        and result.get("infeasible_day") is expected_infeasible
    )


def _dart_sign_context(settlement: ReferencedNYISOReplay) -> dict[str, Any]:
    values = np.asarray(settlement.dart_da_minus_rt, dtype=float)
    positive_count = int(np.count_nonzero(values > 0.0))
    negative_count = int(np.count_nonzero(values < 0.0))
    zero_count = int(np.count_nonzero(values == 0.0))
    if positive_count and negative_count:
        label = "MIXED"
    elif positive_count:
        label = "POSITIVE"
    elif negative_count:
        label = "NEGATIVE"
    else:
        label = "ZERO"
    return {
        "label": label,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "event_count": int(values.size),
        "canonical_definition": "dart_da_minus_rt = da_lbmp - rt_lbmp",
        "source_relationship": settlement.source_relationship,
        "settlement_sha256": settlement.input_sha256,
        "post_decision_scoring_context_only": True,
    }


def _run_external_cell(
    *,
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    settlement: ReferencedNYISOReplay,
    forcing: FrozenReplayForcing,
    candidate_id: str,
    experiment_id: str,
    seed: int,
    evidence_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    checkpoint_root: Path,
    first_stage_solver: Callable[..., Mapping[str, Any]],
    recourse_solver: Callable[..., Mapping[str, Any]],
    component_cost_hook: Callable[[int, Mapping[str, Any]], Mapping[str, float]] | None,
    actionability: ActionabilityDay | None,
    model_complexity: float,
    forecast_time_seconds: float,
) -> dict[str, Any]:
    plan, first_stage_result = freeze_stochastic_first_stage(
        bundle,
        physical,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
        seed=seed,
        policy_id="B6",
        config=config,
        evidence_hashes=evidence_hashes,
        solver=first_stage_solver,
    )
    first_stage_audit = _solver_audit(first_stage_result, stage="first_stage")
    sidecar_path = checkpoint_root / "operational_sidecar.json"
    sidecar_fingerprint = sha256_json(
        {
            "plan_sha256": plan.plan_sha256,
            "settlement_sha256": settlement.input_sha256,
            "forcing_sha256": forcing.input_sha256,
            "recourse_solver": {
                "module": getattr(recourse_solver, "__module__", "unknown"),
                "qualname": getattr(recourse_solver, "__qualname__", type(recourse_solver).__name__),
            },
            "solver_feasibility_rule": (
                "incumbent_required_and_finite_relative_mip_gap_le_0_05"
            ),
        }
    )
    if sidecar_path.is_file():
        sidecar = _read_json(sidecar_path)
        if sidecar.get("input_sha256") != sidecar_fingerprint:
            raise DARTRealEvidenceError(f"operational sidecar fingerprint changed: {sidecar_path}")
    else:
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "status": "IN_PROGRESS",
            "input_sha256": sidecar_fingerprint,
            "records": {},
        }
        atomic_json(sidecar_path, sidecar)
    timestamp_to_interval = {
        str(value): index for index, value in enumerate(bundle.timestamps_utc)
    }

    def instrumented_solver(
        suffix: ScenarioBundle,
        physical_artifact: FrozenPhysicalArtifact,
        policy_id: str,
        solver_config: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        result = dict(recourse_solver(suffix, physical_artifact, policy_id, solver_config))
        dispatch = _first_interval_dispatch(result)
        interval = timestamp_to_interval[str(suffix.timestamps_utc[0])]
        audit = _solver_audit(result, stage=f"recourse_interval_{interval}")
        record = {
            **audit,
            "interval": interval,
            "shortfall_work_units": float(
                np.asarray(dispatch.get("shortfall_work_units", 0.0), dtype=float).sum()
            ),
        }
        sidecar["records"][str(interval)] = record
        atomic_json(sidecar_path, sidecar)
        return result

    settings = dict(config)

    def costs(interval: int, dispatch: Mapping[str, Any]) -> Mapping[str, float]:
        derived = _component_costs(dispatch, settings)
        if component_cost_hook is not None:
            supplied = dict(component_cost_hook(interval, dispatch))
            overlap = set(derived).intersection(supplied)
            if overlap:
                raise DARTRealEvidenceError(
                    f"component cost hook duplicates derived costs: {sorted(overlap)}"
                )
            derived.update({str(key): float(value) for key, value in supplied.items()})
        return derived

    replay = run_external_replay_day(
        bundle,
        physical,
        plan,
        settlement,
        forcing,
        config=settings,
        checkpoint_path=checkpoint_root / "external_replay.json",
        recourse_solver=instrumented_solver,
        component_cost_hook=costs,
    )
    if replay.get("status") != "COMPLETE":
        raise DARTRealEvidenceError("external replay did not complete every interval")
    sidecar = _read_json(sidecar_path)
    records = [sidecar["records"].get(str(index)) for index in range(len(bundle.timestamps_utc))]
    if any(record is None for record in records):
        raise DARTRealEvidenceError("operational sidecar lacks one or more replay intervals")
    typed_records = [dict(record) for record in records if record is not None]
    sidecar["status"] = "COMPLETE"
    sidecar["records_sha256"] = sha256_json(typed_records)
    atomic_json(sidecar_path, sidecar)
    causal = all(
        interval.get("future_realized_targets_used") is False
        and interval.get("causal_audit", {}).get("future_realized_targets_used") is False
        and interval.get("causal_audit", {}).get("future_issue_time_hashes")
        == interval.get("causal_audit", {}).get("replay_future_hashes")
        for interval in replay["intervals"]
    )
    if not causal:
        raise DARTRealEvidenceError("causal recourse audit failed")
    workload_total = float(np.sum(forcing.workload))
    shortfall_total = float(sum(record["shortfall_work_units"] for record in typed_records))
    sla = 1.0 if workload_total <= 0.0 else max(0.0, min(1.0, 1.0 - shortfall_total / workload_total))
    recourse_timeout_count = sum(
        int(bool(record["time_limit_reached"])) for record in typed_records
    )
    recourse_nonzero_gap_count = sum(
        int(bool(record["nonzero_gap"])) for record in typed_records
    )
    recourse_gap_over_cap_count = sum(
        int(bool(record["gap_over_cap"])) for record in typed_records
    )
    recourse_no_incumbent_count = sum(
        int(record["has_incumbent"] is not True) for record in typed_records
    )
    recourse_infeasible_count = sum(
        int(bool(record["rejected_under_bounded_gap_rule"]))
        for record in typed_records
    )
    solver_any_timeout = bool(
        first_stage_audit["time_limit_reached"] or recourse_timeout_count
    )
    solver_any_nonzero_gap = bool(
        first_stage_audit["nonzero_gap"] or recourse_nonzero_gap_count
    )
    solver_any_gap_over_cap = bool(
        first_stage_audit["gap_over_cap"] or recourse_gap_over_cap_count
    )
    solver_any_no_incumbent = bool(
        first_stage_audit["has_incumbent"] is not True
        or recourse_no_incumbent_count
    )
    solver_any_infeasible = bool(
        first_stage_audit["rejected_under_bounded_gap_rule"]
        or recourse_infeasible_count
    )
    binding_labels = sorted(
        {
            str(label)
            for audit in (first_stage_audit, *typed_records)
            for label in (audit.get("binding_constraint_labels") or [])
        }
    )
    invalid = bool(
        replay.get("maximum_settlement_identity_residual_mw", np.inf) > 1e-8
        or not causal
        or solver_any_infeasible
    )
    components = dict(replay.get("component_costs_usd", {}))
    result = {
        **components,
        "simulated_total_cost_usd": float(replay["simulated_total_cost_usd"]),
        "weighted_sla_satisfaction": float(sla),
        "weighted_sla_satisfaction_role": (
            "engineered_replay_shortfall_fraction_not_observed_business_sla"
        ),
        "deadline_violations": shortfall_total,
        "deadline_violations_role": (
            "engineered_recourse_shortfall_work_units_not_observed_deadline_events"
        ),
        "infeasible_day": solver_any_infeasible,
        "invalid_day": bool(invalid),
        "forecast_time_seconds": float(max(forecast_time_seconds, 0.0)),
        "forecast_time_role": "serialized_oos_artifact_materialization_latency",
        "solve_time_seconds": float(
            first_stage_result.get("solve_time_seconds", 0.0)
            + sum(record["solve_time_seconds"] for record in typed_records)
        ),
        "model_complexity": float(max(model_complexity, 0.0)),
        "model_complexity_role": "combined_da_rt_checkpoint_bytes",
        "gate_time_feasible_flexible_energy_mwh": float(
            0.0 if actionability is None else actionability.total_feasible_mwh
        ),
        "source_evidence_sha256": sha256_json(
            {
                "plan": plan.plan_sha256,
                "settlement": settlement.input_sha256,
                "forcing": forcing.input_sha256,
                "replay": replay["replay_rows_sha256"],
                "operations": sidecar["records_sha256"],
            }
        ),
        "first_stage_plan_sha256": plan.plan_sha256,
        "first_stage_solver_audit": first_stage_audit,
        "recourse_solver_audit": {
            "interval_count": len(typed_records),
            "timeout_count": recourse_timeout_count,
            "nonzero_gap_count": recourse_nonzero_gap_count,
            "gap_over_cap_count": recourse_gap_over_cap_count,
            "no_incumbent_count": recourse_no_incumbent_count,
            "infeasible_under_frozen_rule_count": recourse_infeasible_count,
            "mip_gap_missing_count": sum(
                int(not bool(record["mip_gap_reported"])) for record in typed_records
            ),
            "records_sha256": sidecar["records_sha256"],
        },
        "solver_any_time_limit_reached": solver_any_timeout,
        "solver_any_nonzero_gap": solver_any_nonzero_gap,
        "solver_any_gap_over_cap": solver_any_gap_over_cap,
        "solver_any_missing_gap": bool(
            first_stage_audit["missing_mip_gap"]
            or any(not bool(record["mip_gap_reported"]) for record in typed_records)
        ),
        "solver_any_no_incumbent": solver_any_no_incumbent,
        "solver_any_infeasible_under_frozen_rule": solver_any_infeasible,
        "solver_feasibility_rule": (
            "incumbent_required_and_finite_relative_mip_gap_le_0_05"
        ),
        "maximum_accepted_relative_mip_gap": MAXIMUM_ACCEPTED_MIP_GAP,
        "planner_objective_bound_scope": (
            "issue_time_first_stage_B6_objective_not_realized_settlement_cost"
        ),
        "planner_objective_lower_bound": first_stage_audit.get(
            "solver_objective_lower_bound"
        ),
        "planner_objective_upper_bound": first_stage_audit.get(
            "solver_objective_upper_bound"
        ),
        "a3_context": {
            "dart_sign": _dart_sign_context(settlement),
            "binding_constraint": binding_labels or None,
            "binding_constraint_status": (
                "SOLVER_REPORTED" if binding_labels else "UNAVAILABLE"
            ),
            "binding_constraint_undefined_reason": (
                None
                if binding_labels
                else "SOLVER_DID_NOT_REPORT_DUAL_OR_ACTIVITY_DIAGNOSTICS"
            ),
            "binding_constraint_evidence_role": (
                "explicit_solver_report_only_no_post_hoc_inference"
            ),
            "post_decision_context_only": True,
        },
        "first_stage_frozen_before_replay": True,
        "causal_recourse_no_future_reveal": True,
        "external_historical_nyiso_replay": True,
        "selected_scenario_used_as_realized": False,
        "counterfactual": True,
        "future_realized_targets_used": False,
        "causal_provenance": {
            "plan_sha256": plan.plan_sha256,
            "settlement_sha256": settlement.input_sha256,
            "forcing_sha256": forcing.input_sha256,
            "issue_bundle_sha256": sha256_json(
                {
                    "timestamps": _array_sha256(bundle.timestamps_utc),
                    "da": _array_sha256(bundle.da_price),
                    "rt_ensemble": _array_sha256(bundle.rt_price),
                    "workload_ensemble": _array_sha256(bundle.workload),
                    "pv_ensemble": _array_sha256(bundle.pv),
                    "physical_multiplier": _array_sha256(bundle.physical_multiplier),
                    "probabilities": _array_sha256(bundle.probabilities),
                }
            ),
            "evidence_hashes_sha256": sha256_json(dict(evidence_hashes)),
            "external_replay_rows_sha256": replay["replay_rows_sha256"],
            "operational_sidecar_records_sha256": sidecar["records_sha256"],
            "first_stage_inputs_known_at_gate": True,
            "realized_settlement_revealed_after_first_stage_freeze": True,
            "rolling_recourse_current_interval_only": True,
            "future_issue_time_hashes_match_replay": True,
            "candidate_independent_forcing": True,
            "selected_scenario_as_realized": False,
            "future_realized_targets_used": False,
        },
    }
    return result


def _perfect_information_bundle(
    base: ScenarioBundle,
    settlement: ReferencedNYISOReplay,
    forcing: FrozenReplayForcing,
) -> ScenarioBundle:
    scenarios = base.reduced_count
    output = replace(
        base,
        da_price=np.asarray(settlement.da_lbmp, dtype=float),
        rt_price=np.broadcast_to(
            settlement.rt_lbmp, (scenarios, *settlement.rt_lbmp.shape)
        ).copy(),
        workload=np.broadcast_to(
            forcing.workload, (scenarios, len(forcing.workload))
        ).copy(),
        pv=np.broadcast_to(
            forcing.pv, (scenarios, *forcing.pv.shape)
        ).copy(),
        physical_multiplier=np.broadcast_to(
            forcing.physical_multiplier,
            (scenarios, len(forcing.physical_multiplier)),
        ).copy(),
    )
    output.validate()
    return output


def _ratio_stat(
    numerator: float,
    denominator: float,
    *,
    zero_reason: str,
    count: int,
) -> dict[str, Any]:
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 0.0:
        raise DARTRealEvidenceError("metric sufficient statistics are invalid")
    if denominator == 0.0:
        return {
            "value": None,
            "status": zero_reason,
            "undefined_reason": zero_reason,
            "numerator": numerator,
            "denominator": denominator,
            "count": int(count),
        }
    return {
        "value": numerator / denominator,
        "status": "OK",
        "undefined_reason": None,
        "numerator": numerator,
        "denominator": denominator,
        "count": int(count),
    }


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    probability = np.asarray(scores, dtype=float).reshape(-1)
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        return {
            "value": None,
            "status": "NO_POSITIVE_SPIKES",
            "undefined_reason": "NO_POSITIVE_SPIKES",
            "positive_count": 0,
            "event_count": int(truth.size),
        }
    order = np.argsort(-probability, kind="stable")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return {
        "value": float(np.sum(precision[ordered]) / positives),
        "status": "OK",
        "undefined_reason": None,
        "positive_count": positives,
        "event_count": int(truth.size),
    }


def _interval_diagnostics(actual: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for coverage in (0.50, 0.80, 0.90, 0.95):
        alpha = (1.0 - coverage) / 2.0
        lower = np.quantile(draws, alpha, axis=0, method="linear")
        upper = np.quantile(draws, 1.0 - alpha, axis=0, method="linear")
        diagnostic = interval_calibration(
            actual, lower, upper, nominal_coverage=coverage
        )
        count = int(diagnostic["count"])
        covered = (
            0.0
            if diagnostic["coverage"] is None
            else float(diagnostic["coverage"]) * count
        )
        width_total = (
            0.0
            if diagnostic["mean_width"] is None
            else float(diagnostic["mean_width"]) * count
        )
        label = str(int(round(coverage * 100)))
        output[label] = {
            **diagnostic,
            "nominal_coverage": coverage,
            "coverage_sufficient_statistics": _ratio_stat(
                covered,
                count,
                zero_reason="NO_VALID_INTERVAL_ROWS",
                count=count,
            ),
            "width_sufficient_statistics": _ratio_stat(
                width_total,
                count,
                zero_reason="NO_VALID_INTERVAL_ROWS",
                count=count,
            ),
            "bounds_source": "empirical_joint_draw_quantiles",
        }
    return output


def _pit_diagnostics(actual: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pit = np.asarray(
        pit_from_ensemble(actual, draws, scenario_axis=0), dtype=float
    )
    finite = pit[np.isfinite(pit)]
    if finite.size == 0:
        return pit, {
            "status": "NO_VALID_PIT_ROWS",
            "undefined_reason": "NO_VALID_PIT_ROWS",
            "count": 0,
            "mean": None,
            "variance": None,
            "uniform_ks_distance": None,
            "histogram_bin_edges": np.linspace(0.0, 1.0, 11).tolist(),
            "histogram_counts": [0] * 10,
        }
    ordered = np.sort(finite)
    count = len(ordered)
    upper_distance = np.max(np.arange(1, count + 1) / count - ordered)
    lower_distance = np.max(ordered - np.arange(0, count) / count)
    histogram, edges = np.histogram(finite, bins=np.linspace(0.0, 1.0, 11))
    return pit, {
        "status": "OK",
        "undefined_reason": None,
        "count": int(count),
        "mean": float(np.mean(finite)),
        "variance": float(np.var(finite)),
        "uniform_ks_distance": float(max(upper_distance, lower_distance)),
        "histogram_bin_edges": edges.tolist(),
        "histogram_counts": histogram.astype(int).tolist(),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(valid)) < 2:
        return None
    x = np.asarray(left[valid], dtype=float)
    y = np.asarray(right[valid], dtype=float)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise DARTRealEvidenceError("row-correlation inputs must be aligned matrices")
    valid_rows = np.all(np.isfinite(x) & np.isfinite(y), axis=1)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    output = np.full(x.shape[0], np.nan, dtype=float)
    usable = valid_rows & (denominator > 0.0)
    output[usable] = numerator[usable] / denominator[usable]
    return output


def _temporal_acf_diagnostic(actual: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    horizon = actual.shape[1]
    maximum_lag = min(12, horizon - 1)
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for zone_index, zone in enumerate(NYISO_ZONES):
        for lag in range(1, maximum_lag + 1):
            observed = _correlation(actual[zone_index, :-lag], actual[zone_index, lag:])
            scenario_values = _row_correlations(
                draws[:, zone_index, :-lag], draws[:, zone_index, lag:]
            )
            scenario_values = scenario_values[np.isfinite(scenario_values)]
            if observed is None or scenario_values.size == 0:
                continue
            expected = float(np.mean(scenario_values))
            error = abs(expected - observed)
            errors.append(error)
            rows.append(
                {
                    "zone": zone,
                    "lag_intervals": lag,
                    "observed_acf": observed,
                    "forecast_mean_acf": expected,
                    "absolute_error": error,
                    "valid_scenario_count": int(scenario_values.size),
                }
            )
    statistics = _ratio_stat(
        sum(errors),
        len(errors),
        zero_reason="INSUFFICIENT_TEMPORAL_ACF_VARIATION",
        count=len(errors),
    )
    return {
        **statistics,
        "maximum_lag_intervals": maximum_lag,
        "zone_lag_records": rows,
        "definition": "mean absolute observed-versus-ensemble-mean ACF error",
    }


def _cross_zone_rank_diagnostic(actual: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    actual_ranks = rankdata(actual, axis=1, method="average", nan_policy="omit")
    scenario_ranks = rankdata(draws, axis=2, method="average", nan_policy="omit")
    for first in range(len(NYISO_ZONES)):
        for second in range(first + 1, len(NYISO_ZONES)):
            observed = _correlation(
                actual_ranks[first], actual_ranks[second]
            )
            scenario_values = _row_correlations(
                scenario_ranks[:, first, :], scenario_ranks[:, second, :]
            )
            scenario_values = scenario_values[np.isfinite(scenario_values)]
            if observed is None or scenario_values.size == 0:
                continue
            expected = float(np.mean(scenario_values))
            error = abs(expected - observed)
            errors.append(error)
            rows.append(
                {
                    "zone_pair": [NYISO_ZONES[first], NYISO_ZONES[second]],
                    "observed_spearman": observed,
                    "forecast_mean_spearman": expected,
                    "absolute_error": error,
                    "valid_scenario_count": int(scenario_values.size),
                }
            )
    statistics = _ratio_stat(
        sum(errors),
        len(errors),
        zero_reason="INSUFFICIENT_CROSS_ZONE_RANK_VARIATION",
        count=len(errors),
    )
    return {
        **statistics,
        "zone_pair_records": rows,
        "definition": "mean absolute observed-versus-ensemble-mean Spearman error",
    }


def _longest_true_runs(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=bool)
    if matrix.ndim != 2:
        raise DARTRealEvidenceError("longest-run input must be a row matrix")
    current = np.zeros(matrix.shape[0], dtype=int)
    longest = np.zeros(matrix.shape[0], dtype=int)
    for interval in range(matrix.shape[1]):
        current = np.where(matrix[:, interval], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def _spike_duration_diagnostic(
    actual: np.ndarray,
    draws: np.ndarray,
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[str, Any]:
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for direction, observed_mask, scenario_mask in (
        ("negative", actual < lower_threshold, draws < lower_threshold),
        ("positive", actual > upper_threshold, draws > upper_threshold),
    ):
        observed_durations = _longest_true_runs(observed_mask)
        scenario_durations_by_zone = _longest_true_runs(
            scenario_mask.reshape(-1, scenario_mask.shape[-1])
        ).reshape(scenario_mask.shape[0], scenario_mask.shape[1])
        for zone_index, zone in enumerate(NYISO_ZONES):
            observed_duration = int(observed_durations[zone_index])
            scenario_durations = np.asarray(
                scenario_durations_by_zone[:, zone_index], dtype=float
            )
            expected_duration = float(np.mean(scenario_durations))
            error = abs(expected_duration - observed_duration)
            errors.append(error)
            rows.append(
                {
                    "direction": direction,
                    "zone": zone,
                    "observed_longest_run_intervals": observed_duration,
                    "forecast_mean_longest_run_intervals": expected_duration,
                    "absolute_error_intervals": error,
                }
            )
    statistics = _ratio_stat(
        sum(errors),
        len(errors),
        zero_reason="NO_SPIKE_DURATION_ROWS",
        count=len(errors),
    )
    return {
        **statistics,
        "direction_zone_records": rows,
        "definition": "mean absolute longest-tail-run duration error in intervals",
    }


def _joint_spike_diagnostic(
    actual: np.ndarray,
    draws: np.ndarray,
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[str, Any]:
    actual_tail = (actual < lower_threshold) | (actual > upper_threshold)
    scenario_tail = (draws < lower_threshold) | (draws > upper_threshold)
    actual_joint = np.count_nonzero(actual_tail, axis=0) >= 2
    scenario_joint = np.count_nonzero(scenario_tail, axis=1) >= 2
    probabilities = np.mean(scenario_joint, axis=0)
    count = actual_joint.size
    frequency_error = abs(float(np.mean(probabilities)) - float(np.mean(actual_joint)))
    brier_numerator = float(
        np.sum((probabilities - actual_joint.astype(float)) ** 2)
    )
    return {
        "status": "OK" if count else "NO_JOINT_SPIKE_ROWS",
        "undefined_reason": None if count else "NO_JOINT_SPIKE_ROWS",
        "count": int(count),
        "observed_frequency": None if not count else float(np.mean(actual_joint)),
        "forecast_mean_probability": None if not count else float(np.mean(probabilities)),
        "frequency_absolute_error": None if not count else frequency_error,
        "brier_sufficient_statistics": _ratio_stat(
            brier_numerator,
            count,
            zero_reason="NO_JOINT_SPIKE_ROWS",
            count=count,
        ),
        "definition": "simultaneous tail event in at least two of four zones",
    }


def _persist_metric_diagnostics(
    root: Path,
    *,
    day: str,
    candidate_id: str,
    seed: int,
    pit_values: np.ndarray,
    actual: np.ndarray,
    draws: np.ndarray,
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
    path = root / day / candidate_id / f"seed_{seed}.npz"
    fingerprint = sha256_json(
        {
            "pit_values": _array_sha256(pit_values),
            "actual": _array_sha256(actual),
            "draws": _array_sha256(draws),
            "metadata": dict(metadata),
        }
    )
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            stored_fingerprint = str(stored["input_sha256"].item())
        if stored_fingerprint != fingerprint:
            raise DARTRealEvidenceError(
                f"metric diagnostic checkpoint fingerprint changed: {path}"
            )
    else:
        _atomic_npz(
            path,
            pit_values=np.asarray(pit_values, dtype=float),
            actual_dart_da_minus_rt_sha256=np.asarray(_array_sha256(actual)),
            forecast_dart_draws_sha256=np.asarray(_array_sha256(draws)),
            input_sha256=np.asarray(fingerprint),
            metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
        )
    return path.as_posix(), sha256_file(path)


def _metric_diagnostic_complete(record: Mapping[str, Any]) -> bool:
    layers = record.get("diagnostic_layers")
    if not isinstance(layers, Mapping) or set(layers) != {"layer1", "layer2"}:
        return False
    path = Path(str(record.get("diagnostic_artifact_path", "")))
    if not path.is_file():
        return False
    try:
        expected = _valid_sha256(
            record.get("diagnostic_artifact_sha256"),
            label="metric diagnostic artifact",
        )
    except DARTRealEvidenceError:
        return False
    return sha256_file(path) == expected


def _action_sufficient_statistics(
    probabilities: np.ndarray,
    realized: np.ndarray,
    weights: np.ndarray,
    action_metrics: Mapping[str, Any],
    *,
    wrong_action_economic_loss_usd: float,
    missed_opportunity_economic_loss_usd: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    class_values = np.asarray((-1, 0, 1), dtype=int)
    predicted = class_values[np.argmax(probabilities, axis=-1)]
    one_hot = realized[..., None] == class_values
    row_brier = np.sum((probabilities - one_hot.astype(float)) ** 2, axis=-1)
    total_weight = float(np.sum(weights))
    statistics: dict[str, Any] = {
        "economic_action_brier": _ratio_stat(
            float(np.sum(weights * row_brier)),
            total_weight,
            zero_reason="NO_FEASIBLE_FLEXIBILITY",
            count=int(np.count_nonzero(weights > 0.0)),
        )
    }
    for field_name, source_name in (
        ("action_precision", "flexibility_weighted_precision"),
        ("action_recall", "flexibility_weighted_recall"),
        ("action_f1", "flexibility_weighted_f1"),
    ):
        value = action_metrics.get(source_name)
        statistics[field_name] = _ratio_stat(
            0.0 if value is None else float(value) * total_weight,
            0.0 if value is None else total_weight,
            zero_reason="NO_FEASIBLE_FLEXIBILITY",
            count=int(np.count_nonzero(weights > 0.0)),
        )
    opportunity_denominator = float(
        action_metrics["opportunity_weight_denominator"]
    )
    statistics["wrong_action_loss"] = _ratio_stat(
        float(action_metrics["wrong_action_loss_total"]),
        opportunity_denominator,
        zero_reason="NO_FEASIBLE_OPPORTUNITY",
        count=int(np.count_nonzero(weights > 0.0)),
    )
    statistics["missed_opportunity_loss"] = _ratio_stat(
        float(action_metrics["missed_opportunity_loss_total"]),
        opportunity_denominator,
        zero_reason="NO_FEASIBLE_OPPORTUNITY",
        count=int(np.count_nonzero(weights > 0.0)),
    )
    statistics["wrong_action_rate"] = _ratio_stat(
        float(action_metrics["wrong_action_rate_numerator"]),
        float(action_metrics["wrong_action_rate_denominator"]),
        zero_reason="NO_EXECUTED_ACTION",
        count=int(np.count_nonzero(predicted != 0)),
    )
    statistics["missed_opportunity_rate"] = _ratio_stat(
        float(action_metrics["missed_opportunity_rate_numerator"]),
        float(action_metrics["missed_opportunity_rate_denominator"]),
        zero_reason="NO_REALIZED_ECONOMIC_OPPORTUNITY",
        count=int(np.count_nonzero(realized != 0)),
    )
    # A unit denominator makes the phase aggregator report the mean daily USD
    # loss after averaging the three common seeds within each date.  Physical
    # MWh, rather than normalized F, is deliberately used in these two totals.
    statistics["wrong_action_economic_loss_usd"] = _ratio_stat(
        wrong_action_economic_loss_usd,
        1.0,
        zero_reason="NO_VALID_DAY",
        count=int(np.count_nonzero(weights > 0.0)),
    )
    statistics["missed_opportunity_economic_loss_usd"] = _ratio_stat(
        missed_opportunity_economic_loss_usd,
        1.0,
        zero_reason="NO_VALID_DAY",
        count=int(np.count_nonzero(weights > 0.0)),
    )
    confusion: dict[str, Any] = {}
    for action_class in class_values:
        truth = realized == action_class
        chosen = predicted == action_class
        confusion[str(int(action_class))] = {
            "weighted_true_positive": float(np.sum(weights[truth & chosen])),
            "weighted_false_positive": float(np.sum(weights[~truth & chosen])),
            "weighted_false_negative": float(np.sum(weights[truth & ~chosen])),
            "weighted_support": float(np.sum(weights[truth])),
        }
    return statistics, confusion


def _spike_sufficient_statistics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[dict[str, Any], dict[str, int]]:
    true_positive = int(np.count_nonzero(actual & predicted))
    false_positive = int(np.count_nonzero(~actual & predicted))
    false_negative = int(np.count_nonzero(actual & ~predicted))
    actual_positive_count = true_positive + false_negative
    predicted_positive_count = true_positive + false_positive
    precision = _ratio_stat(
        true_positive,
        predicted_positive_count,
        zero_reason="NO_PREDICTED_POSITIVE_SPIKES",
        count=predicted_positive_count,
    )
    recall = _ratio_stat(
        true_positive,
        actual_positive_count,
        zero_reason="NO_POSITIVE_SPIKES",
        count=actual_positive_count,
    )
    if actual_positive_count == 0:
        f1 = {
            "value": None,
            "status": "NO_POSITIVE_SPIKES",
            "undefined_reason": "NO_POSITIVE_SPIKES",
            "numerator": float(2 * true_positive),
            "denominator": float(2 * true_positive + false_positive + false_negative),
            "count": 0,
        }
    else:
        f1 = _ratio_stat(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
            zero_reason="NO_POSITIVE_OR_PREDICTED_SPIKES",
            count=actual_positive_count,
        )
    return (
        {"spike_precision": precision, "spike_recall": recall, "spike_f1": f1},
        {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "actual_positive_count": actual_positive_count,
            "predicted_positive_count": predicted_positive_count,
        },
    )


def _metric_record(
    *,
    day: str,
    candidate_id: str,
    seed: int,
    daily: DailyDART,
    settlement: ReferencedNYISOReplay,
    actionability: ActionabilityDay,
    thresholds: Mapping[str, Any],
    spike_probability_threshold: float,
    diagnostics_root: Path,
) -> dict[str, Any]:
    actual = settlement.dart_da_minus_rt
    quantiles = daily.dart_quantiles
    draws = daily.dart_draws
    if actual.shape != quantiles.shape[:2] or actual.shape != draws.shape[1:]:
        raise DARTRealEvidenceError("DART forecast and historical outcome shapes differ")
    median = quantiles[..., QUANTILES.index(0.50)]
    per_event_crps = np.asarray(
        crps_from_quantiles(actual, quantiles, QUANTILES, reduction="none"), dtype=float
    )
    if tuple(map(str, thresholds["zone_order"])) != tuple(NYISO_ZONES):
        raise DARTRealEvidenceError("tail-threshold zone order changed")
    lower_extreme_by_zone = np.asarray(
        thresholds["dart_lower_q01_by_zone"], dtype=float
    )
    lower_tail_by_zone = np.asarray(
        thresholds["dart_lower_q05_by_zone"], dtype=float
    )
    upper_tail_by_zone = np.asarray(
        thresholds["dart_upper_q95_by_zone"], dtype=float
    )
    upper_extreme_by_zone = np.asarray(
        thresholds["dart_upper_q99_by_zone"], dtype=float
    )
    if (
        lower_extreme_by_zone.shape != (len(NYISO_ZONES),)
        or lower_tail_by_zone.shape != lower_extreme_by_zone.shape
        or upper_tail_by_zone.shape != lower_extreme_by_zone.shape
        or upper_extreme_by_zone.shape != lower_extreme_by_zone.shape
        or not np.all(np.isfinite(lower_extreme_by_zone))
        or not np.all(np.isfinite(lower_tail_by_zone))
        or not np.all(np.isfinite(upper_tail_by_zone))
        or not np.all(np.isfinite(upper_extreme_by_zone))
        or np.any(lower_extreme_by_zone > lower_tail_by_zone)
        or np.any(lower_tail_by_zone >= upper_tail_by_zone)
        or np.any(upper_tail_by_zone > upper_extreme_by_zone)
    ):
        raise DARTRealEvidenceError(
            "per-zone thresholds must satisfy q01 <= q05 < q95 <= q99"
        )
    lower_extreme_threshold = np.broadcast_to(
        lower_extreme_by_zone[:, None], actual.shape
    )
    lower_tail_threshold = np.broadcast_to(
        lower_tail_by_zone[:, None], actual.shape
    )
    upper_tail_threshold = np.broadcast_to(
        upper_tail_by_zone[:, None], actual.shape
    )
    upper_extreme_threshold = np.broadcast_to(
        upper_extreme_by_zone[:, None], actual.shape
    )
    lower_probability = np.mean(draws < lower_extreme_threshold, axis=0)
    upper_probability = np.mean(draws > upper_extreme_threshold, axis=0)
    lower_calibration = exceedance_calibration(
        actual,
        lower_probability,
        threshold=lower_extreme_threshold,
        direction="below",
    )
    upper_calibration = exceedance_calibration(
        actual,
        upper_probability,
        threshold=upper_extreme_threshold,
        direction="above",
    )
    lower_extreme_score = np.asarray(
        threshold_weighted_crps(
            actual,
            draws,
            lower_threshold=lower_extreme_threshold,
            region="lower",
            scenario_axis=0,
            reduction="none",
        ),
        dtype=float,
    )
    upper_extreme_score = np.asarray(
        threshold_weighted_crps(
            actual,
            draws,
            upper_threshold=upper_extreme_threshold,
            region="upper",
            scenario_axis=0,
            reduction="none",
        ),
        dtype=float,
    )
    lower_tail_score = np.asarray(
        threshold_weighted_crps(
            actual,
            draws,
            lower_threshold=lower_tail_threshold,
            region="lower",
            scenario_axis=0,
            reduction="none",
        ),
        dtype=float,
    )
    upper_tail_score = np.asarray(
        threshold_weighted_crps(
            actual,
            draws,
            upper_threshold=upper_tail_threshold,
            region="upper",
            scenario_axis=0,
            reduction="none",
        ),
        dtype=float,
    )
    extreme_tail_score = lower_extreme_score + upper_extreme_score
    q05_q95_tail_score = lower_tail_score + upper_tail_score
    emphasized_tail_score = np.asarray(
        emphasized_tail_threshold_weighted_crps(
            actual,
            draws,
            lower_extreme_threshold=lower_extreme_threshold,
            lower_tail_threshold=lower_tail_threshold,
            upper_tail_threshold=upper_tail_threshold,
            upper_extreme_threshold=upper_extreme_threshold,
            scenario_axis=0,
            reduction="none",
        ),
        dtype=float,
    )
    if not np.allclose(
        emphasized_tail_score,
        extreme_tail_score + q05_q95_tail_score,
        equal_nan=True,
    ):
        raise DARTRealEvidenceError("emphasized tail score decomposition changed")
    negative_tail_score = lower_extreme_score + lower_tail_score
    positive_tail_score = upper_extreme_score + upper_tail_score
    valid_crps = (
        np.isfinite(negative_tail_score)
        & np.isfinite(positive_tail_score)
        & np.isfinite(actual)
    )
    negative_tail_crps = _ratio_stat(
        float(np.sum(negative_tail_score[valid_crps])),
        float(np.count_nonzero(valid_crps)),
        zero_reason="NO_VALID_NEGATIVE_TAIL_CRPS_ROWS",
        count=int(np.count_nonzero(valid_crps)),
    )
    positive_tail_crps = _ratio_stat(
        float(np.sum(positive_tail_score[valid_crps])),
        float(np.count_nonzero(valid_crps)),
        zero_reason="NO_VALID_POSITIVE_TAIL_CRPS_ROWS",
        count=int(np.count_nonzero(valid_crps)),
    )
    intervals = _interval_diagnostics(actual, draws)
    pit_values, pit = _pit_diagnostics(actual, draws)
    action_probabilities = economic_action_probabilities(
        np.moveaxis(draws, 0, -1),
        actionability.lower_no_action_bound,
        actionability.upper_no_action_bound,
        scenario_axis=-1,
    )
    actual_actions = economic_action_labels(
        actual,
        actionability.lower_no_action_bound,
        actionability.upper_no_action_bound,
    )
    normalized_f = np.asarray(
        actionability.normalized_feasibility_weight, dtype=float
    )
    realized_net_gain = economic_net_gain_signal(
        actual,
        actionability.lower_no_action_bound,
        actionability.upper_no_action_bound,
    )
    predicted_net_gain = economic_net_gain_signal(
        median,
        actionability.lower_no_action_bound,
        actionability.upper_no_action_bound,
    )
    action = economic_action_metrics(
        action_probabilities,
        actual_actions,
        normalized_f,
        opportunity_magnitude=np.abs(realized_net_gain),
        source_fields=actionability.source_fields,
        available_at_gate=actionability.available_at_gate,
    )
    weighted = action_weighted_crps(
        per_event_crps,
        normalized_f,
        source_fields=actionability.source_fields,
        available_at_gate=actionability.available_at_gate,
    )
    feasibility_q05_q95_tail = (
        feasibility_conditioned_threshold_weighted_crps(
            actual,
            draws,
            normalized_f,
            lower_threshold=lower_tail_threshold,
            upper_threshold=upper_tail_threshold,
            scenario_axis=0,
            source_fields=actionability.source_fields,
            available_at_gate=actionability.available_at_gate,
        )
    )
    feasibility_q01_q99_extreme = (
        feasibility_conditioned_threshold_weighted_crps(
            actual,
            draws,
            normalized_f,
            lower_threshold=lower_extreme_threshold,
            upper_threshold=upper_extreme_threshold,
            scenario_axis=0,
            source_fields=actionability.source_fields,
            available_at_gate=actionability.available_at_gate,
        )
    )
    if (
        feasibility_q05_q95_tail.status != feasibility_q01_q99_extreme.status
        or feasibility_q05_q95_tail.count != feasibility_q01_q99_extreme.count
        or not np.isclose(
            feasibility_q05_q95_tail.denominator,
            feasibility_q01_q99_extreme.denominator,
        )
    ):
        raise DARTRealEvidenceError(
            "feasibility-conditioned tail components used different F panels"
        )
    feasibility_tail_denominator = feasibility_q05_q95_tail.denominator
    feasibility_tail_numerator = (
        feasibility_q05_q95_tail.numerator
        + feasibility_q01_q99_extreme.numerator
    )
    feasibility_tail_value = (
        feasibility_tail_numerator / feasibility_tail_denominator
        if feasibility_tail_denominator > 0.0
        else None
    )
    feasibility_tail_statistics = {
        "value": feasibility_tail_value,
        "status": feasibility_q05_q95_tail.status,
        "numerator": feasibility_tail_numerator,
        "denominator": feasibility_tail_denominator,
        "count": feasibility_q05_q95_tail.count,
        "undefined_reason": (
            None if feasibility_tail_value is not None else feasibility_q05_q95_tail.status
        ),
    }
    feasibility_q05_q95_statistics = {
        **feasibility_q05_q95_tail.as_dict(),
        "undefined_reason": (
            None
            if feasibility_q05_q95_tail.value is not None
            else feasibility_q05_q95_tail.status
        ),
    }
    feasibility_q01_q99_statistics = {
        **feasibility_q01_q99_extreme.as_dict(),
        "undefined_reason": (
            None
            if feasibility_q01_q99_extreme.value is not None
            else feasibility_q01_q99_extreme.status
        ),
    }
    forl = feasible_opportunity_ranking_loss(
        normalized_f,
        realized_net_gain,
        predicted_net_gain,
    )
    spike_probability = np.clip(lower_probability + upper_probability, 0.0, 1.0)
    spike_actual = (actual < lower_extreme_threshold) | (
        actual > upper_extreme_threshold
    )
    spike_predicted = spike_probability >= spike_probability_threshold
    spike_statistics, spike_counts = _spike_sufficient_statistics(
        spike_actual, spike_predicted
    )
    average_precision = _average_precision(spike_actual, spike_probability)
    q99_brier = _ratio_stat(
        float(np.sum((spike_probability - spike_actual.astype(float)) ** 2)),
        float(spike_actual.size),
        zero_reason="NO_Q99_EVENT_ROWS",
        count=int(spike_actual.size),
    )
    predicted_actions = np.asarray(action["predicted_actions"], dtype=int)
    decrease_da_gain = actual - actionability.upper_no_action_bound
    increase_da_gain = actionability.lower_no_action_bound - actual
    chosen_gain = np.where(
        predicted_actions == -1,
        decrease_da_gain,
        np.where(predicted_actions == 1, increase_da_gain, 0.0),
    )
    wrong_mask = (predicted_actions != 0) & (predicted_actions != actual_actions)
    missed_mask = (predicted_actions == 0) & (actual_actions != 0)
    physical_mwh = np.asarray(actionability.feasible_by_event_mwh, dtype=float)
    wrong_action_economic_loss_usd = float(
        np.sum(physical_mwh * np.maximum(-chosen_gain, 0.0) * wrong_mask)
    )
    missed_opportunity_economic_loss_usd = float(
        np.sum(physical_mwh * np.abs(realized_net_gain) * missed_mask)
    )
    weights = normalized_f
    action_statistics, action_confusion = _action_sufficient_statistics(
        action_probabilities,
        actual_actions,
        weights,
        action,
        wrong_action_economic_loss_usd=wrong_action_economic_loss_usd,
        missed_opportunity_economic_loss_usd=(
            missed_opportunity_economic_loss_usd
        ),
    )
    feasibility_weighted_global_crps_statistics = {
        **weighted.as_dict(),
        "undefined_reason": None if weighted.value is not None else weighted.status,
    }
    # AWCRPS follows the preregistered local-tail definition: the proper
    # q05/q95 threshold score plus the extra q01/q99 extreme component,
    # weighted only by candidate-independent normalized F.
    awcrps_statistics = dict(feasibility_tail_statistics)
    forl_statistics = {
        **forl.as_dict(),
        "undefined_reason": None if forl.value is not None else forl.status,
    }
    q99_miss_rate = _ratio_stat(
        float(spike_counts["false_negative"]),
        float(spike_counts["actual_positive_count"]),
        zero_reason="NO_POSITIVE_Q99_EVENTS",
        count=int(spike_counts["actual_positive_count"]),
    )
    sufficient_statistics = {
        "awcrps": awcrps_statistics,
        "feasibility_conditioned_tail_crps": feasibility_tail_statistics,
        "feasibility_conditioned_q05_q95_tail_crps": (
            feasibility_q05_q95_statistics
        ),
        "feasibility_conditioned_q01_q99_extreme_crps": (
            feasibility_q01_q99_statistics
        ),
        "forl": forl_statistics,
        "q99_event_brier": q99_brier,
        "q99_precision": spike_statistics["spike_precision"],
        "q99_recall": spike_statistics["spike_recall"],
        "q99_f1": spike_statistics["spike_f1"],
        "q99_miss_rate": q99_miss_rate,
        **action_statistics,
        **spike_statistics,
    }
    # One multivariate path case per zone avoids mixing the four spatial paths.
    observations = actual
    scenario_paths = np.moveaxis(draws, 0, 1)
    energy = energy_score(observations, scenario_paths, scenario_axis=1)
    variogram = variogram_score(observations, scenario_paths, scenario_axis=1)
    temporal_acf = _temporal_acf_diagnostic(actual, draws)
    cross_zone_rank = _cross_zone_rank_diagnostic(actual, draws)
    spike_duration = _spike_duration_diagnostic(
        actual,
        draws,
        lower_threshold=lower_extreme_threshold,
        upper_threshold=upper_extreme_threshold,
    )
    joint_spike = _joint_spike_diagnostic(
        actual,
        draws,
        lower_threshold=lower_extreme_threshold,
        upper_threshold=upper_extreme_threshold,
    )
    temporal_acf_summary = {
        key: value
        for key, value in temporal_acf.items()
        if key != "zone_lag_records"
    }
    cross_zone_rank_summary = {
        key: value
        for key, value in cross_zone_rank.items()
        if key != "zone_pair_records"
    }
    spike_duration_summary = {
        key: value
        for key, value in spike_duration.items()
        if key != "direction_zone_records"
    }
    base_source_sha = sha256_json(
        {
            "dart_artifact": daily.source_sha256,
            "nyiso_replay": settlement.input_sha256,
            "actionability": actionability.source_sha256,
            "thresholds": dict(thresholds),
        }
    )
    diagnostic_path, diagnostic_sha = _persist_metric_diagnostics(
        diagnostics_root,
        day=day,
        candidate_id=candidate_id,
        seed=seed,
        pit_values=pit_values,
        actual=actual,
        draws=draws,
        metadata={
            "source_evidence_sha256": base_source_sha,
            "canonical_dart_definition": "DA_MINUS_RT",
            "future_realized_inputs_used": False,
            "locked_test_outcome_used_for_gate_features": False,
            "diagnostic_detail_records": {
                "temporal_acf": temporal_acf["zone_lag_records"],
                "cross_zone_rank": cross_zone_rank["zone_pair_records"],
                "spike_duration": spike_duration["direction_zone_records"],
            },
        },
    )
    source_sha = sha256_json(
        {
            "underlying_sources_sha256": base_source_sha,
            "diagnostic_artifact_sha256": diagnostic_sha,
        }
    )
    negative_calibration = {
        **lower_calibration,
        "direction": "below",
        "zone_order": list(NYISO_ZONES),
        "threshold_quantile": 0.01,
        "threshold_by_zone": lower_extreme_by_zone.tolist(),
    }
    positive_calibration = {
        **upper_calibration,
        "direction": "above",
        "zone_order": list(NYISO_ZONES),
        "threshold_quantile": 0.99,
        "threshold_by_zone": upper_extreme_by_zone.tolist(),
    }
    layer1 = {
        "intervals": intervals,
        "pit": pit,
        "negative_tail_weighted_crps": negative_tail_crps,
        "positive_tail_weighted_crps": positive_tail_crps,
        "q05_q95_threshold_weighted_crps": float(
            np.mean(q05_q95_tail_score)
        ),
        "q01_q99_extreme_threshold_weighted_crps": float(
            np.mean(extreme_tail_score)
        ),
        "emphasized_tail_threshold_weighted_crps": float(
            np.mean(emphasized_tail_score)
        ),
        "negative_exceedance_calibration": negative_calibration,
        "positive_exceedance_calibration": positive_calibration,
        "tail_threshold_source": "per_zone_pre_origin_training_history_only",
        "tail_threshold_contract_sha256": thresholds[
            "threshold_contract_sha256"
        ],
        "proper_threshold_weighted_scoring": True,
    }
    layer2 = {
        "energy_score": float(energy),
        "variogram_score": float(variogram),
        "temporal_autocorrelation_error": temporal_acf_summary,
        "cross_zone_rank_correlation_error": cross_zone_rank_summary,
        "spike_duration_error": spike_duration_summary,
        "joint_spike_frequency_error": joint_spike,
    }
    return {
        "operating_date": day,
        "candidate_id": candidate_id,
        "seed": seed,
        "mae": float(np.mean(np.abs(actual - median))),
        "rmse": float(np.sqrt(np.mean((actual - median) ** 2))),
        "crps": float(crps_from_quantiles(actual, quantiles, QUANTILES)),
        "pinball_loss": float(pinball_loss(actual, quantiles, QUANTILES)),
        "tail_crps": float(np.mean(emphasized_tail_score)),
        "threshold_weighted_crps": float(np.mean(emphasized_tail_score)),
        "emphasized_tail_threshold_weighted_crps": float(
            np.mean(emphasized_tail_score)
        ),
        "q05_q95_threshold_weighted_crps": float(
            np.mean(q05_q95_tail_score)
        ),
        "q01_q99_extreme_threshold_weighted_crps": float(
            np.mean(extreme_tail_score)
        ),
        "tail_threshold_contract_sha256": thresholds[
            "threshold_contract_sha256"
        ],
        "negative_tail_crps": negative_tail_crps["value"],
        "negative_tail_crps_status": negative_tail_crps["status"],
        "positive_tail_crps": positive_tail_crps["value"],
        "positive_tail_crps_status": positive_tail_crps["status"],
        "tail_calibration_error": float(
            np.mean(
                [
                    float(lower_calibration["calibration_error"]),
                    float(upper_calibration["calibration_error"]),
                ]
            )
        ),
        "awcrps": awcrps_statistics["value"],
        "awcrps_status": awcrps_statistics["status"],
        "awcrps_undefined_reason": awcrps_statistics["undefined_reason"],
        "feasibility_weighted_global_crps": weighted.value,
        "feasibility_weighted_global_crps_status": weighted.status,
        "feasibility_weighted_global_crps_undefined_reason": (
            feasibility_weighted_global_crps_statistics["undefined_reason"]
        ),
        "feasibility_conditioned_tail_crps": feasibility_tail_statistics["value"],
        "feasibility_conditioned_tail_crps_status": feasibility_tail_statistics[
            "status"
        ],
        "feasibility_conditioned_tail_crps_undefined_reason": (
            feasibility_tail_statistics["undefined_reason"]
        ),
        "feasibility_conditioned_q05_q95_tail_crps": (
            feasibility_q05_q95_statistics["value"]
        ),
        "feasibility_conditioned_q05_q95_tail_crps_status": (
            feasibility_q05_q95_statistics["status"]
        ),
        "feasibility_conditioned_q01_q99_extreme_crps": (
            feasibility_q01_q99_statistics["value"]
        ),
        "feasibility_conditioned_q01_q99_extreme_crps_status": (
            feasibility_q01_q99_statistics["status"]
        ),
        "economic_action_brier": action_statistics["economic_action_brier"]["value"],
        "economic_action_brier_status": action_statistics["economic_action_brier"]["status"],
        "action_precision": action_statistics["action_precision"]["value"],
        "action_precision_status": action_statistics["action_precision"]["status"],
        "action_recall": action_statistics["action_recall"]["value"],
        "action_recall_status": action_statistics["action_recall"]["status"],
        "action_f1": action_statistics["action_f1"]["value"],
        "action_f1_status": action_statistics["action_f1"]["status"],
        "wrong_action_loss": action_statistics["wrong_action_loss"]["value"],
        "wrong_action_loss_status": action_statistics["wrong_action_loss"]["status"],
        "missed_opportunity_loss": action_statistics["missed_opportunity_loss"]["value"],
        "missed_opportunity_loss_status": action_statistics["missed_opportunity_loss"]["status"],
        "wrong_action_rate": action_statistics["wrong_action_rate"]["value"],
        "wrong_action_rate_status": action_statistics["wrong_action_rate"]["status"],
        "missed_opportunity_rate": action_statistics["missed_opportunity_rate"]["value"],
        "missed_opportunity_rate_status": action_statistics[
            "missed_opportunity_rate"
        ]["status"],
        "wrong_action_economic_loss_usd": wrong_action_economic_loss_usd,
        "wrong_action_economic_loss_usd_status": "OK",
        "missed_opportunity_economic_loss_usd": (
            missed_opportunity_economic_loss_usd
        ),
        "missed_opportunity_economic_loss_usd_status": "OK",
        "action_confusion_sufficient_statistics": action_confusion,
        "spike_precision": spike_statistics["spike_precision"]["value"],
        "spike_precision_status": spike_statistics["spike_precision"]["status"],
        "spike_recall": spike_statistics["spike_recall"]["value"],
        "spike_recall_status": spike_statistics["spike_recall"]["status"],
        "spike_f1": spike_statistics["spike_f1"]["value"],
        "spike_f1_status": spike_statistics["spike_f1"]["status"],
        "spike_counts": spike_counts,
        "spike_pr_auc": average_precision["value"],
        "spike_pr_auc_status": average_precision["status"],
        "q99_event_brier": q99_brier["value"],
        "q99_event_brier_status": q99_brier["status"],
        "q99_precision": spike_statistics["spike_precision"]["value"],
        "q99_precision_status": spike_statistics["spike_precision"]["status"],
        "q99_recall": spike_statistics["spike_recall"]["value"],
        "q99_recall_status": spike_statistics["spike_recall"]["status"],
        "q99_f1": spike_statistics["spike_f1"]["value"],
        "q99_f1_status": spike_statistics["spike_f1"]["status"],
        "q99_miss_rate": q99_miss_rate["value"],
        "q99_miss_rate_status": q99_miss_rate["status"],
        "q99_pr_auc": average_precision["value"],
        "q99_pr_auc_status": average_precision["status"],
        "q99_probability_decision_threshold": float(spike_probability_threshold),
        "interval_coverage_50": intervals["50"]["coverage"],
        "interval_width_50": intervals["50"]["mean_width"],
        "interval_coverage_80": intervals["80"]["coverage"],
        "interval_width_80": intervals["80"]["mean_width"],
        "interval_calibration_error_80": intervals["80"][
            "absolute_calibration_error"
        ],
        "interval_coverage_90": intervals["90"]["coverage"],
        "interval_width_90": intervals["90"]["mean_width"],
        "interval_calibration_error_90": intervals["90"][
            "absolute_calibration_error"
        ],
        "interval_coverage_95": intervals["95"]["coverage"],
        "interval_width_95": intervals["95"]["mean_width"],
        "energy_score": float(energy),
        "variogram_score": float(variogram),
        "forl": forl.value,
        "forl_status": forl.status,
        "forl_undefined_reason": forl_statistics["undefined_reason"],
        "negative_exceedance_calibration": negative_calibration,
        "positive_exceedance_calibration": positive_calibration,
        "temporal_autocorrelation_error": temporal_acf["value"],
        "temporal_autocorrelation_error_status": temporal_acf["status"],
        "cross_zone_rank_correlation_error": cross_zone_rank["value"],
        "cross_zone_rank_correlation_error_status": cross_zone_rank["status"],
        "spike_duration_error": spike_duration["value"],
        "spike_duration_error_status": spike_duration["status"],
        "joint_spike_frequency_error": joint_spike["frequency_absolute_error"],
        "joint_spike_frequency_error_status": joint_spike["status"],
        "metric_sufficient_statistics": sufficient_statistics,
        "diagnostic_layers": {"layer1": layer1, "layer2": layer2},
        "diagnostic_artifact_path": diagnostic_path,
        "diagnostic_artifact_sha256": diagnostic_sha,
        "pit_values_sha256": _array_sha256(pit_values),
        "normalized_feasibility_weight_sha256": _array_sha256(normalized_f),
        "normalized_feasibility_weight_sum": float(np.sum(normalized_f)),
        "normalized_feasibility_weight_min": float(np.min(normalized_f)),
        "normalized_feasibility_weight_max": float(np.max(normalized_f)),
        "physical_flexible_mwh_sha256": _array_sha256(physical_mwh),
        "physical_flexible_mwh_total": float(np.sum(physical_mwh)),
        "_q99_event_labels": spike_actual.reshape(-1).astype(bool),
        "_q99_event_probabilities": spike_probability.reshape(-1).astype(float),
        "source_evidence_sha256": source_sha,
        "future_realized_inputs_used": False,
        "locked_test_outcome_used_for_gate_features": False,
        "counterfactual_scoring": True,
    }


_RATIO_AGGREGATION_FIELDS = (
    "awcrps",
    "feasibility_conditioned_tail_crps",
    "feasibility_conditioned_q05_q95_tail_crps",
    "feasibility_conditioned_q01_q99_extreme_crps",
    "forl",
    "economic_action_brier",
    "action_precision",
    "action_recall",
    "action_f1",
    "wrong_action_loss",
    "missed_opportunity_loss",
    "wrong_action_rate",
    "missed_opportunity_rate",
    "wrong_action_economic_loss_usd",
    "missed_opportunity_economic_loss_usd",
    "q99_event_brier",
    "q99_precision",
    "q99_recall",
    "q99_f1",
    "q99_miss_rate",
    "spike_precision",
    "spike_recall",
    "spike_f1",
)


def _aggregate_metric_sufficient_statistics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate ratios by day-seed sufficient stats, never mean ratios."""

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        candidate = str(record["candidate_id"])
        day = str(record["operating_date"])
        grouped.setdefault(candidate, {}).setdefault(day, []).append(record)
    candidates: dict[str, Any] = {}
    expected_seeds = set(DART_VALUE_SELECT_SEEDS)
    for candidate in sorted(grouped):
        days = grouped[candidate]
        candidate_metrics: dict[str, Any] = {}
        for metric_name in _RATIO_AGGREGATION_FIELDS:
            day_rows: list[dict[str, Any]] = []
            total_numerator = 0.0
            total_denominator = 0.0
            total_count = 0.0
            undefined_reasons: set[str] = set()
            for day in sorted(days):
                rows = days[day]
                observed_seeds = {int(row["seed"]) for row in rows}
                if observed_seeds != expected_seeds or len(rows) != len(expected_seeds):
                    raise DARTRealEvidenceError(
                        f"{candidate}/{day} lacks the exact three-seed metric panel"
                    )
                stats = [
                    row.get("metric_sufficient_statistics", {}).get(metric_name)
                    for row in rows
                ]
                if any(not isinstance(item, Mapping) for item in stats):
                    raise DARTRealEvidenceError(
                        f"{candidate}/{day} lacks sufficient statistics for {metric_name}"
                    )
                typed = [dict(item) for item in stats if isinstance(item, Mapping)]
                day_numerator = float(
                    np.mean([float(item["numerator"]) for item in typed])
                )
                day_denominator = float(
                    np.mean([float(item["denominator"]) for item in typed])
                )
                day_count = float(np.mean([float(item["count"]) for item in typed]))
                reasons = sorted(
                    {
                        str(item["undefined_reason"])
                        for item in typed
                        if item.get("undefined_reason")
                    }
                )
                undefined_reasons.update(reasons)
                day_rows.append(
                    {
                        "operating_date": day,
                        "seed_count": len(rows),
                        "seed_average_numerator": day_numerator,
                        "seed_average_denominator": day_denominator,
                        "seed_average_count": day_count,
                        "undefined_reasons": reasons,
                    }
                )
                total_numerator += day_numerator
                total_denominator += day_denominator
                total_count += day_count
            if metric_name == "spike_f1" and total_count == 0.0:
                aggregate = {
                    "value": None,
                    "status": "NO_POSITIVE_SPIKES",
                    "undefined_reason": "NO_POSITIVE_SPIKES",
                    "numerator": total_numerator,
                    "denominator": total_denominator,
                    "count": 0,
                }
            else:
                aggregate = _ratio_stat(
                    total_numerator,
                    total_denominator,
                    zero_reason=(
                        ";".join(sorted(undefined_reasons))
                        if undefined_reasons
                        else "ZERO_AGGREGATE_DENOMINATOR"
                    ),
                    count=int(round(total_count)),
                )
            candidate_metrics[metric_name] = {
                **aggregate,
                "per_day_seed_averaged_sufficient_statistics": day_rows,
            }
        pooled_labels: list[np.ndarray] = []
        pooled_probabilities: list[np.ndarray] = []
        pooled_day_rows: list[dict[str, Any]] = []
        probability_thresholds: set[float] = set()
        for day in sorted(days):
            rows = days[day]
            labels_by_seed = [
                np.asarray(row.get("_q99_event_labels"), dtype=bool).reshape(-1)
                for row in rows
            ]
            probabilities_by_seed = [
                np.asarray(
                    row.get("_q99_event_probabilities"), dtype=float
                ).reshape(-1)
                for row in rows
            ]
            if (
                any(values.size == 0 for values in labels_by_seed)
                or len({values.shape for values in labels_by_seed}) != 1
                or len({values.shape for values in probabilities_by_seed}) != 1
                or labels_by_seed[0].shape != probabilities_by_seed[0].shape
                or any(
                    not np.array_equal(values, labels_by_seed[0])
                    for values in labels_by_seed[1:]
                )
                or any(not np.all(np.isfinite(values)) for values in probabilities_by_seed)
            ):
                raise DARTRealEvidenceError(
                    f"{candidate}/{day} has an invalid pooled q99 event panel"
                )
            mean_probability = np.mean(
                np.stack(probabilities_by_seed, axis=0), axis=0
            )
            pooled_labels.append(labels_by_seed[0])
            pooled_probabilities.append(mean_probability)
            probability_thresholds.update(
                float(row["q99_probability_decision_threshold"]) for row in rows
            )
            pooled_day_rows.append(
                {
                    "operating_date": day,
                    "seed_count": len(rows),
                    "event_count": int(mean_probability.size),
                    "positive_count": int(np.count_nonzero(labels_by_seed[0])),
                    "seed_mean_probability_sha256": _array_sha256(mean_probability),
                }
            )
        if len(probability_thresholds) != 1:
            raise DARTRealEvidenceError(
                f"{candidate} changed the q99 action probability threshold"
            )
        pooled_truth = np.concatenate(pooled_labels)
        pooled_score = np.concatenate(pooled_probabilities)
        probability_threshold = next(iter(probability_thresholds))
        pooled_prediction = pooled_score >= probability_threshold
        pooled_classification, pooled_counts = _spike_sufficient_statistics(
            pooled_truth, pooled_prediction
        )
        pooled_brier = _ratio_stat(
            float(np.sum((pooled_score - pooled_truth.astype(float)) ** 2)),
            float(pooled_truth.size),
            zero_reason="NO_Q99_EVENT_ROWS",
            count=int(pooled_truth.size),
        )
        pooled_pr_auc = _average_precision(pooled_truth, pooled_score)
        pooled_miss = _ratio_stat(
            float(pooled_counts["false_negative"]),
            float(pooled_counts["actual_positive_count"]),
            zero_reason="NO_POSITIVE_Q99_EVENTS",
            count=int(pooled_counts["actual_positive_count"]),
        )
        candidate_metrics.update(
            {
                "q99_event_brier": pooled_brier,
                "q99_pr_auc": pooled_pr_auc,
                "q99_precision": pooled_classification["spike_precision"],
                "q99_recall": pooled_classification["spike_recall"],
                "q99_f1": pooled_classification["spike_f1"],
                "q99_miss_rate": pooled_miss,
                "q99_pooled_panel": {
                    "aggregation_order": (
                        "average_three_seed_event_probabilities_within_date;"
                        "pool_exact_date_event_panel;score_once"
                    ),
                    "probability_decision_threshold": probability_threshold,
                    "event_count": int(pooled_truth.size),
                    "positive_count": int(np.count_nonzero(pooled_truth)),
                    "event_panel_sha256": sha256_json(
                        {
                            "labels": _array_sha256(pooled_truth.astype(np.uint8)),
                            "probabilities": _array_sha256(pooled_score),
                        }
                    ),
                    "per_day_seed_average": pooled_day_rows,
                },
            }
        )
        candidates[candidate] = candidate_metrics
    contract = {
        "schema_version": SCHEMA_VERSION,
        "aggregation_order": (
            "average_numerator_and_denominator_across_three_seeds_within_day;"
            "sum_day_sufficient_statistics;divide_once"
        ),
        "mean_of_ratios_used": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    return {**contract, "aggregation_sha256": sha256_json(contract)}


def _regime_labels(
    *,
    actual: ReferencedNYISOReplay,
    forcing: FrozenReplayForcing,
    issue: FrozenForecastArtifact,
    thresholds: Mapping[str, Any],
) -> tuple[str, ...]:
    labels = ["ordinary"]
    dart = actual.dart_da_minus_rt
    lower = np.asarray(thresholds["dart_lower_q01_by_zone"], dtype=float)[:, None]
    upper = np.asarray(thresholds["dart_upper_q99_by_zone"], dtype=float)[:, None]
    if np.any(dart > upper):
        labels.append("positive_dart_spike")
    if np.any(dart < lower):
        labels.append("negative_dart_spike")
    workload_reference = np.mean(issue.workload_pool, axis=1)
    if float(np.mean(forcing.workload)) > float(np.quantile(workload_reference, 0.90)):
        labels.append("high_workload")
    pv_reference = np.mean(issue.pv_pool, axis=(1, 2))
    if float(np.mean(forcing.pv)) < float(np.quantile(pv_reference, 0.10)):
        labels.append("low_pv")
    congestion = np.ptp(actual.rt_lbmp, axis=0)
    if np.any(congestion > float(thresholds["congestion_upper"])):
        labels.append("high_congestion")
    return tuple(labels)


def _candidate_evidence(
    *,
    inventory: OOSInventory,
    dates: Sequence[str],
    issue: Mapping[str, FrozenForecastArtifact],
    settlement: Mapping[str, ReferencedNYISOReplay],
    forcing: Mapping[str, FrozenReplayForcing],
    actionability: Mapping[str, ActionabilityDay],
    thresholds: Mapping[int, Mapping[str, Any]],
    physical: FrozenPhysicalArtifact,
    config: Mapping[str, Any],
    repository_root: Path,
    candidate_specs: Sequence[Any] = CANDIDATE_POOL,
) -> dict[str, tuple[CandidateEvidence, ...]]:
    def common_origin_hash(field_name: str) -> str:
        by_origin: dict[str, set[str]] = {}
        for record in inventory.records:
            by_origin.setdefault(record.origin_name, set()).add(
                str(getattr(record, field_name))
            )
        if any(len(values) != 1 for values in by_origin.values()):
            raise DARTRealEvidenceError(
                f"candidate-independent {field_name} changed within an origin"
            )
        return sha256_json(
            {
                origin: next(iter(values))
                for origin, values in sorted(by_origin.items())
            }
        )

    issue_contract = sha256_json(
        [_forecast_fingerprint(issue[day]) for day in dates]
    )
    issue_time_panel = common_origin_hash("issue_time_panel_sha256")
    calibration_panel = common_origin_hash("calibration_panel_sha256")
    threshold_contract = sha256_json(
        {
            str(year): _valid_sha256(
                record.get("threshold_contract_sha256"),
                label=f"tail threshold contract {year}",
            )
            for year, record in sorted(thresholds.items())
        }
    )
    workload_forecast = sha256_json(
        [_array_sha256(issue[day].workload_pool) for day in dates]
    )
    pv_forecast = sha256_json([_array_sha256(issue[day].pv_pool) for day in dates])
    nyiso_replay = sha256_json([settlement[day].input_sha256 for day in dates])
    workload_replay = sha256_json([_array_sha256(forcing[day].workload) for day in dates])
    pv_replay = sha256_json([_array_sha256(forcing[day].pv) for day in dates])
    physical_replay = sha256_json(
        [_array_sha256(forcing[day].physical_multiplier) for day in dates]
    )
    twin = sha256_json(_physical_fingerprint(physical))
    user = sha256_json(
        {
            "identical_replay": config.get("identical_replay", {}),
            "feasibility": config.get("feasibility", {}),
            "actionability_sources": [actionability[day].source_sha256 for day in dates],
        }
    )
    optimizer = sha256_json(
        {
            "code": sha256_file(repository_root / "src/dcvpp/optimization/stochastic_milp.py"),
            "real_evidence_adapter_code": sha256_file(Path(__file__).resolve()),
            "runtime": config.get("optimization_runtime", {}),
            "policy": "B6",
        }
    )
    replay_engine = sha256_json(
        {
            "code": sha256_file(repository_root / "src/dcvpp/optimization/replay_engine.py"),
            "sign": "dart_da_minus_rt=da_lbmp-rt_lbmp",
        }
    )
    common = {
        "load_forecast_sha256": workload_forecast,
        "workload_forecast_sha256": workload_forecast,
        "pv_forecast_sha256": pv_forecast,
        "nyiso_replay_sha256": nyiso_replay,
        "workload_replay_sha256": workload_replay,
        "pv_replay_sha256": pv_replay,
        "physical_replay_sha256": physical_replay,
        "user_sha256": user,
        "twin_sha256": twin,
        "optimizer_sha256": optimizer,
        "replay_engine_sha256": replay_engine,
        "issue_time_panel_sha256": issue_time_panel,
        "calibration_panel_sha256": calibration_panel,
        "tail_threshold_contract_sha256": threshold_contract,
    }
    a1_common = {
        **common,
        "da_forecast_sha256": issue_contract,
        "calibration_sha256": sha256_json(
            {"rule": "candidate_dart_distribution_only", "quantiles": list(QUANTILES)}
        ),
        "dependence_sha256": sha256_json(
            {"rule": "common_parent_indices_and_fixed_non_price_blocks"}
        ),
        "scenario_builder_sha256": sha256_json(
            {"algorithm": "common_farthest_first_reduction", "count": 20}
        ),
    }
    output: dict[str, list[CandidateEvidence]] = {"A1": [], "A2": []}
    for candidate in candidate_specs:
        records = [record for record in inventory.records if record.candidate_name == candidate.name]
        artifact_hashes = sorted(record.artifact_sha256 for record in records)
        model_hashes = sorted(record.model_sha256 for record in records)
        calibration_hashes = sorted(record.calibration_sha256 for record in records)
        dependence_hashes = sorted(record.dependence_sha256 for record in records)
        da_hashes = sorted(record.da_prediction_sha256 for record in records)
        model_sha = sha256_json(model_hashes)
        dart_forecast_sha = sha256_json(artifact_hashes)
        a1_hashes = {
            **a1_common,
            "dart_model_sha256": model_sha,
            "dart_forecast_sha256": dart_forecast_sha,
            "rt_scenario_sha256": sha256_json(
                {"experiment": "A1", "dart": artifact_hashes, "common_issue": issue_contract}
            ),
        }
        output["A1"].append(CandidateEvidence(candidate.name, "A1", a1_hashes))
        a2_hashes = {
            **common,
            "dart_model_sha256": model_sha,
            "dart_forecast_sha256": dart_forecast_sha,
            "da_forecast_sha256": sha256_json(da_hashes),
            "calibration_sha256": sha256_json(calibration_hashes),
            "dependence_sha256": sha256_json(dependence_hashes),
            "scenario_builder_sha256": sha256_json(
                {
                    "algorithm": "candidate_distribution_common_parent_transform",
                    "candidate": candidate.name,
                    "dart": artifact_hashes,
                }
            ),
            "rt_scenario_sha256": sha256_json(
                {"experiment": "A2", "candidate_joint_draws": artifact_hashes}
            ),
        }
        output["A2"].append(CandidateEvidence(candidate.name, "A2", a2_hashes))
    normalized = {key: tuple(value) for key, value in output.items()}
    a1 = assert_experiment_identity(normalized["A1"])
    a2 = assert_experiment_identity(normalized["A2"])
    if set(a1["observed_candidate_varying_hashes"]).difference(A1_ALLOWED_TO_DIFFER):
        raise DARTRealEvidenceError("A1 evidence changed a forbidden block")
    if set(a2["observed_candidate_varying_hashes"]).difference(A2_ALLOWED_TO_DIFFER):
        raise DARTRealEvidenceError("A2 evidence changed a forbidden block")
    for key in CANDIDATE_INDEPENDENT_HASHES | _REAL_EVIDENCE_COMMON_HASHES:
        values = {
            candidate.hashes[key]
            for experiment in normalized.values()
            for candidate in experiment
        }
        if len(values) != 1:
            raise DARTRealEvidenceError(f"candidate-independent hash changed: {key}")
    return normalized


def _frozen_dependencies(stage_root: Path, phase: str) -> dict[str, str]:
    if phase != "locked_test":
        return {}
    freeze_path = stage_root / "selection_freeze_manifest.json"
    if not freeze_path.is_file():
        raise MissingRealEvidenceError(
            "locked evidence requires the pre-2025 selection freeze manifest"
        )
    freeze = _read_json(freeze_path)
    contract = freeze.get("frozen_contract")
    if not isinstance(contract, Mapping) or freeze.get("freeze_fingerprint") != sha256_json(contract):
        raise DARTRealEvidenceError("selection freeze manifest fingerprint changed")
    dependencies = contract.get("dependency_fingerprints")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(FREEZE_FINGERPRINT_KEYS):
        raise DARTRealEvidenceError("selection freeze lacks the exact dependency fingerprints")
    return {
        str(key): _valid_sha256(value, label=f"frozen dependency {key}")
        for key, value in dependencies.items()
    }


def _empty_gates(*, operating_date_count: int = 0, panel_cell_count: int = 0) -> dict[str, Any]:
    return {
        **{key: False for key in REQUIRED_GATE_KEYS},
        "operating_date_count": int(operating_date_count),
        "panel_cell_count": int(panel_cell_count),
        "v1_2025_metrics_consumed": False,
        "selected_scenario_used_as_realized": False,
    }


def _write_partial(
    *,
    output_root: Path,
    phase: str,
    request_fingerprint: str,
    error: Exception,
    gates: Mapping[str, Any],
    inventory: OOSInventory | None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{phase}_real_evidence_manifest.json"
    pending_count = int(getattr(error, "pending_count", 1))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PARTIAL",
        "phase": phase,
        "request_fingerprint": request_fingerprint,
        "paper_mode_fail_closed": True,
        "synthetic_fallback_used": False,
        "v1_2025_metrics_consumed": False,
        "evidence": None,
        "metric_aggregates_path": None,
        "diagnostic_availability_path": None,
        "source_inventory_sha256": None if inventory is None else inventory.fingerprint,
        "completion_gates": dict(gates),
        "actionable_pending_count": pending_count,
        "pending": [
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "action": "provide or repair the exact missing/hash-aligned real evidence and resume",
            }
        ],
        "updated_at_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return {
        "status": "PARTIAL",
        "evidence": None,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "metric_aggregates_path": None,
        "diagnostic_availability_path": None,
        "completion_gates": dict(gates),
        "actionable_pending_count": pending_count,
    }


def run_real_dart_phase_evidence(
    root: Path,
    dart_forecast_root: Path,
    physical_result: Mapping[str, Any] | FrozenPhysicalArtifact,
    forecast_result: Mapping[str, Any],
    config: Mapping[str, Any],
    phase: str,
    output_root: Path,
    resume: bool = True,
    *,
    first_stage_solver: Callable[..., Mapping[str, Any]] = solve_policy,
    recourse_solver: Callable[..., Mapping[str, Any]] = solve_policy,
    perfect_information_solver: Callable[..., Mapping[str, Any]] = solve_policy,
    component_cost_hook: Callable[[int, Mapping[str, Any]], Mapping[str, float]] | None = None,
    candidate_pool_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build exact real DART phase evidence and run/resume external replay.

    Parameters are deliberately CLI/orchestrator friendly: ``root`` is the
    repository root, ``dart_forecast_root`` is the phase training manifest's
    ``stage_root``, and ``output_root`` is a stable directory within the DART
    run.  The optional solver callbacks exist for focused tests and alternative
    installed solver backends; paper calls use the real repository solvers.

    ``forecast_result`` is consumed only for its frozen issue-time day index
    and the explicit ``dart_real_evidence_inputs`` block.  Legacy v1 2025
    metrics, selected winners, and score tables are never read.
    """

    repository_root = Path(root).resolve()
    stage_root = Path(dart_forecast_root).resolve()
    destination = Path(output_root).resolve()
    if phase not in {"selection", "locked_test"}:
        raise ValueError("phase must be selection or locked_test")
    if not isinstance(forecast_result, Mapping) or not isinstance(config, Mapping):
        raise TypeError("forecast_result and config must be mappings")
    normalized_pool_contract = (
        validate_partial_candidate_pool_contract(candidate_pool_contract)
        if candidate_pool_contract is not None
        else None
    )
    candidate_specs = candidate_specs_for_contract(normalized_pool_contract)
    physical = (
        physical_result
        if isinstance(physical_result, FrozenPhysicalArtifact)
        else FrozenPhysicalArtifact.from_mapping(physical_result)
    )
    inputs: dict[str, Any] = {}
    inventory: OOSInventory | None = None
    baseline_dependency_inventory: OOSInventory | None = None
    dates: tuple[str, ...] = ()
    request_fingerprint = sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "repository_root": repository_root.as_posix(),
            "dart_forecast_root": stage_root.as_posix(),
            "phase": phase,
            "physical": _physical_fingerprint(physical),
            "config": _fingerprint_value(config),
            "candidate_pool_contract": normalized_pool_contract,
            "forecast_locator": _fingerprint_value(
                {
                    key: forecast_result.get(key)
                    for key in (
                        "frozen_day_index",
                        "forecast_day_index",
                        "frozen_days_path",
                        "frozen_days",
                        "days",
                        INPUT_BLOCK,
                    )
                    if key in forecast_result
                }
            ),
        }
    )
    gates = _empty_gates()
    existing_manifest = destination / f"{phase}_real_evidence_manifest.json"
    if not resume and existing_manifest.exists():
        raise DARTRealEvidenceError(
            f"real-evidence output already exists and resume=False: {existing_manifest}"
        )
    try:
        if not stage_root.is_dir():
            raise MissingRealEvidenceError(f"DART forecast stage root is absent: {stage_root}")
        inputs = _input_block(config, forecast_result)
        real_contract = config.get("real_evidence_contract", {})
        if not isinstance(real_contract, Mapping):
            raise DARTRealEvidenceError("real_evidence_contract must be a mapping")
        inventory = _discover_oos_inventory(
            repository_root=repository_root,
            stage_root=stage_root,
            phase=phase,
            real_contract=real_contract,
            candidate_specs=candidate_specs,
        )
        dates = _selected_dates(inventory, phase=phase, real_contract=real_contract)
        baseline_source = str(
            config.get("decision_pipeline_contract", {})
            .get("preregistered_baseline", {})
            .get("forecast_source", "seasonal_naive")
        )
        candidate_names = {candidate.name for candidate in candidate_specs}
        if baseline_source in candidate_names:
            baseline_dependency_inventory = inventory
        else:
            try:
                baseline_spec = _CANDIDATE_BY_NAME[baseline_source]
            except KeyError as exc:
                raise DARTRealEvidenceError(
                    f"preregistered baseline source is unknown: {baseline_source}"
                ) from exc
            baseline_dependency_inventory = _discover_oos_inventory(
                repository_root=repository_root,
                stage_root=stage_root,
                phase=phase,
                real_contract=real_contract,
                candidate_specs=(baseline_spec,),
            )
            if set(dates).difference(baseline_dependency_inventory.dates):
                raise MissingRealEvidenceError(
                    "shared baseline dependency lacks one or more replay dates"
                )
        gates = _empty_gates(
            operating_date_count=len(dates),
            panel_cell_count=len(dates) * len(candidate_specs) * 2 * len(DART_VALUE_SELECT_SEEDS),
        )
        gates["no_future_or_locked_leakage"] = bool(
            inventory.all_leakage_gates_passed
            and baseline_dependency_inventory.all_leakage_gates_passed
        )
        issue_store = _issue_store(
            repository_root=repository_root,
            forecast_result=forecast_result,
            inputs=inputs,
        )
        settlement_records = _expand_day_records(
            inputs.get("nyiso_replay_by_date"),
            root=repository_root,
            label="nyiso_replay_by_date",
        )
        forcing_records = _expand_day_records(
            inputs.get("replay_forcing_by_date"),
            root=repository_root,
            label="replay_forcing_by_date",
        )
        action_records = _expand_day_records(
            inputs.get("actionability_by_date"),
            root=repository_root,
            label="actionability_by_date",
        )
        nyiso_data_root_raw = inputs.get("referenced_nyiso_data_root")
        nyiso_data_root = (
            None
            if nyiso_data_root_raw is None
            else _resolve_path(nyiso_data_root_raw, root=repository_root)
        )
        settlement_store = _ExternalInputStore(
            repository_root=repository_root,
            phase=phase,
            records=settlement_records,
            nyiso_data_root=nyiso_data_root,
        )
        forcing_store = _ForcingStore(
            repository_root=repository_root, records=forcing_records
        )
        action_store = _ActionabilityStore(
            repository_root=repository_root, records=action_records
        )
        issue, settlement, forcing, actionability, thresholds = _preflight_inputs(
            dates,
            inventory=inventory,
            issue_store=issue_store,
            settlement_store=settlement_store,
            forcing_store=forcing_store,
            action_store=action_store,
            threshold_records=dict(inputs.get("training_thresholds_by_year", {})),
        )
        shards = _write_daily_shards(
            inventory,
            dates=dates,
            output_root=destination,
            candidate_specs=candidate_specs,
        )
        baseline_spec = _CANDIDATE_BY_NAME[baseline_source]
        if baseline_source not in candidate_names:
            dependency_shards = _write_daily_shards(
                baseline_dependency_inventory,
                dates=dates,
                output_root=destination / "shared_baseline_dependency_only",
                candidate_specs=(baseline_spec,),
            )
            overlap = set(shards).intersection(dependency_shards)
            if overlap:
                raise DARTRealEvidenceError(
                    "shared baseline dependency overlaps evaluated candidate shards"
                )
            shards.update(dependency_shards)
        candidates = _candidate_evidence(
            inventory=inventory,
            dates=dates,
            issue=issue,
            settlement=settlement,
            forcing=forcing,
            actionability=actionability,
            thresholds=thresholds,
            physical=physical,
            config=config,
            repository_root=repository_root,
            candidate_specs=candidate_specs,
        )
        if baseline_source in candidate_names:
            baseline_candidates = candidates
        else:
            baseline_candidates = _candidate_evidence(
                inventory=baseline_dependency_inventory,
                dates=dates,
                issue=issue,
                settlement=settlement,
                forcing=forcing,
                actionability=actionability,
                thresholds=thresholds,
                physical=physical,
                config=config,
                repository_root=repository_root,
                candidate_specs=(baseline_spec,),
            )
            for experiment in ("A1", "A2"):
                evaluated_common = candidates[experiment][0].hashes
                baseline_common = baseline_candidates[experiment][0].hashes
                changed = sorted(
                    key
                    for key in CANDIDATE_INDEPENDENT_HASHES
                    | _REAL_EVIDENCE_COMMON_HASHES
                    if evaluated_common[key] != baseline_common[key]
                )
                if changed:
                    raise DARTRealEvidenceError(
                        "shared baseline dependency changed candidate-independent "
                        f"evidence: {changed}"
                    )
        a1_audit = assert_experiment_identity(candidates["A1"])
        a2_audit = assert_experiment_identity(candidates["A2"])
        gates["a1_identity_exact"] = bool(a1_audit["passed"])
        gates["a2_allowed_changes_exact"] = bool(a2_audit["passed"])
        panel = build_common_replay_panel(dates, candidates)
        if panel["cell_count"] != gates["panel_cell_count"]:
            raise DARTRealEvidenceError("replay panel count changed after preflight")
        evidence_by_experiment = {
            experiment: {item.candidate_id: item for item in values}
            for experiment, values in candidates.items()
        }
        reduced_count = int(
            config.get("decision_pipeline_contract", {}).get("reduced_scenario_count", 20)
        )
        if reduced_count != 20:
            raise DARTRealEvidenceError("primary DART replay requires exactly 20 scenarios")
        runtime_config = config.get("optimization_runtime", {})
        if not isinstance(runtime_config, Mapping):
            raise DARTRealEvidenceError("optimization_runtime must be a mapping")
        runtime_config = dict(runtime_config)
        daily_cache: dict[tuple[str, str, int], DailyDART] = {}
        metric_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

        def daily_for(day: str, candidate_id: str, seed: int) -> tuple[DailyDART, OOSArtifactRecord]:
            key = (day, candidate_id, seed)
            year = date.fromisoformat(day).year
            slot = _slot_for_seed(candidate_id, seed)
            record_inventory = (
                inventory
                if candidate_id in candidate_names
                else baseline_dependency_inventory
            )
            record = record_inventory.records_by_key[(year, candidate_id, slot)]
            if key not in daily_cache:
                daily_cache[key] = _load_daily_dart(shards[(year, candidate_id, slot, day)])
            return daily_cache[key], record

        def bundle_for(day: str, candidate_id: str, experiment_id: str, seed: int) -> tuple[ScenarioBundle, DailyDART, OOSArtifactRecord]:
            daily, record = daily_for(day, candidate_id, seed)
            base = _base_issue_bundle(
                physical,
                issue[day],
                forcing[day],
                seed=seed,
                reduced_count=reduced_count,
            )
            return (
                _candidate_issue_bundle(base, daily, experiment_id=experiment_id),
                daily,
                record,
            )

        def metric_for(day: str, candidate_id: str, seed: int) -> dict[str, Any]:
            key = (day, candidate_id, seed)
            if key not in metric_cache:
                daily, _record = daily_for(day, candidate_id, seed)
                metric_cache[key] = _metric_record(
                    day=day,
                    candidate_id=candidate_id,
                    seed=seed,
                    daily=daily,
                    settlement=settlement[day],
                    actionability=actionability[day],
                    thresholds=thresholds[date.fromisoformat(day).year],
                    spike_probability_threshold=float(
                        real_contract.get("spike_probability_threshold", 0.50)
                    ),
                    diagnostics_root=destination / "metric_diagnostics",
                )
            return metric_cache[key]

        def candidate_runner(cell: ReplayCell) -> Mapping[str, Any]:
            bundle, daily, record = bundle_for(
                cell.operating_date, cell.candidate_id, cell.experiment_id, cell.seed
            )
            result = _run_external_cell(
                bundle=bundle,
                physical=physical,
                settlement=settlement[cell.operating_date],
                forcing=forcing[cell.operating_date],
                candidate_id=cell.candidate_id,
                experiment_id=cell.experiment_id,
                seed=cell.seed,
                evidence_hashes=cell.evidence_hashes,
                config=runtime_config,
                checkpoint_root=(
                    destination
                    / "external"
                    / cell.operating_date
                    / cell.candidate_id
                    / cell.experiment_id
                    / f"seed_{cell.seed}"
                ),
                first_stage_solver=first_stage_solver,
                recourse_solver=recourse_solver,
                component_cost_hook=component_cost_hook,
                actionability=actionability[cell.operating_date],
                model_complexity=float(record.model_byte_count),
                forecast_time_seconds=daily.load_seconds,
            )
            metric = metric_for(cell.operating_date, cell.candidate_id, cell.seed)
            result["accuracy_metrics"] = {
                "mae": metric["mae"],
                "rmse": metric["rmse"],
                "crps": metric["crps"],
            }
            return result

        baseline_id = str(
            config.get("decision_pipeline_contract", {})
            .get("preregistered_baseline", {})
            .get("baseline_id", "DART_BASELINE_B6_SEASONAL_NAIVE_V1")
        )

        def baseline_runner(day: str, _experiment: str, seed: int) -> Mapping[str, Any]:
            bundle, daily, record = bundle_for(day, baseline_source, "A2", seed)
            shared = {
                item.candidate_id: item
                for item in baseline_candidates["A2"]
            }[baseline_source].hashes
            result = _run_external_cell(
                bundle=bundle,
                physical=physical,
                settlement=settlement[day],
                forcing=forcing[day],
                candidate_id=baseline_id,
                experiment_id="BASELINE",
                seed=seed,
                evidence_hashes=shared,
                config=runtime_config,
                checkpoint_root=destination / "external_baseline" / day / f"seed_{seed}",
                first_stage_solver=first_stage_solver,
                recourse_solver=recourse_solver,
                component_cost_hook=component_cost_hook,
                actionability=None,
                model_complexity=float(record.model_byte_count),
                forecast_time_seconds=daily.load_seconds,
            )
            # run_dart_replay_panel and downstream identity tests compare the
            # complete mapping across A1/A2, so do not embed experiment_id.
            return result

        def pi_runner(day: str) -> Mapping[str, Any]:
            base = _base_issue_bundle(
                physical,
                issue[day],
                forcing[day],
                seed=DART_VALUE_SELECT_SEEDS[0],
                reduced_count=reduced_count,
            )
            oracle = _perfect_information_bundle(base, settlement[day], forcing[day])
            solved = solve_perfect_information_bound(
                oracle,
                physical,
                config=runtime_config,
                solver=perfect_information_solver,
            )
            solver_audit = _solver_audit(
                solved["result"], stage="perfect_information"
            )
            pi_metrics_valid = not bool(
                solver_audit["rejected_under_bounded_gap_rule"]
            )
            if pi_metrics_valid:
                pi_invalid_reason = None
            elif solver_audit["mip_gap_reported"] is not True:
                pi_invalid_reason = "MISSING_PI_MIP_GAP"
            elif solver_audit["has_incumbent"] is not True:
                pi_invalid_reason = "PI_SOLVER_NO_INCUMBENT"
            elif solver_audit["gap_over_cap"]:
                pi_invalid_reason = "PI_SOLVER_GAP_OVER_CAP"
            elif solver_audit["termination_infeasible"]:
                pi_invalid_reason = "PI_SOLVER_NONFEASIBLE_TERMINATION"
            else:
                pi_invalid_reason = "PI_SOLVER_INFEASIBLE_UNDER_FROZEN_RULE"
            return {
                "simulated_total_cost_usd": float(
                    solver_audit["solver_objective_lower_bound"]
                    if solver_audit["solver_objective_lower_bound"] is not None
                    else solved["simulated_cost_lower_bound_usd"]
                ),
                "policy_id": "B8",
                "operational_value_upper_bound": True,
                "deployable": False,
                "solver_audit": solver_audit,
                "pi_metrics_valid": pi_metrics_valid,
                "pi_metrics_invalid_reason": pi_invalid_reason,
                "source_evidence_sha256": sha256_json(
                    {
                        "settlement": settlement[day].input_sha256,
                        "forcing": forcing[day].input_sha256,
                    }
                ),
            }

        replay = run_dart_replay_panel(
            panel,
            destination / "panel_checkpoints",
            candidate_runner=candidate_runner,
            baseline_runner=baseline_runner,
            perfect_information_runner=pi_runner,
        )
        if replay.get("status") != "COMPLETE" or replay.get("cell_count") != panel["cell_count"]:
            raise DARTRealEvidenceError("common replay panel is incomplete")
        metric_records: list[dict[str, Any]] = []
        for day in dates:
            for candidate in candidate_specs:
                for seed in DART_VALUE_SELECT_SEEDS:
                    metric_records.append(metric_for(day, candidate.name, seed))
        metric_aggregates = _aggregate_metric_sufficient_statistics(metric_records)
        for record in metric_records:
            candidate_metrics = metric_aggregates["candidates"][
                str(record["candidate_id"])
            ]
            for field_name in (
                "q99_event_brier",
                "q99_pr_auc",
                "q99_precision",
                "q99_recall",
                "q99_f1",
                "q99_miss_rate",
            ):
                aggregate = candidate_metrics[field_name]
                record[field_name] = aggregate["value"]
                record[f"{field_name}_status"] = aggregate["status"]
                record[f"{field_name}_undefined_reason"] = aggregate.get(
                    "undefined_reason"
                )
                if {
                    "numerator",
                    "denominator",
                    "count",
                }.issubset(aggregate):
                    record["metric_sufficient_statistics"][field_name] = {
                        key: aggregate.get(key)
                        for key in (
                            "value",
                            "status",
                            "undefined_reason",
                            "numerator",
                            "denominator",
                            "count",
                        )
                    }
                    record["metric_sufficient_statistics"][field_name][
                        "aggregation_scope"
                    ] = "candidate_pooled_date_event_panel_after_seed_probability_mean"
            # Preserve historical selector aliases while ensuring every q99
            # diagnostic comes from the exact same pooled date-event panel.
            for alias_name, q99_name in (
                ("spike_precision", "q99_precision"),
                ("spike_recall", "q99_recall"),
                ("spike_f1", "q99_f1"),
                ("spike_pr_auc", "q99_pr_auc"),
            ):
                record[alias_name] = record[q99_name]
                record[f"{alias_name}_status"] = record[f"{q99_name}_status"]
                record[f"{alias_name}_undefined_reason"] = record[
                    f"{q99_name}_undefined_reason"
                ]
                pooled_statistics = record["metric_sufficient_statistics"].get(
                    q99_name
                )
                if isinstance(pooled_statistics, Mapping):
                    record["metric_sufficient_statistics"][alias_name] = dict(
                        pooled_statistics
                    )
            record.pop("_q99_event_labels", None)
            record.pop("_q99_event_probabilities", None)
        metric_aggregates_path = destination / f"{phase}_metric_aggregates.json"
        atomic_json(metric_aggregates_path, metric_aggregates)
        unavailable_diagnostics: list[dict[str, Any]] = []
        availability_fields = (
            "awcrps",
            "feasibility_conditioned_tail_crps",
            "forl",
            "economic_action_brier",
            "action_precision",
            "action_recall",
            "action_f1",
            "wrong_action_loss",
            "missed_opportunity_loss",
            "wrong_action_rate",
            "missed_opportunity_rate",
            "wrong_action_economic_loss_usd",
            "missed_opportunity_economic_loss_usd",
            "q99_event_brier",
            "q99_pr_auc",
            "q99_precision",
            "q99_recall",
            "q99_f1",
            "q99_miss_rate",
            "spike_precision",
            "spike_recall",
            "spike_f1",
            "spike_pr_auc",
            "temporal_autocorrelation_error",
            "cross_zone_rank_correlation_error",
            "spike_duration_error",
            "joint_spike_frequency_error",
        )
        for record in metric_records:
            for field_name in availability_fields:
                if record.get(field_name) is None:
                    unavailable_diagnostics.append(
                        {
                            "operating_date": record["operating_date"],
                            "candidate_id": record["candidate_id"],
                            "seed": record["seed"],
                            "metric": field_name,
                            "reason": record.get(f"{field_name}_status")
                            or "UNAVAILABLE_FROM_SUPPLIED_EVIDENCE",
                        }
                    )
        diagnostic_availability = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE_WITH_EXPLICIT_NULLS"
            if unavailable_diagnostics
            else "COMPLETE",
            "invented_or_zero_imputed_metrics": False,
            "unavailable_record_count": len(unavailable_diagnostics),
            "unavailable_records": unavailable_diagnostics,
        }
        diagnostic_availability_path = (
            destination / f"{phase}_diagnostic_availability.json"
        )
        atomic_json(diagnostic_availability_path, diagnostic_availability)
        regimes = {
            day: _regime_labels(
                actual=settlement[day],
                forcing=forcing[day],
                issue=issue[day],
                thresholds=thresholds[date.fromisoformat(day).year],
            )
            for day in dates
        }
        dependencies = _frozen_dependencies(stage_root, phase)
        source_evidence_sha = sha256_json(
            {
                "inventory": inventory.fingerprint,
                "panel": panel["panel_sha256"],
                "replay": replay["results_sha256"],
                "metrics": sha256_json(metric_records),
                "metric_aggregates": metric_aggregates["aggregation_sha256"],
                "diagnostic_availability": sha256_file(diagnostic_availability_path),
                "regimes": regimes,
                "thresholds": thresholds,
                "v1_2025_metrics_consumed": False,
            }
        )
        evidence = DARTPhaseEvidence(
            phase=phase,  # type: ignore[arg-type]
            operating_dates=tuple(dates),
            candidates_by_experiment=candidates,
            replay_results=tuple(replay["results"]),
            metric_records=tuple(metric_records),
            regime_labels_by_date=regimes,
            source_evidence_sha256=source_evidence_sha,
            frozen_dependency_fingerprints=dependencies,
            contextual_selector=None,
            candidate_pool_contract=normalized_pool_contract,
            fixture=False,
        )
        gates.update(
            {
                "external_historical_nyiso_replay": all(
                    row["candidate_result"].get("external_historical_nyiso_replay") is True
                    for row in replay["results"]
                ),
                "first_stage_frozen_before_replay": all(
                    row["candidate_result"].get("first_stage_frozen_before_replay") is True
                    for row in replay["results"]
                ),
                "causal_recourse_no_future_reveal": all(
                    row["candidate_result"].get("causal_recourse_no_future_reveal") is True
                    for row in replay["results"]
                ),
                "counterfactual_labels": all(
                    row["decision_value"].get("counterfactual") is True
                    and row["decision_value"].get("realized_commercial_profit_claim") is False
                    for row in replay["results"]
                ),
                "external_replay_causal_provenance_complete": all(
                    _causal_provenance_complete(row["candidate_result"])
                    and _causal_provenance_complete(row["baseline_result"])
                    for row in replay["results"]
                ),
                "bounded_gap_solver_feasibility_rule_applied": all(
                    _bounded_gap_rule_applied(row["candidate_result"])
                    and _bounded_gap_rule_applied(row["baseline_result"])
                    for row in replay["results"]
                ),
                "ratio_of_sums_metric_aggregation_complete": bool(
                    metric_aggregates.get("mean_of_ratios_used") is False
                    and metric_aggregates.get("candidate_count")
                    == len(candidate_specs)
                    and metric_aggregates_path.is_file()
                ),
                "layer1_layer2_diagnostics_emitted": all(
                    _metric_diagnostic_complete(row)
                    for row in metric_records
                )
                and diagnostic_availability_path.is_file(),
                "three_layers_complete": True,
            }
        )
        if not all(bool(gates[key]) for key in REQUIRED_GATE_KEYS):
            raise DARTRealEvidenceError(
                "one or more mandatory real-evidence completion gates failed"
            )
        evidence_path = destination / f"{phase}_phase_evidence.json"
        atomic_json(evidence_path, evidence.to_dict())
        manifest_path = destination / f"{phase}_real_evidence_manifest.json"
        inventory_path = destination / f"{phase}_oos_inventory.json"
        atomic_json(
            inventory_path,
            {
                "schema_version": SCHEMA_VERSION,
                "phase": phase,
                "status": "COMPLETE",
                "records": [
                    record.to_dict()
                    for record in sorted(inventory.records, key=lambda item: item.key)
                ],
                "inventory_sha256": inventory.fingerprint,
            },
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "phase": phase,
            "request_fingerprint": request_fingerprint,
            "paper_mode_fail_closed": True,
            "synthetic_fallback_used": False,
            "v1_2025_metrics_consumed": False,
            "candidate_pool_contract": normalized_pool_contract,
            "terminal_status_cap": (
                normalized_pool_contract.get("terminal_status_cap")
                if normalized_pool_contract is not None
                else None
            ),
            "source_inventory_path": inventory_path.as_posix(),
            "source_inventory_sha256": inventory.fingerprint,
            "evidence_path": evidence_path.as_posix(),
            "evidence_file_sha256": sha256_file(evidence_path),
            "evidence_fingerprint": evidence.fingerprint,
            "frozen_dependency_fingerprints": dependencies,
            "completion_gates": gates,
            "actionable_pending_count": 0,
            "replay_results_sha256": replay["results_sha256"],
            "metric_records_sha256": sha256_json(metric_records),
            "metric_aggregates_path": metric_aggregates_path.as_posix(),
            "metric_aggregates_file_sha256": sha256_file(metric_aggregates_path),
            "metric_aggregates_sha256": metric_aggregates["aggregation_sha256"],
            "diagnostic_availability_path": diagnostic_availability_path.as_posix(),
            "diagnostic_availability_sha256": sha256_file(
                diagnostic_availability_path
            ),
            "unsupported_or_undefined_metric_record_count": len(
                unavailable_diagnostics
            ),
            "updated_at_utc": utc_now(),
        }
        atomic_json(manifest_path, manifest)
        return {
            "status": "COMPLETE",
            "evidence": evidence,
            "candidate_pool_contract": normalized_pool_contract,
            "terminal_status_cap": (
                normalized_pool_contract.get("terminal_status_cap")
                if normalized_pool_contract is not None
                else None
            ),
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "metric_aggregates_path": metric_aggregates_path.as_posix(),
            "diagnostic_availability_path": diagnostic_availability_path.as_posix(),
            "completion_gates": gates,
            "actionable_pending_count": 0,
        }
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _write_partial(
            output_root=destination,
            phase=phase,
            request_fingerprint=request_fingerprint,
            error=exc,
            gates=gates,
            inventory=inventory,
        )


__all__ = [
    "DARTRealEvidenceError",
    "MissingRealEvidenceError",
    "SCHEMA_VERSION",
    "run_real_dart_phase_evidence",
]
