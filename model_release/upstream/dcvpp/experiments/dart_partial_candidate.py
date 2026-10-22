"""Strict evidence contract for the one admissible DART candidate failure.

This is deliberately not a general partial-run escape hatch.  The only
scientifically admissible reduced comparison is the preregistered nine-model
attempt in which the three 2022 RT-at-DA-gate iTransformer seed units all end
with ``NO_HEALTHY_EPOCH``.  All dates and every unaffected OOS unit must remain
present.  A run satisfying this contract may continue through replay and
optimization, but it is permanently capped at ``PARTIAL_PAPER``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_forecast_oos import (
    CANDIDATE_POOL,
    FREEZE_FINGERPRINT_KEYS,
    LockedTestContractError,
)
from dcvpp.experiments.dart_replay import DART_VALUE_SELECT_SEEDS


PARTIAL_CANDIDATE_STATUS = "PARTIAL_WITH_FAILED_CANDIDATE"
PARTIAL_CANDIDATE_TERMINAL_CAP = "PARTIAL_PAPER"
EXCLUDED_CANDIDATE = "itransformer"
EXCLUDED_ORIGIN = "through_2021_to_2022_oos"
EXCLUDED_FORECAST_YEAR = 2022
EXCLUDED_TARGET = "rt_lbmp_at_da_gate"
EXCLUDED_REASON = "NO_HEALTHY_EPOCH"
ATTEMPTED_CANDIDATES = tuple(candidate.name for candidate in CANDIDATE_POOL)
ELIGIBLE_CANDIDATES = tuple(
    candidate for candidate in ATTEMPTED_CANDIDATES if candidate != EXCLUDED_CANDIDATE
)
EXPECTED_FAILED_SEEDS = tuple(int(seed) for seed in DART_VALUE_SELECT_SEEDS)
EXPECTED_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
SELECTION_EXECUTION_UNIT_COUNT = 123
SELECTION_ELIGIBLE_UNIT_COUNT = 108
LOCKED_ELIGIBLE_UNIT_COUNT = 36
LOCKED_ELIGIBLE_NEURAL_GROUP_COUNT = 10
LOCKED_SCHEDULER_DIRECTORY = "locked_checkpoint_scheduler"


def _with_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(core)
    normalized["contract_sha256"] = sha256_json(normalized)
    return normalized


def _normalized_dependencies(values: Mapping[str, str]) -> dict[str, str]:
    normalized = {str(key): str(value).lower() for key, value in values.items()}
    if set(normalized) != set(FREEZE_FINGERPRINT_KEYS) or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in normalized.values()
    ):
        raise ValueError(
            "freeze requires the exact dependency keys and SHA-256 values"
        )
    return dict(sorted(normalized.items()))


def build_partial_candidate_pool_contract() -> dict[str, Any]:
    """Build the sole admissible reduced-pool contract and pin its content."""

    return _with_sha256(
        {
            "schema_version": "dcvpp.dart_partial_candidate.v1",
            "status": PARTIAL_CANDIDATE_STATUS,
            "attempted_candidate_count": len(ATTEMPTED_CANDIDATES),
            "attempted_candidates": list(ATTEMPTED_CANDIDATES),
            "eligible_candidate_count": len(ELIGIBLE_CANDIDATES),
            "eligible_candidates": list(ELIGIBLE_CANDIDATES),
            "globally_excluded_candidates": [
                {
                    "candidate_name": EXCLUDED_CANDIDATE,
                    "reason_code": EXCLUDED_REASON,
                    "failed_group_count": 1,
                    "failed_unit_count": len(EXPECTED_FAILED_SEEDS),
                }
            ],
            "failed_unit_count": len(EXPECTED_FAILED_SEEDS),
            "date_drop_count": 0,
            "common_date_panel_preserved": True,
            "health_or_calibration_gate_relaxed": False,
            "synthetic_fallback_used": False,
            "selection_results_or_thresholds_changed_after_failure": False,
            "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
        }
    )


def _as_sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return tuple(value)


def validate_partial_candidate_pool_contract(
    value: Mapping[str, Any] | None,
    *,
    failed_units: Sequence[Mapping[str, Any]] | None = None,
    require_failed_units: bool = False,
) -> dict[str, Any]:
    """Return a normalized strict contract or raise ``ValueError``.

    ``failed_units`` is mandatory at the pipeline admission boundary.  It is
    optional in downstream artifacts because those artifacts pin the already
    admitted contract hash and the pipeline manifest pins the per-seed ledger.
    """

    if not isinstance(value, Mapping):
        raise ValueError("candidate_pool_contract is absent")
    contract = dict(value)
    declared_sha = contract.get("contract_sha256")
    core = {key: item for key, item in contract.items() if key != "contract_sha256"}
    actual_sha = sha256_json(core)
    if declared_sha != actual_sha:
        raise ValueError("candidate_pool_contract hash is invalid")

    attempted = tuple(
        map(str, _as_sequence(contract.get("attempted_candidates"), field="attempted_candidates"))
    )
    eligible = tuple(
        map(str, _as_sequence(contract.get("eligible_candidates"), field="eligible_candidates"))
    )
    excluded = _as_sequence(
        contract.get("globally_excluded_candidates"),
        field="globally_excluded_candidates",
    )
    if attempted != ATTEMPTED_CANDIDATES:
        raise ValueError("attempted candidate pool is not the frozen nine-model pool")
    if eligible != ELIGIBLE_CANDIDATES:
        raise ValueError("eligible candidate pool is not the exact eight-model pool")
    if int(contract.get("attempted_candidate_count", -1)) != 9:
        raise ValueError("attempted_candidate_count must be 9")
    if int(contract.get("eligible_candidate_count", -1)) != 8:
        raise ValueError("eligible_candidate_count must be 8")
    if int(contract.get("failed_unit_count", -1)) != 3:
        raise ValueError("failed_unit_count must be 3")
    if int(contract.get("date_drop_count", -1)) != 0:
        raise ValueError("date_drop_count must be zero")
    if contract.get("common_date_panel_preserved") is not True:
        raise ValueError("common_date_panel_preserved must be true")
    if len(excluded) != 1 or not isinstance(excluded[0], Mapping):
        raise ValueError("exactly one globally excluded candidate is required")
    exclusion = dict(excluded[0])
    if (
        exclusion.get("candidate_name") != EXCLUDED_CANDIDATE
        or exclusion.get("reason_code") != EXCLUDED_REASON
        or int(exclusion.get("failed_group_count", -1)) != 1
        or int(exclusion.get("failed_unit_count", -1)) != 3
    ):
        raise ValueError("global exclusion is not the unique iTransformer failure")
    if contract != build_partial_candidate_pool_contract():
        raise ValueError("candidate_pool_contract differs from the sole admissible contract")

    normalized_units: list[dict[str, Any]] = []
    if failed_units is not None:
        for raw in _as_sequence(failed_units, field="failed_units"):
            if not isinstance(raw, Mapping):
                raise ValueError("failed_units contains a malformed record")
            try:
                normalized = {
                    "origin": str(raw.get("origin")),
                    "forecast_year": int(raw.get("forecast_year")),
                    "target": str(raw.get("target")),
                    "candidate": str(raw.get("candidate")),
                    "seed": int(raw.get("seed")),
                    "reason_code": str(raw.get("reason_code")),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError("failed_units contains a malformed identity") from exc
            normalized_units.append(normalized)
        expected_identities = {
            (
                EXCLUDED_ORIGIN,
                EXCLUDED_FORECAST_YEAR,
                EXCLUDED_TARGET,
                EXCLUDED_CANDIDATE,
                seed,
                EXCLUDED_REASON,
            )
            for seed in EXPECTED_FAILED_SEEDS
        }
        observed_identities = {
            (
                item["origin"],
                item["forecast_year"],
                item["target"],
                item["candidate"],
                item["seed"],
                item["reason_code"],
            )
            for item in normalized_units
        }
        if len(normalized_units) != 3 or observed_identities != expected_identities:
            raise ValueError(
                "failed units are not exactly the three 2022 RT iTransformer seeds"
            )
    elif require_failed_units:
        raise ValueError("the three failed-unit records are required")

    return {**core, "contract_sha256": actual_sha}


def candidate_specs_for_contract(
    value: Mapping[str, Any] | None,
) -> tuple[Any, ...]:
    """Resolve the effective candidate specs; ``None`` retains all nine."""

    if value is None:
        return tuple(CANDIDATE_POOL)
    normalized = validate_partial_candidate_pool_contract(value)
    eligible = set(map(str, normalized["eligible_candidates"]))
    return tuple(candidate for candidate in CANDIDATE_POOL if candidate.name in eligible)


def failed_units_from_scheduler_manifest(
    scheduler_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Map the one failed scheduler group to its three frozen seed units."""

    if scheduler_manifest.get("status") != "COMPLETE":
        raise ValueError("scheduler lifecycle is not complete")
    if scheduler_manifest.get("scheduler_scientific_status") != (
        "COMPLETE_WITH_FAILED_GROUPS"
    ):
        raise ValueError("scheduler does not declare isolated failed groups")
    if (
        int(scheduler_manifest.get("group_count", -1)) != 36
        or int(scheduler_manifest.get("completed_group_count", -1)) != 35
        or int(scheduler_manifest.get("failed_group_count", -1)) != 1
        or int(scheduler_manifest.get("resolved_group_count", -1)) != 36
    ):
        raise ValueError("scheduler group counts do not prove 35 complete plus one failed")
    manifest_sha = scheduler_manifest.get("manifest_sha256")
    core = {
        key: item
        for key, item in scheduler_manifest.items()
        if key != "manifest_sha256"
    }
    if manifest_sha != sha256_json(core):
        raise ValueError("scheduler manifest hash is invalid")
    failed_groups = scheduler_manifest.get("failed_groups")
    if not isinstance(failed_groups, Mapping) or len(failed_groups) != 1:
        raise ValueError("scheduler must preserve exactly one failed group")
    group_key, raw = next(iter(failed_groups.items()))
    if not isinstance(raw, Mapping):
        raise ValueError("scheduler failed-group record is malformed")
    group = dict(raw)
    if (
        group.get("group_fingerprint") != group_key
        or group.get("task") != EXCLUDED_TARGET
        or group.get("model_name") != EXCLUDED_CANDIDATE
        or group.get("failure_status") != EXCLUDED_REASON
        or int(group.get("completed_common_epoch", -1)) != 100
        or group.get("all_seeds_reached_hard_maximum_before_selection_failure")
        is not True
        or group.get("failure_preserved_without_gate_relaxation") is not True
        or group.get("market_replay_or_locked_test_consumed") is not False
    ):
        raise ValueError("scheduler failure is not the admissible 2022 RT iTransformer group")
    group_root = Path(str(group.get("output_root", "")))
    if group_root.parts[-3:] != (
        EXCLUDED_ORIGIN,
        EXCLUDED_TARGET,
        EXCLUDED_CANDIDATE,
    ):
        raise ValueError("failed scheduler group has the wrong annual origin")
    failure_path = Path(str(group.get("checkpoint_group_failure_path", "")))
    if (
        not failure_path.is_file()
        or sha256_file(failure_path) != group.get("checkpoint_group_failure_sha256")
    ):
        raise ValueError("failed-group evidence path/hash is invalid")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (
        failure.get("group_fingerprint") != group_key
        or failure.get("status") != EXCLUDED_REASON
        or failure.get("selection_result") != group.get("selection_result")
    ):
        raise ValueError("failed-group evidence differs from the scheduler manifest")
    return [
        {
            "unit_id": "__".join(
                (
                    EXCLUDED_ORIGIN,
                    EXCLUDED_TARGET,
                    EXCLUDED_CANDIDATE,
                    f"study_seed_{seed}",
                )
            ),
            "origin": EXCLUDED_ORIGIN,
            "forecast_year": EXCLUDED_FORECAST_YEAR,
            "target": EXCLUDED_TARGET,
            "candidate": EXCLUDED_CANDIDATE,
            "seed": seed,
            "seed_slot": f"study_seed_{seed}",
            "reason_code": EXCLUDED_REASON,
            "group_fingerprint": group_key,
            "checkpoint_group_failure_path": failure_path.as_posix(),
            "checkpoint_group_failure_sha256": sha256_file(failure_path),
        }
        for seed in EXPECTED_FAILED_SEEDS
    ]


def eligible_units_for_contract(
    units: Sequence[Any], value: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Return only units belonging to the hash-validated eligible pool."""

    normalized = validate_partial_candidate_pool_contract(value)
    eligible = set(normalized["eligible_candidates"])
    result = tuple(unit for unit in units if unit.candidate.name in eligible)
    if any(unit.candidate.name == EXCLUDED_CANDIDATE for unit in result):
        raise ValueError("excluded candidate survived eligible-unit filtering")
    return result


def selection_execution_units_for_contract(
    units: Sequence[Any],
    value: Mapping[str, Any],
    *,
    failed_units: Sequence[Mapping[str, Any]],
) -> tuple[Any, ...]:
    """Skip exactly three failed seed units while retaining other iTransformer years."""

    validate_partial_candidate_pool_contract(
        value, failed_units=failed_units, require_failed_units=True
    )
    failed_ids = {str(item["unit_id"]) for item in failed_units}
    result = tuple(unit for unit in units if unit.unit_id not in failed_ids)
    observed_skips = {unit.unit_id for unit in units} - {unit.unit_id for unit in result}
    if observed_skips != failed_ids or len(result) != SELECTION_EXECUTION_UNIT_COUNT:
        raise ValueError("selection execution filter did not skip exactly three seed units")
    return result


def _expected_utc_dates(forecast_year: int) -> tuple[str, ...]:
    start = datetime(forecast_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(forecast_year + 1, 1, 1, tzinfo=timezone.utc)
    values: list[str] = []
    current = start
    while current < end:
        values.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return tuple(values)


def _event_date_contract(prediction: Mapping[str, Any], forecast_year: int) -> dict[str, Any]:
    event_keys = tuple(map(str, prediction["event_keys"]))
    target_times = tuple(map(str, prediction["target_times_utc"]))
    if len(event_keys) != len(target_times):
        raise ValueError("event keys and target times are not aligned")
    parsed_dates = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%d")
        for value in target_times
    )
    expected_dates = _expected_utc_dates(forecast_year)
    observed_dates = tuple(dict.fromkeys(parsed_dates))
    counts = Counter(parsed_dates)
    expected_events_per_date = 96 * len(EXPECTED_ZONES)
    if observed_dates != expected_dates or set(counts.values()) != {
        expected_events_per_date
    }:
        raise ValueError(
            f"forecast year {forecast_year} does not contain its complete UTC date panel"
        )
    expected_count = len(expected_dates) * expected_events_per_date
    if len(event_keys) != expected_count:
        raise ValueError("forecast event count does not match date x zone x interval grid")
    key_zones: list[str] = []
    for key in event_keys:
        matching_zones = [zone for zone in EXPECTED_ZONES if zone in key]
        if len(matching_zones) != 1:
            raise ValueError("forecast event key does not identify one expected NYISO zone")
        key_zones.append(matching_zones[0])
    date_zone_counts = Counter(zip(parsed_dates, key_zones, strict=True))
    if set(date_zone_counts.values()) != {96} or set(key_zones) != set(
        EXPECTED_ZONES
    ):
        raise ValueError("forecast panel is not exactly date x zone x 96 intervals")
    return {
        "forecast_year": forecast_year,
        "date_count": len(expected_dates),
        "event_count": expected_count,
        "date_sha256": sha256_json(list(expected_dates)),
        "event_keys_sha256": sha256_json(list(event_keys)),
        "target_times_sha256": sha256_json(list(target_times)),
    }


def collect_eligible_oos_evidence(
    producer: Any,
    *,
    role: str,
    candidate_pool_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect and hash-pin a complete eligible candidate OOS panel.

    This uses the producer's frozen readers and sign/coupling gates, but never
    asks the original nine-candidate collector to treat the failed model as
    complete.
    """

    if role not in {"selection_oos", "locked_test"}:
        raise ValueError("eligible OOS evidence role must be selection_oos or locked_test")
    normalized = validate_partial_candidate_pool_contract(candidate_pool_contract)
    eligible = set(normalized["eligible_candidates"])
    units = tuple(
        unit
        for unit in producer.plan.work_units(role=role)
        if unit.candidate.name in eligible
    )
    expected_count = (
        SELECTION_ELIGIBLE_UNIT_COUNT
        if role == "selection_oos"
        else LOCKED_ELIGIBLE_UNIT_COUNT
    )
    if len(units) != expected_count:
        raise ValueError("eligible OOS unit count differs from the frozen panel")
    prediction_records: list[dict[str, Any]] = []
    dart_records: dict[str, dict[str, str]] = {}
    information_records: dict[str, str] = {}
    uniform_records: dict[str, str] = {}
    coupling_records: dict[str, str] = {}
    coverage_by_unit: dict[str, dict[str, Any]] = {}
    coverage_reference_by_year_target: dict[tuple[int, str], dict[str, Any]] = {}
    for unit in units:
        request_fingerprint = producer._request_fingerprint(unit)
        completed = producer._load_completed_unit(
            unit, request_fingerprint=request_fingerprint
        )
        if completed is None:
            raise ValueError(f"eligible unit is incomplete: {unit.unit_id}")
        prediction_path = Path(completed["checkpoint"]["prediction_artifact"]["path"])
        prediction_sha = str(completed["checkpoint"]["prediction_artifact"]["sha256"])
        if sha256_file(prediction_path) != prediction_sha:
            raise ValueError(f"eligible prediction hash changed: {unit.unit_id}")
        coverage = _event_date_contract(
            completed["prediction"], unit.origin.forecast_year
        )
        key = (unit.origin.forecast_year, unit.target.name)
        reference = coverage_reference_by_year_target.setdefault(key, coverage)
        if coverage != reference:
            raise ValueError("eligible candidates do not share the same full date/event panel")
        coverage_by_unit[unit.unit_id] = coverage
        prediction_records.append(
            {
                "unit_id": unit.unit_id,
                "path": prediction_path.as_posix(),
                "sha256": prediction_sha,
                "forecast_year": unit.origin.forecast_year,
                "target": unit.target.name,
                "candidate": unit.candidate.name,
                "seed_slot": unit.seed_slot.name,
                "coverage_sha256": sha256_json(coverage),
            }
        )
        dart_path = producer._ensure_dart_artifact(unit)
        if dart_path is None:
            raise ValueError(f"eligible paired DART artifact is incomplete: {unit.unit_id}")
        dart_records[dart_path.as_posix()] = {
            "path": dart_path.as_posix(),
            "sha256": sha256_file(dart_path),
        }
        info_path = producer._information_manifest_path(unit.origin)
        _, uniform_manifest_path = producer._uniform_paths(unit.origin)
        _, coupling_manifest_path = producer._joint_uniform_paths(unit)
        information_records[unit.origin.name] = sha256_file(info_path)
        uniform_records[unit.origin.name] = sha256_file(uniform_manifest_path)
        coupling_records[
            f"{unit.origin.name}/{unit.candidate.name}/{unit.seed_slot.name}"
        ] = sha256_file(coupling_manifest_path)
    candidate_counts = Counter(item["candidate"] for item in prediction_records)
    origin_count = 3 if role == "selection_oos" else 1
    expected_candidate_counts = Counter(
        {
            candidate.name: origin_count
            * len(producer.plan.targets)
            * len(candidate.seed_slots)
            for candidate in candidate_specs_for_contract(normalized)
        }
    )
    if candidate_counts != expected_candidate_counts:
        raise ValueError("eligible candidate/year/target/seed coverage is incomplete")
    evidence = {
        "candidate_pool_contract": {
            key: item
            for key, item in normalized.items()
            if key not in {"normalized_failed_units", "failed_units_sha256"}
        },
        "role": role,
        "eligible_unit_count": len(units),
        "prediction_artifacts": sorted(
            prediction_records, key=lambda item: item["unit_id"]
        ),
        "dart_artifacts": sorted(dart_records.values(), key=lambda item: item["path"]),
        "common_information_manifest_sha256": dict(sorted(information_records.items())),
        "common_uniform_manifest_sha256": dict(sorted(uniform_records.items())),
        "joint_coupling_manifest_sha256": dict(sorted(coupling_records.items())),
        "coverage_contract": {
            "required_selection_years": [2022, 2023, 2024]
            if role == "selection_oos"
            else [2025],
            "required_targets": ["da_lbmp", "rt_lbmp_at_da_gate"],
            "required_study_seeds": list(EXPECTED_FAILED_SEEDS),
            "eligible_candidates": list(ELIGIBLE_CANDIDATES),
            "date_drop_count": 0,
            "common_date_panel_preserved": True,
            "unit_coverage_sha256": sha256_json(coverage_by_unit),
            "year_target_reference_sha256": sha256_json(
                {
                    f"{year}/{target}": value
                    for (year, target), value in sorted(
                        coverage_reference_by_year_target.items()
                    )
                }
            ),
        },
    }
    evidence["evidence_sha256"] = sha256_json(evidence)
    return evidence


def freeze_partial_selection(
    producer: Any,
    dependency_fingerprints: Mapping[str, str],
    candidate_pool_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the complete eight-candidate evidence before opening 2025."""

    dependencies = _normalized_dependencies(dependency_fingerprints)
    if producer._locked_output_exists() or producer.locked_test_ledger_path.exists():
        raise LockedTestContractError(
            "selection cannot be frozen or changed after any 2025 output exists"
        )
    evidence = collect_eligible_oos_evidence(
        producer,
        role="selection_oos",
        candidate_pool_contract=candidate_pool_contract,
    )
    frozen_contract = {
        "schema": "dcvpp.dart_partial_candidate.v1",
        "plan_fingerprint": producer.plan.fingerprint,
        "input_fingerprints": producer.input_fingerprints,
        "dependency_fingerprints": dependencies,
        "candidate_pool_contract": evidence["candidate_pool_contract"],
        "selection_evidence": evidence,
        "selection_evidence_fingerprint": sha256_json(evidence),
        "selection_years": [2022, 2023, 2024],
        "final_refit_through_year": 2024,
        "locked_test_year": 2025,
        "locked_test_runs_allowed": 1,
        "locked_test_results_may_modify_selection": False,
        "common_pre_da_gate_information": True,
        "daily_retraining": False,
        "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
    }
    fingerprint = sha256_json(frozen_contract)
    path = producer.freeze_manifest_path
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("freeze_fingerprint") != fingerprint:
            raise LockedTestContractError(
                "immutable partial selection freeze already exists with different inputs"
            )
        return existing
    manifest = {
        "status": "FROZEN_BEFORE_2025_WITH_FAILED_CANDIDATE",
        "frozen_contract": frozen_contract,
        "freeze_fingerprint": fingerprint,
        "frozen_at_utc": utc_now(),
    }
    atomic_json(path, manifest)
    return manifest


def finalize_partial_selection_stage_manifest(
    stage_manifest: Mapping[str, Any],
    *,
    producer: Any,
    scheduler_manifest_path: Path,
) -> dict[str, Any]:
    """Promote a raw fail-closed partial stage to strict partial readiness."""

    raw = dict(stage_manifest)
    if raw.get("phase") != "selection" or int(raw.get("expected_unit_count", -1)) != 126:
        raise ValueError("partial finalization requires the full 126-unit selection stage")
    if raw.get("status") != "PARTIAL" or raw.get("downstream_ready") is not False:
        raise ValueError("partial finalization requires the frozen fail-closed stage state")
    scheduler_path = Path(scheduler_manifest_path)
    if not scheduler_path.is_file():
        raise ValueError("completed scheduler manifest is absent")
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    failed_units = failed_units_from_scheduler_manifest(scheduler)
    contract = build_partial_candidate_pool_contract()
    validate_partial_candidate_pool_contract(
        contract, failed_units=failed_units, require_failed_units=True
    )
    all_selection_units = producer.plan.work_units(role="selection_oos")
    expected_execution_units = selection_execution_units_for_contract(
        all_selection_units, contract, failed_units=failed_units
    )
    expected_completed_ids = {unit.unit_id for unit in expected_execution_units}
    completed_ids = tuple(map(str, raw.get("completed_units", ())))
    if (
        len(completed_ids) != SELECTION_EXECUTION_UNIT_COUNT
        or len(set(completed_ids)) != len(completed_ids)
        or set(completed_ids) != expected_completed_ids
    ):
        raise ValueError("raw stage did not complete exactly the 123 unaffected units")
    raw_failed_ids = {
        str(item.get("unit_id"))
        for item in raw.get("failed_units", ())
        if isinstance(item, Mapping)
    }
    expected_failed_ids = {str(item["unit_id"]) for item in failed_units}
    admissible_raw_failed_ids = {
        unit.unit_id
        for unit in all_selection_units
        if unit.origin.name == EXCLUDED_ORIGIN
        and unit.target.name == EXCLUDED_TARGET
        and unit.candidate.name == EXCLUDED_CANDIDATE
        and int(unit.seed_slot.value) in EXPECTED_FAILED_SEEDS
    }
    if (
        raw_failed_ids != expected_failed_ids
        or raw_failed_ids != admissible_raw_failed_ids
        or int(raw.get("failed_unit_count", -1)) != 3
    ):
        raise ValueError("raw stage failures differ from the three preserved seed units")
    evidence = collect_eligible_oos_evidence(
        producer,
        role="selection_oos",
        candidate_pool_contract=contract,
    )
    eligible_ids = {
        item["unit_id"] for item in evidence["prediction_artifacts"]
    }
    expected_eligible_ids = {
        unit.unit_id
        for unit in eligible_units_for_contract(
            producer.plan.work_units(role="selection_oos"), contract
        )
    }
    if eligible_ids != expected_eligible_ids:
        raise ValueError("eligible OOS evidence does not cover the exact eight-model pool")
    archive_inventory: Mapping[str, Any] | None = None
    if raw.get("anchor_overlay_recovery") is not None:
        from dcvpp.experiments.dart_checkpoint_replay_compat import (
            validate_v4_epoch_archives,
        )

        archive_inventory = validate_v4_epoch_archives(
            v4_stage_root=producer.output_root,
            expected_track_count=108,
            repository_root=Path(str(raw["repository_root"])),
        )
    finalized = {
        **raw,
        "status": PARTIAL_CANDIDATE_STATUS,
        "downstream_ready": False,
        "partial_candidate_downstream_ready": True,
        "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
        "completed_unit_count": SELECTION_EXECUTION_UNIT_COUNT,
        "failed_unit_count": len(failed_units),
        "failed_units": failed_units,
        "eligible_expected_unit_count": SELECTION_ELIGIBLE_UNIT_COUNT,
        "eligible_completed_unit_count": len(eligible_ids),
        "globally_excluded_completed_unit_count": (
            SELECTION_EXECUTION_UNIT_COUNT - len(eligible_ids)
        ),
        "candidate_pool_contract": contract,
        "date_drop_count": 0,
        "common_date_panel_preserved": True,
        "v4_full_epoch_archive_inventory": (
            archive_inventory
            if archive_inventory is not None
            else raw.get("v4_full_epoch_archive_inventory")
        ),
        "partial_candidate_compat_code_sha256": sha256_file(Path(__file__)),
        "scheduler_scientific_status": scheduler["scheduler_scientific_status"],
        "scheduler_manifest_path": scheduler_path.as_posix(),
        "scheduler_manifest_sha256": sha256_file(scheduler_path),
        "phase_evidence": evidence,
        "finished_at_utc": utc_now(),
    }
    finalized.pop("manifest_fingerprint", None)
    finalized["manifest_fingerprint"] = sha256_json(finalized)
    stage_root = Path(str(finalized.get("stage_root", producer.output_root)))
    manifest_path = stage_root / "selection_stage_manifest.json"
    atomic_json(manifest_path, finalized)
    return finalized


def run_partial_candidate_selection_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Resume the frozen stage, then admit only the exact isolated failure."""

    from dcvpp.experiments.dart_forecast_oos import DARTForecastOOSProducer
    from dcvpp.experiments.dart_training import (
        ValidationResidualGaussianCopulaProvider,
        run_dart_forecast_training_stage,
    )

    if kwargs.get("phase", "selection") != "selection":
        raise ValueError("partial selection wrapper accepts phase=selection only")
    if len(args) >= 4:
        artifact_root = Path(args[1]).resolve()
        run_id = str(args[3])
    else:
        artifact_root = Path(kwargs.get("artifact_root", "")).resolve()
        run_id = str(kwargs.get("run_id", ""))
    existing_path = (
        artifact_root
        / "dart_value_select"
        / f"run_id={run_id}"
        / "forecast_oos"
        / "selection_stage_manifest.json"
    )
    if kwargs.get("resume", True) and existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("status") == PARTIAL_CANDIDATE_STATUS:
            repository_root = (
                Path(args[0]).resolve()
                if args
                else Path(kwargs.get("repository_root", "")).resolve()
            )
            config_path = (
                Path(args[2]).resolve()
                if len(args) >= 3
                else Path(kwargs.get("config_path", "")).resolve()
            )
            validate_partial_candidate_pool_contract(
                existing.get("candidate_pool_contract"),
                failed_units=existing.get("failed_units"),
                require_failed_units=True,
            )
            existing_core = {
                key: item
                for key, item in existing.items()
                if key != "manifest_fingerprint"
            }
            evidence = existing.get("phase_evidence", {})
            evidence_core = (
                {
                    key: item
                    for key, item in evidence.items()
                    if key != "evidence_sha256"
                }
                if isinstance(evidence, Mapping)
                else {}
            )
            if (
                existing.get("manifest_fingerprint") != sha256_json(existing_core)
                or str(existing.get("run_id")) != run_id
                or Path(str(existing.get("stage_root", ""))).resolve()
                != existing_path.parent.resolve()
                or Path(str(existing.get("repository_root", ""))).resolve()
                != repository_root
                or Path(str(existing.get("artifact_root", ""))).resolve()
                != artifact_root
                or not config_path.is_file()
                or existing.get("config_sha256") != sha256_file(config_path)
                or not isinstance(existing.get("input_fingerprints"), Mapping)
                or existing.get("input_fingerprints", {}).get(
                    "immutable_v1_manifest"
                )
                != kwargs.get("input_fingerprints", {}).get(
                    "immutable_v1_manifest"
                )
                or existing.get("input_fingerprints", {}).get(
                    "v1_readiness_manifest"
                )
                != kwargs.get("input_fingerprints", {}).get(
                    "v1_readiness_manifest"
                )
                or existing.get("input_fingerprints", {}).get(
                    "v1_testing_selection_policy"
                )
                != kwargs.get("input_fingerprints", {}).get(
                    "v1_testing_selection_policy"
                )
                or existing.get("partial_candidate_downstream_ready") is not True
                or existing.get("downstream_ready") is not False
                or int(existing.get("completed_unit_count", -1))
                != SELECTION_EXECUTION_UNIT_COUNT
                or int(existing.get("eligible_completed_unit_count", -1))
                != SELECTION_ELIGIBLE_UNIT_COUNT
                or int(existing.get("failed_unit_count", -1)) != 3
                or int(existing.get("date_drop_count", -1)) != 0
                or existing.get("common_date_panel_preserved") is not True
                or not isinstance(evidence, Mapping)
                or evidence.get("evidence_sha256") != sha256_json(evidence_core)
                or int(evidence.get("eligible_unit_count", -1))
                != SELECTION_ELIGIBLE_UNIT_COUNT
            ):
                raise ValueError("existing strict partial selection manifest is invalid")
            return existing
    stage_runner = kwargs.pop("stage_runner", run_dart_forecast_training_stage)
    if not callable(stage_runner):
        raise TypeError("stage_runner must be callable")
    raw = stage_runner(*args, **kwargs)
    if raw.get("status") == "COMPLETE":
        return raw
    strict_keys = {
        "stage_root",
        "input_fingerprints",
        "expected_unit_count",
        "completed_units",
        "failed_units",
    }
    if (
        raw.get("status") != "PARTIAL"
        or raw.get("phase") != "selection"
        or not strict_keys.issubset(raw)
        or int(raw.get("expected_unit_count", -1)) != 126
    ):
        return raw
    stage_root = Path(str(raw["stage_root"]))
    provider = ValidationResidualGaussianCopulaProvider(residual_root=stage_root)
    producer = DARTForecastOOSProducer(
        stage_root,
        input_fingerprints=raw["input_fingerprints"],
        joint_draw_provider=provider,
    )
    producer.initialize()
    return finalize_partial_selection_stage_manifest(
        raw,
        producer=producer,
        scheduler_manifest_path=(
            stage_root / "checkpoint_scheduler/hybrid_scheduler_manifest.json"
        ),
    )


def _partial_stack_from_selection_manifest(
    repository_root: Path,
    artifact_root: Path,
    run_id: str,
) -> tuple[Any, Any, dict[str, Any], Path]:
    from dcvpp.experiments.dart_forecast_oos import DARTForecastOOSProducer
    from dcvpp.experiments.dart_training import (
        DARTTrainingConfig,
        ReferencedNYISODARTTrainingRunner,
        ReferencedNYISOPanelSource,
        ValidationResidualGaussianCopulaProvider,
    )

    repository = Path(repository_root).resolve()
    artifacts = Path(artifact_root).resolve()
    stage_root = artifacts / "dart_value_select" / f"run_id={run_id}" / "forecast_oos"
    selection_path = stage_root / "selection_stage_manifest.json"
    if not selection_path.is_file():
        raise ValueError("partial selection stage manifest is absent")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != PARTIAL_CANDIDATE_STATUS
        or selection.get("partial_candidate_downstream_ready") is not True
    ):
        raise ValueError("selection stage is not strict partial-candidate ready")
    contract = validate_partial_candidate_pool_contract(
        selection.get("candidate_pool_contract"),
        failed_units=selection.get("failed_units"),
        require_failed_units=True,
    )
    manifest_core = {
        key: item
        for key, item in selection.items()
        if key != "manifest_fingerprint"
    }
    if selection.get("manifest_fingerprint") != sha256_json(manifest_core):
        raise ValueError("partial selection manifest hash changed")
    expected_stage_root = (
        artifacts / "dart_value_select" / f"run_id={run_id}" / "forecast_oos"
    ).resolve()
    if Path(str(selection.get("stage_root", ""))).resolve() != expected_stage_root:
        raise ValueError("partial selection manifest points to another stage root")
    if str(selection.get("run_id")) != str(run_id):
        raise ValueError("partial selection manifest belongs to another run")
    if Path(str(selection.get("repository_root", ""))).resolve() != repository:
        raise ValueError("partial selection manifest belongs to another repository")
    if Path(str(selection.get("artifact_root", ""))).resolve() != artifacts:
        raise ValueError("partial selection manifest belongs to another artifact root")
    effective_training = DARTTrainingConfig()
    if (
        selection.get("training_config") != asdict(effective_training)
        or selection.get("training_config_sha256") != effective_training.fingerprint
    ):
        raise ValueError("partial selection training configuration changed")
    source = ReferencedNYISOPanelSource(repository_root=repository)
    runner = ReferencedNYISODARTTrainingRunner(
        output_root=stage_root,
        source=source,
        config=effective_training,
    )
    provider = ValidationResidualGaussianCopulaProvider(
        residual_root=stage_root,
        minimum_residuals=effective_training.minimum_coupling_residuals,
    )
    producer = DARTForecastOOSProducer(
        stage_root,
        input_fingerprints=selection["input_fingerprints"],
        joint_draw_provider=provider,
    )
    producer.initialize()
    return producer, runner, contract, stage_root


def _stage_identity_fields(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: selection[key]
        for key in (
            "repository_root",
            "artifact_root",
            "config_path",
            "config_sha256",
            "training_config",
            "training_config_sha256",
            "input_fingerprints",
            "required_candidate_pool",
            "paper_mode_fail_closed",
            "synthetic_fallback_used",
        )
        if key in selection
    }


def _locked_neural_group_jobs(
    runner: Any,
    units: Sequence[Any],
    *,
    plan_fingerprint: str,
    quantiles: Sequence[float],
) -> tuple[tuple[dict[str, Any], ...], tuple[tuple[str, str, str], ...]]:
    """Build exactly one 2025 job per eligible candidate/target neural group."""

    from dcvpp.experiments.dart_forecast_oos import OOSForecastRequest, STUDY_SEEDS

    unique: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    for unit in units:
        if unit.candidate.family != "neural":
            continue
        identity = (unit.origin.name, unit.target.name, unit.candidate.name)
        if identity not in seen:
            seen.add(identity)
            unique.append(unit)
    identities = tuple(
        (unit.origin.name, unit.target.name, unit.candidate.name) for unit in unique
    )
    expected_models = {
        candidate.name
        for candidate in candidate_specs_for_contract(
            build_partial_candidate_pool_contract()
        )
        if candidate.family == "neural"
    }
    if (
        len(unique) != LOCKED_ELIGIBLE_NEURAL_GROUP_COUNT
        or {unit.candidate.name for unit in unique} != expected_models
        or {unit.target.name for unit in unique}
        != {"da_lbmp", "rt_lbmp_at_da_gate"}
        or {unit.origin.forecast_year for unit in unique} != {2025}
    ):
        raise ValueError("locked neural pretraining pool is not the exact 10 groups")
    device = runner._neural_device()
    jobs: list[dict[str, Any]] = []
    for unit in unique:
        request = OOSForecastRequest(
            unit_id=f"partial_locked_pretrain__{unit.unit_id}",
            request_fingerprint=sha256_json(
                {
                    "role": "partial_locked_eligible_neural_pretraining",
                    "plan_fingerprint": plan_fingerprint,
                    "origin": unit.origin.name,
                    "target": unit.target.name,
                    "candidate": unit.candidate.name,
                }
            ),
            plan_fingerprint=plan_fingerprint,
            origin_name=unit.origin.name,
            fit_through_year=unit.origin.fit_through_year,
            forecast_year=unit.origin.forecast_year,
            target_name=unit.target.name,
            target_field=unit.target.target_field,
            candidate_name=unit.candidate.name,
            candidate_family=unit.candidate.family,
            seed_slot_name=unit.seed_slot.name,
            seed=STUDY_SEEDS[0],
            quantiles=tuple(map(float, quantiles)),
            training_start_utc=unit.origin.training_start_utc,
            training_cutoff_exclusive_utc=(
                unit.origin.training_cutoff_exclusive_utc
            ),
            forecast_start_utc=unit.origin.forecast_start_utc,
            forecast_end_exclusive_utc=unit.origin.forecast_end_exclusive_utc,
            checkpoint_directory=(
                Path(runner.output_root)
                / LOCKED_SCHEDULER_DIRECTORY
                / "request_views"
                / unit.target.name
                / unit.candidate.name
            ),
            resume_checkpoint=None,
            expected_information_set_fingerprint=None,
        )
        runner._validate_request(request)
        data = runner._data(request)
        jobs.append(runner._neural_group_kwargs(request, data, device=device))
    return tuple(jobs), identities


def pretrain_partial_locked_neural_groups(
    runner: Any,
    units: Sequence[Any],
    *,
    plan_fingerprint: str,
    quantiles: Sequence[float],
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Run and verify the independent exact-10 2025 neural scheduler."""

    from dcvpp.experiments import dart_checkpoint_training as checkpoint_training

    jobs, identities = _locked_neural_group_jobs(
        runner,
        units,
        plan_fingerprint=plan_fingerprint,
        quantiles=quantiles,
    )
    schedule = scheduler or checkpoint_training.pretrain_dart_energy_groups_hybrid
    if not callable(schedule):
        raise TypeError("locked neural scheduler must be callable")
    scheduler_root = Path(runner.output_root) / LOCKED_SCHEDULER_DIRECTORY
    manifest = schedule(jobs, scheduler_root=scheduler_root)
    path = scheduler_root / "hybrid_scheduler_manifest.json"
    if not path.is_file():
        raise ValueError("locked neural scheduler manifest is absent")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    core = {key: item for key, item in persisted.items() if key != "manifest_sha256"}
    completed = persisted.get("completed_groups", {})
    completed_identities = {
        (
            Path(str(record.get("output_root", ""))).parts[-3],
            str(record.get("task")),
            str(record.get("model_name")),
        )
        for record in completed.values()
        if isinstance(record, Mapping)
        and len(Path(str(record.get("output_root", ""))).parts) >= 3
    }
    if (
        manifest != persisted
        or persisted.get("status") != "COMPLETE"
        or persisted.get("efficient_hybrid_group_scheduler") is not True
        or tuple(persisted.get("study_seeds", ()))
        != tuple(DART_VALUE_SELECT_SEEDS)
        or persisted.get("market_replay_or_locked_test_consumed") is not False
        or persisted.get("manifest_sha256") != sha256_json(core)
        or int(persisted.get("group_count", -1))
        != LOCKED_ELIGIBLE_NEURAL_GROUP_COUNT
        or len(completed) != LOCKED_ELIGIBLE_NEURAL_GROUP_COUNT
        or completed_identities != set(identities)
        or persisted.get("failed_groups") not in (None, {})
    ):
        raise ValueError("locked neural scheduler did not complete the exact 10 groups")
    return {
        "schema_version": "dcvpp.dart_partial_locked_scheduler.v1",
        "status": "COMPLETE",
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "manifest_sha256": persisted["manifest_sha256"],
        "scheduler_fingerprint": persisted["scheduler_fingerprint"],
        "group_count": LOCKED_ELIGIBLE_NEURAL_GROUP_COUNT,
        "eligible_neural_groups": [list(identity) for identity in identities],
        "candidate_pool_contract": build_partial_candidate_pool_contract(),
        "failed_group_count": 0,
    }


@contextmanager
def _locked_scheduler_verification(runner: Any, evidence: Mapping[str, Any]):
    """Route the frozen runner's read-only gate to the independent scheduler."""

    scheduler_path = Path(str(evidence.get("path", "")))
    if not scheduler_path.is_file() or sha256_file(scheduler_path) != evidence.get(
        "sha256"
    ):
        raise ValueError("locked neural scheduler evidence hash changed")
    original = runner._require_hybrid_pretraining

    def require(group_root: Path) -> None:
        manifest = json.loads(scheduler_path.read_text(encoding="utf-8"))
        core = {
            key: item for key, item in manifest.items() if key != "manifest_sha256"
        }
        matching = [
            record
            for record in manifest.get("completed_groups", {}).values()
            if Path(str(record.get("output_root", ""))) == Path(group_root)
        ]
        if (
            manifest.get("status") != "COMPLETE"
            or tuple(manifest.get("study_seeds", ()))
            != tuple(DART_VALUE_SELECT_SEEDS)
            or manifest.get("market_replay_or_locked_test_consumed") is not False
            or manifest.get("manifest_sha256") != sha256_json(core)
            or int(manifest.get("group_count", -1))
            != LOCKED_ELIGIBLE_NEURAL_GROUP_COUNT
            or len(matching) != 1
        ):
            raise ValueError("neural group is absent from the locked scheduler")
        checkpoint = Path(str(matching[0].get("checkpoint_group_manifest_path", "")))
        if not checkpoint.is_file() or sha256_file(checkpoint) != matching[0].get(
            "checkpoint_group_manifest_sha256"
        ):
            raise ValueError("locked neural group checkpoint hash changed")

    runner._require_hybrid_pretraining = require
    try:
        yield
    finally:
        runner._require_hybrid_pretraining = original


def run_partial_candidate_freeze_stage(
    repository_root: Path,
    artifact_root: Path,
    config_path: Path,
    run_id: str,
    *,
    dependency_fingerprints: Mapping[str, str],
    **_: Any,
) -> dict[str, Any]:
    """Freeze the hash-complete eight-candidate 2022--2024 panel."""

    phase = _.get("phase", "freeze")
    if phase != "freeze":
        raise ValueError("partial freeze wrapper accepts phase=freeze only")

    producer, _runner, contract, stage_root = _partial_stack_from_selection_manifest(
        repository_root, artifact_root, run_id
    )
    selection = json.loads(
        (stage_root / "selection_stage_manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(Path(config_path).resolve()) != selection.get("config_sha256"):
        raise ValueError("partial freeze config differs from selection")
    started = utc_now()
    freeze = freeze_partial_selection(producer, dependency_fingerprints, contract)
    manifest = {
        "schema": "dart_training_adapter_v1",
        "stage": "dart_annual_forecast_training",
        "phase": "freeze",
        "run_id": run_id,
        **_stage_identity_fields(selection),
        "status": "COMPLETE",
        "downstream_ready": True,
        "partial_candidate_downstream_ready": True,
        "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
        "candidate_pool_contract": contract,
        "expected_unit_count": 1,
        "completed_unit_count": 1,
        "completed_units": ["selection_freeze"],
        "failed_unit_count": 0,
        "failed_units": [],
        "date_drop_count": 0,
        "phase_evidence": freeze,
        "stage_root": stage_root.as_posix(),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    atomic_json(stage_root / "freeze_stage_manifest.json", manifest)
    return manifest


def run_partial_candidate_locked_stage(
    repository_root: Path,
    artifact_root: Path,
    config_path: Path,
    run_id: str,
    *,
    dependency_fingerprints: Mapping[str, str],
    resume: bool = True,
    **_: Any,
) -> dict[str, Any]:
    """Open 2025 once and execute only the frozen eligible 36-unit pool."""

    phase = _.get("phase", "locked_test")
    if phase != "locked_test":
        raise ValueError("partial locked wrapper accepts phase=locked_test only")

    producer, runner, contract, stage_root = _partial_stack_from_selection_manifest(
        repository_root, artifact_root, run_id
    )
    selection = json.loads(
        (stage_root / "selection_stage_manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(Path(config_path).resolve()) != selection.get("config_sha256"):
        raise ValueError("partial locked-test config differs from selection")
    dependencies = _normalized_dependencies(dependency_fingerprints)
    freeze_path = producer.freeze_manifest_path
    if not freeze_path.is_file():
        raise LockedTestContractError("partial selection freeze manifest is absent")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_contract = freeze.get("frozen_contract", {})
    if (
        freeze.get("status") != "FROZEN_BEFORE_2025_WITH_FAILED_CANDIDATE"
        or frozen_contract.get("dependency_fingerprints") != dependencies
        or frozen_contract.get("candidate_pool_contract") != contract
        or freeze.get("freeze_fingerprint") != sha256_json(frozen_contract)
    ):
        raise LockedTestContractError("partial selection freeze contract is invalid")
    freeze_file_sha = sha256_file(freeze_path)
    locked_origin = next(
        item for item in producer.plan.origins if item.role == "locked_test"
    )
    locked_contract = {
        "schema": "dcvpp.dart_partial_candidate.v1",
        "plan_fingerprint": producer.plan.fingerprint,
        "freeze_fingerprint": freeze["freeze_fingerprint"],
        "freeze_manifest_sha256": freeze_file_sha,
        "dependency_fingerprints": dependencies,
        "candidate_pool_contract": contract,
        "origin": {
            **asdict(locked_origin),
            "training_start_utc": locked_origin.training_start_utc,
            "training_cutoff_exclusive_utc": (
                locked_origin.training_cutoff_exclusive_utc
            ),
            "forecast_start_utc": locked_origin.forecast_start_utc,
            "forecast_end_exclusive_utc": locked_origin.forecast_end_exclusive_utc,
        },
        "locked_test_runs_allowed": 1,
        "resume_same_fingerprint_allowed": True,
        "locked_test_feedback_forbidden": True,
    }
    locked_fingerprint = sha256_json(locked_contract)
    ledger_path = producer.locked_test_ledger_path
    started = utc_now()
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if (
            ledger.get("locked_test_fingerprint") != locked_fingerprint
            or ledger.get("freeze_manifest_sha256") != freeze_file_sha
        ):
            raise LockedTestContractError("2025 partial locked-test fingerprint changed")
        if ledger.get("status") == "COMPLETE":
            if not resume:
                raise LockedTestContractError("the once-only 2025 test is already complete")
            evidence = collect_eligible_oos_evidence(
                producer, role="locked_test", candidate_pool_contract=contract
            )
            if sha256_json(evidence) != ledger.get("evidence_fingerprint"):
                raise LockedTestContractError("locked-test evidence changed")
            stage_manifest_path = stage_root / "locked_test_stage_manifest.json"
            if not stage_manifest_path.is_file():
                raise LockedTestContractError(
                    "complete locked ledger lacks its stage manifest"
                )
            resumed_manifest = json.loads(
                stage_manifest_path.read_text(encoding="utf-8")
            )
            resumed_core = {
                key: item
                for key, item in resumed_manifest.items()
                if key != "manifest_fingerprint"
            }
            if (
                resumed_manifest.get("manifest_fingerprint")
                != sha256_json(resumed_core)
            ):
                raise LockedTestContractError("locked stage manifest hash changed")
            return resumed_manifest
        if not resume:
            raise LockedTestContractError("the once-only 2025 test is already open")
    else:
        if producer._locked_output_exists():
            raise LockedTestContractError("2025 outputs exist without the opening ledger")
        ledger = {
            "schema": "dcvpp.dart_partial_candidate.v1",
            "status": "STARTED",
            "locked_test_fingerprint": locked_fingerprint,
            "locked_contract": locked_contract,
            "freeze_manifest_sha256": freeze_file_sha,
            "candidate_pool_contract": contract,
            "run_count": 1,
            "started_at_utc": started,
            "locked_test_results_may_modify_selection": False,
        "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
    }
        atomic_json(ledger_path, ledger)
    units = eligible_units_for_contract(
        producer.plan.work_units(role="locked_test"), contract
    )
    if len(units) != LOCKED_ELIGIBLE_UNIT_COUNT:
        raise ValueError("locked eligible pool must contain exactly 36 units")
    eligible_neural_groups = {
        (unit.origin.name, unit.target.name, unit.candidate.name)
        for unit in units
        if unit.candidate.family == "neural"
    }
    if len(eligible_neural_groups) != LOCKED_ELIGIBLE_NEURAL_GROUP_COUNT:
        raise ValueError("locked eligible pool must contain exactly 10 neural groups")
    completed: list[str] = []
    try:
        locked_scheduler = pretrain_partial_locked_neural_groups(
            runner,
            units,
            plan_fingerprint=producer.plan.fingerprint,
            quantiles=producer.plan.quantiles,
        )
        with _locked_scheduler_verification(runner, locked_scheduler):
            for unit in units:
                producer.execute_unit(
                    unit, runner, resume=resume, _allow_locked_test=True
                )
                completed.append(unit.unit_id)
        evidence = collect_eligible_oos_evidence(
            producer, role="locked_test", candidate_pool_contract=contract
        )
        complete_ledger = {
            **ledger,
            "status": "COMPLETE",
            "completed_at_utc": utc_now(),
            "execution_count": len(completed),
            "locked_neural_pretraining": locked_scheduler,
            "evidence": evidence,
            "evidence_fingerprint": sha256_json(evidence),
            "run_count": 1,
            "selection_updated_from_2025": False,
        }
        atomic_json(ledger_path, complete_ledger)
    except Exception as exc:
        atomic_json(
            ledger_path,
            {
                **ledger,
                "status": "STARTED",
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "last_error_at_utc": utc_now(),
                "run_count": 1,
            },
        )
        raise
    manifest = {
        "schema": "dart_training_adapter_v1",
        "stage": "dart_annual_forecast_training",
        "phase": "locked_test",
        "run_id": run_id,
        **_stage_identity_fields(selection),
        "status": "COMPLETE",
        "downstream_ready": True,
        "partial_candidate_downstream_ready": True,
        "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
        "candidate_pool_contract": contract,
        "expected_unit_count": LOCKED_ELIGIBLE_UNIT_COUNT,
        "completed_unit_count": len(completed),
        "completed_units": completed,
        "failed_unit_count": 0,
        "failed_units": [],
        "date_drop_count": 0,
        "common_date_panel_preserved": True,
        "phase_evidence": complete_ledger,
        "stage_root": stage_root.as_posix(),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    atomic_json(stage_root / "locked_test_stage_manifest.json", manifest)
    return manifest


__all__ = [
    "ATTEMPTED_CANDIDATES",
    "ELIGIBLE_CANDIDATES",
    "EXCLUDED_CANDIDATE",
    "PARTIAL_CANDIDATE_STATUS",
    "PARTIAL_CANDIDATE_TERMINAL_CAP",
    "build_partial_candidate_pool_contract",
    "candidate_specs_for_contract",
    "collect_eligible_oos_evidence",
    "eligible_units_for_contract",
    "failed_units_from_scheduler_manifest",
    "finalize_partial_selection_stage_manifest",
    "freeze_partial_selection",
    "pretrain_partial_locked_neural_groups",
    "run_partial_candidate_selection_stage",
    "run_partial_candidate_freeze_stage",
    "run_partial_candidate_locked_stage",
    "selection_execution_units_for_contract",
    "validate_partial_candidate_pool_contract",
]
