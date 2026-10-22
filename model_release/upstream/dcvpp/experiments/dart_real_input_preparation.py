"""Prepare hash-pinned real-input indices for formal DART replay.

This module is intentionally an assembler and auditor, not a data generator.
It never substitutes a scenario draw for a realised path and never invents a
missing workload, PV, or gate-state record.  A config-ready
``dart_real_evidence_inputs`` block is emitted only when the exact requested
date panel is complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_training import ReferencedNYISOPanelSource
from dcvpp.research_branch.contracts import NYISO_ZONES


SCHEMA_VERSION = "dcvpp.dart_real_input_preparation.v1"
DESIGN_SCHEMA_VERSION = "dcvpp.dart_counterfactual_input_design.v1"
SETTLEMENT_COVERAGE_SCHEMA_VERSION = "dcvpp.nyiso_settlement_coverage.v1"
_REQUIRED_ACTION_SOURCE_FIELDS = {
    "available_task_queue",
    "task_deadlines",
    "server_capacity",
    "current_battery_soc_and_power_limits",
    "migration_eligibility_and_bandwidth",
    "battery_degradation_cost",
    "migration_energy_and_bandwidth_cost",
    "transaction_cost",
    "engineered_sla_risk",
    "minimum_dispatch_quantity",
    "settlement_asymmetry",
}


class DARTRealInputPreparationError(RuntimeError):
    """Raised when an index is malformed; missing evidence is reported in-manifest."""


def _relative_or_absolute(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def _hashed_source(path: str | Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, root=root)
    return {
        "path": _relative_or_absolute(resolved, root),
        "exists": resolved.is_file(),
        "byte_count": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def write_counterfactual_input_design_manifest(
    *,
    repository_root: Path,
    output_root: Path,
    settlement_coverage_index: str | Path,
    nlr295_workload_path: str | Path,
    nasa_derived_pv_path: str | Path,
    frozen_physical_artifact_path: str | Path,
    frozen_twin_path: str | Path,
    locked_issue_index_path: str | Path,
    common_seed: int = 20260809,
    draw_count: int = 100,
) -> dict[str, Any]:
    """Write a lightweight, fail-closed design for counterfactual DART inputs.

    This intentionally does not materialize daily NPZ files.  In particular,
    an incomplete historical settlement panel is assessed before spending time
    on workload/PV bootstrap materialization or any optimization.  The design
    records the exact causal construction that can be run later without
    changing the date contract or turning a separately observed NLR trace into
    an observed NYISO facility trajectory.
    """

    if draw_count != 100:
        raise DARTRealInputPreparationError("formal DART design requires exactly 100 draws")
    root = Path(repository_root).resolve()
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    coverage_path = _resolve(settlement_coverage_index, root=root)
    if not coverage_path.is_file():
        raise DARTRealInputPreparationError("settlement coverage index is absent")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not isinstance(coverage, Mapping) or not isinstance(coverage.get("phases"), Mapping):
        raise DARTRealInputPreparationError("settlement coverage index has no phase mapping")
    expected_counts = {"selection": 1096, "locked_test": 365}
    coverage_summary: dict[str, Any] = {}
    settlement_blocked = False
    for phase, expected_count in expected_counts.items():
        raw = coverage["phases"].get(phase)
        if not isinstance(raw, Mapping):
            raise DARTRealInputPreparationError(f"coverage lacks {phase}")
        retained = int(raw.get("retained_operating_date_count", -1))
        expected = int(raw.get("expected_operating_date_count", -1))
        invalid = int(raw.get("settlement_invalid_date_count", -1))
        if expected != expected_count or retained != expected_count or invalid < 0:
            raise DARTRealInputPreparationError(
                f"{phase} coverage changed the exact full-calendar date contract"
            )
        phase_path = _resolve(raw["path"], root=root) if raw.get("path") else None
        if phase_path is None or not phase_path.is_file():
            raise DARTRealInputPreparationError(f"{phase} coverage evidence is absent")
        phase_sha = sha256_file(phase_path)
        if raw.get("sha256") is not None and str(raw["sha256"]) != phase_sha:
            raise DARTRealInputPreparationError(f"{phase} coverage evidence hash changed")
        settlement_blocked = settlement_blocked or invalid > 0
        coverage_summary[phase] = {
            "status": str(raw.get("status", "")),
            "expected_operating_date_count": expected,
            "retained_operating_date_count": retained,
            "settlement_evaluable_date_count": int(
                raw.get("settlement_evaluable_date_count", expected - invalid)
            ),
            "settlement_invalid_date_count": invalid,
            "missing_settlement_interval_count": int(
                raw.get("missing_settlement_interval_count", 0)
            ),
            "date_drop_count": int(raw.get("date_drop_count", 0)),
            "coverage_path": _relative_or_absolute(phase_path, root),
            "coverage_sha256": phase_sha,
        }
    sources = {
        "nlr295_workload_15m": _hashed_source(nlr295_workload_path, root=root),
        "nasa_derived_pv_15m": _hashed_source(nasa_derived_pv_path, root=root),
        "frozen_physical_artifact": _hashed_source(
            frozen_physical_artifact_path, root=root
        ),
        "frozen_nyiso_twins": _hashed_source(frozen_twin_path, root=root),
        "locked_2025_issue_index": _hashed_source(locked_issue_index_path, root=root),
        "settlement_coverage_index": _hashed_source(coverage_path, root=root),
    }
    absent_sources = sorted(name for name, record in sources.items() if not record["exists"])
    if absent_sources:
        raise DARTRealInputPreparationError(
            f"counterfactual input design sources are absent: {absent_sources}"
        )

    physical_path = _resolve(frozen_physical_artifact_path, root=root)
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    if physical.get("frozen") is not True or tuple(physical.get("zones", ())) != tuple(NYISO_ZONES):
        raise DARTRealInputPreparationError("physical artifact is not the frozen four-zone twin")
    battery_power = np.asarray(physical.get("battery_power_mw"), dtype=float)
    battery_energy = np.asarray(physical.get("battery_energy_mwh"), dtype=float)
    multiplier_pool = np.asarray(physical.get("physical_multiplier_pool"), dtype=float)
    if (
        battery_power.shape != (len(NYISO_ZONES),)
        or battery_energy.shape != battery_power.shape
        or multiplier_pool.ndim != 2
        or multiplier_pool.shape[1] != len(NYISO_ZONES)
        or multiplier_pool.shape[0] < draw_count
        or not np.isfinite(multiplier_pool).all()
    ):
        raise DARTRealInputPreparationError("frozen physical arrays do not satisfy the design")
    physical_center = np.median(multiplier_pool, axis=0)
    initial_soc = 0.5 * battery_energy
    interval_hours = 0.25
    battery_flexible_mwh = np.minimum.reduce(
        [battery_power * interval_hours, initial_soc, battery_energy - initial_soc]
    )
    normalizer = np.maximum(battery_power * interval_hours, np.finfo(float).eps)
    normalized_f = np.clip(battery_flexible_mwh / normalizer, 0.0, 1.0)

    design_state = (
        "BLOCKED_BY_SETTLEMENT_COVERAGE"
        if settlement_blocked
        else "DESIGN_READY_FOR_MATERIALIZATION"
    )
    manifest: dict[str, Any] = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "status": design_state,
        "created_at": utc_now(),
        "paper_terminal_cap": "PARTIAL_PAPER" if settlement_blocked else None,
        "formal_a1_a2_mcs_lcb_winner_defined": not settlement_blocked,
        "locked_2025_opened": False,
        "locked_2025_opening_rule": (
            "forbidden_until_selection_full_panel_replay_and_static_rule_freeze_are_complete"
        ),
        "efficiency_stop": {
            "daily_npz_materialized_by_this_step": 0,
            "training_runs_launched_by_this_step": 0,
            "solver_runs_launched_by_this_step": 0,
            "replay_cells_launched_by_this_step": 0,
            "reason": (
                "settlement completeness was audited before expensive counterfactual input materialization"
            ),
        },
        "date_contract": {
            "selection": {"years": [2022, 2023, 2024], "day_count": 1096},
            "locked_test": {"years": [2025], "day_count": 365},
            "dates_dropped": 0,
            "missing_settlement_imputed": False,
        },
        "settlement_coverage": coverage_summary,
        "source_inventory": sources,
        "common_randomness": {
            "draw_count": draw_count,
            "root_seed": int(common_seed),
            "daily_seed_formula": (
                "uint32(first_8_bytes(SHA256(schema_version|root_seed|operating_date|component)))"
            ),
            "candidate_independent": True,
            "same_draw_ids_across_candidates": True,
        },
        "constructible_inputs": {
            "issue_time_workload_ensemble": {
                "status": "DESIGN_READY_NOT_MATERIALIZED",
                "source": "nlr295_workload_15m",
                "method": "causal_seasonal_circular_day_bootstrap_with_replacement",
                "history_gate": "all source intervals in a sampled day are strictly before issue_time_utc",
                "seasonal_pool": (
                    "complete 96-interval NLR days ranked by circular day-of-year distance; "
                    "same weekday class first, then nearest dates until nonempty"
                ),
                "draws": draw_count,
                "mapping_to_nyiso": (
                    "cross_domain_counterfactual_only; NLR workload is not a synchronized or observed NYISO-site load"
                ),
                "future_values_used": False,
            },
            "issue_time_pv_ensemble": {
                "status": "DESIGN_READY_NOT_MATERIALIZED",
                "source": "nasa_derived_pv_15m",
                "method": "causal_zonewise_seasonal_climatology_day_bootstrap_with_replacement",
                "history_gate": (
                    "all sampled rows satisfy deployable_available_at_utc <= issue_time_utc "
                    "and all source intervals precede the operating day"
                ),
                "seasonal_pool": "complete four-zone 96-interval days ranked by circular day-of-year distance",
                "draws": draw_count,
                "evidence_role": "reconstructed",
                "local_measurement_claimed": False,
                "future_values_used": False,
            },
            "replay_forcing": {
                "status": "DESIGN_READY_NOT_MATERIALIZED",
                "candidate_independent": True,
                "selected_scenario_as_realized": False,
                "workload": (
                    "frozen circular mapping over complete referenced NLR days; replayed cross-domain counterfactual"
                ),
                "pv": (
                    "target-day NASA-derived PV path; reconstructed forcing revealed one interval at a time after decision freeze"
                ),
                "physical_multiplier": {
                    "policy": "preregistered_median_of_frozen_physical_multiplier_pool",
                    "value_by_zone": physical_center.tolist(),
                    "candidate_independent": True,
                    "selected_after_outcome": False,
                    "evidence_role": "counterfactual",
                },
            },
            "actionability": {
                "status": "DESIGN_READY_NOT_MATERIALIZED",
                "shape": [len(NYISO_ZONES), 96],
                "policy": "conservative_battery_only_when_gate_time_queue_and_deadline_state_is_absent",
                "queue_migration_flexibility_mwh": 0.0,
                "initial_soc_policy": "engineering_assumption_50_percent_of_frozen_energy_capacity",
                "initial_soc_mwh_by_zone": initial_soc.tolist(),
                "battery_flexible_energy_mwh_per_interval_by_zone": battery_flexible_mwh.tolist(),
                "normalized_F_by_zone_repeated_over_96_intervals": normalized_f.tolist(),
                "formula": (
                    "F[z,t]=min(P[z]*0.25,SOC0[z],E[z]-SOC0[z])/(P[z]*0.25), clipped to [0,1]"
                ),
                "available_at_gate": True,
                "candidate_independent": True,
                "future_realized_price_workload_pv_used": False,
                "source_fields": sorted(_REQUIRED_ACTION_SOURCE_FIELDS),
                "absent_queue_deadline_and_migration_fields_policy": (
                    "known conservative zero contribution, not imputed operational observations"
                ),
                "no_action_band": {
                    "units": "USD_per_MWh_equivalent",
                    "lower_formula": "-(battery_degradation + migration + transaction + SLA + minimum_dispatch + settlement_asymmetry)",
                    "upper_formula": "+(battery_degradation + migration + transaction + SLA + minimum_dispatch + settlement_asymmetry)",
                    "battery_degradation_usd_per_mwh": 1.0,
                    "battery_degradation_source": "frozen optimizer default",
                    "migration_energy_and_bandwidth_usd_per_mwh": 0.0,
                    "engineered_sla_risk_usd_per_mwh": 0.0,
                    "zero_components_reason": "battery-only action does not migrate, defer, or drop workload",
                    "transaction_cost_usd_per_mwh": 0.0,
                    "minimum_dispatch_adder_usd_per_mwh": 0.0,
                    "settlement_asymmetry_adder_usd_per_mwh": 0.0,
                    "zero_cost_components_evidence_role": "preregistered_engineering_assumption",
                },
            },
        },
        "claim_boundaries": {
            "synthetic_fallback_used": False,
            "nlr_nyiso_joint_observation_claimed": False,
            "counterfactual": True,
            "nasa_power_semantics": "third_party_referenced_reconstruction_not_local_measurement",
            "nlr_semantics": "third_party_referenced_trace_not_nyiso_facility_operation",
            "nyiso_semantics": "official_historical_market_record_not_investigator_measurement",
            "commercial_profit_claimed": False,
        },
        "materialization_gate": {
            "allowed": not settlement_blocked,
            "requirements": [
                "complete unchanged selection settlement panel",
                "complete candidate-independent issue-time/forcing/actionability indices",
                "selection replay and static rule freeze before locked 2025 is opened",
            ],
        },
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    output_path = destination / "counterfactual_input_design_manifest.json"
    atomic_json(output_path, manifest)
    manifest["manifest_path"] = _relative_or_absolute(output_path, root)
    manifest["manifest_sha256"] = sha256_file(output_path)
    return manifest


def _settlement_coverage_from_frames(
    *,
    phase: str,
    dates: Sequence[str],
    da_15m: pd.DataFrame,
    rt_15m: pd.DataFrame,
    source_sha256: str,
) -> dict[str, Any]:
    """Audit exact calendar settlement coverage without imputing or dropping dates.

    A missing source interval makes that operating date unavailable for economic
    settlement.  It is not a solver infeasibility: no candidate solver or cost
    calculation is allowed to run for such a date.
    """

    expected_dates = tuple(map(str, dates))
    if not expected_dates or len(set(expected_dates)) != len(expected_dates):
        raise DARTRealInputPreparationError(
            "settlement coverage dates must be nonempty and unique"
        )
    _ = exact_phase_dates(phase)
    source_digest = str(source_sha256).lower()
    if len(source_digest) != 64 or any(
        character not in "0123456789abcdef" for character in source_digest
    ):
        raise DARTRealInputPreparationError("settlement source SHA-256 is invalid")

    matrices: dict[str, pd.DataFrame] = {}
    for field, frame in (("da_lbmp", da_15m), ("rt_lbmp", rt_15m)):
        required = {"timestamp_utc", "zone", field}
        missing = required.difference(frame.columns)
        if missing:
            raise DARTRealInputPreparationError(
                f"settlement {field} panel lacks {sorted(missing)}"
            )
        rows = frame[["timestamp_utc", "zone", field]].copy()
        rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
        rows["zone"] = rows["zone"].astype(str)
        if rows.duplicated(["timestamp_utc", "zone"]).any():
            raise DARTRealInputPreparationError(
                f"settlement {field} panel contains duplicate zone-times"
            )
        matrices[field] = rows.pivot(
            index="timestamp_utc", columns="zone", values=field
        ).reindex(columns=NYISO_ZONES)

    records: list[dict[str, Any]] = []
    complete_dates: list[str] = []
    incomplete_dates: list[str] = []
    missing_union_total = 0
    missing_da_total = 0
    missing_rt_total = 0
    missing_value_total = 0
    for day in expected_dates:
        timestamps = pd.date_range(day, periods=96, freq="15min", tz="UTC")
        da = matrices["da_lbmp"].reindex(timestamps)
        rt = matrices["rt_lbmp"].reindex(timestamps)
        da_missing = da.isna().any(axis=1).to_numpy(dtype=bool)
        rt_missing = rt.isna().any(axis=1).to_numpy(dtype=bool)
        any_missing = da_missing | rt_missing
        missing_union = int(np.count_nonzero(any_missing))
        missing_da = int(np.count_nonzero(da_missing))
        missing_rt = int(np.count_nonzero(rt_missing))
        missing_values = int(da.isna().to_numpy().sum() + rt.isna().to_numpy().sum())
        evaluable = missing_union == 0
        (complete_dates if evaluable else incomplete_dates).append(day)
        missing_union_total += missing_union
        missing_da_total += missing_da
        missing_rt_total += missing_rt
        missing_value_total += missing_values
        records.append(
            {
                "operating_date": day,
                "expected_interval_count": 96,
                "missing_settlement_interval_count": missing_union,
                "missing_da_interval_count": missing_da,
                "missing_rt_interval_count": missing_rt,
                "missing_zone_value_count": missing_values,
                "settlement_evaluable": evaluable,
                "decision_replay_cell_status": (
                    "ELIGIBLE" if evaluable else "INVALID_MISSING_SETTLEMENT"
                ),
                "solver_launched_for_missing_settlement": False,
                "candidate_cost_emitted_for_missing_settlement": False,
            }
        )

    status = "COMPLETE" if not incomplete_dates else "PARTIAL_SOURCE_COVERAGE"
    payload = {
        "schema_version": SETTLEMENT_COVERAGE_SCHEMA_VERSION,
        "status": status,
        "phase": phase,
        "expected_operating_date_count": len(expected_dates),
        "retained_operating_date_count": len(records),
        "date_drop_count": 0,
        "settlement_evaluable_date_count": len(complete_dates),
        "settlement_invalid_date_count": len(incomplete_dates),
        "missing_settlement_interval_count": missing_union_total,
        "missing_da_interval_count": missing_da_total,
        "missing_rt_interval_count": missing_rt_total,
        "missing_zone_value_count": missing_value_total,
        "operating_date_panel_sha256": sha256_json(list(expected_dates)),
        "source_sha256": source_digest,
        "missing_settlement_imputed": False,
        "missing_dates_silently_dropped": False,
        "missing_settlement_is_solver_infeasibility": False,
        "decision_value_defined_only_for_complete_settlement_dates": True,
        "formal_full_panel_decision_selection_complete": not incomplete_dates,
        "mcs_lcb_and_winner_defined": not incomplete_dates,
        "terminal_paper_cap": None if not incomplete_dates else "PARTIAL_PAPER",
        "continuation_policy": {
            "prediction_quality_and_candidate_independent_evidence_may_continue": True,
            "economic_replay_may_run_only_for_settlement_evaluable_dates": True,
            "partial-date_replay_is_diagnostic_not_formal_model_selection": True,
            "formal_a1_a2_mcs_lcb_winner_blocked_when_any_date_is_invalid": bool(
                incomplete_dates
            ),
            "no_candidate_cost_is_fabricated_for_invalid_dates": True,
        },
        "incomplete_operating_dates": incomplete_dates,
        "records": records,
    }
    payload["coverage_fingerprint"] = sha256_json(payload)
    return payload


def audit_referenced_nyiso_settlement_coverage(
    *,
    repository_root: Path,
    phase: str,
    dates: Sequence[str],
    referenced_nyiso_data_root: str | Path,
) -> dict[str, Any]:
    """Load referenced partitions and return an exact-date settlement ledger."""

    root = Path(repository_root).resolve()
    data_root = _resolve(referenced_nyiso_data_root, root=root)
    maximum_year = 2024 if phase == "selection" else 2025 if phase == "locked_test" else None
    if maximum_year is None:
        raise ValueError("phase must be selection or locked_test")
    bundle = ReferencedNYISOPanelSource(
        repository_root=root, data_root=data_root
    ).load_through_year(maximum_year)
    return _settlement_coverage_from_frames(
        phase=phase,
        dates=dates,
        da_15m=bundle.da_15m,
        rt_15m=bundle.rt_15m,
        source_sha256=str(bundle.source_manifest["partitions_sha256"]),
    )


def exact_phase_dates(phase: str) -> tuple[str, ...]:
    years = (2022, 2023, 2024) if phase == "selection" else (2025,) if phase == "locked_test" else ()
    if not years:
        raise ValueError("phase must be selection or locked_test")
    dates: list[str] = []
    for year in years:
        values = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="UTC")
        dates.extend(value.date().isoformat() for value in values)
    return tuple(dates)


def _resolve(path: str | Path, *, root: Path, sibling: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    root_value = (root / value).resolve()
    if root_value.exists() or sibling is None:
        return root_value
    return (sibling / value).resolve()


def _load_index(locator: str | Path | Mapping[str, Any] | None, *, root: Path) -> tuple[dict[str, Any], Path | None]:
    if locator is None:
        return {}, None
    if isinstance(locator, Mapping):
        if "index_path" not in locator:
            payload = dict(locator)
            return payload, None
        locator = locator["index_path"]
    path = _resolve(locator, root=root)
    if not path.is_file():
        return {}, path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DARTRealInputPreparationError(f"index root must be a mapping: {path}")
    return payload, path


def _records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("days", payload)
    if isinstance(raw, Mapping):
        return {str(day): value for day, value in raw.items() if isinstance(value, Mapping)}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        output: dict[str, Mapping[str, Any]] = {}
        for value in raw:
            if not isinstance(value, Mapping) or "operating_date" not in value:
                raise DARTRealInputPreparationError("daily index record lacks operating_date")
            day = str(value["operating_date"])
            if day in output:
                raise DARTRealInputPreparationError(f"duplicate daily index record: {day}")
            output[day] = value
        return output
    raise DARTRealInputPreparationError("index days must be a mapping or list")


def _npz_record(
    record: Mapping[str, Any], *, root: Path, index_path: Path | None, label: str
) -> tuple[Path, str, tuple[str, ...], dict[str, np.ndarray]]:
    if "path" not in record:
        raise DARTRealInputPreparationError(f"{label} record lacks path")
    path = _resolve(record["path"], root=root, sibling=index_path.parent if index_path else None)
    if not path.is_file():
        raise DARTRealInputPreparationError(f"{label} artifact is absent: {path}")
    digest = sha256_file(path)
    if record.get("sha256") is not None and str(record["sha256"]) != digest:
        raise DARTRealInputPreparationError(f"{label} artifact hash changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        fields = tuple(data.files)
        arrays = {name: data[name] for name in fields}
    return path, digest, fields, arrays


def _source_hashes(value: np.ndarray, *, label: str) -> dict[str, str]:
    parsed = json.loads(str(value.item()))
    if not isinstance(parsed, dict) or not parsed:
        raise DARTRealInputPreparationError(f"{label} lacks source hashes")
    for name, digest in parsed.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise DARTRealInputPreparationError(f"{label} source hash {name} is invalid")
    return {str(name): str(digest).lower() for name, digest in parsed.items()}


def _timestamp_panel_sha(timestamps: pd.DatetimeIndex) -> str:
    return sha256_json([value.isoformat() for value in timestamps])


def _validate_issue(day: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    required = {
        "issue_time_utc", "operating_date", "forecast_id", "timestamps_utc",
        "da_price", "rt_price_pool", "workload_pool", "pv_pool", "zones",
        "source_hashes_json",
        "locked_test",
    }
    missing = required.difference(arrays)
    if missing:
        raise DARTRealInputPreparationError(f"issue-time {day} lacks {sorted(missing)}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(arrays["timestamps_utc"], utc=True))
    issue = pd.Timestamp(str(arrays["issue_time_utc"].item()))
    if issue.tzinfo is None:
        raise DARTRealInputPreparationError(f"issue-time {day} is timezone-naive")
    issue = issue.tz_convert("UTC")
    if str(arrays["operating_date"].item()) != day or not len(timestamps) == 96:
        raise DARTRealInputPreparationError(f"issue-time {day} date/horizon mismatch")
    if not (issue < timestamps.min()) or not timestamps.is_unique:
        raise DARTRealInputPreparationError(f"issue-time {day} violates causal chronology")
    draws = int(np.asarray(arrays["rt_price_pool"]).shape[0])
    expected = (len(NYISO_ZONES), 96)
    if (
        tuple(map(str, arrays["zones"].tolist())) != tuple(NYISO_ZONES)
        or np.asarray(arrays["da_price"]).shape != expected
        or draws < 100
        or np.asarray(arrays["rt_price_pool"]).shape != (draws, *expected)
        or np.asarray(arrays["workload_pool"]).shape != (draws, 96)
        or np.asarray(arrays["pv_pool"]).shape != (draws, *expected)
    ):
        raise DARTRealInputPreparationError(f"issue-time {day} array schema mismatch")
    _source_hashes(arrays["source_hashes_json"], label=f"issue-time {day}")
    if bool(arrays["locked_test"].item()) is not (date.fromisoformat(day).year == 2025):
        raise DARTRealInputPreparationError(f"issue-time {day} phase role is wrong")
    return {
        "issue_time_utc": issue.isoformat(),
        "draw_count": draws,
        "timestamp_panel_sha256": _timestamp_panel_sha(timestamps),
    }


def _validate_forcing(day: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    required = {
        "operating_date", "timestamps_utc", "workload", "pv", "physical_multiplier",
        "source_hashes_json", "candidate_independent", "selected_scenario_as_realized",
        "evidence_roles_json",
    }
    missing = required.difference(arrays)
    if missing:
        raise DARTRealInputPreparationError(f"forcing {day} lacks {sorted(missing)}")
    if str(arrays["operating_date"].item()) != day:
        raise DARTRealInputPreparationError(f"forcing {day} operating_date mismatch")
    if bool(arrays["candidate_independent"].item()) is not True:
        raise DARTRealInputPreparationError(f"forcing {day} is candidate-dependent")
    if bool(arrays["selected_scenario_as_realized"].item()) is not False:
        raise DARTRealInputPreparationError(f"forcing {day} selects a scenario as realised")
    timestamps = pd.DatetimeIndex(pd.to_datetime(arrays["timestamps_utc"], utc=True))
    if len(timestamps) != 96 or not timestamps.is_unique:
        raise DARTRealInputPreparationError(f"forcing {day} horizon mismatch")
    if (
        np.asarray(arrays["workload"]).shape != (96,)
        or np.asarray(arrays["pv"]).shape != (len(NYISO_ZONES), 96)
        or np.asarray(arrays["physical_multiplier"]).shape != (len(NYISO_ZONES),)
    ):
        raise DARTRealInputPreparationError(f"forcing {day} array schema mismatch")
    _source_hashes(arrays["source_hashes_json"], label=f"forcing {day}")
    evidence_roles = json.loads(str(arrays["evidence_roles_json"].item()))
    if set(evidence_roles) != {"workload", "pv", "physical_multiplier"}:
        raise DARTRealInputPreparationError(f"forcing {day} evidence roles are incomplete")
    return {"timestamp_panel_sha256": _timestamp_panel_sha(timestamps)}


def _validate_actionability(day: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    required = {
        "operating_date", "timestamps_utc", "feasible_flexible_energy_mwh_by_event",
        "normalized_feasibility_weight_by_event", "lower_no_action_bound",
        "upper_no_action_bound", "available_at_gate", "source_fields_json",
        "source_hashes_json",
    }
    missing = required.difference(arrays)
    if missing:
        raise DARTRealInputPreparationError(f"actionability {day} lacks {sorted(missing)}")
    shape = (len(NYISO_ZONES), 96)
    feasible = np.asarray(arrays["feasible_flexible_energy_mwh_by_event"], dtype=float)
    weight = np.asarray(arrays["normalized_feasibility_weight_by_event"], dtype=float)
    lower = np.asarray(arrays["lower_no_action_bound"], dtype=float)
    upper = np.asarray(arrays["upper_no_action_bound"], dtype=float)
    available = np.asarray(arrays["available_at_gate"], dtype=bool)
    if any(value.shape != shape for value in (feasible, weight, lower, upper, available)):
        raise DARTRealInputPreparationError(f"actionability {day} array schema mismatch")
    if (
        np.any(feasible < 0) or np.any((weight < 0) | (weight > 1))
        or np.any((feasible <= 0) & (weight > 0))
        or np.any(lower > 0) or np.any(upper < 0) or np.any(lower > upper)
    ):
        raise DARTRealInputPreparationError(f"actionability {day} feasibility/band gate failed")
    fields = json.loads(str(arrays["source_fields_json"].item()))
    _source_hashes(arrays["source_hashes_json"], label=f"actionability {day}")
    missing_fields = sorted(_REQUIRED_ACTION_SOURCE_FIELDS.difference(map(str, fields)))
    if missing_fields:
        raise DARTRealInputPreparationError(
            f"actionability {day} lacks required source fields {missing_fields}"
        )
    if not np.all(available):
        raise DARTRealInputPreparationError(f"actionability {day} uses unavailable gate fields")
    timestamps = pd.DatetimeIndex(pd.to_datetime(arrays["timestamps_utc"], utc=True))
    if len(timestamps) != 96 or not timestamps.is_unique:
        raise DARTRealInputPreparationError(f"actionability {day} timestamp horizon mismatch")
    return {
        "positive_feasibility_events": int(np.count_nonzero(feasible > 0)),
        "timestamp_panel_sha256": _timestamp_panel_sha(timestamps),
    }


def _audit_kind(
    kind: str,
    *,
    locator: str | Path | Mapping[str, Any] | None,
    expected_dates: Sequence[str],
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload, index_path = _load_index(locator, root=root)
    records = _records(payload) if payload else {}
    expected = tuple(map(str, expected_dates))
    missing = sorted(set(expected).difference(records))
    extras = sorted(set(records).difference(expected))
    complete: list[dict[str, Any]] = []
    failures: list[str] = []
    validator = {"issue_time": _validate_issue, "forcing": _validate_forcing, "actionability": _validate_actionability}[kind]
    for day in expected:
        record = records.get(day)
        if record is None:
            continue
        try:
            path, digest, _, arrays = _npz_record(
                record, root=root, index_path=index_path, label=f"{kind} {day}"
            )
            diagnostics = validator(day, arrays)
            complete.append(
                {
                    "operating_date": day,
                    "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix(),
                    "sha256": digest,
                    **diagnostics,
                }
            )
        except Exception as exc:
            failures.append(f"{day}: {type(exc).__name__}: {exc}")
    status = "COMPLETE" if not missing and not extras and not failures and len(complete) == len(expected) else "BLOCKED"
    return {
        "status": status,
        "index_path": None if index_path is None else index_path.as_posix(),
        "expected_date_count": len(expected),
        "valid_date_count": len(complete),
        "missing_date_count": len(missing),
        "missing_dates_preview": missing[:12],
        "extra_date_count": len(extras),
        "extra_dates_preview": extras[:12],
        "invalid_date_count": len(failures),
        "failures_preview": failures[:12],
    }, complete


def prepare_dart_real_evidence_inputs(
    *,
    repository_root: Path,
    output_root: Path,
    phase_dates: Mapping[str, Sequence[str]],
    issue_time_index_by_phase: Mapping[str, Any],
    replay_forcing_index_by_phase: Mapping[str, Any],
    actionability_index_by_phase: Mapping[str, Any],
    referenced_nyiso_data_root: str | Path = "data/processed/referenced/nyiso",
    require_full_phase_dates: bool = True,
    source_inventory: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Audit exact panels and emit a config-ready block only if all pass."""

    root = Path(repository_root).resolve()
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    nyiso_root = _resolve(referenced_nyiso_data_root, root=root)
    nyiso_available = nyiso_root.is_dir()
    phase_reports: dict[str, Any] = {}
    merged: dict[str, list[dict[str, Any]]] = {
        "issue_time_ensembles_by_date": [],
        "replay_forcing_by_date": [],
        "actionability_by_date": [],
    }
    for phase in ("selection", "locked_test"):
        dates = tuple(map(str, phase_dates.get(phase, ())))
        if not dates:
            continue
        expected_dates = exact_phase_dates(phase)
        years = {date.fromisoformat(day).year for day in dates}
        expected_years = {2022, 2023, 2024} if phase == "selection" else {2025}
        if (
            years != expected_years
            or len(dates) != len(set(dates))
            or (require_full_phase_dates and dates != expected_dates)
        ):
            raise DARTRealInputPreparationError(f"{phase} dates do not cover exact years")
        reports: dict[str, Any] = {}
        phase_records: dict[str, list[dict[str, Any]]] = {}
        for kind, locators, key in (
            ("issue_time", issue_time_index_by_phase, "issue_time_ensembles_by_date"),
            ("forcing", replay_forcing_index_by_phase, "replay_forcing_by_date"),
            ("actionability", actionability_index_by_phase, "actionability_by_date"),
        ):
            reports[kind], records = _audit_kind(
                kind, locator=locators.get(phase), expected_dates=dates, root=root
            )
            phase_records[kind] = records
            merged[key].extend(records)
        timestamp_hashes: dict[str, dict[str, str]] = {
            kind: {
                record["operating_date"]: record["timestamp_panel_sha256"]
                for record in records
            }
            for kind, records in phase_records.items()
        }
        panel_failures = [
            day
            for day in dates
            if len(
                {
                    timestamp_hashes[kind].get(day)
                    for kind in ("issue_time", "forcing", "actionability")
                }
            )
            != 1
            or None in {
                timestamp_hashes[kind].get(day)
                for kind in ("issue_time", "forcing", "actionability")
            }
        ]
        cross_input_panel = {
            "status": "COMPLETE" if not panel_failures else "BLOCKED",
            "mismatch_or_missing_date_count": len(panel_failures),
            "mismatch_or_missing_dates_preview": panel_failures[:12],
        }
        if nyiso_available:
            try:
                settlement_coverage = audit_referenced_nyiso_settlement_coverage(
                    repository_root=root,
                    phase=phase,
                    dates=dates,
                    referenced_nyiso_data_root=nyiso_root,
                )
            except Exception as exc:
                settlement_coverage = {
                    "schema_version": SETTLEMENT_COVERAGE_SCHEMA_VERSION,
                    "status": "BLOCKED",
                    "phase": phase,
                    "expected_operating_date_count": len(dates),
                    "retained_operating_date_count": len(dates),
                    "date_drop_count": 0,
                    "missing_settlement_imputed": False,
                    "missing_dates_silently_dropped": False,
                    "formal_full_panel_decision_selection_complete": False,
                    "mcs_lcb_and_winner_defined": False,
                    "terminal_paper_cap": "PARTIAL_PAPER",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        else:
            settlement_coverage = {
                "schema_version": SETTLEMENT_COVERAGE_SCHEMA_VERSION,
                "status": "BLOCKED",
                "phase": phase,
                "expected_operating_date_count": len(dates),
                "retained_operating_date_count": len(dates),
                "date_drop_count": 0,
                "missing_settlement_imputed": False,
                "missing_dates_silently_dropped": False,
                "formal_full_panel_decision_selection_complete": False,
                "mcs_lcb_and_winner_defined": False,
                "terminal_paper_cap": "PARTIAL_PAPER",
                "error_type": "FileNotFoundError",
                "error": f"referenced NYISO root is absent: {nyiso_root}",
            }
        settlement_path = destination / f"{phase}_nyiso_settlement_coverage.json"
        atomic_json(settlement_path, settlement_coverage)
        independent_complete = bool(
            all(value["status"] == "COMPLETE" for value in reports.values())
            and not panel_failures
        )
        if independent_complete and settlement_coverage["status"] == "COMPLETE":
            phase_status = "COMPLETE"
        elif independent_complete and settlement_coverage["status"] == "PARTIAL_SOURCE_COVERAGE":
            phase_status = "PARTIAL_SOURCE_COVERAGE"
        else:
            phase_status = "BLOCKED"
        phase_reports[phase] = {
            "status": phase_status,
            "date_count": len(dates),
            "date_panel_sha256": sha256_json(list(dates)),
            "inputs": reports,
            "cross_input_timestamp_panel": cross_input_panel,
            "settlement_coverage": settlement_coverage,
            "settlement_coverage_path": (
                settlement_path.relative_to(root).as_posix()
                if settlement_path.is_relative_to(root)
                else settlement_path.as_posix()
            ),
        }
    source_files: dict[str, Any] = {}
    for label, raw_path in sorted(dict(source_inventory or {}).items()):
        source_path = _resolve(raw_path, root=root)
        source_files[str(label)] = {
            "path": source_path.as_posix(),
            "exists": source_path.is_file(),
            "byte_count": source_path.stat().st_size if source_path.is_file() else None,
            "sha256": sha256_file(source_path) if source_path.is_file() else None,
        }
    phase_statuses = {value["status"] for value in phase_reports.values()}
    if nyiso_available and phase_reports and phase_statuses == {"COMPLETE"}:
        status = "COMPLETE"
    elif phase_reports and phase_statuses.issubset(
        {"COMPLETE", "PARTIAL_SOURCE_COVERAGE"}
    ) and "PARTIAL_SOURCE_COVERAGE" in phase_statuses:
        status = "PARTIAL_SOURCE_COVERAGE"
    else:
        status = "BLOCKED"
    independent_inputs_complete = bool(phase_reports) and all(
        all(
            report["status"] == "COMPLETE"
            for report in value["inputs"].values()
        )
        and value["cross_input_timestamp_panel"]["status"] == "COMPLETE"
        for value in phase_reports.values()
    )
    output_indices: dict[str, str] = {}
    if independent_inputs_complete:
        for key, records in merged.items():
            path = destination / f"{key}_index.json"
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE",
                "day_count": len(records),
                "days": sorted(records, key=lambda item: item["operating_date"]),
            }
            payload["records_sha256"] = sha256_json(payload["days"])
            atomic_json(path, payload)
            output_indices[key] = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
    inputs = None
    if independent_inputs_complete:
        inputs = {
            "issue_time_ensembles_by_date": {"index_path": output_indices["issue_time_ensembles_by_date"]},
            "replay_forcing_by_date": {"index_path": output_indices["replay_forcing_by_date"]},
            "actionability_by_date": {"index_path": output_indices["actionability_by_date"]},
            "referenced_nyiso_data_root": nyiso_root.relative_to(root).as_posix() if nyiso_root.is_relative_to(root) else nyiso_root.as_posix(),
            "settlement_coverage_by_phase": {
                phase: {
                    "path": report["settlement_coverage_path"],
                    "sha256": sha256_file(
                        _resolve(report["settlement_coverage_path"], root=root)
                    ),
                    "status": report["settlement_coverage"]["status"],
                }
                for phase, report in phase_reports.items()
            },
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": utc_now(),
        "synthetic_fallback_used": False,
        "selected_scenario_as_realized": False,
        "missing_settlement_imputed": False,
        "missing_dates_silently_dropped": False,
        "formal_model_selection_requires_complete_settlement_panel": True,
        "candidate_independent_input_indices_emitted": independent_inputs_complete,
        "terminal_paper_cap": (
            "PARTIAL_PAPER" if status == "PARTIAL_SOURCE_COVERAGE" else None
        ),
        "candidate_independent_inputs_required": True,
        "full_calendar_phase_dates_required": require_full_phase_dates,
        "referenced_nyiso": {"status": "AVAILABLE" if nyiso_available else "MISSING", "root": nyiso_root.as_posix()},
        "source_inventory": source_files,
        "phases": phase_reports,
        "dart_real_evidence_inputs": inputs,
        "output_indices": output_indices,
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    atomic_json(destination / "dart_real_evidence_input_preparation_manifest.json", manifest)
    return manifest
