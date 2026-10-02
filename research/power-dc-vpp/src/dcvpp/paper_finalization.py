from __future__ import annotations

import calendar
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from dcvpp.core import PipelineState, atomic_json, atomic_text, sha256_file, utc_now
from dcvpp.paper_status import DEMO_COMPLETE_PAPER_BLOCKED, PAPER_PREFLIGHT_READY


COMPLETE_PAPER = "COMPLETE_PAPER"
PARTIAL_PAPER = "PARTIAL_PAPER"
FAILED = "FAILED"

_REAL_SOURCE = "third_party_referenced_public_data"
_NOT_STUDY_MEASURED = "not_measured_by_this_study"
_HYPOTHESIS_IDS = tuple(f"H{index}" for index in range(7))
_FINISHED_HYPOTHESIS_STATUSES = {
    "supported",
    "not_supported",
    "evaluated",
    "rejected",
}

_OFAT_DIMENSIONS = {
    "flexible_share",
    "deadline_steps",
    "migration_eligibility",
    "migration_bandwidth",
    "transfer_energy_checkpoint_overhead",
    "workload_scale",
    "battery_duration",
    "pv_capacity",
    "pue_slope",
    "physical_parameter_ensemble",
    "climate_support_distance",
    "site_capacity_hardware_transfer",
    "rt_volatility",
    "negative_price_clipping",
    "rt_sell_back_haircut",
    "scenario_count",
    "risk_alpha",
    "risk_weight_frontier",
    "adverse_coupling",
    "stress_scenario_mass",
    "ramp_concentration",
}
_SENSITIVITY_SETTING_KEYS = {
    "flexible_share",
    "deadline_steps",
    "migration_eligible",
    "migration_bandwidth_multiplier",
    "migration_bandwidth_base_work_units_per_interval",
    "transfer_energy_mwh_per_work_unit",
    "checkpoint_overhead_fraction",
    "workload_scale",
    "battery_duration_hours",
    "pv_capacity_multiplier",
    "pue_slope_multiplier",
    "physical_ensemble_selector",
    "physical_multiplier_scale",
    "support_distance_threshold",
    "out_of_support_capacity_derating",
    "site_capacity_multiplier",
    "hardware_transfer_multiplier",
    "rt_volatility_scale",
    "negative_price_floor_usd_per_mwh",
    "rt_sell_back_haircut",
    "cvar_alpha",
    "risk_weight",
    "stress_mass",
    "ramp_limit_mw_per_step",
    "concentration_share_limit",
}
_POLICY_INFERENCE_METRICS = {
    "expected_total_cost_usd",
    "cvar_total_cost_usd",
    "expected_sla_shortfall_work_units",
    "expected_grid_energy_mwh",
    "rebound_peak_mw",
    "maximum_zone_concentration_share",
}


def _status(payload: Mapping[str, Any] | None) -> str:
    return str((payload or {}).get("status", "missing")).strip().lower()


def _resolve(root: Path, raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, Mapping):
        raw = raw.get("path") or raw.get("local_path")
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    preferred_fields: Sequence[str] = (),
) -> Path:
    fields = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    if not fields:
        raise ValueError(f"cannot write a CSV without a schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    temporary.replace(path)
    return path


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    required_for_complete: bool = True,
    evidence: Any = None,
    message: str = "",
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "passed" if passed else "failed",
            "required_for_complete": required_for_complete,
            "message": message,
            "evidence": evidence,
        }
    )


def _result_path(
    root: Path,
    result: Mapping[str, Any],
    *keys: str,
) -> Path | None:
    for key in keys:
        value: Any = result
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        path = _resolve(root, value)
        if path is not None:
            return path
    return None


def _verify_output_map(
    root: Path,
    outputs: Mapping[str, Any] | None,
    hashes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    outputs = outputs if isinstance(outputs, Mapping) else {}
    hashes = hashes if isinstance(hashes, Mapping) else {}
    failures: list[dict[str, Any]] = []
    verified = 0
    for name, raw_path in outputs.items():
        path = _resolve(root, raw_path)
        expected = hashes.get(name)
        if path is None or not path.is_file():
            failures.append({"name": name, "reason": "missing_output", "path": str(raw_path)})
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            failures.append(
                {"name": name, "reason": "missing_expected_sha256", "path": _relative(root, path)}
            )
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                {
                    "name": name,
                    "reason": "sha256_mismatch",
                    "path": _relative(root, path),
                    "expected": expected,
                    "actual": actual,
                }
            )
            continue
        verified += 1
    return {
        "passed": bool(outputs) and verified == len(outputs) and not failures,
        "verified_count": verified,
        "declared_count": len(outputs),
        "failures": failures,
    }


def _verify_output_records(root: Path, records: Any) -> dict[str, Any]:
    if not isinstance(records, list):
        records = []
    failures: list[dict[str, Any]] = []
    verified = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            failures.append({"index": index, "reason": "invalid_output_record"})
            continue
        path = _resolve(root, record.get("path"))
        expected = record.get("sha256")
        if path is None or not path.is_file():
            failures.append(
                {"index": index, "reason": "missing_output", "path": record.get("path")}
            )
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            failures.append(
                {"index": index, "reason": "missing_expected_sha256", "path": record.get("path")}
            )
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                {
                    "index": index,
                    "reason": "sha256_mismatch",
                    "path": record.get("path"),
                    "expected": expected,
                    "actual": actual,
                }
            )
            continue
        verified += 1
    return {
        "passed": bool(records) and verified == len(records) and not failures,
        "verified_count": verified,
        "declared_count": len(records),
        "failures": failures,
    }


def _true_synthetic_flags(value: Any, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            child = f"{path}.{key}"
            normalized = key.lower()
            if normalized in {
                "synthetic",
                "synthetic_fallback",
                "synthetic_fallback_used",
                "uses_synthetic_data",
            } and nested is True:
                found.append(child)
            elif normalized == "profile" and str(nested).lower() == "demo":
                found.append(child)
            found.extend(_true_synthetic_flags(nested, child))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_true_synthetic_flags(nested, f"{path}[{index}]"))
    return found


def _source_semantics_audit(source_manifest: Any) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(source_manifest, Mapping):
        return {"passed": False, "failures": ["source manifest is missing or invalid"]}
    policy = source_manifest.get("source_relationship_policy", {})
    if not isinstance(policy, Mapping) or policy.get("study_measurement_claimed") is not False:
        failures.append("source relationship policy does not explicitly deny study measurement")
    sources = source_manifest.get("sources", [])
    if not isinstance(sources, list) or not sources:
        failures.append("source records are absent")
        sources = []
    nasa_count = 0
    nyiso_count = 0
    xcel_count = 0
    for record in sources:
        if not isinstance(record, Mapping):
            failures.append("non-object source record")
            continue
        source_id = str(record.get("source_id", ""))
        relationship = str(record.get("source_relationship", ""))
        measurement = record.get("measurement_relationship")
        if "measured_by_this_study" in str(measurement) and measurement != _NOT_STUDY_MEASURED:
            failures.append(f"{source_id}: contradictory measurement relationship {measurement!r}")
        if source_id.startswith("nasa_power_"):
            nasa_count += 1
            if relationship != _REAL_SOURCE or record.get("record_type") != "referenced_gridded_reconstruction":
                failures.append(f"{source_id}: NASA POWER semantics are not reconstructed/reference-only")
        elif source_id.startswith("nyiso_"):
            nyiso_count += 1
            if relationship != _REAL_SOURCE or record.get("record_type") != "referenced_official_historical_market_or_load_record":
                failures.append(f"{source_id}: NYISO semantics are not official historical reference-only")
        elif source_id.startswith("xcel_"):
            xcel_count += 1
            if relationship != "third_party_referenced_public_document" or measurement != _NOT_STUDY_MEASURED:
                failures.append(f"{source_id}: Xcel semantics are not official referenced-rule only")
    if nasa_count == 0:
        failures.append("NASA POWER source records are absent")
    if nyiso_count == 0:
        failures.append("NYISO source records are absent")
    if xcel_count == 0:
        failures.append("Xcel source records are absent")
    return {
        "passed": not failures,
        "failures": failures,
        "record_counts": {"nasa_power": nasa_count, "nyiso": nyiso_count, "xcel": xcel_count},
    }


def _physical_frozen_audit(
    root: Path,
    result: Mapping[str, Any],
    calibration_manifest: Any,
    transfer_manifest: Any,
) -> dict[str, Any]:
    failures: list[str] = []
    frozen = result.get("frozen_artifact")
    if not isinstance(frozen, Mapping):
        failures.append("physical result has no frozen artifact")
        frozen = {}
    if frozen.get("frozen") is not True:
        failures.append("physical artifact is not marked frozen")
    if not frozen.get("calibration_id"):
        failures.append("frozen physical artifact lacks calibration_id")
    if len(frozen.get("twin_ids", [])) != 4:
        failures.append("frozen physical artifact does not contain four transfer twins")
    if not isinstance(frozen.get("source_hashes"), Mapping) or not frozen.get("source_hashes"):
        failures.append("frozen physical artifact lacks source hashes")
    if not isinstance(calibration_manifest, Mapping) or calibration_manifest.get("frozen") is not True:
        failures.append("physical calibration manifest is not frozen")
    if not isinstance(transfer_manifest, Mapping):
        failures.append("digital-twin transfer manifest is missing")
    else:
        if transfer_manifest.get("counterfactual") is not True:
            failures.append("transfer manifest does not preserve the counterfactual boundary")
        transfer_hashes = _verify_output_map(
            root,
            transfer_manifest.get("outputs"),
            transfer_manifest.get("output_hashes"),
        )
        if not transfer_hashes["passed"]:
            failures.append("transfer output hashes do not verify")
    return {"passed": not failures, "failures": failures}


def _load_hypotheses(root: Path, optimization_result: Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    path = _result_path(root, optimization_result, "paths.hypotheses", "hypotheses_path")
    payload = _load_json(path)
    hypotheses = payload.get("hypotheses", {}) if isinstance(payload, Mapping) else {}
    return (dict(hypotheses) if isinstance(hypotheses, Mapping) else {}, path)


def _hypothesis_audit(hypotheses: Mapping[str, Any]) -> dict[str, Any]:
    states: dict[str, str] = {}
    failures: list[str] = []
    for hypothesis_id in _HYPOTHESIS_IDS:
        record = hypotheses.get(hypothesis_id)
        if not isinstance(record, Mapping):
            states[hypothesis_id] = "missing"
            failures.append(f"{hypothesis_id} is missing")
            continue
        status = str(record.get("status", "missing")).lower()
        states[hypothesis_id] = status
        if status not in _FINISHED_HYPOTHESIS_STATUSES:
            failures.append(f"{hypothesis_id} is {status}")
    return {"passed": not failures, "states": states, "failures": failures}


def _cell_audit(
    root: Path,
    config: Mapping[str, Any],
    optimization_result: Mapping[str, Any],
) -> dict[str, Any]:
    path = _result_path(root, optimization_result, "paths.cell_status", "cell_status_path")
    raw = _load_json(path)
    cells = [dict(item) for item in raw] if isinstance(raw, list) else []
    by_phase: dict[str, Counter[str]] = {}
    dates_by_phase: dict[str, set[str]] = {}
    cells_by_phase: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        phase = str(cell.get("phase", "unknown"))
        status = str(cell.get("current_status", "PENDING")).upper()
        by_phase.setdefault(phase, Counter())[status] += 1
        dates_by_phase.setdefault(phase, set()).add(str(cell.get("operating_date")))
        cells_by_phase.setdefault(phase, []).append(cell)

    primary = by_phase.get("primary_96_day", Counter())
    primary_dates = dates_by_phase.get("primary_96_day", set())
    primary_count = sum(primary.values())
    primary_month_counts = Counter(
        value[:7] for value in primary_dates if len(value) >= 7 and value[:4].isdigit()
    )
    primary_complete = (
        len(primary_dates) == 96
        and len(primary_month_counts) == 12
        and set(primary_month_counts.values()) == {8}
        and primary_count > 0
        and primary.get("SUCCESS", 0) == primary_count
    )

    optimization_cfg = config.get("optimization", {})
    required_extension = bool(
        optimization_cfg.get("full_year_extension", False)
        if isinstance(optimization_cfg, Mapping)
        else False
    )
    extension = by_phase.get("full_year_extension", Counter())
    extension_dates = dates_by_phase.get("full_year_extension", set())
    target_year = 2025
    branch_cfg = config.get("optimization_branch", {})
    if isinstance(branch_cfg, Mapping):
        target_year = int(branch_cfg.get("target_year", target_year))
    expected_year_days = 366 if calendar.isleap(target_year) else 365
    all_dates = primary_dates | extension_dates
    expected_dates = {
        value.date().isoformat()
        for value in pd.date_range(
            f"{target_year:04d}-01-01",
            f"{target_year:04d}-12-31",
            freq="D",
        )
    }
    extension_complete = not required_extension or (
        bool(optimization_result.get("full_year_extension_requested"))
        and bool(optimization_result.get("full_year_extension_gate_passed"))
        and bool(optimization_result.get("full_year_extension_started"))
        and bool(extension)
        and extension.get("SUCCESS", 0) == sum(extension.values())
        and all_dates == expected_dates
    )
    reported_planned = int(optimization_result.get("planned_cell_count", 0) or 0)
    reported_complete = int(optimization_result.get("completed_count", 0) or 0)
    reported_failures = int(optimization_result.get("failure_count", 0) or 0)
    reported_pending = int(optimization_result.get("pending_count", 0) or 0)
    observed_success = sum(counter.get("SUCCESS", 0) for counter in by_phase.values())
    observed_failed = sum(counter.get("FAILED", 0) for counter in by_phase.values())
    observed_pending = sum(
        counter.get("PENDING", 0) + counter.get("RUNNING", 0)
        for counter in by_phase.values()
    )
    consistency = bool(cells) and (
        reported_planned == len(cells)
        and reported_complete == observed_success
        and reported_failures == observed_failed
        and reported_pending == observed_pending
    )
    branch_cfg = config.get("optimization_branch", {})
    branch_cfg = branch_cfg if isinstance(branch_cfg, Mapping) else {}
    expected_seeds = tuple(
        map(
            int,
            branch_cfg.get(
                "seed_list",
                config.get("seed_list", [config.get("seed", 20260809)]),
            ),
        )
    )
    expected_counts = tuple(map(int, branch_cfg.get("scenario_counts", (10, 20, 40, 80))))
    raw_regimes = branch_cfg.get("regimes", ("independent", "kappa_0_3", "kappa_0_6"))
    expected_regimes = tuple(
        str(item.get("name")) if isinstance(item, Mapping) else str(item)
        for item in raw_regimes
    )
    include_h6 = isinstance(branch_cfg.get("h6_constraints"), Mapping)
    primary_cells = cells_by_phase.get("primary_96_day", [])
    observed_seeds = {int(cell.get("seed")) for cell in primary_cells if cell.get("seed") is not None}
    observed_counts = {
        int(cell.get("reduced_count"))
        for cell in primary_cells
        if cell.get("reduced_count") is not None
    }
    observed_regimes = {str(cell.get("regime")) for cell in primary_cells}
    observed_policies = {str(cell.get("policy_id")) for cell in primary_cells}
    h6_cells = [
        cell
        for cell in primary_cells
        if cell.get("policy_id") == "B6"
        and cell.get("regime") == "independent"
        and int(cell.get("reduced_count", -1)) == 20
        and cell.get("constraint_variant") == "ramp_concentration"
    ]
    # Mirror runner._plan_cells exactly.  The nominal 20-scenario reference has
    # B0-B6 and B8 (B7 is adverse-mixture only), other counts contain nominal
    # B6, dependence sweeps occur only at 20 scenarios, and H6 adds one variant.
    expected_per_seed: set[tuple[str, int, str, str]] = set()
    for scenario_count in expected_counts:
        if "independent" in expected_regimes:
            nominal_policies = (
                tuple(f"B{index}" for index in range(7)) + ("B8",)
                if scenario_count == 20
                else ("B6",)
            )
            expected_per_seed.update(
                ("independent", scenario_count, policy, "base")
                for policy in nominal_policies
            )
            if scenario_count == 20 and include_h6:
                expected_per_seed.add(
                    ("independent", scenario_count, "B6", "ramp_concentration")
                )
        if scenario_count == 20:
            for regime in set(expected_regimes) - {"independent"}:
                expected_per_seed.add((regime, scenario_count, "B6", "base"))
                expected_per_seed.add((regime, scenario_count, "B7", "base"))
    expected_cells_per_day = len(expected_seeds) * len(expected_per_seed)
    observed_primary_per_day = Counter(
        str(cell.get("operating_date")) for cell in primary_cells
    )
    observed_extension_per_day = Counter(
        str(cell.get("operating_date"))
        for cell in cells_by_phase.get("full_year_extension", [])
    )

    def phase_design_exact(phase_cells: Sequence[Mapping[str, Any]]) -> bool:
        grouped: dict[tuple[str, int], Counter[tuple[str, int, str, str]]] = {}
        for cell in phase_cells:
            try:
                group_key = (str(cell.get("operating_date")), int(cell.get("seed")))
                signature = (
                    str(cell.get("regime")),
                    int(cell.get("reduced_count")),
                    str(cell.get("policy_id")),
                    str(cell.get("constraint_variant", "base")),
                )
            except (TypeError, ValueError):
                return False
            grouped.setdefault(group_key, Counter())[signature] += 1
        expected_counter = Counter(expected_per_seed)
        expected_group_count = len(
            {str(cell.get("operating_date")) for cell in phase_cells}
        ) * len(expected_seeds)
        return (
            bool(grouped)
            and len(grouped) == expected_group_count
            and {seed for _, seed in grouped} == set(expected_seeds)
            and all(counter == expected_counter for counter in grouped.values())
        )

    primary_design_exact = phase_design_exact(primary_cells)
    extension_design_exact = phase_design_exact(
        cells_by_phase.get("full_year_extension", [])
    )
    design_complete = (
        branch_cfg.get("policy_override") is None
        and len(expected_seeds) == 3
        and observed_seeds == set(expected_seeds)
        and observed_counts == {10, 20, 40, 80} == set(expected_counts)
        and observed_regimes == {"independent", "kappa_0_3", "kappa_0_6"}
        and observed_regimes == set(expected_regimes)
        and observed_policies == {f"B{index}" for index in range(9)}
        and include_h6
        and len(h6_cells) == len(primary_dates) * len(expected_seeds)
        and bool(observed_primary_per_day)
        and set(observed_primary_per_day.values()) == {expected_cells_per_day}
        and primary_design_exact
        and (
            not required_extension
            or (
                bool(observed_extension_per_day)
                and set(observed_extension_per_day.values()) == {expected_cells_per_day}
                and extension_design_exact
            )
        )
    )
    return {
        "path": _relative(root, path) if path and path.is_file() else None,
        "cell_count": len(cells),
        "by_phase": {phase: dict(counter) for phase, counter in by_phase.items()},
        "primary_day_count": len(primary_dates),
        "primary_month_counts": dict(sorted(primary_month_counts.items())),
        "primary_complete": primary_complete,
        "full_year_extension_required": required_extension,
        "extension_day_count": len(extension_dates),
        "year_day_count": len(all_dates),
        "expected_year_day_count": expected_year_days,
        "extension_complete": extension_complete,
        "summary_consistent": consistency,
        "preregistered_design_complete": design_complete,
        "expected_seeds": list(expected_seeds),
        "observed_seeds": sorted(observed_seeds),
        "expected_scenario_counts": list(expected_counts),
        "observed_scenario_counts": sorted(observed_counts),
        "expected_regimes": list(expected_regimes),
        "observed_regimes": sorted(observed_regimes),
        "observed_policies": sorted(observed_policies),
        "h6_primary_cell_count": len(h6_cells),
        "expected_cells_per_day": expected_cells_per_day,
        "expected_cells_per_seed_per_day": len(expected_per_seed),
        "primary_design_exact": primary_design_exact,
        "extension_design_exact": extension_design_exact,
        "observed": {
            "success": observed_success,
            "failed": observed_failed,
            "pending_or_running": observed_pending,
        },
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _declared_path_audit(
    root: Path,
    result: Mapping[str, Any],
    name: str,
) -> tuple[Path | None, bool, str | None]:
    path = _result_path(root, result, f"paths.{name}")
    expected = (
        result.get("path_hashes", {}).get(name)
        if isinstance(result.get("path_hashes"), Mapping)
        else None
    )
    if path is None or not path.is_file():
        return path, False, "artifact is missing"
    if not isinstance(expected, str) or len(expected) != 64:
        return path, False, "declared SHA-256 is missing"
    actual = sha256_file(path)
    if actual != expected:
        return path, False, "declared SHA-256 does not match"
    return path, True, None


def _ofat_sensitivity_audit(
    root: Path,
    optimization_result: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    plan_path, plan_hash_ok, plan_hash_reason = _declared_path_audit(
        root, optimization_result, "sensitivity_plan"
    )
    results_path, results_hash_ok, results_hash_reason = _declared_path_audit(
        root, optimization_result, "sensitivity_results"
    )
    table_path, table_hash_ok, table_hash_reason = _declared_path_audit(
        root, optimization_result, "sensitivity_table"
    )
    for name, passed, reason in (
        ("plan", plan_hash_ok, plan_hash_reason),
        ("results", results_hash_ok, results_hash_reason),
        ("table", table_hash_ok, table_hash_reason),
    ):
        if not passed:
            failures.append(f"sensitivity {name}: {reason}")
    plan = _load_json(plan_path)
    results = _load_json(results_path)
    variants = plan.get("variants", []) if isinstance(plan, Mapping) else []
    variants = [dict(item) for item in variants if isinstance(item, Mapping)]
    dimensions = {str(item.get("dimension")) for item in variants}
    variant_ids = [str(item.get("variant_id")) for item in variants]
    baselines = Counter(
        str(item.get("dimension"))
        for item in variants
        if item.get("baseline") is True
    )
    if not isinstance(plan, Mapping) or plan.get("enabled") is not True:
        failures.append("OFAT plan is absent or disabled")
    if int((plan or {}).get("primary_day_count", 0) or 0) != 96:
        failures.append("OFAT plan is not restricted to all 96 primary days")
    if (plan or {}).get("full_year_extension_includes_sensitivities") is not False:
        failures.append("OFAT sensitivities entered the full-year extension")
    if dimensions != _OFAT_DIMENSIONS:
        failures.append("OFAT dimensions do not match all preregistered section 14.4 dimensions")
    if len(variant_ids) != len(set(variant_ids)) or not variant_ids:
        failures.append("OFAT variant IDs are absent or non-unique")
    if set(baselines) != _OFAT_DIMENSIONS or set(baselines.values()) != {1}:
        failures.append("each OFAT dimension must have exactly one baseline")
    if any(not item.get("consumed_by") for item in variants):
        failures.append("one or more OFAT variants do not declare a real consumer")
    planned = len(variants) * 96
    if int((plan or {}).get("planned_cell_count", -1)) != planned:
        failures.append("OFAT planned-cell count does not equal 96 times the variant count")

    if not isinstance(results, Mapping) or results.get("status") != "COMPLETE" or results.get("completed") is not True:
        failures.append("OFAT result manifest is not complete")
    else:
        if int(results.get("planned_cell_count", -1)) != planned:
            failures.append("OFAT result planned-cell count disagrees with the plan")
        if int(results.get("success_result_count", -1)) != planned:
            failures.append("OFAT result manifest does not contain one success per planned cell")
        if results.get("full_year_extension_includes_sensitivities") is not False:
            failures.append("OFAT result manifest includes full-year sensitivity cells")
        if plan_path and results.get("plan_sha256") != sha256_file(plan_path):
            failures.append("OFAT result manifest plan hash does not verify")
        if table_path and results.get("result_table_sha256") != sha256_file(table_path):
            failures.append("OFAT result manifest table hash does not verify")
        status_by_dimension = results.get("status_by_dimension", {})
        if not isinstance(status_by_dimension, Mapping) or set(status_by_dimension) != _OFAT_DIMENSIONS:
            failures.append("OFAT result dimension statuses are incomplete")
        else:
            variant_counts = Counter(str(item.get("dimension")) for item in variants)
            for dimension, variant_count in variant_counts.items():
                status = status_by_dimension.get(dimension, {})
                if not isinstance(status, Mapping) or int(status.get("SUCCESS", -1)) != 96 * variant_count:
                    failures.append(f"{dimension}: success count does not cover all dates and levels")
                if any(int(status.get(key, 0) or 0) for key in ("FAILED", "RUNNING", "PENDING")):
                    failures.append(f"{dimension}: unfinished or failed cells remain")

    cell_path = _result_path(root, optimization_result, "paths.cell_status")
    cell_payload = _load_json(cell_path)
    sensitivity_cells = [
        dict(item)
        for item in (cell_payload if isinstance(cell_payload, list) else [])
        if isinstance(item, Mapping) and item.get("phase") == "primary_sensitivity"
    ]
    if len(sensitivity_cells) != planned:
        failures.append("cell status does not contain the complete OFAT plan")
    else:
        if {str(item.get("current_status")).upper() for item in sensitivity_cells} != {"SUCCESS"}:
            failures.append("not every OFAT cell succeeded")
        if len({str(item.get("operating_date")) for item in sensitivity_cells}) != 96:
            failures.append("OFAT cell status does not cover 96 unique dates")
        if len({int(item.get("seed", -1)) for item in sensitivity_cells}) != 1:
            failures.append("OFAT cell status does not use one declared study seed")
        pairs = {
            (str(item.get("operating_date")), str(item.get("variant_id")))
            for item in sensitivity_cells
        }
        if len(pairs) != planned:
            failures.append("OFAT date-by-variant cells are duplicated or missing")

    table = pd.DataFrame()
    if table_path and table_path.is_file():
        try:
            table = pd.read_csv(table_path)
        except Exception as exc:
            failures.append(f"sensitivity table cannot be read: {type(exc).__name__}: {exc}")
    required_columns = {
        "operating_date",
        "study_seed",
        "phase",
        "sensitivity_dimension",
        "variant_id",
        "sensitivity_setting_keys_supported",
        "sensitivity_settings_consumed",
        "sensitivity_variant",
        "sensitivity_effective_inputs",
        "sensitivity_parameters",
        "transformed_scenario_hashes",
    }
    if len(table) != planned or not required_columns.issubset(table.columns):
        failures.append("machine sensitivity table is incomplete or lacks consumption evidence")
    else:
        variants_by_id = {str(item["variant_id"]): item for item in variants}
        for index, row in table.iterrows():
            variant_id = str(row["variant_id"])
            declared = variants_by_id.get(variant_id)
            supported = _json_value(row["sensitivity_setting_keys_supported"])
            consumed = _json_value(row["sensitivity_settings_consumed"])
            variant_record = _json_value(row["sensitivity_variant"])
            effective = _json_value(row["sensitivity_effective_inputs"])
            parameters = _json_value(row["sensitivity_parameters"])
            hashes = _json_value(row["transformed_scenario_hashes"])
            valid = bool(
                declared
                and row["phase"] == "primary_sensitivity"
                and row["sensitivity_dimension"] == declared.get("dimension")
                and isinstance(supported, list)
                and set(map(str, supported)) == _SENSITIVITY_SETTING_KEYS
                and isinstance(consumed, Mapping)
                and isinstance(variant_record, Mapping)
                and variant_record.get("variant_id") == variant_id
                and isinstance(effective, Mapping)
                and bool(effective)
                and isinstance(parameters, Mapping)
                and parameters.get("consumed_by") == declared.get("consumed_by")
                and isinstance(hashes, Mapping)
                and set(hashes)
                == {"da_price", "rt_price", "workload", "pv", "physical_multiplier"}
                and all(isinstance(value, str) and len(value) == 64 for value in hashes.values())
            )
            expected_consumed = _SENSITIVITY_SETTING_KEYS.intersection(
                set((declared or {}).get("solve_overrides", {}))
                | set((declared or {}).get("scenario_transforms", {}))
                | set((declared or {}).get("physical_transforms", {}))
            )
            if expected_consumed and not expected_consumed.issubset(set(consumed or {})):
                valid = False
            if not valid:
                failures.append(f"sensitivity table row {index} lacks verified effective consumption")
                if len(failures) >= 25:
                    break
    return {
        "passed": not failures,
        "failures": failures,
        "dimension_count": len(dimensions),
        "variant_count_per_day": len(variants),
        "planned_cell_count": planned,
        "table_row_count": len(table),
        "paths": {
            "plan": _relative(root, plan_path) if plan_path and plan_path.is_file() else None,
            "results": _relative(root, results_path) if results_path and results_path.is_file() else None,
            "table": _relative(root, table_path) if table_path and table_path.is_file() else None,
        },
    }


def _statistical_inference_audit(
    root: Path,
    optimization_result: Mapping[str, Any],
    *,
    expected_seed_count: int,
) -> dict[str, Any]:
    failures: list[str] = []
    manifest_path, hash_ok, hash_reason = _declared_path_audit(
        root, optimization_result, "statistical_inference"
    )
    if not hash_ok:
        failures.append(f"statistical manifest: {hash_reason}")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("status") != "COMPLETE" or manifest.get("completed") is not True:
        failures.append("statistical inference manifest is not complete")
        manifest = {}
    if manifest.get("paired_daily_comparisons") is not True:
        failures.append("paired daily comparisons are not declared")
    block = manifest.get("moving_block_bootstrap", {})
    if not isinstance(block, Mapping) or block.get("circular_blocks") is not True or int(block.get("replicates", 0) or 0) < 100:
        failures.append("moving circular block bootstrap evidence is incomplete")
    if manifest.get("holm_correction_applied") is not True:
        failures.append("Holm multiplicity correction is absent")
    if not manifest.get("effect_size"):
        failures.append("paired effect-size definition is absent")
    if int(manifest.get("expected_primary_days", 0) or 0) != 96:
        failures.append("statistical manifest does not require 96 primary days")

    frames: dict[str, pd.DataFrame] = {}
    outputs = manifest.get("outputs", {})
    for name in (
        "paired_policy_inference",
        "sensitivity_effects",
        "forecast_dm_tests",
        "forecast_decision_linkage",
    ):
        item = outputs.get(name, {}) if isinstance(outputs, Mapping) else {}
        path = _resolve(root, item.get("path") if isinstance(item, Mapping) else None)
        expected = item.get("sha256") if isinstance(item, Mapping) else None
        if path is None or not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            failures.append(f"{name}: output or SHA-256 is invalid")
            frames[name] = pd.DataFrame()
            continue
        try:
            frames[name] = pd.read_csv(path)
        except Exception as exc:
            failures.append(f"{name}: unreadable CSV: {type(exc).__name__}: {exc}")
            frames[name] = pd.DataFrame()

    policy = frames.get("paired_policy_inference", pd.DataFrame())
    required_inference_columns = {
        "comparison_id",
        "metric",
        "analysis_role",
        "paired_day_count",
        "paired_date_seed_count",
        "block_bootstrap_lower",
        "block_bootstrap_upper",
        "p_value_raw",
        "p_value_holm",
        "effect_size_paired_dz",
        "null_hypothesis",
        "alternative_hypothesis",
    }
    expected_comparisons = {
        "B6_vs_B0",
        "B6_vs_B1",
        "B6_vs_B2",
        "B6_vs_B4",
        "B7_vs_B6_kappa_0_3",
        "B7_vs_B6_kappa_0_6",
        "B6_kappa_0_3_vs_nominal",
        "B6_kappa_0_6_vs_nominal",
        "B6_H6_vs_B6_base",
    }
    if policy.empty or not required_inference_columns.issubset(policy.columns):
        failures.append("paired policy inference table lacks registered machine fields")
    else:
        available_pairs = set(zip(policy["comparison_id"].astype(str), policy["metric"].astype(str)))
        required_pairs = {
            (comparison, metric)
            for comparison in expected_comparisons
            for metric in _POLICY_INFERENCE_METRICS
        }
        if not required_pairs.issubset(available_pairs):
            failures.append("paired policy inference is missing H3/H5/H6 comparison-metric rows")
        if not pd.to_numeric(policy["paired_day_count"], errors="coerce").eq(96).all():
            failures.append("paired policy inference does not use all 96 days")
        expected_pairs = 96 * int(expected_seed_count)
        if not pd.to_numeric(policy["paired_date_seed_count"], errors="coerce").eq(expected_pairs).all():
            failures.append("paired policy inference does not preserve every registered seed")
        if not policy["analysis_role"].astype(str).eq("confirmatory").all():
            failures.append("paired policy inference is not labeled confirmatory")
        numeric = policy[["block_bootstrap_lower", "block_bootstrap_upper", "p_value_raw", "p_value_holm", "effect_size_paired_dz"]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            failures.append("paired policy inference contains missing intervals, p-values, or effects")

    sensitivity = frames.get("sensitivity_effects", pd.DataFrame())
    if sensitivity.empty or not required_inference_columns.issubset(sensitivity.columns):
        failures.append("sensitivity inference table lacks registered machine fields")
    else:
        if not pd.to_numeric(sensitivity["paired_day_count"], errors="coerce").eq(96).all():
            failures.append("sensitivity inference does not use all 96 days")
        if not sensitivity["analysis_role"].astype(str).eq("exploratory").all():
            failures.append("sensitivity inference is not labeled exploratory")
        if set(sensitivity.get("sensitivity_dimension", pd.Series(dtype=str)).astype(str)) != _OFAT_DIMENSIONS:
            failures.append("sensitivity inference omits one or more OFAT dimensions")
        if int(manifest.get("exploratory_sensitivity_row_count", -1)) != len(sensitivity):
            failures.append("sensitivity inference row count disagrees with its manifest")

    forecast = frames.get("forecast_dm_tests", pd.DataFrame())
    if forecast.empty or set(forecast.get("task", pd.Series(dtype=str)).astype(str)) != {"nyiso_da", "nyiso_rt"}:
        failures.append("DA/RT DM-style forecast tests are incomplete")
    elif not {
        "horizon_aware_lag",
        "dm_style_statistic",
        "p_value_holm",
        "effect_size_standardized_mean_difference",
        "prediction_sha256",
    }.issubset(forecast.columns):
        failures.append("forecast DM table lacks horizon, Holm, effect, or provenance fields")

    decision = frames.get("forecast_decision_linkage", pd.DataFrame())
    if len(decision) != 96 or decision.get("operating_date", pd.Series(dtype=str)).astype(str).nunique() != 96:
        failures.append("forecast-decision linkage does not contain exactly 96 unique days")
    else:
        required_decision = {
            "point_error",
            "tail_calibration_error",
            "scenario_quality_error",
            "decision_value_usd",
            "decision_value_definition",
            "paired_seed_count",
            "forecast_prediction_evidence",
            "scenario_quality_evidence",
            "optimization_result_evidence",
        }
        if not required_decision.issubset(decision.columns):
            failures.append("forecast-decision linkage lacks traceable H4 fields")
        elif not pd.to_numeric(decision["paired_seed_count"], errors="coerce").eq(expected_seed_count).all():
            failures.append("forecast-decision linkage overwrote or omitted registered seeds")
        else:
            for index, row in decision.iterrows():
                forecast_evidence = _json_value(row["forecast_prediction_evidence"])
                optimization_evidence = _json_value(row["optimization_result_evidence"])
                scenario_evidence = _json_value(row["scenario_quality_evidence"])
                if not isinstance(forecast_evidence, list) or len(forecast_evidence) != 2:
                    failures.append(f"forecast-decision row {index} lacks DA/RT provenance")
                    break
                if not isinstance(optimization_evidence, list) or len(optimization_evidence) != 2 * expected_seed_count:
                    failures.append(f"forecast-decision row {index} lacks paired optimization provenance")
                    break
                if any(not isinstance(item, Mapping) or len(str(item.get("sha256", ""))) != 64 for item in forecast_evidence + optimization_evidence):
                    failures.append(f"forecast-decision row {index} has unverifiable source hashes")
                    break
                if not isinstance(scenario_evidence, Mapping):
                    failures.append(
                        f"forecast-decision row {index} lacks scenario-quality provenance"
                    )
                    break
                scenario_path = _resolve(root, scenario_evidence.get("path"))
                if (
                    scenario_path is None
                    or not scenario_path.is_file()
                    or sha256_file(scenario_path)
                    != scenario_evidence.get("sha256")
                ):
                    failures.append(
                        f"forecast-decision row {index} scenario-quality artifact fails SHA-256 verification"
                    )
                    break
                for item in forecast_evidence + optimization_evidence:
                    evidence_path = _resolve(root, item.get("path"))
                    if (
                        evidence_path is None
                        or not evidence_path.is_file()
                        or sha256_file(evidence_path) != item.get("sha256")
                    ):
                        failures.append(
                            f"forecast-decision row {index} source artifact does not match its hash"
                        )
                        break
                if failures:
                    break
    return {
        "passed": not failures,
        "failures": failures,
        "policy_row_count": len(policy),
        "sensitivity_row_count": len(sensitivity),
        "forecast_dm_row_count": len(forecast),
        "forecast_decision_record_count": len(decision),
        "manifest": _relative(root, manifest_path) if manifest_path and manifest_path.is_file() else None,
    }


def _scenario_diagnostics_audit(
    root: Path,
    optimization_result: Mapping[str, Any],
    cell_audit: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    path, hash_ok, hash_reason = _declared_path_audit(
        root, optimization_result, "scenario_diagnostics"
    )
    if not hash_ok:
        failures.append(f"scenario diagnostics: {hash_reason}")
    payload = _load_json(path)
    records = [dict(item) for item in payload] if isinstance(payload, list) else []
    days = int(
        cell_audit.get("year_day_count", 0)
        if cell_audit.get("full_year_extension_required")
        else cell_audit.get("primary_day_count", 0)
    )
    seed_count = len(cell_audit.get("expected_seeds", []))
    # Per date/seed: four nominal reduction counts, two adverse pure bundles,
    # and two B7 mixture bundles. OFAT may add further primary-only mixtures.
    expected_minimum = days * seed_count * 8
    if len(records) < expected_minimum:
        failures.append(
            f"scenario diagnostics cover {len(records)} bundles; at least {expected_minimum} reference bundles are required"
        )
    required = {
        "scenario_schema_version",
        "within_domain_dependence",
        "marginal_coverage_90",
        "mean_quantile_score_q05_q50_q95",
        "energy_score",
        "variogram_score",
        "rank_correlation_mean_absolute_error",
        "autocorrelation_mean_absolute_error",
        "autocorrelation_by_block",
        "extreme_event",
        "physical_transfer_support",
        "full_vs_reduced_distribution_distance",
        "decision_value",
        "full_parent_mapping_complete",
        "full_parent_mapping_count",
        "full_parent_component_counts",
    }

    def contains_gaussian_copula(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(contains_gaussian_copula(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_gaussian_copula(item) for item in value)
        return "gaussian_copula" in str(value)

    for index, record in enumerate(records):
        valid = required.issubset(record)
        valid &= record.get("scenario_schema_version") == "gaussian_copula_rank_v2"
        valid &= contains_gaussian_copula(record.get("within_domain_dependence"))
        valid &= record.get("full_parent_mapping_complete") is True
        valid &= int(record.get("full_parent_mapping_count", -1)) == 100
        components = record.get("full_parent_component_counts", {})
        try:
            valid &= isinstance(components, Mapping) and sum(
                map(int, components.values())
            ) == 100
        except (TypeError, ValueError):
            valid = False
        numeric_keys = (
            "marginal_coverage_90",
            "mean_quantile_score_q05_q50_q95",
            "energy_score",
            "variogram_score",
            "rank_correlation_mean_absolute_error",
            "autocorrelation_mean_absolute_error",
            "full_vs_reduced_distribution_distance",
        )
        try:
            valid &= all(math.isfinite(float(record[key])) for key in numeric_keys)
        except (KeyError, TypeError, ValueError):
            valid = False
        event = record.get("extreme_event", {})
        valid &= isinstance(event, Mapping) and {
            "full_cooccurrence_rate",
            "reduced_cooccurrence_rate",
            "full_expected_longest_duration_intervals",
            "reduced_expected_longest_duration_intervals",
        }.issubset(event)
        support = record.get("physical_transfer_support", {})
        distances = support.get("climate_support_distance", {}) if isinstance(support, Mapping) else {}
        valid &= isinstance(distances, Mapping) and len(distances) == 4
        decision = record.get("decision_value", {})
        valid &= isinstance(decision, Mapping) and math.isfinite(
            float(decision.get("decision_value_gap_mean_absolute_mw", math.nan))
        )
        if not valid:
            failures.append(f"scenario diagnostic row {index} is incomplete or invalid")
            if len(failures) >= 25:
                break
    decision_path, decision_hash_ok, decision_hash_reason = _declared_path_audit(
        root, optimization_result, "scenario_decision_value_subset"
    )
    decision_manifest = _load_json(decision_path)
    if not decision_hash_ok:
        failures.append(f"scenario decision-value subset: {decision_hash_reason}")
    if (
        not isinstance(decision_manifest, Mapping)
        or decision_manifest.get("status") != "COMPLETE"
        or decision_manifest.get("completed") is not True
    ):
        failures.append("scenario decision-value optimization subset is incomplete")
        decision_records: list[Mapping[str, Any]] = []
    else:
        decision_records = [
            item
            for item in decision_manifest.get("records", [])
            if isinstance(item, Mapping)
        ]
        if (
            not decision_records
            or len(decision_records)
            != int(decision_manifest.get("requested_day_count", -1))
            or len({str(item.get("operating_date")) for item in decision_records})
            != len(decision_records)
        ):
            failures.append("scenario decision-value subset dates are incomplete or duplicated")
        for index, record in enumerate(decision_records):
            try:
                valid = bool(
                    record.get("status") == "COMPLETE"
                    and int(record.get("reduced_scenario_count", -1)) == 20
                    and int(record.get("evaluation_scenario_count", -1)) == 100
                    and record.get("same_full_100_evaluation_distribution") is True
                    and record.get(
                        "full_population_bundle_restricted_to_decision_value_evaluation"
                    )
                    is True
                    and all(
                        math.isfinite(float(record[key]))
                        for key in (
                            "expected_cost_decision_value_gap_usd",
                            "cvar_decision_value_gap_usd",
                            "objective_decision_value_gap_usd",
                        )
                    )
                )
                artifact = _resolve(root, record.get("full_100_scenario_path"))
                valid &= bool(
                    artifact
                    and artifact.is_file()
                    and sha256_file(artifact)
                    == record.get("full_100_scenario_sha256")
                )
                solver = record.get("solver_evidence", {})
                valid &= isinstance(solver, Mapping) and len(solver) == 3
                for item in solver.values() if isinstance(solver, Mapping) else []:
                    gap = float(item.get("mip_gap", math.nan))
                    target = float(item.get("mip_gap_target", math.nan))
                    valid &= bool(
                        str(item.get("termination_condition", "")).lower()
                        == "optimal"
                        and math.isfinite(gap)
                        and math.isfinite(target)
                        and gap <= target + 1e-12
                    )
                fixed = record.get("fixed_schedule_validation", {})
                valid &= isinstance(fixed, Mapping) and set(fixed) == {
                    "reduced",
                    "full_100",
                }
                for item in fixed.values() if isinstance(fixed, Mapping) else []:
                    valid &= bool(
                        item.get("passed") is True
                        and float(
                            item.get(
                                "fixed_day_ahead_schedule_max_abs_mw", math.inf
                            )
                        )
                        <= 1e-7
                    )
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                failures.append(
                    f"scenario decision-value subset row {index} is invalid"
                )
                break
    return {
        "passed": not failures,
        "failures": failures,
        "record_count": len(records),
        "expected_minimum_reference_bundle_count": expected_minimum,
        "path": _relative(root, path) if path and path.is_file() else None,
        "decision_value_subset_path": (
            _relative(root, decision_path)
            if decision_path and decision_path.is_file()
            else None
        ),
        "decision_value_subset_day_count": len(decision_records),
    }


def _exact_validation_audit(
    root: Path, optimization_result: Mapping[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    path, hash_ok, hash_reason = _declared_path_audit(
        root, optimization_result, "exact_validation_gate"
    )
    if not hash_ok:
        failures.append(f"exact validation gate: {hash_reason}")
    manifest = _load_json(path)
    cases = manifest.get("cases", []) if isinstance(manifest, Mapping) else []
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("status") != "COMPLETE"
        or manifest.get("passed") is not True
        or len(cases) != 2
    ):
        failures.append("one-site and two-site exact-vs-cohort gates did not both pass")
    else:
        for case in cases:
            if not isinstance(case, Mapping) or case.get("passed") is not True:
                failures.append("an exact-vs-cohort case is incomplete")
                break
            if case.get("comparison", {}).get("passed") is not True:
                failures.append("exact and cohort formulations disagree beyond tolerance")
                break
            if case.get("hand_check", {}).get("passed") is not True:
                failures.append("a hand-checkable objective does not verify")
                break
    return {
        "passed": not failures,
        "failures": failures,
        "case_count": len(cases),
        "path": _relative(root, path) if path and path.is_file() else None,
    }


def _rolling_execution_audit(
    root: Path,
    raw_optimization: pd.DataFrame,
) -> dict[str, Any]:
    failures: list[str] = []
    required = {
        "phase",
        "sensitivity_dimension",
        "policy_id",
        "rolling_operation",
        "rolling_interval_minutes",
        "rolling_completed_interval_count",
        "rolling_checkpoint_path",
        "rolling_checkpoint_sha256",
        "future_realized_targets_used",
    }
    reference = pd.DataFrame()
    if raw_optimization.empty or not {"phase", "policy_id"}.issubset(
        raw_optimization.columns
    ):
        failures.append("optimization result table is absent")
    else:
        sensitivity = raw_optimization.get(
            "sensitivity_dimension",
            pd.Series("reference", index=raw_optimization.index),
        )
        reference = raw_optimization.loc[
            raw_optimization["phase"].astype(str).isin(
                ["primary_96_day", "full_year_extension"]
            )
            & sensitivity.astype(str).eq("reference")
        ]
        nonoracle = reference.loc[reference["policy_id"].astype(str).ne("B8")]
        oracle = reference.loc[reference["policy_id"].astype(str).eq("B8")]
        if nonoracle.empty or not required.issubset(raw_optimization.columns):
            failures.append("rolling reference rows or machine fields are missing")
        else:
            if not nonoracle["rolling_operation"].astype(str).str.lower().eq("true").all():
                failures.append("one or more non-oracle reference cells did not run rolling operation")
            if not pd.to_numeric(nonoracle["rolling_interval_minutes"], errors="coerce").eq(15).all():
                failures.append("rolling operation is not fixed at 15-minute intervals")
            if not pd.to_numeric(nonoracle["rolling_completed_interval_count"], errors="coerce").eq(96).all():
                failures.append("one or more rolling days do not contain all 96 intervals")
            if not nonoracle["future_realized_targets_used"].astype(str).str.lower().eq("false").all():
                failures.append("rolling execution used a future realized target")
            for index, row in nonoracle.iterrows():
                path = _resolve(root, row["rolling_checkpoint_path"])
                if (
                    path is None
                    or not path.is_file()
                    or sha256_file(path) != row["rolling_checkpoint_sha256"]
                ):
                    failures.append(f"rolling checkpoint row {index} fails SHA-256 verification")
                    break
        if oracle.empty:
            failures.append("B8 oracle lower-bound reference rows are absent")
        elif "rolling_operation" in oracle and oracle["rolling_operation"].fillna(False).astype(bool).any():
            failures.append("B8 oracle rows incorrectly entered the causal rolling interface")
    return {
        "passed": not failures,
        "failures": failures,
        "reference_row_count": len(reference),
        "rolling_nonoracle_row_count": (
            int(reference["policy_id"].astype(str).ne("B8").sum())
            if not reference.empty
            else 0
        ),
        "oracle_row_count": (
            int(reference["policy_id"].astype(str).eq("B8").sum())
            if not reference.empty
            else 0
        ),
    }


def _economic_timezone_sensitivity(
    root: Path,
    optimization_result: Mapping[str, Any],
    *,
    strict: bool = False,
    expected_seed_count: int = 0,
    expected_day_count: int = 96,
) -> tuple[bool, Any]:
    if strict:
        failures: list[str] = []
        manifest_path, manifest_hash_ok, manifest_hash_reason = _declared_path_audit(
            root, optimization_result, "timezone_sensitivity"
        )
        if not manifest_hash_ok:
            failures.append(
                f"timezone sensitivity manifest: {manifest_hash_reason}"
            )
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, Mapping):
            failures.append("timezone sensitivity manifest is missing or invalid")
            manifest = {}
        expected_seed_count = int(expected_seed_count)
        expected_cell_count = int(expected_day_count) * expected_seed_count * 2
        expected_pair_count = int(expected_day_count) * expected_seed_count
        expected_scalars = {
            "schema_version": "fixed_mst_economic_sensitivity_v1",
            "status": "COMPLETE",
            "completed": True,
            "economic_endpoint_compared": True,
            "scenario_regime": "independent",
            "reduced_scenario_count": 20,
            "rolling_interval_minutes": 15,
            "same_selected_days_forecasts_scenarios_and_realized_parent_ids": True,
            "primary_day_count": int(expected_day_count),
            "seed_count": expected_seed_count,
            "planned_cell_count": expected_cell_count,
            "success_cell_count": expected_cell_count,
            "failed_cell_count": 0,
            "pending_cell_count": 0,
            "paired_date_seed_count": expected_pair_count,
        }
        for key, expected in expected_scalars.items():
            if manifest.get(key) != expected:
                failures.append(
                    f"timezone sensitivity {key} is {manifest.get(key)!r}, expected {expected!r}"
                )
        if set(map(str, manifest.get("policy_ids", []))) != {"B0", "B6"}:
            failures.append("timezone sensitivity must pair exactly B0 and B6")
        seeds = manifest.get("seeds", [])
        try:
            manifest_seeds = {int(value) for value in seeds}
        except (TypeError, ValueError):
            manifest_seeds = set()
        if len(manifest_seeds) != expected_seed_count:
            failures.append("timezone sensitivity seed identities are incomplete")
        if "fixed_mst" not in str(manifest.get("fixed_mst_clock", "")).lower():
            failures.append("fixed-MST source-clock interpretation is not explicit")
        if "utc" not in str(manifest.get("primary_clock", "")).lower():
            failures.append("primary UTC source-clock interpretation is not explicit")
        if optimization_result.get("timezone_economic_sensitivity_complete") is not True:
            failures.append("optimization summary does not mark timezone sensitivity complete")

        hash_cache: dict[Path, str] = {}
        json_cache: dict[Path, Any] = {}

        def verify_file(raw_path: Any, expected_hash: Any, label: str) -> Path | None:
            path = _resolve(root, raw_path)
            if path is None or not path.is_file():
                failures.append(f"{label} is missing")
                return None
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                failures.append(f"{label} SHA-256 is missing")
                return path
            actual = hash_cache.get(path)
            if actual is None:
                actual = sha256_file(path)
                hash_cache[path] = actual
            if actual != expected_hash:
                failures.append(f"{label} SHA-256 does not match")
            return path

        def cached_json(path: Path) -> Any:
            if path not in json_cache:
                json_cache[path] = _load_json(path)
            return json_cache[path]

        fixed_evidence = manifest.get("fixed_physical_evidence", {})
        if not isinstance(fixed_evidence, Mapping):
            failures.append("fixed-MST physical evidence is missing")
            fixed_evidence = {}
        expected_fixed_flags = {
            "method": "rebuild_transfer_from_independently_fitted_fixed_MST_accepted_surrogate",
            "surrogate_frozen": True,
            "surrogate_accepted_for_optimization": True,
        }
        for key, expected in expected_fixed_flags.items():
            if fixed_evidence.get(key) != expected:
                failures.append(
                    f"fixed-MST physical evidence {key} is {fixed_evidence.get(key)!r}"
                )
        calibration_path = verify_file(
            fixed_evidence.get("calibration_manifest_path"),
            fixed_evidence.get("calibration_manifest_sha256"),
            "fixed-MST calibration manifest",
        )
        transfer_path = verify_file(
            fixed_evidence.get("transfer_table_path"),
            fixed_evidence.get("transfer_table_sha256"),
            "fixed-MST transfer table",
        )
        fixed_artifact_path = verify_file(
            fixed_evidence.get("artifact_path"),
            fixed_evidence.get("artifact_sha256"),
            "fixed-MST transferred physical artifact",
        )
        fixed_artifact = _load_json(fixed_artifact_path)
        if not isinstance(fixed_artifact, Mapping):
            failures.append("fixed-MST transferred physical artifact is invalid")
            fixed_artifact = {}
        for key, expected in {
            "schema_version": "fixed_mst_transferred_physical_artifact_v1",
            "frozen": True,
            "counterfactual": True,
            "measurement_relationship": _NOT_STUDY_MEASURED,
        }.items():
            if fixed_artifact.get(key) != expected:
                failures.append(
                    f"fixed-MST transferred physical artifact {key} is invalid"
                )
        for key in (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
        ):
            values = fixed_artifact.get(key)
            try:
                valid = len(values) == 4 and all(
                    math.isfinite(float(value)) for value in values
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                failures.append(f"fixed-MST artifact {key} is not four finite values")
        fixed_source_hashes = fixed_artifact.get("source_hashes", {})
        fixed_source_hashes = (
            fixed_source_hashes if isinstance(fixed_source_hashes, Mapping) else {}
        )
        if calibration_path is not None and transfer_path is not None:
            if fixed_source_hashes.get(
                "fixed_mst_physical_calibration_manifest"
            ) != sha256_file(calibration_path):
                failures.append("fixed artifact does not retain calibration SHA-256")
            if fixed_source_hashes.get(
                "fixed_transfer_table"
            ) != sha256_file(transfer_path):
                failures.append("fixed artifact does not retain transfer SHA-256")

        paired = manifest.get("paired_economic_table", {})
        if not isinstance(paired, Mapping):
            failures.append("paired timezone economic table declaration is missing")
            paired = {}
        paired_path = verify_file(
            paired.get("path"), paired.get("sha256"), "paired timezone economic table"
        )
        declared_paired_path, paired_hash_ok, paired_hash_reason = _declared_path_audit(
            root, optimization_result, "timezone_sensitivity_paired_table"
        )
        if not paired_hash_ok:
            failures.append(
                f"optimization paired timezone table: {paired_hash_reason}"
            )
        elif paired_path is not None and declared_paired_path != paired_path:
            failures.append("paired timezone table declarations disagree")

        table = pd.DataFrame()
        if paired_path is not None:
            try:
                table = pd.read_csv(paired_path)
            except Exception as exc:
                failures.append(f"paired timezone table cannot be read: {exc}")
        required_columns = {
            "operating_date",
            "study_seed",
            "primary_b0_expected_cost_usd",
            "primary_b6_expected_cost_usd",
            "fixed_mst_b0_expected_cost_usd",
            "fixed_mst_b6_expected_cost_usd",
            "primary_policy_value_usd",
            "fixed_mst_policy_value_usd",
            "clock_effect_on_policy_value_usd",
            "primary_cvar_policy_value_usd",
            "fixed_mst_cvar_policy_value_usd",
            "clock_effect_on_cvar_policy_value_usd",
            "same_forecast_scenario_seed",
            "scenario_artifact_path",
            "scenario_artifact_sha256",
            "primary_result_evidence_json",
            "fixed_mst_result_evidence_json",
        }
        if not required_columns.issubset(table.columns):
            failures.append("paired timezone table lacks required evidence columns")
        elif len(table) != expected_pair_count:
            failures.append(
                f"paired timezone table has {len(table)} rows, expected {expected_pair_count}"
            )
        else:
            pair_keys = table[["operating_date", "study_seed"]].astype(str)
            if pair_keys.duplicated().any():
                failures.append("paired timezone table has duplicate date/seed rows")
            if table["operating_date"].astype(str).nunique() != expected_day_count:
                failures.append("paired timezone table does not cover all primary days")
            observed_table_seeds = set(
                pd.to_numeric(table["study_seed"], errors="coerce")
                .dropna()
                .astype(int)
            )
            if observed_table_seeds != manifest_seeds:
                failures.append("paired timezone table seed identities disagree")
            if not table["same_forecast_scenario_seed"].astype(str).str.lower().eq(
                "true"
            ).all():
                failures.append("one or more timezone pairs changed forecast/scenario seed")
            numeric_columns = sorted(
                required_columns
                - {
                    "operating_date",
                    "study_seed",
                    "same_forecast_scenario_seed",
                    "scenario_artifact_path",
                    "scenario_artifact_sha256",
                    "primary_result_evidence_json",
                    "fixed_mst_result_evidence_json",
                }
            )
            numeric = table[numeric_columns].apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any():
                failures.append("paired timezone economics contain unavailable values")
            for row_index, row in table.iterrows():
                verify_file(
                    row.get("scenario_artifact_path"),
                    row.get("scenario_artifact_sha256"),
                    f"paired timezone row {row_index} scenario artifact",
                )
                evidence_groups: dict[str, dict[str, Mapping[str, Any]]] = {}
                for column, label in (
                    ("primary_result_evidence_json", "primary"),
                    ("fixed_mst_result_evidence_json", "fixed-MST"),
                ):
                    try:
                        records = json.loads(str(row[column]))
                    except (KeyError, TypeError, json.JSONDecodeError):
                        records = None
                    if not isinstance(records, list) or len(records) != 2:
                        failures.append(
                            f"paired timezone row {row_index} {label} evidence is invalid"
                        )
                        continue
                    policy_results: dict[str, Mapping[str, Any]] = {}
                    for evidence_index, record in enumerate(records):
                        if not isinstance(record, Mapping):
                            failures.append(
                                f"paired timezone row {row_index} {label} evidence is invalid"
                            )
                            continue
                        result_path = verify_file(
                            record.get("path"),
                            record.get("sha256"),
                            f"paired timezone row {row_index} {label} result {evidence_index}",
                        )
                        payload = (
                            cached_json(result_path)
                            if result_path is not None
                            else None
                        )
                        if not isinstance(payload, Mapping):
                            failures.append(
                                f"paired timezone row {row_index} {label} result is invalid"
                            )
                            continue
                        policy_id = str(payload.get("policy_id", ""))
                        if policy_id not in {"B0", "B6"} or policy_id in policy_results:
                            failures.append(
                                f"paired timezone row {row_index} {label} policies are invalid"
                            )
                            continue
                        try:
                            identity_agrees = (
                                str(payload.get("operating_date"))
                                == str(row["operating_date"])
                                and int(payload.get("study_seed", -1))
                                == int(row["study_seed"])
                            )
                        except (TypeError, ValueError):
                            identity_agrees = False
                        if not identity_agrees:
                            failures.append(
                                f"paired timezone row {row_index} {label} result identity disagrees"
                            )
                        policy_results[policy_id] = payload
                    evidence_groups[label] = policy_results
                primary_results = evidence_groups.get("primary", {})
                fixed_results = evidence_groups.get("fixed-MST", {})
                if set(primary_results) == {"B0", "B6"} and set(fixed_results) == {
                    "B0",
                    "B6",
                }:
                    expected_values = {
                        "primary_b0_expected_cost_usd": primary_results["B0"].get(
                            "expected_total_cost_usd"
                        ),
                        "primary_b6_expected_cost_usd": primary_results["B6"].get(
                            "expected_total_cost_usd"
                        ),
                        "fixed_mst_b0_expected_cost_usd": fixed_results["B0"].get(
                            "expected_total_cost_usd"
                        ),
                        "fixed_mst_b6_expected_cost_usd": fixed_results["B6"].get(
                            "expected_total_cost_usd"
                        ),
                    }
                    for column, expected_value in expected_values.items():
                        try:
                            agrees = math.isclose(
                                float(row[column]),
                                float(expected_value),
                                rel_tol=1e-10,
                                abs_tol=1e-8,
                            )
                        except (TypeError, ValueError):
                            agrees = False
                        if not agrees:
                            failures.append(
                                f"paired timezone row {row_index} {column} disagrees with result evidence"
                            )
                    try:
                        recomputed = {
                            "primary_policy_value_usd": float(
                                primary_results["B0"]["expected_total_cost_usd"]
                            )
                            - float(primary_results["B6"]["expected_total_cost_usd"]),
                            "fixed_mst_policy_value_usd": float(
                                fixed_results["B0"]["expected_total_cost_usd"]
                            )
                            - float(fixed_results["B6"]["expected_total_cost_usd"]),
                            "primary_cvar_policy_value_usd": float(
                                primary_results["B0"]["cvar_total_cost_usd"]
                            )
                            - float(primary_results["B6"]["cvar_total_cost_usd"]),
                            "fixed_mst_cvar_policy_value_usd": float(
                                fixed_results["B0"]["cvar_total_cost_usd"]
                            )
                            - float(fixed_results["B6"]["cvar_total_cost_usd"]),
                        }
                    except (KeyError, TypeError, ValueError):
                        failures.append(
                            f"paired timezone row {row_index} evidence lacks economic endpoints"
                        )
                        recomputed = {}
                    if recomputed:
                        recomputed["clock_effect_on_policy_value_usd"] = (
                            recomputed["fixed_mst_policy_value_usd"]
                            - recomputed["primary_policy_value_usd"]
                        )
                        recomputed["clock_effect_on_cvar_policy_value_usd"] = (
                            recomputed["fixed_mst_cvar_policy_value_usd"]
                            - recomputed["primary_cvar_policy_value_usd"]
                        )
                    for column, expected_value in recomputed.items():
                        if not math.isclose(
                            float(row[column]),
                            expected_value,
                            rel_tol=1e-10,
                            abs_tol=1e-8,
                        ):
                            failures.append(
                                f"paired timezone row {row_index} {column} is not recomputable"
                            )

        cells = manifest.get("cells", [])
        if not isinstance(cells, list) or len(cells) != expected_cell_count:
            failures.append("timezone sensitivity cell inventory is incomplete")
            cells = []
        observed_cell_keys: set[tuple[str, int, str]] = set()
        scenarios_by_pair: dict[tuple[str, int], set[str]] = {}
        for index, cell in enumerate(cells):
            if not isinstance(cell, Mapping):
                failures.append(f"timezone cell {index} is invalid")
                continue
            try:
                key = (
                    str(cell["operating_date"]),
                    int(cell["study_seed"]),
                    str(cell["policy_id"]),
                )
            except (KeyError, TypeError, ValueError):
                failures.append(f"timezone cell {index} identity is invalid")
                continue
            if key in observed_cell_keys:
                failures.append(f"timezone cell identity is duplicated: {key}")
            observed_cell_keys.add(key)
            if key[2] not in {"B0", "B6"}:
                failures.append(f"timezone cell {key} uses an unregistered policy")
            if cell.get("status") != "COMPLETE" or len(
                str(cell.get("fingerprint", ""))
            ) != 64:
                failures.append(f"timezone cell {key} is not hash-fingerprinted COMPLETE")
            checkpoint_path = _resolve(root, cell.get("checkpoint_path"))
            checkpoint = _load_json(checkpoint_path)
            if not isinstance(checkpoint, Mapping) or checkpoint.get("status") != "COMPLETE":
                failures.append(f"timezone cell {key} checkpoint is not COMPLETE")
            elif checkpoint.get("fingerprint") != cell.get("fingerprint"):
                failures.append(f"timezone cell {key} checkpoint fingerprint disagrees")
            for path_key, hash_key, label in (
                ("primary_result_path", "primary_result_sha256", "primary result"),
                ("scenario_path", "scenario_sha256", "scenario artifact"),
                ("fixed_result_path", "fixed_result_sha256", "fixed-MST result"),
                (
                    "rolling_checkpoint_path",
                    "rolling_checkpoint_sha256",
                    "rolling checkpoint",
                ),
            ):
                verify_file(
                    cell.get(path_key), cell.get(hash_key), f"timezone cell {key} {label}"
                )
            if cell.get("fixed_physical_artifact_sha256") != fixed_evidence.get(
                "artifact_sha256"
            ):
                failures.append(f"timezone cell {key} used another physical artifact")
            scenarios_by_pair.setdefault(key[:2], set()).add(
                str(cell.get("scenario_sha256", ""))
            )
        if any(len(values) != 1 or "" in values for values in scenarios_by_pair.values()):
            failures.append("B0/B6 timezone cells did not share one scenario per date/seed")
        if not table.empty and {"operating_date", "study_seed"}.issubset(table):
            expected_cell_keys = {
                (str(row["operating_date"]), int(row["study_seed"]), policy)
                for _, row in table.iterrows()
                for policy in ("B0", "B6")
            }
            if observed_cell_keys != expected_cell_keys:
                failures.append(
                    "timezone cell identities do not match the paired table date/seed design"
                )

        inference = manifest.get("inference", {})
        if not isinstance(inference, Mapping):
            failures.append("timezone paired statistical inference is missing")
            inference = {}
        endpoint_inferences = inference.get("endpoint_inferences", {})
        if not isinstance(endpoint_inferences, Mapping) or set(endpoint_inferences) != {
            "expected_cost_policy_value",
            "cvar_policy_value",
        }:
            failures.append("timezone expected-cost/CVaR endpoint inferences are incomplete")
            endpoint_inferences = {}
        numeric_inference_fields = {
            "primary_mean_policy_value_usd",
            "fixed_mst_mean_policy_value_usd",
            "mean_clock_effect_usd",
            "block_bootstrap_lower_usd",
            "block_bootstrap_upper_usd",
            "p_value",
            "p_value_raw",
            "p_value_holm",
            "alpha",
            "economic_effect_fraction_of_primary_value",
            "economic_significance_threshold_fraction",
        }
        if expected_day_count >= 2:
            numeric_inference_fields.update(
                {"newey_west_standard_error", "test_statistic"}
            )
        for endpoint_id, endpoint in endpoint_inferences.items():
            if not isinstance(endpoint, Mapping):
                failures.append(f"timezone {endpoint_id} inference is invalid")
                continue
            if int(endpoint.get("paired_date_seed_count", 0) or 0) != expected_pair_count:
                failures.append(f"timezone {endpoint_id} paired row count is incomplete")
            if int(endpoint.get("paired_day_count", 0) or 0) != expected_day_count:
                failures.append(f"timezone {endpoint_id} paired day count is incomplete")
            if "circular_block_bootstrap_and_Newey_West" not in str(
                endpoint.get("method", "")
            ):
                failures.append(f"timezone {endpoint_id} inference method is invalid")
            if endpoint.get("multiple_testing_method") != "Holm" or int(
                endpoint.get("holm_family_size", 0) or 0
            ) != 2:
                failures.append(f"timezone {endpoint_id} Holm correction is invalid")
            for field in numeric_inference_fields:
                try:
                    finite = math.isfinite(float(endpoint.get(field)))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    failures.append(f"timezone {endpoint_id} {field} is unavailable")
            for field in (
                "direction_changed",
                "statistically_significant",
                "economically_significant",
                "timezone_sensitive",
                "substantive_conclusion_changed",
            ):
                if not isinstance(endpoint.get(field), bool):
                    failures.append(f"timezone {endpoint_id} {field} is not Boolean")
        conclusion = inference.get("timezone_sensitive")
        if not isinstance(conclusion, bool):
            failures.append("timezone sensitivity conclusion is not Boolean")
        if manifest.get("timezone_sensitive") != conclusion or manifest.get(
            "substantive_conclusion_changed"
        ) != conclusion:
            failures.append("timezone sensitivity conclusion fields disagree")
        return not failures, {
            "passed": not failures,
            "strict": True,
            "failures": failures,
            "path": _relative(root, manifest_path)
            if manifest_path is not None and manifest_path.is_file()
            else None,
            "primary_day_count": manifest.get("primary_day_count"),
            "seed_count": manifest.get("seed_count"),
            "planned_cell_count": manifest.get("planned_cell_count"),
            "paired_date_seed_count": manifest.get("paired_date_seed_count"),
            "timezone_sensitive": manifest.get("timezone_sensitive"),
            "substantive_conclusion_changed": manifest.get(
                "substantive_conclusion_changed"
            ),
            "inference": dict(inference),
            "fixed_physical_evidence": dict(fixed_evidence),
        }

    candidates: list[Any] = [
        optimization_result.get("timezone_sensitivity"),
        optimization_result.get("time_encoding_sensitivity"),
        optimization_result.get("clock_sensitivity"),
    ]
    for raw_path in (
        "artifacts/manifests/timezone_sensitivity_results.json",
        "artifacts/manifests/timezone_sensitivity_manifest.json",
        "artifacts/report/timezone_sensitivity.json",
    ):
        candidates.append(_load_json(root / raw_path))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        status = str(candidate.get("status", "")).lower()
        explicit = candidate.get("economic_endpoint_compared") is True
        key_text = " ".join(map(str, candidate.keys())).lower()
        has_endpoint = any(token in key_text for token in ("economic", "cost", "revenue"))
        has_two_clocks = (
            any(token in key_text for token in ("primary", "utc"))
            and any(token in key_text for token in ("fixed_mst", "fixed-mst", "utc_minus_07"))
        )
        if status in {"complete", "completed", "evaluated", "passed"} and (
            explicit or (has_endpoint and has_two_clocks)
        ):
            return True, dict(candidate)
    return False, None


def _xcel_replay_audit(
    root: Path,
    optimization_result: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    path = _result_path(root, optimization_result, "paths.xcel_local_replay_manifest")
    manifest = _load_json(path)
    result_status = str(
        optimization_result.get("xcel_local_subcase_status", "")
    ).upper()
    accepted_statuses = {"COMPLETE", "PARTIAL_SOURCE_COVERAGE"}
    if result_status not in accepted_statuses:
        failures.append(
            "Xcel local tariff replay status is "
            + str(optimization_result.get("xcel_local_subcase_status", "missing"))
        )
    if not isinstance(manifest, Mapping):
        failures.append("Xcel local replay manifest is missing or invalid")
        manifest = {}
    synthetic_flags = _true_synthetic_flags(manifest, "xcel_local_replay_manifest")
    if synthetic_flags:
        failures.append("Xcel local replay manifest contains true synthetic flags")
    expected = {
        "schema_version": "xcel_local_tariff_replay_v2",
        "primary_rate_snapshot": "2024-05-01",
        "current_2026_rates_used": False,
        "execution_complete": True,
        "all_available_source_intervals_replayed": True,
        "missing_intervals_imputed": False,
        "event_calls": 0,
        "event_timestamps_synthesized": False,
        "price_elasticity_estimated": False,
        "tariff_defined_counterfactual": True,
        "realized_response_evidence": False,
        "realized_savings_claimed": False,
        "source_relationship": "tariff_defined_counterfactual",
        "load_source_relationship": _REAL_SOURCE,
        "measurement_relationship": _NOT_STUDY_MEASURED,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            failures.append(f"Xcel replay {key} is {manifest.get(key)!r}, expected {value!r}")
    manifest_status = str(manifest.get("status", "")).upper()
    if manifest_status not in accepted_statuses or manifest_status != result_status:
        failures.append(
            f"Xcel replay manifest/result statuses disagree or are not terminal: "
            f"manifest={manifest_status or 'missing'}, result={result_status or 'missing'}"
        )
    implementation_path = root / "src/dcvpp/research_branch/xcel_counterfactual.py"
    implementation_sha256 = manifest.get("implementation_sha256")
    if (
        not implementation_path.is_file()
        or not isinstance(implementation_sha256, str)
        or sha256_file(implementation_path) != implementation_sha256
    ):
        failures.append("Xcel replay implementation hash does not verify")
    coverage = manifest.get("coverage", {})
    if not isinstance(coverage, Mapping):
        failures.append("Xcel replay coverage is missing")
        coverage = {}
    tariff_effective_month_count = int(
        coverage.get(
            "tariff_effective_month_count",
            coverage.get("expected_month_count", 0),
        )
        or 0
    )
    supported_month_count = int(
        coverage.get(
            "supported_overlap_month_count",
            coverage.get("expected_month_count", 0),
        )
        or 0
    )
    complete_month_count = int(
        coverage.get(
            "complete_overlap_month_count",
            coverage.get("complete_month_count", 0),
        )
        or 0
    )
    observed_intervals = int(coverage.get("observed_interval_count", 0) or 0)
    expected_intervals = int(coverage.get("expected_interval_count", 0) or 0)
    missing_intervals = int(
        coverage.get(
            "source_missing_interval_count",
            coverage.get(
                "missing_interval_count",
                max(expected_intervals - observed_intervals, 0),
            ),
        )
        or 0
    )
    source_unsupported_month_count = int(
        coverage.get("source_unsupported_month_count", 0) or 0
    )
    def month_set(key: str) -> set[str]:
        raw_months = coverage.get(key, [])
        return (
            {str(value) for value in raw_months}
            if isinstance(raw_months, (list, tuple, set))
            else set()
        )

    supported_months = month_set("supported_overlap_months")
    complete_months = month_set("complete_overlap_months")
    unsupported_months = month_set("source_unsupported_months")
    coverage_fraction = coverage.get("source_coverage_fraction")
    expected_fraction = (
        observed_intervals / expected_intervals if expected_intervals else 0.0
    )
    # Source gaps are a disclosed data limitation, not evidence that the replay
    # failed.  Completion means every available source interval was processed
    # and the absent intervals remained absent.
    if (
        tariff_effective_month_count <= 0
        or not 0 < supported_month_count <= tariff_effective_month_count
        or not 0 <= complete_month_count <= supported_month_count
        or source_unsupported_month_count
        != tariff_effective_month_count - supported_month_count
        or not 0 < observed_intervals <= expected_intervals
        or missing_intervals != expected_intervals - observed_intervals
        or (result_status == "COMPLETE" and missing_intervals != 0)
        or (result_status == "PARTIAL_SOURCE_COVERAGE" and missing_intervals <= 0)
        or len(supported_months) != supported_month_count
        or len(complete_months) != complete_month_count
        or len(unsupported_months) != source_unsupported_month_count
        or not complete_months.issubset(supported_months)
        or bool(supported_months & unsupported_months)
        or coverage.get("available_source_start") in (None, "")
        or coverage.get("available_source_end") in (None, "")
        or not isinstance(coverage_fraction, (int, float))
        or not math.isclose(float(coverage_fraction), expected_fraction, rel_tol=1e-9)
        or manifest.get("source_coverage_complete")
        is not (
            missing_intervals == 0
            and source_unsupported_month_count == 0
            and complete_month_count == tariff_effective_month_count
        )
        or (
            result_status == "COMPLETE"
            and manifest.get("source_coverage_complete") is not True
        )
        or (
            result_status == "PARTIAL_SOURCE_COVERAGE"
            and manifest.get("source_coverage_complete") is not False
        )
    ):
        failures.append(
            "Xcel replay coverage/status does not quantify the tariff/source "
            "overlap and retained missing intervals consistently"
        )
    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, Mapping):
        inputs = {}
    for name, path_key, hash_key in (
        ("NLR300 facility", "nlr300_facility_path", "nlr300_facility_sha256"),
        ("Xcel rates", "xcel_rate_input_path", "xcel_rate_input_sha256"),
    ):
        source = _resolve(root, inputs.get(path_key))
        expected_hash = inputs.get(hash_key)
        if (
            source is None
            or not source.is_file()
            or not isinstance(expected_hash, str)
            or sha256_file(source) != expected_hash
        ):
            failures.append(f"{name} input hash does not verify")
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, Mapping):
        outputs = {}
    for name, path_key, hash_key in (
        ("replay Parquet", "parquet", "parquet_sha256"),
        ("replay CSV", "csv", "csv_sha256"),
    ):
        output = _resolve(root, outputs.get(path_key))
        expected_hash = outputs.get(hash_key)
        if (
            output is None
            or not output.is_file()
            or not isinstance(expected_hash, str)
            or sha256_file(output) != expected_hash
        ):
            failures.append(f"Xcel {name} hash does not verify")

    replay_path = _resolve(root, outputs.get("parquet"))
    if replay_path is not None and replay_path.is_file():
        try:
            replay = pd.read_parquet(replay_path)
        except Exception as exc:
            failures.append(f"Xcel replay Parquet cannot be read: {type(exc).__name__}: {exc}")
            replay = pd.DataFrame()
    else:
        csv_path = _resolve(root, outputs.get("csv"))
        try:
            replay = pd.read_csv(csv_path) if csv_path is not None and csv_path.is_file() else pd.DataFrame()
        except Exception as exc:
            failures.append(f"Xcel replay CSV cannot be read: {type(exc).__name__}: {exc}")
            replay = pd.DataFrame()

    required_columns = {
        "month",
        "schedule",
        "tariff_applicable",
        "complete_month",
        "schedule_eligible",
        "observed_interval_count",
        "expected_interval_count",
        "fixed_charge_usd",
        "observed_interval_energy_charge_usd",
        "observed_peak_demand_charge_usd",
        "observed_interval_tariff_cost_usd",
        "event_calls",
        "tariff_defined_counterfactual",
        "realized_response_evidence",
        "realized_savings_claimed",
    }
    if replay.empty or not required_columns.issubset(replay.columns):
        failures.append("Xcel replay rows or required evidence columns are missing")
    else:
        expected_months = {f"2024-{month:02d}" for month in range(1, 13)}
        expected_schedules = {"SG-CPP", "PG-CPP", "TG-CPP"}
        pairs = set(
            zip(replay["month"].astype(str), replay["schedule"].astype(str), strict=False)
        )
        if (
            set(replay["month"].astype(str)) != expected_months
            or set(replay["schedule"].astype(str)) != expected_schedules
            or pairs
            != {
                (month, schedule)
                for month in expected_months
                for schedule in expected_schedules
            }
            or len(replay) != 36
        ):
            failures.append("Xcel replay is not one row per 2024 month and retained schedule")

        def bool_series(column: str) -> pd.Series:
            values = replay[column]
            if pd.api.types.is_bool_dtype(values):
                return values.fillna(False).astype(bool)
            return values.astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )

        tariff_applicable = bool_series("tariff_applicable")
        complete_month = bool_series("complete_month")
        eligible = bool_series("schedule_eligible")
        tariff_defined = bool_series("tariff_defined_counterfactual")
        response = bool_series("realized_response_evidence")
        savings = bool_series("realized_savings_claimed")
        pre_effective = replay["month"].astype(str).isin(
            {"2024-01", "2024-02", "2024-03", "2024-04"}
        )
        tariff_effective = ~pre_effective
        if (
            tariff_applicable.isna().any()
            or tariff_applicable.loc[pre_effective].any()
            or not tariff_applicable.loc[tariff_effective].all()
        ):
            failures.append("Xcel tariff applicability does not respect the 2024-05-01 effective date")
        cost_columns = [
            "fixed_charge_usd",
            "observed_interval_energy_charge_usd",
            "observed_peak_demand_charge_usd",
            "observed_interval_tariff_cost_usd",
        ]
        if replay.loc[pre_effective, cost_columns].notna().any(axis=None):
            failures.append("Xcel January-April out-of-period rows contain tariff costs")
        if (
            complete_month.isna().any()
            or eligible.isna().any()
            or not eligible.loc[tariff_effective].all()
        ):
            failures.append("Xcel applicable replay rows are incomplete or schedule-ineligible")
        monthly = replay.loc[tariff_effective].groupby("month", sort=True).first()
        row_observed = int(
            pd.to_numeric(monthly["observed_interval_count"], errors="coerce").sum()
        )
        row_expected = int(
            pd.to_numeric(monthly["expected_interval_count"], errors="coerce").sum()
        )
        monthly_complete = bool_series("complete_month").loc[tariff_effective].groupby(
            replay.loc[tariff_effective, "month"].astype(str)
        ).first()
        row_complete = int(monthly_complete.sum())
        row_supported_months = set(
            monthly.index[
                pd.to_numeric(monthly["observed_interval_count"], errors="coerce").gt(0)
            ].astype(str)
        )
        row_complete_months = set(monthly_complete.index[monthly_complete].astype(str))
        row_unsupported_months = set(monthly.index.astype(str)) - row_supported_months
        if (
            row_observed != observed_intervals
            or row_expected != expected_intervals
            or row_complete != complete_month_count
            or len(monthly) != tariff_effective_month_count
            or row_supported_months != supported_months
            or row_complete_months != complete_months
            or row_unsupported_months != unsupported_months
        ):
            failures.append("Xcel replay row coverage does not reconcile to the manifest")
        event_calls = pd.to_numeric(replay["event_calls"], errors="coerce")
        if (
            event_calls.isna().any()
            or not event_calls.eq(0).all()
            or tariff_defined.isna().any()
            or not tariff_defined.all()
            or response.isna().any()
            or response.any()
            or savings.isna().any()
            or savings.any()
        ):
            failures.append("Xcel row-level counterfactual/event/response labels are contradictory")
    return {
        "passed": not failures,
        "failures": failures,
        "true_synthetic_flags": synthetic_flags,
        "status": manifest.get("status"),
        "coverage": dict(coverage),
        "supported_overlap_month_count": supported_month_count,
        "tariff_effective_month_count": tariff_effective_month_count,
        "complete_overlap_month_count": complete_month_count,
        "source_unsupported_month_count": source_unsupported_month_count,
        "supported_overlap_months": sorted(supported_months),
        "complete_overlap_months": sorted(complete_months),
        "source_unsupported_months": sorted(unsupported_months),
        "missing_interval_count": missing_intervals,
        "manifest": _relative(root, path) if path and path.is_file() else None,
        "outputs": {
            key: _relative(root, resolved)
            for key in ("parquet", "csv")
            if (resolved := _resolve(root, outputs.get(key))) is not None
            and resolved.is_file()
        },
        "implementation": (
            _relative(root, implementation_path)
            if implementation_path.is_file()
            else None
        ),
    }


def _physical_timezone_sensitivity(
    physical_result: Mapping[str, Any],
    calibration_manifest: Any,
) -> dict[str, Any]:
    clock = calibration_manifest.get("clock_sensitivity", {}) if isinstance(calibration_manifest, Mapping) else {}
    counts = physical_result.get("counts", {})
    passed = (
        isinstance(clock, Mapping)
        and clock.get("primary_locked_test_rmse_kw") is not None
        and clock.get("fixed_mst_locked_test_rmse_kw") is not None
        and int(counts.get("primary_overlap_rows", 0) or 0) > 0
        and int(counts.get("fixed_mst_overlap_rows", 0) or 0) > 0
    )
    return {
        "passed": passed,
        "primary_overlap_rows": counts.get("primary_overlap_rows"),
        "fixed_mst_overlap_rows": counts.get("fixed_mst_overlap_rows"),
        "locked_test_metrics": dict(clock) if isinstance(clock, Mapping) else {},
    }


def _forecast_metrics_path(root: Path, manifest: Any, result: Mapping[str, Any]) -> Path | None:
    direct = _result_path(root, result, "metrics_path")
    if direct is not None and direct.is_file():
        return direct
    if isinstance(manifest, Mapping):
        for record in manifest.get("outputs", []):
            if isinstance(record, Mapping) and str(record.get("path", "")).replace("\\", "/").endswith(
                "/metrics/metrics.json"
            ):
                path = _resolve(root, record.get("path"))
                if path is not None and path.is_file():
                    return path
    return None


def _manifest_output_path(root: Path, manifest: Any, suffix: str) -> Path | None:
    if not isinstance(manifest, Mapping):
        return None
    normalized_suffix = suffix.replace("\\", "/")
    for record in manifest.get("outputs", []):
        if not isinstance(record, Mapping):
            continue
        raw = str(record.get("path", "")).replace("\\", "/")
        if raw.endswith(normalized_suffix):
            path = _resolve(root, record.get("path"))
            if path is not None and path.is_file():
                return path
    return None


def _forecast_scope_audit(
    root: Path,
    config: Mapping[str, Any],
    manifest: Any,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(manifest, Mapping):
        return {"passed": False, "failures": ["forecast run manifest is missing"]}
    branch_cfg = config.get("forecast_branch", {})
    branch_cfg = branch_cfg if isinstance(branch_cfg, Mapping) else {}
    forecast_cfg = config.get("forecast", {})
    forecast_cfg = forecast_cfg if isinstance(forecast_cfg, Mapping) else {}
    expected_tasks = tuple(
        map(str, branch_cfg.get("tasks", ("nyiso_da", "nyiso_rt", "nlr_workload")))
    )
    expected_seeds = tuple(
        map(
            int,
            branch_cfg.get(
                "seeds",
                config.get("seed_list", [config.get("seed", 20260809)]),
            ),
        )
    )
    expected_epochs = int(branch_cfg.get("torch_epochs", forecast_cfg.get("epochs", 100)))
    expected_estimators = int(branch_cfg.get("advanced_n_estimators", 500))
    expected_draws = int(branch_cfg.get("forecast_draw_count", 100))
    effective = manifest.get("effective_config", {})
    if not isinstance(effective, Mapping):
        effective = {}
    if set(map(str, effective.get("tasks", ()))) != set(expected_tasks):
        failures.append("forecast task set differs from the paper configuration")
    if tuple(map(int, effective.get("seeds", ()))) != expected_seeds or len(expected_seeds) != 4:
        failures.append("v1 forecast seeds do not match its completed pre-three-seed four-seed panel")
    if effective.get("max_train_rows") is not None:
        failures.append("paper forecast used a row cap")
    if int(effective.get("torch_epochs", -1)) != expected_epochs:
        failures.append("advanced forecast epoch count differs from the paper configuration")
    if int(effective.get("advanced_n_estimators", -1)) != expected_estimators:
        failures.append("tree estimator count differs from the paper configuration")
    if int(effective.get("forecast_draw_count", -1)) != expected_draws:
        failures.append("forecast draw count differs from the paper configuration")

    branch_cfg_opt = config.get("optimization_branch", {})
    target_year = int(
        branch_cfg_opt.get("target_year", 2025)
        if isinstance(branch_cfg_opt, Mapping)
        else 2025
    )
    expected_dates = {
        value.date().isoformat()
        for value in pd.date_range(
            f"{target_year:04d}-01-01",
            f"{target_year:04d}-12-31",
            freq="D",
        )
    }
    result = result if isinstance(result, Mapping) else {}
    result_day_index = result.get("forecast_day_index")
    day_index_record = (
        result_day_index
        if isinstance(result_day_index, Mapping)
        else manifest.get("forecast_day_index", {})
    )
    if not isinstance(day_index_record, Mapping):
        day_index_record = {}
    index_path = _resolve(root, day_index_record.get("index_path"))
    index_payload = _load_json(index_path)
    index_days = index_payload.get("days", []) if isinstance(index_payload, Mapping) else []
    indexed_dates = {
        str(record.get("operating_date"))
        for record in index_days
        if isinstance(record, Mapping)
    }
    indexed_paths = {
        str(record.get("path"))
        for record in index_days
        if isinstance(record, Mapping) and record.get("path")
    }
    index_hashes_complete = bool(index_days) and all(
        isinstance(record, Mapping)
        and isinstance(record.get("sha256"), str)
        and len(str(record.get("sha256"))) == 64
        for record in index_days
    )
    if str(day_index_record.get("status", "")).upper() != "COMPLETE":
        failures.append("frozen forecast-day index is not COMPLETE")
    if int(day_index_record.get("day_count", -1)) != len(expected_dates):
        failures.append("forecast-day count does not cover the full target year")
    if int(day_index_record.get("draw_count", -1)) != expected_draws:
        failures.append("forecast-day draw count differs from the paper configuration")
    if indexed_dates != expected_dates or len(indexed_paths) != len(expected_dates):
        failures.append("frozen forecast-day index does not contain one unique artifact per target-year date")
    if not index_hashes_complete:
        failures.append("one or more frozen forecast-day hashes are missing")

    split_path = _manifest_output_path(root, manifest, "/manifests/split_manifest.json")
    split_manifest = _load_json(split_path)
    split_tasks = split_manifest.get("tasks", {}) if isinstance(split_manifest, Mapping) else {}
    if set(split_tasks) != set(expected_tasks):
        failures.append("chronological split manifest task set is incomplete")
    else:
        for task in expected_tasks:
            record = split_tasks.get(task, {})
            if not isinstance(record, Mapping):
                failures.append(f"{task} split record is missing")
                continue
            if record.get("overlap_allowed") is not False:
                failures.append(f"{task} does not explicitly prohibit split overlap")
            if record.get("locked_test_used_for_model_or_seed_selection") is not False:
                failures.append(f"{task} does not explicitly protect the locked test")
            splits = record.get("splits", {})
            if not isinstance(splits, Mapping) or set(splits) != {
                "train",
                "validation",
                "locked_test",
            }:
                failures.append(f"{task} chronological split partitions are incomplete")
            elif any(
                int(splits[name].get("row_count", 0) or 0) <= 0
                or len(str(splits[name].get("key_sha256", ""))) != 64
                for name in ("train", "validation", "locked_test")
                if isinstance(splits.get(name), Mapping)
            ):
                failures.append(f"{task} split rows or hashes are incomplete")
    return {
        "passed": not failures,
        "failures": failures,
        "expected_tasks": list(expected_tasks),
        "expected_seeds": list(expected_seeds),
        "effective_config": dict(effective),
        "target_year": target_year,
        "forecast_day_count": len(indexed_dates),
        "forecast_day_unique_path_count": len(indexed_paths),
        "forecast_day_index": _relative(root, index_path) if index_path and index_path.is_file() else None,
        "split_manifest": _relative(root, split_path) if split_path and split_path.is_file() else None,
    }


def _forecast_extension_bridge_audit(
    root: Path,
    result: Mapping[str, Any],
    v1_manifest_path: Path | None,
) -> dict[str, Any]:
    failures: list[str] = []
    expected = {
        "status": "COMPLETE",
        "optimization_allowed": True,
        "actionable_pending_count": 0,
        "integrity_failure_count": 0,
    }
    for key, value in expected.items():
        observed = result.get(key)
        if key == "status":
            observed = str(observed).upper()
        if observed != value:
            failures.append(
                f"forecast extension bridge {key} is {observed!r}, expected {value!r}"
            )
    extension_path = _resolve(
        root, result.get("extension_run_manifest", result.get("path"))
    )
    expected_extension_hash = result.get(
        "extension_manifest_sha256", result.get("sha256")
    )
    extension_manifest = _load_json(extension_path)
    if extension_path is None or not extension_path.is_file():
        failures.append("forecast extension manifest is missing")
    elif not isinstance(expected_extension_hash, str) or len(
        expected_extension_hash
    ) != 64:
        failures.append("forecast extension manifest SHA-256 is missing")
    elif sha256_file(extension_path) != expected_extension_hash:
        failures.append("forecast extension manifest SHA-256 does not match")
    if not isinstance(extension_manifest, Mapping):
        failures.append("forecast extension manifest is invalid")
        extension_manifest = {}
    if str(extension_manifest.get("status", "")).upper() != "COMPLETE":
        failures.append("forecast extension manifest is not COMPLETE")
    if extension_manifest.get("terminal") is not True:
        failures.append("forecast extension manifest is not terminal")
    if extension_manifest.get("v1_outputs_rewritten") is not False:
        failures.append("forecast extension does not prove v1 remained read-only")
    gates = extension_manifest.get("completion_gates", {})
    if not isinstance(gates, Mapping) or not gates or not all(
        value is True for value in gates.values()
    ):
        failures.append("forecast extension completion gates are incomplete")

    v1_sha = (
        sha256_file(v1_manifest_path)
        if v1_manifest_path is not None and v1_manifest_path.is_file()
        else None
    )
    v1_lineage = result.get("v1_lineage", {})
    v1_lineage = v1_lineage if isinstance(v1_lineage, Mapping) else {}
    extension_baseline = extension_manifest.get("baseline_lineage", {})
    extension_baseline = (
        extension_baseline if isinstance(extension_baseline, Mapping) else {}
    )
    if (
        not v1_sha
        or v1_lineage.get("manifest_sha256") != v1_sha
        or extension_baseline.get("baseline_manifest_sha256") != v1_sha
        or extension_manifest.get("baseline_manifest_sha256_after") != v1_sha
    ):
        failures.append("forecast extension does not pin the current v1 manifest hash")

    counts = result.get("counts", {})
    counts = counts if isinstance(counts, Mapping) else {}
    expected_counts = {
        "task_variant_cells": 21,
        "seeds": 3,
        "seed_runs": 63,
        "deployment_days": 365,
        "scenario_npz_verified": 365,
        "failed_seeds": 0,
    }
    for key, value in expected_counts.items():
        if int(counts.get(key, -1) or 0) != value:
            failures.append(
                f"forecast extension {key} is {counts.get(key)!r}, expected {value}"
            )
    day_index = result.get("forecast_day_index", {})
    day_index = day_index if isinstance(day_index, Mapping) else {}
    if not (
        str(day_index.get("status", "")).upper() == "COMPLETE"
        and int(day_index.get("day_count", 0) or 0) == 365
        and int(day_index.get("deployment_day_count", 0) or 0) == 365
        and day_index.get("deployment_coverage_complete") is True
        and int(day_index.get("scenario_npz_verified_count", 0) or 0) == 365
    ):
        failures.append("forecast extension day-index coverage is incomplete")
    index_path = _resolve(root, day_index.get("index_path"))
    if (
        index_path is None
        or not index_path.is_file()
        or not isinstance(day_index.get("index_sha256"), str)
        or sha256_file(index_path) != day_index.get("index_sha256")
    ):
        failures.append("forecast extension day-index SHA-256 does not verify")
    index_payload = _load_json(index_path)
    index_days = (
        index_payload.get("days", [])
        if isinstance(index_payload, Mapping)
        else []
    )
    indexed_dates = {
        str(record.get("operating_date"))
        for record in index_days
        if isinstance(record, Mapping)
    }
    if not (
        isinstance(index_payload, Mapping)
        and str(index_payload.get("status", "")).upper() == "COMPLETE"
        and int(index_payload.get("day_count", 0) or 0) == 365
        and int(index_payload.get("deployment_day_count", 0) or 0) == 365
        and index_payload.get("deployment_coverage_complete") is True
        and isinstance(index_days, list)
        and len(index_days) == 365
        and len(indexed_dates) == 365
        and all(
            isinstance(record, Mapping)
            and len(str(record.get("sha256", ""))) == 64
            and record.get("realized_targets_used_to_generate_forecast") is False
            and record.get("missing_realized_targets_imputed") is False
            for record in index_days
        )
    ):
        failures.append("forecast extension day-index content is invalid")
    energy_records = result.get("energy_hypothesis_metrics", [])
    workload_records = result.get("workload_hypothesis_metrics", [])
    if not isinstance(energy_records, list) or {
        str(record.get("task"))
        for record in energy_records
        if isinstance(record, Mapping)
    } != {"nyiso_da", "nyiso_rt"}:
        failures.append("forecast extension H1 records do not cover DA and RT")
    if not isinstance(workload_records, list) or len(workload_records) != 1:
        failures.append("forecast extension H2 record is missing or duplicated")
    return {
        "passed": not failures,
        "failures": failures,
        "extension_manifest": (
            _relative(root, extension_path)
            if extension_path is not None and extension_path.is_file()
            else None
        ),
        "v1_manifest_sha256": v1_sha,
        "counts": dict(counts),
        "day_index": dict(day_index),
    }


def _build_physical_table(
    root: Path,
    run_id: str,
    physical_result: Mapping[str, Any],
    calibration_manifest: Any,
) -> tuple[Path, pd.DataFrame]:
    source = _result_path(root, physical_result, "outputs.facility_metrics")
    frame = pd.DataFrame()
    if source is not None and source.is_file():
        frame = pd.read_parquet(source)
    elif isinstance(calibration_manifest, Mapping):
        rows: list[dict[str, Any]] = []
        for key in ("primary", "fixed_mst_sensitivity"):
            branch = calibration_manifest.get(key, {})
            if isinstance(branch, Mapping):
                rows.extend(item for item in branch.get("metrics", []) if isinstance(item, Mapping))
        frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.insert(0, "run_id", run_id)
        frame.insert(1, "status", "completed_real_experiment")
    path = root / "artifacts/tables/nlr_twin_validation.csv"
    fields = [
        "run_id",
        "status",
        "clock_interpretation",
        "split",
        "model",
        "row_count",
        "facility_mae_kw",
        "facility_rmse_kw",
        "pue_mae",
        "source_relationship",
        "measurement_relationship",
        "evidence_role",
    ]
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records") if not frame.empty else []
    _atomic_csv(path, rows, fields)
    return path, frame


def _forecast_rows(metrics: Any, task_names: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = metrics.get("tasks", {}) if isinstance(metrics, Mapping) else {}
    if not isinstance(tasks, Mapping):
        return rows
    for task, task_payload in tasks.items():
        if task not in task_names or not isinstance(task_payload, Mapping):
            continue
        for split in ("validation", "locked_test"):
            for record in task_payload.get(split, []):
                if isinstance(record, Mapping):
                    rows.append({**record, "task": task, "split": split, "model_family": "baseline_or_tree"})
        for model_key in ("dgpt_e", "cfc_w"):
            model = task_payload.get(model_key)
            if not isinstance(model, Mapping):
                continue
            for split in ("validation", "locked_test"):
                record = model.get(split)
                if not isinstance(record, Mapping):
                    continue
                if task == "nlr_workload" and any(isinstance(record.get(key), Mapping) for key in ("count", "volume")):
                    for target in ("count", "volume"):
                        nested = record.get(target)
                        if isinstance(nested, Mapping):
                            rows.append(
                                {
                                    "task": task,
                                    "split": split,
                                    "model": model_key,
                                    "model_family": "preregistered_advanced",
                                    "target": target,
                                    **nested,
                                }
                            )
                    if record.get("joint_log_rmse") is not None:
                        rows.append(
                            {
                                "task": task,
                                "split": split,
                                "model": model_key,
                                "model_family": "preregistered_advanced",
                                "target": "joint",
                                "joint_log_rmse": record.get("joint_log_rmse"),
                            }
                        )
                else:
                    rows.append(
                        {
                            "task": task,
                            "split": split,
                            "model": model_key,
                            "model_family": "preregistered_advanced",
                            **record,
                        }
                    )
    return rows


def _build_forecast_tables(
    root: Path,
    run_id: str,
    forecast_manifest: Any,
    forecast_result: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, pd.DataFrame], Path | None]:
    metrics_path = _forecast_metrics_path(root, forecast_manifest, forecast_result)
    metrics = _load_json(metrics_path)
    common = {
        "run_id": run_id,
        "status": "completed_real_experiment",
        "evidence_role": "derived",
        "source_relationship": _REAL_SOURCE,
        "measurement_relationship": _NOT_STUDY_MEASURED,
    }
    nyiso_rows = [{**common, **row} for row in _forecast_rows(metrics, {"nyiso_da", "nyiso_rt"})]
    workload_rows = [{**common, **row} for row in _forecast_rows(metrics, {"nlr_workload"})]
    paths = {
        "nyiso": root / "artifacts/tables/nyiso_forecast_main.csv",
        "workload": root / "artifacts/tables/workload_forecast_main.csv",
    }
    fields = [
        "run_id",
        "task",
        "split",
        "model",
        "model_family",
        "target",
        "status",
        "mae",
        "rmse",
        "wape",
        "r2",
        "mean_pinball",
        "coverage_90",
        "n",
        "evidence_role",
        "source_relationship",
        "measurement_relationship",
    ]
    _atomic_csv(paths["nyiso"], nyiso_rows, fields)
    _atomic_csv(paths["workload"], workload_rows, fields)
    frames = {"nyiso": pd.DataFrame(nyiso_rows), "workload": pd.DataFrame(workload_rows)}
    return paths, frames, metrics_path


def _build_optimization_table(
    root: Path,
    run_id: str,
    optimization_result: Mapping[str, Any],
) -> tuple[Path, pd.DataFrame, pd.DataFrame, Path | None]:
    source = _result_path(root, optimization_result, "paths.result_table", "result_table")
    raw = pd.read_csv(source) if source is not None and source.is_file() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "phase",
        "policy_id",
        "regime",
        "reduced_scenario_count",
        "constraint_variant",
    ]
    metrics = [
        "objective_usd",
        "expected_total_cost_usd",
        "cvar_total_cost_usd",
        "expected_sla_shortfall_work_units",
        "expected_grid_energy_mwh",
        "rebound_peak_mw",
        "maximum_zone_concentration_share",
    ]
    if not raw.empty and all(column in raw for column in group_columns):
        for keys, group in raw.groupby(group_columns, dropna=False, sort=True):
            row: dict[str, Any] = {
                "run_id": run_id,
                "status": "completed_counterfactual_cells",
                "cell_count": len(group),
                "evidence_role": "counterfactual",
                "source_relationship": "counterfactual_using_third_party_referenced_public_data",
                "measurement_relationship": _NOT_STUDY_MEASURED,
                "counterfactual": True,
            }
            row.update(dict(zip(group_columns, keys)))
            for metric in metrics:
                if metric in group:
                    numeric = pd.to_numeric(group[metric], errors="coerce").dropna()
                    if not numeric.empty:
                        row[f"{metric}_mean"] = float(numeric.mean())
                        row[f"{metric}_median"] = float(numeric.median())
            rows.append(row)
    path = root / "artifacts/tables/counterfactual_optimization_main.csv"
    fields = [
        "run_id",
        "phase",
        "policy_id",
        "regime",
        "reduced_scenario_count",
        "constraint_variant",
        "status",
        "cell_count",
        "expected_total_cost_usd_mean",
        "cvar_total_cost_usd_mean",
        "expected_sla_shortfall_work_units_mean",
        "rebound_peak_mw_mean",
        "maximum_zone_concentration_share_mean",
        "evidence_role",
        "source_relationship",
        "measurement_relationship",
        "counterfactual",
    ]
    _atomic_csv(path, rows, fields)
    return path, pd.DataFrame(rows), raw, source


def _build_hypothesis_table(
    root: Path,
    run_id: str,
    hypotheses: Mapping[str, Any],
) -> tuple[Path, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for hypothesis_id in _HYPOTHESIS_IDS:
        record = hypotheses.get(hypothesis_id)
        if not isinstance(record, Mapping):
            continue
        scalar = {
            key: value
            for key, value in record.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        nested = {
            key: value
            for key, value in record.items()
            if isinstance(value, (dict, list, tuple))
        }
        rows.append(
            {
                "run_id": run_id,
                "hypothesis_id": hypothesis_id,
                **scalar,
                "details_json": json.dumps(nested, sort_keys=True),
                "evidence_role": "derived",
                "measurement_relationship": _NOT_STUDY_MEASURED,
            }
        )
    path = root / "artifacts/tables/h0_h6_results.csv"
    _atomic_csv(
        path,
        rows,
        [
            "run_id",
            "hypothesis_id",
            "status",
            "claim",
            "reason",
            "details_json",
            "evidence_role",
            "measurement_relationship",
        ],
    )
    return path, pd.DataFrame(rows)


def _flatten_failures(
    root: Path,
    physical_result: Mapping[str, Any],
    forecast_result: Mapping[str, Any],
    forecast_manifest: Any,
    optimization_result: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    cell_audit: Mapping[str, Any],
    raw_optimization: pd.DataFrame,
    transfer_manifest: Any,
    xcel_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(stage: str, work_unit: str, status: str, reason: str, **extra: Any) -> None:
        rows.append(
            {
                "stage": stage,
                "work_unit": work_unit,
                "status": status,
                "reason": reason,
                **extra,
            }
        )

    for blocker in physical_result.get("blockers", []):
        if isinstance(blocker, Mapping):
            add(
                "physical",
                str(blocker.get("code", "physical_branch")),
                _status(physical_result),
                str(blocker.get("message", "")),
                exception_type=blocker.get("exception_type"),
            )
    if not xcel_audit.get("passed"):
        add(
            "xcel_local_replay",
            "nlr300_xcel_2024_tariff_replay",
            str(optimization_result.get("xcel_local_subcase_status", "missing")),
            "; ".join(map(str, xcel_audit.get("failures", [])))
            or "Xcel local tariff replay is incomplete",
        )
    elif int(xcel_audit.get("missing_interval_count", 0) or 0) > 0:
        coverage = xcel_audit.get("coverage", {})
        coverage = coverage if isinstance(coverage, Mapping) else {}
        add(
            "xcel_local_replay",
            "nlr300_2024_source_coverage",
            "data_limitation",
            f"{xcel_audit.get('missing_interval_count')} of "
            f"{coverage.get('expected_interval_count')} tariff-effective source intervals "
            "are absent and were not imputed",
        )
    manifest_failures = forecast_manifest.get("task_failures", []) if isinstance(forecast_manifest, Mapping) else []
    for failure in manifest_failures:
        if isinstance(failure, Mapping):
            add(
                "forecast",
                str(failure.get("task", failure.get("work_unit", "task"))),
                "failed",
                str(failure.get("error", failure.get("reason", failure))),
            )
    if isinstance(forecast_manifest, Mapping):
        for record in forecast_manifest.get("outputs", []):
            if not isinstance(record, Mapping) or not str(record.get("path", "")).endswith("failed_seeds.json"):
                continue
            failed_seeds = _load_json(_resolve(root, record.get("path")))
            for failure in failed_seeds if isinstance(failed_seeds, list) else []:
                if isinstance(failure, Mapping):
                    add(
                        "forecast",
                        f"{failure.get('task', 'task')}:seed={failure.get('seed', 'unknown')}",
                        "failed_seed",
                        str(failure.get("error", failure.get("reason", failure))),
                    )

    cell_status_path = _result_path(root, optimization_result, "paths.cell_status", "cell_status_path")
    cell_records = _load_json(cell_status_path)
    for cell in cell_records if isinstance(cell_records, list) else []:
        if not isinstance(cell, Mapping):
            continue
        status = str(cell.get("current_status", "PENDING")).upper()
        if status == "SUCCESS":
            continue
        checkpoint = _load_json(_resolve(root, cell.get("checkpoint_path")))
        diagnostics = checkpoint.get("diagnostics", {}) if isinstance(checkpoint, Mapping) else {}
        reason = (
            diagnostics.get("message")
            if isinstance(diagnostics, Mapping)
            else None
        ) or f"optimization cell is {status}"
        add(
            "optimization",
            str(cell.get("checkpoint_key", "cell")),
            status.lower(),
            str(reason),
            phase=cell.get("phase"),
            policy_id=cell.get("policy_id"),
            seed=cell.get("seed"),
            regime=cell.get("regime"),
            reduced_scenario_count=cell.get("reduced_count"),
            checkpoint_path=cell.get("checkpoint_path"),
        )

    if not raw_optimization.empty:
        if "termination_condition" in raw_optimization:
            for _, record in raw_optimization.loc[
                ~raw_optimization["termination_condition"].astype(str).str.lower().eq("optimal")
            ].iterrows():
                add(
                    "optimization",
                    str(record.get("checkpoint_key", "nonoptimal_success")),
                    "nonoptimal_termination",
                    f"termination_condition={record.get('termination_condition')}",
                    phase=record.get("phase"),
                    policy_id=record.get("policy_id"),
                    seed=record.get("study_seed", record.get("scenario_seed")),
                    regime=record.get("regime"),
                    reduced_scenario_count=record.get("reduced_scenario_count"),
                    solver_status=record.get("solver_status"),
                )
        gap_column = next(
            (column for column in ("mip_gap", "relative_gap", "solver_gap") if column in raw_optimization),
            None,
        )
        if gap_column:
            gaps = pd.to_numeric(raw_optimization[gap_column], errors="coerce")
            for index in gaps[gaps.gt(0)].index:
                record = raw_optimization.loc[index]
                add(
                    "optimization",
                    str(record.get("checkpoint_key", f"row={index}")),
                    "nonzero_gap",
                    f"{gap_column}={gaps.loc[index]}",
                    phase=record.get("phase"),
                    policy_id=record.get("policy_id"),
                    seed=record.get("study_seed", record.get("scenario_seed")),
                    regime=record.get("regime"),
                    reduced_scenario_count=record.get("reduced_scenario_count"),
                )

    for hypothesis_id in _HYPOTHESIS_IDS:
        record = hypotheses.get(hypothesis_id)
        status = str(record.get("status", "missing")).lower() if isinstance(record, Mapping) else "missing"
        if status not in _FINISHED_HYPOTHESIS_STATUSES:
            reason = record.get("reason", status) if isinstance(record, Mapping) else "missing"
            add("hypotheses", hypothesis_id, status, str(reason))

    if cell_audit.get("full_year_extension_required") and not cell_audit.get("extension_complete"):
        add(
            "optimization",
            "full_year_extension",
            "incomplete",
            (
                f"year-day coverage={cell_audit.get('year_day_count')}/"
                f"{cell_audit.get('expected_year_day_count')}"
            ),
        )

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                child = f"{path}.{key}"
                if "extrapolat" in str(key).lower() and (
                    nested is True or str(nested).lower() in {"true", "extrapolated", "outside_support"}
                ):
                    add("transfer", child, "extrapolated", f"{key}={nested}")
                visit(nested, child)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(transfer_manifest, "transfer_manifest")
    return rows


def _generate_figures(
    root: Path,
    physical: pd.DataFrame,
    forecast_frames: Mapping[str, pd.DataFrame],
    optimization: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> tuple[list[Path], list[dict[str, Any]]]:
    figures: list[Path] = []
    failures: list[dict[str, Any]] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], [{"figure": "all", "reason": f"{type(exc).__name__}: {exc}"}]

    figure_root = root / "artifacts/figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        path = figure_root / name
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        figures.append(path)

    try:
        required = {
            "split",
            "model",
            "clock_interpretation",
            "facility_rmse_kw",
        }
        if not physical.empty and required.issubset(physical.columns):
            selected = physical.loc[
                physical["split"].astype(str).eq("locked_test")
                & physical["model"]
                .astype(str)
                .eq("frozen_piecewise_linear_surrogate")
            ]
            if not selected.empty:
                plt.figure(figsize=(7.2, 4.2))
                plt.bar(
                    selected["clock_interpretation"].astype(str),
                    pd.to_numeric(selected["facility_rmse_kw"], errors="coerce"),
                    color=["#2F5597", "#ED7D31"][: len(selected)],
                )
                plt.ylabel("Locked-test facility RMSE (kW)")
                plt.title("NLR300 source-clock sensitivity")
                plt.xticks(rotation=12, ha="right")
                save("paper_physical_timezone_sensitivity.png")
    except Exception as exc:
        plt.close("all")
        failures.append({"figure": "paper_physical_timezone_sensitivity.png", "reason": str(exc)})

    try:
        combined = pd.concat(
            [frame for frame in forecast_frames.values() if not frame.empty],
            ignore_index=True,
        )
        if not combined.empty and {"task", "model", "split", "mae"}.issubset(combined.columns):
            selected = combined.loc[combined["split"].astype(str).eq("locked_test")].copy()
            selected["label"] = selected["task"].astype(str) + ":" + selected["model"].astype(str)
            selected["mae"] = pd.to_numeric(selected["mae"], errors="coerce")
            selected = selected.dropna(subset=["mae"])
            if not selected.empty:
                plt.figure(figsize=(10.5, 4.8))
                plt.bar(selected["label"], selected["mae"], color="#5B9BD5")
                plt.ylabel("Locked-test MAE")
                plt.title("Source-backed forecasting results")
                plt.xticks(rotation=55, ha="right")
                save("paper_forecast_locked_test_mae.png")
    except Exception as exc:
        plt.close("all")
        failures.append({"figure": "paper_forecast_locked_test_mae.png", "reason": str(exc)})

    try:
        required = {
            "phase",
            "policy_id",
            "regime",
            "reduced_scenario_count",
            "constraint_variant",
            "expected_total_cost_usd_mean",
        }
        if not optimization.empty and required.issubset(optimization.columns):
            selected = optimization.loc[
                optimization["phase"].astype(str).eq("primary_96_day")
                & optimization["regime"].astype(str).eq("independent")
                & pd.to_numeric(optimization["reduced_scenario_count"], errors="coerce").eq(20)
                & optimization["constraint_variant"].astype(str).eq("base")
            ].copy()
            if not selected.empty:
                plt.figure(figsize=(8.5, 4.5))
                plt.bar(
                    selected["policy_id"].astype(str),
                    pd.to_numeric(selected["expected_total_cost_usd_mean"], errors="coerce"),
                    color="#70AD47",
                )
                plt.ylabel("Mean expected total cost (USD)")
                plt.title("96-day NYISO counterfactual policy comparison")
                save("paper_counterfactual_policy_cost.png")
    except Exception as exc:
        plt.close("all")
        failures.append({"figure": "paper_counterfactual_policy_cost.png", "reason": str(exc)})

    try:
        if not hypotheses.empty and {"hypothesis_id", "status"}.issubset(hypotheses.columns):
            palette = {
                "supported": "#70AD47",
                "not_supported": "#C00000",
                "evaluated": "#5B9BD5",
                "not_evaluable": "#A5A5A5",
                "missing": "#7F7F7F",
            }
            statuses = hypotheses["status"].fillna("missing").astype(str)
            plt.figure(figsize=(7.5, 3.8))
            plt.bar(
                hypotheses["hypothesis_id"].astype(str),
                [1] * len(hypotheses),
                color=[palette.get(value, "#FFC000") for value in statuses],
            )
            for index, value in enumerate(statuses):
                plt.text(index, 0.5, value, ha="center", va="center", rotation=90, fontsize=8)
            plt.yticks([])
            plt.ylim(0, 1)
            plt.title("H0-H6 evidence status")
            save("paper_h0_h6_status.png")
    except Exception as exc:
        plt.close("all")
        failures.append({"figure": "paper_h0_h6_status.png", "reason": str(exc)})

    return figures, failures


def _branch_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "status",
            "run_id",
            "run_ids",
            "completed",
            "downstream_ready",
            "frozen_policy_sha256",
            "formal_dart_training_allowed",
            "native_rt_cross_head_label",
            "native_rt_selection_eligible",
            "counts",
            "completed_count",
            "failure_count",
            "pending_count",
            "auxiliary_failure_count",
            "auxiliary_pending_count",
            "actionable_pending_count",
            "gated_pending_count",
            "planned_cell_count",
            "primary_day_count",
            "full_year_extension_requested",
            "full_year_extension_gate_passed",
            "full_year_extension_started",
            "full_year_coverage_complete",
            "full_year_available_day_count",
            "full_year_expected_day_count",
            "full_year_missing_day_count",
            "full_year_missing_date_count",
            "timezone_economic_sensitivity_required",
            "timezone_economic_sensitivity_complete",
            "timezone_economic_sensitivity_status",
            "timezone_economic_sensitivity_planned_cell_count",
            "timezone_economic_sensitivity_success_cell_count",
            "timezone_economic_sensitivity_paired_date_seed_count",
            "timezone_sensitive",
            "timezone_substantive_conclusion_changed",
            "xcel_local_subcase_status",
            "manifest_path",
            "manifest_file_sha256",
            "output_root",
            "dart_overlay_manifest",
            "dart_overlay_manifest_sha256",
            "candidate_pool_contract",
            "terminal_status_cap",
            "run_manifest",
            "error_type",
            "error",
            "blockers",
        )
        if key in result
    }


def _classify(
    *,
    checks: Sequence[Mapping[str, Any]],
    preflight_result: Mapping[str, Any],
    physical_result: Mapping[str, Any],
    forecast_result: Mapping[str, Any],
    optimization_result: Mapping[str, Any],
    genuine_components: Sequence[str],
) -> tuple[str, str]:
    incomplete = [
        str(item["check_id"])
        for item in checks
        if item.get("required_for_complete") is True and item.get("status") != "passed"
    ]
    if not incomplete:
        return COMPLETE_PAPER, "all mandatory paper experiments and integrity gates completed"
    if genuine_components:
        return (
            PARTIAL_PAPER,
            "genuine paper experiments exist, but mandatory items remain incomplete: "
            + ", ".join(incomplete),
        )
    mandatory_gate_failed = (
        _status(preflight_result) == DEMO_COMPLETE_PAPER_BLOCKED.lower()
        or _status(physical_result) == "blocked"
        or any(
            item.get("check_id") in {"preflight_gate", "source_evidence_semantics"}
            and item.get("status") == "failed"
            for item in checks
        )
    )
    if mandatory_gate_failed:
        return (
            DEMO_COMPLETE_PAPER_BLOCKED,
            "no genuine paper experiment could proceed because a mandatory evidence gate failed",
        )
    branch_statuses = {
        _status(physical_result),
        _status(forecast_result),
        _status(optimization_result),
    }
    if "failed" in branch_statuses:
        return FAILED, "no genuine paper experiment completed and an executable branch failed"
    return (
        DEMO_COMPLETE_PAPER_BLOCKED,
        "no genuine paper experiment could be established from the available mandatory evidence",
    )


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not math.isfinite(number):
        return "unavailable"
    return f"{number:.6g}"


def _report_text(
    *,
    status: str,
    justification: str,
    run_id: str,
    genuine_components: Sequence[str],
    checks: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    physical_frame: pd.DataFrame,
    forecast_frames: Mapping[str, pd.DataFrame],
    optimization_frame: pd.DataFrame,
    hypothesis_audit: Mapping[str, Any],
    physical_timezone: Mapping[str, Any],
    economic_timezone: Any,
    cell_audit: Mapping[str, Any],
    xcel_audit: Mapping[str, Any],
    dart_audit: Mapping[str, Any],
) -> str:
    lines = [
        "# Research Run Summary",
        "",
        f"- Terminal status: `{status}`",
        f"- Run ID: `{run_id}`",
        f"- Generated: `{utc_now()}`",
        f"- Justification: {justification}",
        f"- Genuine paper components: `{', '.join(genuine_components) if genuine_components else 'none'}`",
        "- Synthetic paper fallback: `false`",
        "- NLR-to-NYISO relationship: parameter-transfer counterfactual; not an observed deployment, bid, dispatch, or realized revenue.",
        "",
        "## Completion counts",
        "",
        f"- Optimization cells: `{cell_audit.get('observed', {}).get('success', 0)}` successful, `{cell_audit.get('observed', {}).get('failed', 0)}` failed, `{cell_audit.get('observed', {}).get('pending_or_running', 0)}` pending/running.",
        f"- Primary operating days: `{cell_audit.get('primary_day_count', 0)}/96`.",
        f"- Full-year date coverage: `{cell_audit.get('year_day_count', 0)}/{cell_audit.get('expected_year_day_count', 365)}`.",
        f"- Recorded failures/limitations: `{len(failure_rows)}`; the exhaustive inventory is `artifacts/tables/failures.csv`.",
        "",
        "## Physical results",
        "",
    ]
    physical_selected = pd.DataFrame()
    if not physical_frame.empty and {"split", "model"}.issubset(physical_frame.columns):
        physical_selected = physical_frame.loc[
            physical_frame["split"].astype(str).eq("locked_test")
            & physical_frame["model"]
            .astype(str)
            .eq("frozen_piecewise_linear_surrogate")
        ]
    if physical_selected.empty:
        lines.append("- Locked-test physical metrics: unavailable.")
    else:
        for _, row in physical_selected.iterrows():
            lines.append(
                f"- `{row.get('clock_interpretation')}`: facility RMSE "
                f"`{_format_number(row.get('facility_rmse_kw'))} kW`, PUE MAE "
                f"`{_format_number(row.get('pue_mae'))}`."
            )
    lines.extend(["", "## Forecasting results", ""])
    forecast = pd.concat(
        [frame for frame in forecast_frames.values() if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in forecast_frames.values()) else pd.DataFrame()
    if forecast.empty or not {"split", "task", "model"}.issubset(forecast.columns):
        lines.append("- Locked-test forecast metrics: unavailable.")
    else:
        selected = forecast.loc[forecast["split"].astype(str).eq("locked_test")]
        for _, row in selected.iterrows():
            if row.get("mae") is not None and pd.notna(row.get("mae")):
                target_value = row.get("target")
                target = (
                    f"/{target_value}"
                    if target_value not in (None, "") and pd.notna(target_value)
                    else ""
                )
                lines.append(
                    f"- `{row.get('task')}/{row.get('model')}{target}`: MAE "
                    f"`{_format_number(row.get('mae'))}`, RMSE `{_format_number(row.get('rmse'))}`."
                )
    lines.extend(
        [
            "",
            "## DART-ValueSelect",
            "",
            "- Pre-DART immutable-v1 readiness gate: "
            f"`{'passed' if dart_audit.get('v1_readiness', {}).get('passed') else 'failed'}`; "
            "native-RT cross-head DART/action/value outputs remain "
            "`AUXILIARY_NOT_DA_GATE_ELIGIBLE`.",
            f"- Integrity gate: `{'passed' if dart_audit.get('passed') else 'failed'}`.",
            "- Selection/locked panels: "
            f"`{dart_audit.get('panel_counts', {}).get('selection', {}).get('operating_dates', 'unavailable')}`/"
            f"`{dart_audit.get('panel_counts', {}).get('locked_test', {}).get('operating_dates', 'unavailable')}` days.",
            "- H4 statuses: "
            f"`{dart_audit.get('hypothesis_statuses', {})}`.",
            "- All reported DART values are counterfactual replay outputs, not realized commercial revenue.",
            "",
            "## Economic and risk results",
            "",
        ]
    )
    selector_comparison = dart_audit.get("selector_comparison", {})
    selector_rows = (
        selector_comparison.get("locked_test_selector_rows", [])
        if isinstance(selector_comparison, Mapping)
        else []
    )
    if isinstance(selector_rows, list) and selector_rows:
        lines.extend(["", "### Frozen selector comparison", ""])
        for method in ("MAE-Select", "DART-ValueSelect"):
            method_rows = [
                row
                for row in selector_rows
                if isinstance(row, Mapping) and row.get("selector") == method
            ]
            members = sorted(
                {
                    str(row.get("winner_set_member"))
                    for row in method_rows
                    if row.get("winner_set_member") not in (None, "")
                }
            )
            values = [
                float(row["locked_test_A2_mean_daily_incremental_operational_value"])
                for row in method_rows
                if row.get("locked_test_A2_mean_daily_incremental_operational_value")
                is not None
                and pd.notna(
                    row.get(
                        "locked_test_A2_mean_daily_incremental_operational_value"
                    )
                )
            ]
            value_text = (
                f"`${_format_number(min(values))}` to `${_format_number(max(values))}`"
                if values
                else "unavailable"
            )
            lines.append(
                f"- `{method}` winner set: `{members or 'unavailable'}`; "
                f"locked-A2 mean daily counterfactual value range: {value_text}."
            )
        cssa = (
            selector_comparison.get("cssa", {})
            if isinstance(selector_comparison, Mapping)
            else {}
        )
        mae_cssa = (
            cssa.get("by_comparator_selector", {}).get("MAE-Select", {})
            if isinstance(cssa, Mapping)
            and isinstance(cssa.get("by_comparator_selector"), Mapping)
            else {}
        )
        value_lcbs = (
            mae_cssa.get("target_minus_comparator_value_CSSA_lcb_by_block", {})
            if isinstance(mae_cssa, Mapping)
            else {}
        )
        if isinstance(value_lcbs, Mapping):
            lines.append(
                "- Simultaneous set-valued `CSSA_7(DART-ValueSelect, MAE-Select)`: "
                f"`${_format_number(value_lcbs.get('7'))}` per day; 14/28-day "
                f"robustness: `${_format_number(value_lcbs.get('14'))}` / "
                f"`${_format_number(value_lcbs.get('28'))}`."
            )
    if optimization_frame.empty:
        lines.append("- Counterfactual optimization metrics: unavailable.")
    else:
        selected = optimization_frame
        if "phase" in selected:
            selected = selected.loc[selected["phase"].astype(str).eq("primary_96_day")]
        if "regime" in selected:
            selected = selected.loc[selected["regime"].astype(str).eq("independent")]
        if "reduced_scenario_count" in selected:
            selected = selected.loc[
                pd.to_numeric(selected["reduced_scenario_count"], errors="coerce").eq(20)
            ]
        if selected.empty:
            lines.append("- Primary nominal 20-scenario policy metrics: unavailable.")
        else:
            for _, row in selected.iterrows():
                lines.append(
                    f"- `{row.get('policy_id')}`: mean expected cost "
                    f"`${_format_number(row.get('expected_total_cost_usd_mean'))}`, mean CVaR "
                    f"`${_format_number(row.get('cvar_total_cost_usd_mean'))}`, mean SLA shortfall "
                    f"`{_format_number(row.get('expected_sla_shortfall_work_units_mean'))}` work units."
                )
    coverage = xcel_audit.get("coverage", {})
    coverage = coverage if isinstance(coverage, Mapping) else {}
    lines.extend(
        [
            "",
            "## Xcel local tariff replay",
            "",
            f"- Status: `{xcel_audit.get('status', 'missing')}`; integrity gate: "
            f"`{'passed' if xcel_audit.get('passed') else 'failed'}`.",
            "- Verified tariff/source-overlap coverage: "
            f"`{coverage.get('observed_interval_count', 'unavailable')}/"
            f"{coverage.get('expected_interval_count', 'unavailable')}` intervals and "
            f"`{coverage.get('complete_month_count', 'unavailable')}/"
            f"{coverage.get('expected_month_count', 'unavailable')}` complete months.",
            "- The retained 2024 rate overlaps eight May-December months; "
            f"`{xcel_audit.get('complete_overlap_month_count', 'unavailable')}` months are "
            "interval-complete and "
            f"`{xcel_audit.get('missing_interval_count', 'unavailable')}` missing source "
            "intervals remain explicitly unfilled.",
            "- January-April precede the 2024-05-01 rate snapshot and remain "
            "tariff-inapplicable rows with no tariff cost.",
            "- This is a tariff-defined counterfactual: event calls are zero and no "
            "realized response or savings are claimed.",
            "",
            "## H0-H6",
            "",
        ]
    )
    for hypothesis_id in _HYPOTHESIS_IDS:
        lines.append(f"- `{hypothesis_id}`: `{hypothesis_audit.get('states', {}).get(hypothesis_id, 'missing')}`.")
    lines.extend(["", "## Timezone sensitivity", ""])
    if physical_timezone.get("passed"):
        metrics = physical_timezone.get("locked_test_metrics", {})
        lines.append(
            "- Physical comparison completed: primary UTC-source-clock RMSE "
            f"`{_format_number(metrics.get('primary_locked_test_rmse_kw'))} kW`; fixed-MST-source-clock RMSE "
            f"`{_format_number(metrics.get('fixed_mst_locked_test_rmse_kw'))} kW`."
        )
    else:
        lines.append("- Physical source-clock sensitivity: incomplete.")
    if isinstance(economic_timezone, Mapping):
        lines.append(
            "- Primary pooled economic endpoint comparison completed; machine-readable outcome is in the integrity audit."
        )
    else:
        lines.append("- Primary pooled economic endpoint comparison across the two source-clock interpretations: unavailable.")
    lines.extend(["", "## Integrity gates", ""])
    for item in checks:
        lines.append(
            f"- `{item.get('check_id')}`: `{item.get('status')}` — {item.get('message', '')}"
        )
    lines.extend(
        [
            "",
            "## Evidence boundaries",
            "",
            "- NLR295, NLR300, NASA POWER, SRRL, NYISO, and Xcel inputs are third-party referenced public data or official documents; they were not measured by this study.",
            "- NASA POWER is gridded reconstruction/reanalysis, not locally measured weather or PV output.",
            "- Xcel results are tariff-defined counterfactuals with zero observed event calls and no realized customer response or savings evidence.",
            "- NYISO results are counterfactual simulations for transferred virtual nodes; no NLR facility submitted NYISO bids or earned realized market revenue.",
            "- Missing, failed, timed-out, infeasible, extrapolated, nonoptimal, and nonzero-gap records are retained in `artifacts/tables/failures.csv`; unavailable metrics are not imputed.",
            "",
        ]
    )
    return "\n".join(lines)


def _persist_state(
    root: Path,
    state: PipelineState | None,
    *,
    run_id: str,
    status: str,
    paper_results_available: bool,
    paper_training_started: bool,
    branch_summaries: Mapping[str, Any],
    counts: Mapping[str, Any],
    output_paths: Sequence[Path],
    justification: str,
) -> PipelineState:
    state = state or PipelineState(root)
    state.data["run_id"] = run_id
    state.data["profile"] = "paper"
    state.data["terminal_status"] = status
    state.data.setdefault("stages", {})["paper_execution"] = {
        "name": "paper_execution",
        "status": "failed" if status == FAILED else "completed",
        "terminal": True,
        "terminal_status": status,
        "finished_at": utc_now(),
        "outputs": [_relative(root, path) for path in output_paths if path.is_file()],
        "message": justification,
    }
    state.data["paper"] = {
        "run_status": status,
        "terminal": True,
        "terminal_status": status,
        "paper_training_started": paper_training_started,
        "paper_results_available": paper_results_available,
        "synthetic_fallback": False,
        "branch_results": branch_summaries,
        "counts": dict(counts),
    }
    state.data["updated_at"] = utc_now()
    atomic_json(state.path, state.data)
    atomic_text(
        state.status_path,
        "\n".join(
            [
                "# Current Stage Status",
                "",
                f"- Run ID: `{state.data['run_id']}`",
                "- Stage: `paper_execution`",
                f"- Terminal status: `{status}`",
                "- Terminal: `true`",
                f"- Paper results available: `{str(paper_results_available).lower()}`",
                f"- Paper training started: `{str(paper_training_started).lower()}`",
                "- Synthetic paper fallback: `false`",
                f"- Physical branch: `{branch_summaries['physical'].get('status', 'missing')}`",
                f"- Forecast branch: `{branch_summaries['forecast'].get('status', 'missing')}`",
                f"- Optimization branch: `{branch_summaries['optimization'].get('status', 'missing')}`",
                f"- Successful optimization cells: `{counts.get('optimization_success', 0)}`",
                f"- Failed optimization cells: `{counts.get('optimization_failed', 0)}`",
                f"- Justification: {justification}",
                f"- Updated: `{state.data['updated_at']}`",
                "",
            ]
        ),
    )
    return state


def finalize_paper_run(
    root: Path,
    config: dict,
    run_id: str,
    preflight_result: dict,
    physical_result: dict,
    forecast_result: dict,
    optimization_result: dict,
    dart_result: dict | None = None,
    dart_v1_readiness_result: dict | None = None,
    state: PipelineState | None = None,
) -> dict[str, Any]:
    """Create a truthful, evidence-gated terminal paper record.

    ``COMPLETE_PAPER`` is deliberately harder to reach than a collection of
    successful branch return codes.  Missing metrics remain missing, and any
    real partial result is retained as ``PARTIAL_PAPER`` rather than being
    replaced by demo or synthetic evidence.
    """

    root = Path(root).resolve()
    actionable_pending = int(
        optimization_result.get(
            "actionable_pending_count",
            optimization_result.get("pending_count", 0),
        )
        or 0
    )
    if actionable_pending > 0:
        raise RuntimeError("paper finalization requires zero pending optimization cells")

    checks: list[dict[str, Any]] = []
    source_path = _resolve(root, preflight_result.get("source_manifest")) or (
        root / "artifacts/manifests/source_manifest.json"
    )
    source_manifest = _load_json(source_path)
    physical_branch_manifest_path = _result_path(root, physical_result, "manifest_path")
    physical_branch_manifest = _load_json(physical_branch_manifest_path)
    calibration_path = _result_path(root, physical_result, "outputs.physical_manifest")
    calibration_manifest = _load_json(calibration_path)
    transfer_path = _result_path(root, physical_result, "outputs.transfer_manifest")
    transfer_manifest = _load_json(transfer_path)
    forecast_manifest_path = _result_path(root, forecast_result, "run_manifest")
    forecast_manifest = _load_json(forecast_manifest_path)
    xcel_audit = _xcel_replay_audit(root, optimization_result)

    preflight_passed = _status(preflight_result) == PAPER_PREFLIGHT_READY.lower() or (
        isinstance(preflight_result.get("timezone_gate"), Mapping)
        and _status(preflight_result.get("timezone_gate")) == "passed"
    )
    _check(
        checks,
        "preflight_gate",
        preflight_passed,
        evidence={"status": preflight_result.get("status")},
        message="mandatory source/timezone preflight passed" if preflight_passed else "mandatory preflight did not pass",
    )

    semantics = _source_semantics_audit(source_manifest)
    _check(
        checks,
        "source_evidence_semantics",
        semantics["passed"],
        evidence=semantics,
        message="reference-only source semantics are consistent" if semantics["passed"] else "; ".join(semantics["failures"]),
    )

    physical_complete = _status(physical_result) == "completed"
    _check(
        checks,
        "physical_branch_complete",
        physical_complete,
        evidence={"status": physical_result.get("status"), "counts": physical_result.get("counts")},
        message="physical calibration and transfer completed" if physical_complete else "physical calibration/transfer incomplete",
    )
    physical_hashes = _verify_output_map(
        root,
        physical_result.get("outputs"),
        physical_result.get("output_hashes"),
    )
    _check(
        checks,
        "physical_output_hashes",
        physical_hashes["passed"] if physical_complete else False,
        evidence=physical_hashes,
        message=f"verified {physical_hashes['verified_count']}/{physical_hashes['declared_count']} physical outputs",
    )
    frozen_audit = _physical_frozen_audit(
        root, physical_result, calibration_manifest, transfer_manifest
    )
    _check(
        checks,
        "frozen_calibration_and_transfer",
        frozen_audit["passed"] if physical_complete else False,
        evidence=frozen_audit,
        message="physical calibration and four transfer twins are frozen" if frozen_audit["passed"] else "; ".join(frozen_audit["failures"]),
    )

    forecast_complete = _status(forecast_result) == "complete"
    forecast_counts = (
        forecast_manifest.get("counts", {})
        if isinstance(forecast_manifest, Mapping)
        else forecast_result.get("counts", {})
    )
    forecast_count_complete = (
        int(forecast_counts.get("tasks_requested", 0) or 0) > 0
        and int(forecast_counts.get("tasks_completed", 0) or 0)
        == int(forecast_counts.get("tasks_requested", 0) or 0)
        and int(forecast_counts.get("tasks_failed", 0) or 0) == 0
        and int(forecast_counts.get("failed_seeds", 0) or 0) == 0
    )
    _check(
        checks,
        "forecast_branch_complete",
        forecast_complete and forecast_count_complete,
        evidence={"status": forecast_result.get("status"), "counts": forecast_counts},
        message="all requested forecast tasks and seeds completed" if forecast_complete and forecast_count_complete else "forecast task or seed coverage is incomplete",
    )
    forecast_hashes = _verify_output_records(
        root,
        forecast_manifest.get("outputs") if isinstance(forecast_manifest, Mapping) else None,
    )
    _check(
        checks,
        "forecast_output_hashes",
        forecast_hashes["passed"] if forecast_complete else False,
        evidence=forecast_hashes,
        message=f"verified {forecast_hashes['verified_count']}/{forecast_hashes['declared_count']} forecast outputs",
    )
    forecast_scope = _forecast_scope_audit(
        root, config, forecast_manifest, forecast_result
    )
    _check(
        checks,
        "forecast_preregistered_scope_and_chronology",
        forecast_scope["passed"] if forecast_complete else False,
        evidence=forecast_scope,
        message=(
            "v1 forecast tasks, its retained four-seed auxiliary panel, training scope, frozen full-year days, and chronological splits match the completed recovery contract"
            if forecast_scope["passed"]
            else "; ".join(forecast_scope["failures"])
        ),
    )
    forecast_extension_required = bool(
        isinstance(config.get("optimization_branch", {}), Mapping)
        and config.get("optimization_branch", {}).get("run_ofat_sensitivities")
        is True
    )
    forecast_extension_audit = _forecast_extension_bridge_audit(
        root, forecast_result, forecast_manifest_path
    )
    _check(
        checks,
        "forecast_extension_and_365_day_bridge_complete",
        forecast_extension_audit["passed"],
        required_for_complete=forecast_extension_required,
        evidence=forecast_extension_audit,
        message=(
            "post-v1 extension trials, hashes, lineage, and 365 deployment days are complete"
            if forecast_extension_audit["passed"]
            else "; ".join(forecast_extension_audit["failures"])
        ),
    )

    execution_config = config.get("paper_execution", {})
    execution_config = (
        execution_config if isinstance(execution_config, Mapping) else {}
    )
    dart_required = bool(
        execution_config.get("dart_enabled") is True
        or execution_config.get("dart_config_path") not in (None, "")
    )
    if dart_required:
        from dcvpp.experiments.dart_pipeline import (
            audit_dart_pipeline_evidence,
            audit_v1_dart_readiness_evidence,
        )

        dart_v1_readiness_audit = audit_v1_dart_readiness_evidence(
            root,
            dart_v1_readiness_result,
            expected_v1_manifest=forecast_manifest_path,
        )
        dart_audit = audit_dart_pipeline_evidence(root, dart_result)
        dart_audit["v1_readiness"] = dart_v1_readiness_audit
    else:
        dart_v1_readiness_audit = {
            "passed": True,
            "status": "NOT_REQUIRED_LEGACY_COMPATIBILITY",
            "failures": [],
        }
        dart_audit = {
            "passed": True,
            "status": "NOT_REQUIRED_LEGACY_COMPATIBILITY",
            "failures": [],
            "panel_counts": {},
            "hypothesis_statuses": {},
            "dynamic_artifact_inventory_required": False,
        }
    _check(
        checks,
        "dart_v1_readiness_and_frozen_policy_complete",
        bool(dart_v1_readiness_audit.get("passed")),
        required_for_complete=dart_required,
        evidence=dart_v1_readiness_audit,
        message=(
            "immutable v1 was evaluated on the frozen 2024 validation projection; "
            "native-RT cross-head outputs are auxiliary-only"
            if dart_v1_readiness_audit.get("passed")
            else "; ".join(
                map(str, dart_v1_readiness_audit.get("failures", []))
            )
        ),
    )
    _check(
        checks,
        "dart_value_select_complete",
        bool(
            dart_audit.get("passed")
            and dart_audit.get("complete_paper_eligible", True)
        ),
        required_for_complete=dart_required,
        evidence=dart_audit,
        message=(
            "DART sign/common-gate, exact OOS panels, A1/A2 replay, three layers, "
            "feasibility/MCS/LCB, once-lock, and H4 evidence passed"
            if dart_audit.get("passed")
            and dart_audit.get("complete_paper_eligible", True)
            else (
                "valid eight-candidate DART evidence is retained, but the "
                "hash-pinned candidate exclusion caps the paper at PARTIAL_PAPER"
                if dart_audit.get("passed")
                and dart_audit.get("terminal_status_cap") == PARTIAL_PAPER
                else "; ".join(map(str, dart_audit.get("failures", [])))
            )
        ),
    )

    branch_evidence_failures: list[str] = []
    if physical_complete:
        if physical_result.get("source_relationship") != _REAL_SOURCE:
            branch_evidence_failures.append("physical source relationship")
        if physical_result.get("measurement_relationship") != _NOT_STUDY_MEASURED:
            branch_evidence_failures.append("physical measurement relationship")
    if forecast_complete:
        if not isinstance(forecast_manifest, Mapping) or forecast_manifest.get("started_from_real_referenced_data") is not True:
            branch_evidence_failures.append("forecast real-source declaration")
        if not isinstance(forecast_manifest, Mapping) or forecast_manifest.get("measurement_relationship") != _NOT_STUDY_MEASURED:
            branch_evidence_failures.append("forecast measurement relationship")
    if int(optimization_result.get("completed_count", 0) or 0) > 0:
        if optimization_result.get("nyiso_evidence_role") != "counterfactual":
            branch_evidence_failures.append("NYISO counterfactual evidence role")
        if optimization_result.get("measurement_relationship") != _NOT_STUDY_MEASURED:
            branch_evidence_failures.append("optimization measurement relationship")
        if optimization_result.get("xcel_relationship") != "tariff_defined_counterfactual":
            branch_evidence_failures.append("Xcel tariff-defined counterfactual relationship")
    _check(
        checks,
        "branch_evidence_labels",
        not branch_evidence_failures,
        evidence={"failures": branch_evidence_failures},
        message="branch evidence roles preserve reference/counterfactual boundaries" if not branch_evidence_failures else "missing or contradictory labels: " + ", ".join(branch_evidence_failures),
    )

    synthetic_flags = []
    for name, payload in (
        ("physical_result", physical_result),
        ("physical_manifest", physical_branch_manifest),
        ("calibration_manifest", calibration_manifest),
        ("transfer_manifest", transfer_manifest),
        ("forecast_manifest", forecast_manifest),
        ("dart_v1_readiness_result", dart_v1_readiness_result),
        ("dart_result", dart_result),
        ("optimization_result", optimization_result),
    ):
        synthetic_flags.extend(_true_synthetic_flags(payload, name))
    synthetic_flags.extend(xcel_audit.get("true_synthetic_flags", []))
    _check(
        checks,
        "no_synthetic_contamination",
        not synthetic_flags,
        evidence={"true_synthetic_flags": synthetic_flags},
        message="no synthetic paper fallback flag is true" if not synthetic_flags else "synthetic contamination flags found",
    )

    cell_audit = _cell_audit(root, config, optimization_result)
    optimization_complete = (
        _status(optimization_result) == "complete"
        and optimization_result.get("completed") is True
        and int(optimization_result.get("failure_count", 0) or 0) == 0
        and int(optimization_result.get("pending_count", 0) or 0) == 0
        and int(optimization_result.get("auxiliary_failure_count", 0) or 0) == 0
        and int(optimization_result.get("auxiliary_pending_count", 0) or 0) == 0
        and int(optimization_result.get("planned_cell_count", 0) or 0) > 0
    )
    _check(
        checks,
        "optimization_branch_complete",
        optimization_complete,
        evidence={
            "status": optimization_result.get("status"),
            "completed": optimization_result.get("completed"),
            "completed_count": optimization_result.get("completed_count"),
            "failure_count": optimization_result.get("failure_count"),
            "pending_count": optimization_result.get("pending_count"),
            "auxiliary_failure_count": optimization_result.get(
                "auxiliary_failure_count", 0
            ),
            "auxiliary_pending_count": optimization_result.get(
                "auxiliary_pending_count", 0
            ),
        },
        message="all planned optimization cells completed" if optimization_complete else "optimization branch is incomplete or contains failed cells",
    )
    _check(
        checks,
        "optimization_cell_status_consistency",
        cell_audit["summary_consistent"] if int(optimization_result.get("planned_cell_count", 0) or 0) else False,
        evidence=cell_audit,
        message="cell-level statuses reconcile to the branch summary" if cell_audit["summary_consistent"] else "cell-level statuses do not reconcile to the branch summary",
    )
    _check(
        checks,
        "primary_96_day_complete",
        cell_audit["primary_complete"],
        evidence=cell_audit,
        message="all cells across exactly 96 primary days succeeded" if cell_audit["primary_complete"] else "the preregistered 96-day primary experiment is incomplete",
    )
    _check(
        checks,
        "full_year_extension_complete",
        cell_audit["extension_complete"],
        evidence=cell_audit,
        message="required full-year extension completed" if cell_audit["extension_complete"] else "required full-year extension is incomplete",
    )
    _check(
        checks,
        "preregistered_optimization_design_complete",
        cell_audit["preregistered_design_complete"],
        evidence={
            key: cell_audit.get(key)
            for key in (
                "expected_seeds",
                "observed_seeds",
                "expected_scenario_counts",
                "observed_scenario_counts",
                "expected_regimes",
                "observed_regimes",
                "observed_policies",
                "h6_primary_cell_count",
                "expected_cells_per_day",
            )
        },
        message=(
            "B0-B8, three seeds, three dependence regimes, 10/20/40/80 scenarios, and H6 variants match the amended design"
            if cell_audit["preregistered_design_complete"]
            else "optimization cells do not cover the full preregistered design"
        ),
    )

    branch_cfg = config.get("optimization_branch", {})
    branch_cfg = branch_cfg if isinstance(branch_cfg, Mapping) else {}
    formal_extended_evidence_required = bool(
        branch_cfg.get("run_ofat_sensitivities") is True
        or optimization_result.get("ofat_sensitivities_requested") is True
    )
    expected_seed_count = len(cell_audit.get("expected_seeds", []))
    ofat_audit = _ofat_sensitivity_audit(root, optimization_result)
    _check(
        checks,
        "preregistered_ofat_sensitivities_complete",
        ofat_audit["passed"],
        required_for_complete=formal_extended_evidence_required,
        evidence=ofat_audit,
        message=(
            "all preregistered one-factor-at-a-time variants consumed real model inputs on 96 days"
            if ofat_audit["passed"]
            else "; ".join(ofat_audit["failures"])
        ),
    )
    statistical_audit = _statistical_inference_audit(
        root,
        optimization_result,
        expected_seed_count=expected_seed_count,
    )
    _check(
        checks,
        "statistical_inference_complete",
        statistical_audit["passed"],
        required_for_complete=formal_extended_evidence_required,
        evidence=statistical_audit,
        message=(
            "paired/block-bootstrap, DM-style, Holm, effect-size, and H4 linkage evidence verified"
            if statistical_audit["passed"]
            else "; ".join(statistical_audit["failures"])
        ),
    )
    scenario_audit = _scenario_diagnostics_audit(
        root, optimization_result, cell_audit
    )
    _check(
        checks,
        "scenario_generation_and_reduction_diagnostics_complete",
        scenario_audit["passed"],
        required_for_complete=formal_extended_evidence_required,
        evidence=scenario_audit,
        message=(
            "Gaussian-copula generation, dependence, marginal, event-duration, support, and reduction diagnostics verified"
            if scenario_audit["passed"]
            else "; ".join(scenario_audit["failures"])
        ),
    )

    _check(
        checks,
        "xcel_local_tariff_replay_complete",
        xcel_audit["passed"],
        evidence=xcel_audit,
        message=(
            "Xcel local tariff replay verified all inputs/outputs, replayed every available "
            "May-December source interval, and retained quantified source gaps without imputation"
            if xcel_audit["passed"]
            else "; ".join(xcel_audit["failures"])
        ),
    )

    hypotheses, hypothesis_path = _load_hypotheses(root, optimization_result)
    hypothesis_audit = _hypothesis_audit(hypotheses)
    _check(
        checks,
        "h0_h6_complete",
        hypothesis_audit["passed"],
        evidence=hypothesis_audit,
        message="H0-H6 were all evaluated" if hypothesis_audit["passed"] else "; ".join(hypothesis_audit["failures"]),
    )

    physical_timezone = _physical_timezone_sensitivity(
        physical_result, calibration_manifest
    )
    _check(
        checks,
        "timezone_physical_sensitivity_complete",
        physical_timezone["passed"],
        evidence=physical_timezone,
        message="UTC and fixed-MST overlap/locked physical metrics were compared" if physical_timezone["passed"] else "physical source-clock sensitivity is incomplete",
    )
    economic_timezone_complete, economic_timezone = _economic_timezone_sensitivity(
        root,
        optimization_result,
        strict=formal_extended_evidence_required,
        expected_seed_count=expected_seed_count,
        expected_day_count=96,
    )
    _check(
        checks,
        "timezone_economic_sensitivity_complete",
        economic_timezone_complete,
        evidence=economic_timezone,
        message="primary pooled economic endpoint was compared across source-clock interpretations" if economic_timezone_complete else "primary pooled economic endpoint source-clock comparison is missing",
    )

    table_paths: dict[str, Path] = {}
    physical_table, physical_frame = _build_physical_table(
        root, run_id, physical_result, calibration_manifest
    )
    table_paths["physical"] = physical_table
    forecast_paths, forecast_frames, metrics_path = _build_forecast_tables(
        root, run_id, forecast_manifest, forecast_result
    )
    table_paths.update({f"forecast_{key}": value for key, value in forecast_paths.items()})
    optimization_table, optimization_frame, raw_optimization, raw_optimization_path = (
        _build_optimization_table(root, run_id, optimization_result)
    )
    table_paths["optimization"] = optimization_table
    exact_validation_audit = _exact_validation_audit(root, optimization_result)
    _check(
        checks,
        "exact_vs_cohort_validation_gate",
        exact_validation_audit["passed"],
        required_for_complete=formal_extended_evidence_required,
        evidence=exact_validation_audit,
        message=(
            "one-site and two-site hand-checkable exact/cohort formulations agree"
            if exact_validation_audit["passed"]
            else "; ".join(exact_validation_audit["failures"])
        ),
    )
    rolling_audit = _rolling_execution_audit(root, raw_optimization)
    _check(
        checks,
        "causal_15_minute_rolling_execution_complete",
        rolling_audit["passed"],
        required_for_complete=formal_extended_evidence_required,
        evidence=rolling_audit,
        message=(
            "all non-oracle reference cells used causal 15-minute rolling replay and B8 remained an oracle lower bound"
            if rolling_audit["passed"]
            else "; ".join(rolling_audit["failures"])
        ),
    )
    observed_success = int(cell_audit.get("observed", {}).get("success", 0) or 0)
    unique_result_keys = (
        int(raw_optimization["checkpoint_key"].astype(str).nunique())
        if not raw_optimization.empty and "checkpoint_key" in raw_optimization
        else len(raw_optimization)
    )
    result_relationships_valid = True
    if not raw_optimization.empty:
        if "nyiso_evidence_role" in raw_optimization:
            result_relationships_valid &= bool(
                raw_optimization["nyiso_evidence_role"].astype(str).eq("counterfactual").all()
            )
        else:
            result_relationships_valid = False
        if "measurement_relationship" in raw_optimization:
            result_relationships_valid &= bool(
                raw_optimization["measurement_relationship"].astype(str).eq(_NOT_STUDY_MEASURED).all()
            )
        else:
            result_relationships_valid = False
    result_coverage_passed = (
        observed_success > 0
        and len(raw_optimization) == observed_success
        and unique_result_keys == observed_success
        and result_relationships_valid
    )
    _check(
        checks,
        "optimization_result_coverage",
        result_coverage_passed,
        evidence={
            "successful_cells": observed_success,
            "result_rows": len(raw_optimization),
            "unique_checkpoint_keys": unique_result_keys,
            "evidence_labels_valid": result_relationships_valid,
            "source_path": (
                _relative(root, raw_optimization_path)
                if raw_optimization_path is not None and raw_optimization_path.is_file()
                else None
            ),
        },
        message=(
            "every successful optimization cell has one labeled scalar result row"
            if result_coverage_passed
            else "successful optimization cells and result rows do not reconcile"
        ),
    )
    solver_evidence_failures: list[str] = []
    if raw_optimization.empty:
        solver_evidence_failures.append("no successful solver result rows")
    else:
        if "termination_condition" not in raw_optimization:
            solver_evidence_failures.append("termination_condition is missing")
        if "mip_gap" not in raw_optimization:
            solver_evidence_failures.append("mip_gap is missing")
        else:
            gaps = pd.to_numeric(raw_optimization["mip_gap"], errors="coerce")
            if gaps.isna().any():
                solver_evidence_failures.append("one or more mip_gap values are unavailable")
            targets = (
                pd.to_numeric(raw_optimization["mip_gap_target"], errors="coerce")
                if "mip_gap_target" in raw_optimization
                else pd.Series(
                    float(config.get("optimization", {}).get("mip_gap", 0.01)),
                    index=raw_optimization.index,
                )
            )
            if targets.isna().any() or (gaps > targets + 1e-12).any():
                solver_evidence_failures.append("one or more solver gaps exceed the declared target")
    _check(
        checks,
        "optimization_solver_evidence",
        not solver_evidence_failures,
        evidence={
            "failures": solver_evidence_failures,
            "mip_gap_max": (
                float(pd.to_numeric(raw_optimization["mip_gap"], errors="coerce").max())
                if not raw_optimization.empty and "mip_gap" in raw_optimization
                else None
            ),
        },
        message=(
            "termination and MIP-gap evidence are recorded within target"
            if not solver_evidence_failures
            else "; ".join(solver_evidence_failures)
        ),
    )
    hypothesis_table, hypothesis_frame = _build_hypothesis_table(
        root, run_id, hypotheses
    )
    table_paths["hypotheses"] = hypothesis_table

    failure_rows = _flatten_failures(
        root,
        physical_result,
        forecast_result,
        forecast_manifest,
        optimization_result,
        hypotheses,
        cell_audit,
        raw_optimization,
        transfer_manifest,
        xcel_audit,
    )
    for stage, audit in (
        (
            "forecast_extension_bridge",
            forecast_extension_audit
            if forecast_extension_required
            else {"failures": []},
        ),
        ("ofat_sensitivities", ofat_audit),
        ("statistical_inference", statistical_audit),
        ("scenario_diagnostics", scenario_audit),
        ("exact_validation", exact_validation_audit),
        ("rolling_execution", rolling_audit),
        (
            "timezone_economic_sensitivity",
            economic_timezone if isinstance(economic_timezone, Mapping) else {},
        ),
    ):
        for reason in audit.get("failures", []):
            failure_rows.append(
                {
                    "stage": stage,
                    "work_unit": "integrity_gate",
                    "status": "failed",
                    "reason": str(reason),
                }
            )
    if dart_required:
        for reason in dart_v1_readiness_audit.get("failures", []):
            failure_rows.append(
                {
                    "stage": "dart_v1_readiness",
                    "work_unit": "frozen_validation_policy_gate",
                    "status": "failed",
                    "reason": str(reason),
                }
            )
        for reason in dart_audit.get("failures", []):
            failure_rows.append(
                {
                    "stage": "dart_value_select",
                    "work_unit": "integrity_gate",
                    "status": "failed",
                    "reason": str(reason),
                }
            )
        candidate_pool_contract = dart_audit.get("candidate_pool_contract")
        if isinstance(candidate_pool_contract, Mapping):
            for exclusion in candidate_pool_contract.get(
                "globally_excluded_candidates", []
            ):
                if not isinstance(exclusion, Mapping):
                    continue
                failure_rows.append(
                    {
                        "stage": "dart_value_select",
                        "work_unit": str(exclusion.get("candidate_name", "unknown")),
                        "status": "excluded_candidate_partial_paper_cap",
                        "reason": str(exclusion.get("reason_code", "unknown")),
                    }
                )
    # Keep the finalizer's exhaustive ledger outside the canonical table path.
    # The artifact inventory derives the canonical failures table (and its
    # source sidecar) from this immutable-at-each-pass source.  Writing the raw
    # ledger directly to artifacts/tables/failures.csv would let inventory
    # replace it and would then make the finalizer's post-inventory append
    # invalidate the canonical table's recorded hash.
    failure_source_path = (
        root / "artifacts/paper_finalization" / run_id / "failures.csv"
    )
    failures_path = root / "artifacts/tables/failures.csv"
    failure_columns = [
        "stage",
        "work_unit",
        "status",
        "reason",
        "phase",
        "policy_id",
        "seed",
        "regime",
        "reduced_scenario_count",
        "solver_status",
        "checkpoint_path",
        "exception_type",
    ]
    _atomic_csv(
        failure_source_path,
        failure_rows,
        failure_columns,
    )
    # Seed the user-facing path for isolated/mocked inventory callers.  A real
    # inventory pass replaces this with the source-backed canonical bundle.
    _atomic_csv(failures_path, failure_rows, failure_columns)
    table_paths["failures"] = failures_path

    figure_paths, figure_failures = _generate_figures(
        root,
        physical_frame,
        forecast_frames,
        optimization_frame,
        hypothesis_frame,
    )
    supported_figure_count = sum(
        (
            not physical_frame.empty,
            any(not frame.empty for frame in forecast_frames.values()),
            not optimization_frame.empty,
            not hypothesis_frame.empty,
        )
    )
    _check(
        checks,
        "supported_figures_generated",
        len(figure_paths) == supported_figure_count and not figure_failures,
        evidence={
            "supported_source_count": supported_figure_count,
            "generated": [_relative(root, path) for path in figure_paths],
            "failures": figure_failures,
        },
        message=f"generated {len(figure_paths)}/{supported_figure_count} supported paper figures",
    )

    # The canonical artifact inventory is the single authority for the
    # manuscript's required manifests, tables, figures, and reports.  It may
    # materialize only source-backed aliases/conversions; any missing
    # mandatory registry entry remains a hard COMPLETE blocker.
    inventory_inputs: dict[str, Any] = {}
    configured_inventory = config.get("paper_artifacts", {})
    if isinstance(configured_inventory, Mapping):
        inventory_inputs.update(dict(configured_inventory))
    inventory_inputs.update(
        {
            "physical": physical_result,
            "forecast": forecast_result,
            "dart_v1_readiness": dart_v1_readiness_result or {},
            "dart": dart_result or {},
            "optimization": optimization_result,
        }
    )
    paper_table_inputs = inventory_inputs.setdefault("paper_table_inputs", {})
    if not isinstance(paper_table_inputs, dict):
        paper_table_inputs = dict(paper_table_inputs) if isinstance(
            paper_table_inputs, Mapping
        ) else {}
        inventory_inputs["paper_table_inputs"] = paper_table_inputs
    paper_table_inputs["failures.csv"] = {
        "source": _relative(root, failure_source_path),
        "sha256": sha256_file(failure_source_path),
    }

    def run_artifact_inventory() -> dict[str, Any]:
        try:
            from dcvpp.paper_artifacts import finalize_artifact_inventory

            return dict(finalize_artifact_inventory(root, inventory_inputs))
        except Exception as exc:
            return {
                "status": "FAILED",
                "completed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "artifacts": [],
            }

    artifact_inventory = run_artifact_inventory()
    inventory_registry = artifact_inventory.get("artifacts", [])
    inventory_registry = (
        inventory_registry if isinstance(inventory_registry, list) else []
    )
    inventory_gate_passed = bool(
        artifact_inventory.get("completed") is True
        and str(artifact_inventory.get("status", "")).upper() == "COMPLETE"
    )
    inventory_gate_evidence = {
        key: artifact_inventory.get(key)
        for key in (
            "status",
            "completed",
            "canonical_registry_count",
            "optional_registry_count",
            "mandatory_count",
            "materialized_mandatory_count",
            "missing_mandatory_count",
            "optional_skipped_count",
            "inventory_path",
            "pending_and_skips_path",
            "error_type",
            "error",
        )
        if key in artifact_inventory
    }
    inventory_gate_evidence["registry"] = [
        {
            key: item.get(key)
            for key in (
                "artifact_id",
                "relative_path",
                "requirement",
                "status",
                "reason",
            )
        }
        for item in inventory_registry
        if isinstance(item, Mapping)
    ]
    _check(
        checks,
        "canonical_artifact_inventory_complete",
        inventory_gate_passed,
        evidence=inventory_gate_evidence,
        message=(
            "the canonical manuscript artifact registry is fully materialized"
            if inventory_gate_passed
            else (
                "canonical artifact inventory is incomplete: "
                f"{artifact_inventory.get('materialized_mandatory_count', 0)}/"
                f"{artifact_inventory.get('mandatory_count', 0)} mandatory artifacts materialized"
            )
        ),
    )
    inventory_failure_rows = [
        {
            "stage": "paper_artifact_inventory",
            "work_unit": str(item.get("artifact_id", item.get("relative_path", "unknown"))),
            "status": "missing_mandatory",
            "reason": str(item.get("reason", "mandatory canonical artifact is not materialized")),
        }
        for item in inventory_registry
        if isinstance(item, Mapping)
        and item.get("requirement") == "mandatory"
        and item.get("status") != "MATERIALIZED"
    ]
    if inventory_failure_rows:
        failure_rows.extend(inventory_failure_rows)
        _atomic_csv(
            failure_source_path,
            failure_rows,
            failure_columns,
        )
        paper_table_inputs["failures.csv"] = {
            "source": _relative(root, failure_source_path),
            "sha256": sha256_file(failure_source_path),
        }

    genuine_components: list[str] = []
    if physical_complete and frozen_audit["passed"] and not _true_synthetic_flags(physical_result):
        genuine_components.append("physical")
    if (
        int(forecast_counts.get("tasks_completed", 0) or 0) > 0
        and isinstance(forecast_manifest, Mapping)
        and forecast_manifest.get("started_from_real_referenced_data") is True
        and not _true_synthetic_flags(forecast_manifest)
    ):
        genuine_components.append("forecast")
    if dart_required and dart_audit.get("passed"):
        genuine_components.append("dart_value_select_counterfactual_replay")
    if int(optimization_result.get("completed_count", 0) or 0) > 0:
        genuine_components.append("counterfactual_optimization")
    if (
        str(xcel_audit.get("status", "")).upper()
        in {"COMPLETE", "PARTIAL_SOURCE_COVERAGE"}
        and xcel_audit.get("manifest")
        and xcel_audit.get("outputs")
    ):
        genuine_components.append("xcel_local_tariff_replay")

    status, justification = _classify(
        checks=checks,
        preflight_result=preflight_result,
        physical_result=physical_result,
        forecast_result=forecast_result,
        optimization_result=optimization_result,
        genuine_components=genuine_components,
    )
    if (
        dart_required
        and dart_audit.get("terminal_status_cap") == PARTIAL_PAPER
        and status == COMPLETE_PAPER
    ):
        status = PARTIAL_PAPER
        justification = (
            "DART evidence is downstream-valid under the sole hash-pinned "
            "eight-candidate exception, which permanently caps this run at "
            "PARTIAL_PAPER."
        )
    paper_results_available = bool(genuine_components)
    paper_training_started = bool(
        paper_results_available
        or preflight_result.get("paper_training_started") is True
        or physical_result.get("paper_training_started") is True
        or forecast_result.get("paper_training_started") is True
        or (dart_result or {}).get("paper_results_available") is True
    )
    counts = {
        "physical_models_or_metric_rows": len(physical_frame),
        "forecast_metric_rows": sum(len(frame) for frame in forecast_frames.values()),
        "optimization_success": cell_audit.get("observed", {}).get("success", 0),
        "optimization_failed": cell_audit.get("observed", {}).get("failed", 0),
        "optimization_pending_or_running": cell_audit.get("observed", {}).get(
            "pending_or_running", 0
        ),
        "primary_days": cell_audit.get("primary_day_count", 0),
        "full_year_days": cell_audit.get("year_day_count", 0),
        "hypotheses_finished": sum(
            value in _FINISHED_HYPOTHESIS_STATUSES
            for value in hypothesis_audit.get("states", {}).values()
        ),
        "dart_selection_days": int(
            dart_audit.get("panel_counts", {})
            .get("selection", {})
            .get("operating_dates", 0)
            or 0
        ),
        "dart_locked_test_days": int(
            dart_audit.get("panel_counts", {})
            .get("locked_test", {})
            .get("operating_dates", 0)
            or 0
        ),
        "failure_inventory_rows": len(failure_rows),
        "figures": len(figure_paths),
    }

    integrity_path = root / "artifacts/manifests/paper_integrity_audit.json"
    required_failed = [
        item["check_id"]
        for item in checks
        if item["required_for_complete"] and item["status"] != "passed"
    ]
    atomic_json(
        integrity_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": utc_now(),
            "terminal_status": status,
            "complete_paper_gate_passed": status == COMPLETE_PAPER,
            "synthetic_fallback": False,
            "paper_results_available": paper_results_available,
            "checks": checks,
            "required_failed_checks": required_failed,
            "hypothesis_states": hypothesis_audit.get("states", {}),
            "timezone_sensitivity": {
                "physical": physical_timezone,
                "economic": economic_timezone,
                "conclusion_sensitive": (
                    economic_timezone.get("substantive_conclusion_changed")
                    if isinstance(economic_timezone, Mapping)
                    else None
                ),
            },
            "cell_completeness": cell_audit,
            "xcel_local_tariff_replay": xcel_audit,
            "dart_v1_readiness": dart_v1_readiness_audit,
            "dart_value_select": dart_audit,
            "genuine_paper_components": genuine_components,
        },
    )

    blockers_path = root / "artifacts/report/blockers.json"
    atomic_json(
        blockers_path,
        {
            "run_id": run_id,
            "terminal_status": status,
            "terminal": True,
            "paper_results_available": paper_results_available,
            "blockers_or_unresolved_requirements": [
                {
                    "code": item["check_id"],
                    "message": item.get("message", ""),
                }
                for item in checks
                if item["required_for_complete"] and item["status"] != "passed"
            ],
        },
    )

    report_path = root / "artifacts/report/run_summary.md"
    atomic_text(
        report_path,
        _report_text(
            status=status,
            justification=justification,
            run_id=run_id,
            genuine_components=genuine_components,
            checks=checks,
            failure_rows=failure_rows,
            physical_frame=physical_frame,
            forecast_frames=forecast_frames,
            optimization_frame=optimization_frame,
            hypothesis_audit=hypothesis_audit,
            physical_timezone=physical_timezone,
            economic_timezone=economic_timezone,
            cell_audit=cell_audit,
            xcel_audit=xcel_audit,
            dart_audit=dart_audit,
        ),
    )

    branch_summaries = {
        "physical": _branch_summary(physical_result),
        "forecast": _branch_summary(forecast_result),
        "optimization": _branch_summary(optimization_result),
    }
    if dart_required or dart_result is not None:
        branch_summaries["dart_v1_readiness"] = _branch_summary(
            dart_v1_readiness_result or {}
        )
        branch_summaries["dart"] = _branch_summary(dart_result or {})
    execution_manifest_path = root / "artifacts/manifests/paper_execution_manifest.json"
    atomic_json(
        execution_manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": status,
            "terminal": True,
            "paper_results_available": paper_results_available,
            "synthetic_fallback": False,
            "branch_results": branch_summaries,
            "counts": counts,
            "integrity_audit": _artifact_record(root, integrity_path),
            "updated_at_utc": utc_now(),
        },
    )

    # Refresh after the final report and execution manifest exist so the
    # authoritative inventory records their final bytes.  The gate outcome is
    # required to be stable across the two deterministic passes.
    refreshed_artifact_inventory = run_artifact_inventory()
    stable_artifact_inventory = run_artifact_inventory()

    def inventory_gate_identity(payload: Mapping[str, Any]) -> tuple[Any, ...]:
        registry = payload.get("artifacts", [])
        missing_ids = tuple(
            sorted(
                str(item.get("artifact_id"))
                for item in registry
                if isinstance(item, Mapping)
                and item.get("requirement") == "mandatory"
                and item.get("status") != "MATERIALIZED"
            )
        ) if isinstance(registry, list) else ()
        return (
            str(payload.get("status", "")),
            bool(payload.get("completed") is True),
            int(payload.get("canonical_registry_count", 0) or 0),
            int(payload.get("mandatory_count", 0) or 0),
            int(payload.get("missing_mandatory_count", 0) or 0),
            missing_ids,
        )

    if inventory_gate_identity(refreshed_artifact_inventory) != inventory_gate_identity(
        stable_artifact_inventory
    ):
        raise RuntimeError(
            "canonical artifact inventory changed during terminalization; "
            "resume finalization to obtain a stable evidence registry"
        )
    first_gate_outcome = (
        str(artifact_inventory.get("status", "")).upper(),
        bool(artifact_inventory.get("completed") is True),
    )
    stable_gate_outcome = (
        str(stable_artifact_inventory.get("status", "")).upper(),
        bool(stable_artifact_inventory.get("completed") is True),
    )
    if first_gate_outcome != stable_gate_outcome:
        raise RuntimeError(
            "canonical artifact COMPLETE gate changed after terminal outputs were written; "
            "resume finalization before persisting terminal state"
        )
    artifact_inventory = stable_artifact_inventory

    state_outputs = [
        *table_paths.values(),
        *figure_paths,
        integrity_path,
        blockers_path,
        report_path,
        execution_manifest_path,
    ]
    for inventory_value in (
        artifact_inventory.get("inventory_path"),
        artifact_inventory.get("pending_and_skips_path"),
        (
            artifact_inventory.get("final_manifest", {}).get("path")
            if isinstance(artifact_inventory.get("final_manifest"), Mapping)
            else None
        ),
    ):
        inventory_path = _resolve(root, inventory_value)
        if inventory_path is not None and inventory_path.is_file():
            state_outputs.append(inventory_path)
    # The terminal proof manifest must reach disk before terminal state.  If the
    # process is interrupted between these two atomic writes, the next resume
    # can safely repeat finalization; terminal state can never point at an old
    # preflight manifest.
    core_artifacts = list(state_outputs)
    for optional in (
        source_path,
        physical_branch_manifest_path,
        calibration_path,
        transfer_path,
        forecast_manifest_path,
        _resolve(root, (dart_v1_readiness_result or {}).get("manifest_path")),
        _resolve(root, (dart_result or {}).get("manifest_path")),
        _resolve(root, forecast_result.get("dart_overlay_manifest")),
        hypothesis_path,
        metrics_path,
        raw_optimization_path,
        _result_path(root, optimization_result, "paths.cell_status"),
        _result_path(root, optimization_result, "paths.run_summary"),
        _result_path(root, optimization_result, "paths.xcel_local_replay_manifest"),
        _resolve(root, xcel_audit.get("outputs", {}).get("parquet")),
        _resolve(root, xcel_audit.get("outputs", {}).get("csv")),
        _resolve(root, xcel_audit.get("implementation")),
    ):
        if optional is not None and optional.is_file():
            core_artifacts.append(optional)
    artifact_records = [
        _artifact_record(root, path)
        for path in sorted(set(core_artifacts), key=lambda item: _relative(root, item))
        if path.is_file()
    ]
    run_manifest_path = root / "artifacts/manifests/run_manifest.json"
    atomic_json(
        run_manifest_path,
        {
            "schema_version": 2,
            "profile": "paper",
            "run_id": run_id,
            "status": status,
            "terminal_status": status,
            "terminal": True,
            "paper_training_started": paper_training_started,
            "paper_results_available": paper_results_available,
            "synthetic_fallback": False,
            "justification": justification,
            "branch_results": branch_summaries,
            "counts": counts,
            "required_failed_checks": required_failed,
            "source_relationship_policy": {
                "study_measurement_claimed": False,
                "nyiso_operation_observed": False,
                "xcel_event_calls_observed": False,
                "counterfactual_parameter_transfer": True,
            },
            "canonical_artifact_inventory": {
                key: artifact_inventory.get(key)
                for key in (
                    "status",
                    "completed",
                    "canonical_registry_count",
                    "optional_registry_count",
                    "mandatory_count",
                    "materialized_mandatory_count",
                    "missing_mandatory_count",
                    "optional_skipped_count",
                    "inventory_path",
                    "pending_and_skips_path",
                    "final_manifest",
                )
                if key in artifact_inventory
            },
            "artifacts": artifact_records,
            "completed_at_utc": utc_now(),
        },
    )
    state_outputs.append(run_manifest_path)
    state = _persist_state(
        root,
        state,
        run_id=run_id,
        status=status,
        paper_results_available=paper_results_available,
        paper_training_started=paper_training_started,
        branch_summaries=branch_summaries,
        counts=counts,
        output_paths=state_outputs,
        justification=justification,
    )

    return {
        "status": status,
        "terminal_status": status,
        "terminal": True,
        "paper_training_started": paper_training_started,
        "paper_results_available": paper_results_available,
        "synthetic_fallback": False,
        "justification": justification,
        "branch_results": branch_summaries,
        "counts": counts,
        "required_failed_checks": required_failed,
        "failure_count": len(failure_rows),
        "integrity_audit": _relative(root, integrity_path),
        "artifact_inventory": (
            _relative(root, _resolve(root, artifact_inventory.get("inventory_path")))
            if _resolve(root, artifact_inventory.get("inventory_path")) is not None
            else None
        ),
        "artifact_inventory_status": artifact_inventory.get("status"),
        "run_manifest": _relative(root, run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "report": _relative(root, report_path),
        "tables": {key: _relative(root, path) for key, path in table_paths.items()},
        "figures": [_relative(root, path) for path in figure_paths],
        "state": _relative(root, state.path),
    }
