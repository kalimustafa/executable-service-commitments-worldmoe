from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_H0_DECISION_RULES: dict[str, float] = {
    "component_closure_excess_mae_kw_max": 0.25,
    "component_closure_residual_after_documented_load_kw_max": 0.25,
    "ramp_mae_ratio_max": 1.0,
    "source_day_coverage_min": 0.90,
    "uncertainty_coverage_absolute_error_max": 0.10,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_records(
    metrics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if metrics is None:
        return []
    if isinstance(metrics, Mapping):
        return [dict(metrics)]
    if isinstance(metrics, (str, bytes)):
        return []
    return [dict(record) for record in metrics if isinstance(record, Mapping)]


def _upstream_hypothesis(
    hypothesis_id: str,
    metrics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
    required_fields: tuple[str, ...],
    claim: str,
) -> dict[str, Any]:
    records = _metric_records(metrics)
    record = records[0] if len(records) == 1 else {}
    missing = [field for field in required_fields if field not in record]
    if missing:
        return {
            "hypothesis_id": hypothesis_id,
            "status": "not_evaluable",
            "claim": claim,
            "reason": f"frozen upstream artifact did not provide: {', '.join(missing)}",
            "metrics": records,
        }
    if record.get("evaluated") is False:
        return {
            "hypothesis_id": hypothesis_id,
            "status": "not_evaluable",
            "claim": claim,
            "reason": str(record.get("reason", "upstream evidence was not evaluable")),
            "metrics": record,
        }
    explicit = record.get("hypothesis_status")
    status = explicit if explicit in {"supported", "not_supported"} else "evaluated"
    return {
        "hypothesis_id": hypothesis_id,
        "status": status,
        "claim": claim,
        "metrics": record,
    }


def _forecast_hypothesis(
    hypothesis_id: str,
    metrics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
    *,
    expected_tasks: tuple[str, ...],
    required_fields: tuple[str, ...],
    claim: str,
) -> dict[str, Any]:
    records = _metric_records(metrics)
    by_task: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        task = str(record.get("task", ""))
        if task in by_task:
            duplicates.append(task)
        elif task:
            by_task[task] = record
    missing_tasks = [task for task in expected_tasks if task not in by_task]
    field_gaps = {
        task: [field for field in required_fields if field not in by_task.get(task, {})]
        for task in expected_tasks
    }
    field_gaps = {task: gaps for task, gaps in field_gaps.items() if gaps}
    if missing_tasks or duplicates or field_gaps:
        return {
            "hypothesis_id": hypothesis_id,
            "status": "not_evaluable",
            "claim": claim,
            "reason": "complete unique locked-test records are required for every preregistered task",
            "missing_tasks": missing_tasks,
            "duplicate_tasks": sorted(set(duplicates)),
            "missing_fields_by_task": field_gaps,
            "records": records,
        }
    selected = [by_task[task] for task in expected_tasks]
    if any(record.get("evaluated") is not True for record in selected):
        return {
            "hypothesis_id": hypothesis_id,
            "status": "not_evaluable",
            "claim": claim,
            "reason": "one or more upstream task records were not evaluated",
            "records": selected,
        }
    return {
        "hypothesis_id": hypothesis_id,
        "status": (
            "supported"
            if all(record.get("support") is True for record in selected)
            else "not_supported"
        ),
        "claim": claim,
        "required_tasks": list(expected_tasks),
        "all_tasks_evaluated": True,
        "all_tasks_supported": all(
            record.get("support") is True for record in selected
        ),
        "records": selected,
    }


def derive_physical_hypothesis_metrics(
    root: Path,
    physical_result: Mapping[str, Any],
    decision_rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive H0 for the deployed frozen core from locked-test artifacts.

    This does not refit the physical twin.  It evaluates the exact
    ``frozen_piecewise_linear_surrogate`` loaded by neural training, rather than
    its upstream calibrated-statistical-twin teacher, using the already-frozen
    metric, closure, uncertainty, and split outputs named by the physical branch
    result.
    """

    rules = dict(DEFAULT_H0_DECISION_RULES)
    if isinstance(decision_rules, Mapping):
        rules.update(
            {
                key: float(value)
                for key, value in decision_rules.items()
                if key in rules
            }
        )
    result: dict[str, Any] = {
        "hypothesis": "H0",
        "evaluated": False,
        "hypothesis_status": "not_evaluable",
        "decision_rules": rules,
        "claim_scope": "chronologically_held_out_nlr_period",
    }
    try:
        if str(physical_result.get("status", "")).lower() not in {
            "complete",
            "completed",
        }:
            raise ValueError("physical branch is not complete")
        outputs = physical_result.get("outputs", {})
        hashes = physical_result.get("output_hashes", {})
        if not isinstance(outputs, Mapping) or not isinstance(hashes, Mapping):
            raise ValueError("physical branch did not provide output/hash mappings")

        required = (
            "facility_metrics",
            "component_closure",
            "uncertainty_metrics",
            "split_manifest",
            "overlap_manifest",
        )
        paths: dict[str, Path] = {}
        observed_hashes: dict[str, str] = {}
        for name in required:
            raw = outputs.get(name)
            if not raw:
                raise ValueError(f"physical output is missing: {name}")
            path = Path(str(raw))
            if not path.is_absolute():
                path = Path(root) / path
            if not path.is_file():
                raise ValueError(f"physical output file is missing: {path}")
            observed = _file_sha256(path)
            expected = hashes.get(name)
            if not expected or observed != expected:
                raise ValueError(f"physical output hash mismatch: {name}")
            paths[name] = path
            observed_hashes[name] = observed

        metrics = pd.read_parquet(paths["facility_metrics"])
        primary = metrics.loc[
            metrics["clock_interpretation"].astype(str).str.startswith("primary_")
            & metrics["split"].eq("locked_test")
        ]

        def exactly_one(model: str) -> dict[str, Any]:
            selected = primary.loc[primary["model"].eq(model)]
            if len(selected) != 1:
                raise ValueError(
                    f"expected one primary locked-test physical row for {model}, got {len(selected)}"
                )
            return selected.iloc[0].to_dict()

        frozen_core = exactly_one("frozen_piecewise_linear_surrogate")
        uncalibrated = exactly_one("uncalibrated_engineering_curve")
        closure_frame = pd.read_parquet(paths["component_closure"])
        closure_rows = closure_frame.loc[
            closure_frame["interpretation"].astype(str).str.startswith("primary")
        ]
        if len(closure_rows) != 1:
            raise ValueError("expected one primary component-closure row")
        closure = closure_rows.iloc[0].to_dict()
        uncertainty_frame = pd.read_parquet(paths["uncertainty_metrics"])
        uncertainty_rows = uncertainty_frame.loc[
            uncertainty_frame["clock_interpretation"]
            .astype(str)
            .str.startswith("primary_")
        ]
        if len(uncertainty_rows) != 1:
            raise ValueError("expected one primary uncertainty-coverage row")
        uncertainty = uncertainty_rows.iloc[0].to_dict()
        split = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
        locked_split = split.get("nlr300_srrl_primary", {})
        if not locked_split.get("locked_test"):
            raise ValueError("physical split manifest does not mark the final period locked")
        overlap = json.loads(paths["overlap_manifest"].read_text(encoding="utf-8"))
        source_scope = overlap.get("governing_common_support_daily_scope", {})
        primary_overlap = overlap.get("primary_assumed_utc", {})
        expected_source_days = int(source_scope.get("expected_days", 0))
        available_source_days = int(source_scope.get("available_days", 0))
        source_day_coverage = float(
            source_scope.get(
                "usable_day_fraction",
                available_source_days / max(expected_source_days, 1),
            )
        )
        active_job_intervals = int(
            primary_overlap.get("intervals_with_active_nlr295_jobs", 0)
        )
        monitored_job_intervals = int(
            primary_overlap.get("intervals_with_monitoring_energy_reconstruction", 0)
        )
        monitoring_energy_interval_fraction = (
            monitored_job_intervals / active_job_intervals
            if active_job_intervals > 0
            else float("nan")
        )

        core_fields = (
            "facility_mae_kw",
            "facility_rmse_kw",
            "daily_energy_mae_kwh",
            "pue_mae",
        )
        improvements = {
            field: {
                "frozen_core": float(frozen_core[field]),
                "uncalibrated": float(uncalibrated[field]),
                "frozen_core_minus_uncalibrated": float(frozen_core[field])
                - float(uncalibrated[field]),
                "improved": float(frozen_core[field]) < float(uncalibrated[field]),
            }
            for field in core_fields
        }
        documented_load = float(closure["documented_unmetered_pump_kw"])
        closure_excess_mae = max(float(closure["closure_mae_kw"]) - documented_load, 0.0)
        closure_residual_after_documented = abs(
            float(closure["closure_bias_kw"]) + documented_load
        )
        ramp_ratio = float(frozen_core["ramp_mae_kw_per_interval"]) / max(
            float(uncalibrated["ramp_mae_kw_per_interval"]), 1e-12
        )
        uncertainty_error = abs(
            float(uncertainty["locked_test_interval_coverage"])
            - float(uncertainty["nominal_coverage"])
        )
        checks = {
            "all_core_errors_improved": all(
                item["improved"] for item in improvements.values()
            ),
            "component_closure_excess_mae_within_tolerance": closure_excess_mae
            <= rules["component_closure_excess_mae_kw_max"],
            "component_closure_residual_after_documented_load_within_tolerance": closure_residual_after_documented
            <= rules["component_closure_residual_after_documented_load_kw_max"],
            "ramp_mae_within_tolerance": ramp_ratio
            <= rules["ramp_mae_ratio_max"],
            "source_day_coverage_within_tolerance": source_day_coverage
            >= rules["source_day_coverage_min"],
            "uncertainty_coverage_within_tolerance": uncertainty_error
            <= rules["uncertainty_coverage_absolute_error_max"],
        }
        supported = all(checks.values())
        result.update(
            {
                "evaluated": True,
                "hypothesis_status": "supported" if supported else "not_supported",
                "evaluated_model": "frozen_piecewise_linear_surrogate",
                "baseline_model": "uncalibrated_engineering_curve",
                "locked_test": {
                    "locked": True,
                    "method": locked_split.get("method"),
                    "row_count": int(
                        locked_split.get("row_counts", {}).get(
                            "locked_test", frozen_core["row_count"]
                        )
                    ),
                    "train_end_exclusive": locked_split.get("train_end_exclusive"),
                    "validation_end_exclusive": locked_split.get(
                        "validation_end_exclusive"
                    ),
                    "support_end_utc": locked_split.get("support_end_utc"),
                },
                "core_error_comparisons": improvements,
                "balance": {
                    "closure_mae_kw": float(closure["closure_mae_kw"]),
                    "closure_bias_kw": float(closure["closure_bias_kw"]),
                    "documented_unmetered_pump_kw": documented_load,
                    "excess_mae_after_documented_load_kw": closure_excess_mae,
                    "absolute_bias_after_documented_load_kw": closure_residual_after_documented,
                },
                "ramp": {
                    "frozen_core_mae_kw_per_interval": float(
                        frozen_core["ramp_mae_kw_per_interval"]
                    ),
                    "uncalibrated_mae_kw_per_interval": float(
                        uncalibrated["ramp_mae_kw_per_interval"]
                    ),
                    "frozen_core_to_uncalibrated_ratio": ramp_ratio,
                },
                "missingness": {
                    "expected_common_support_days": expected_source_days,
                    "available_common_support_days": available_source_days,
                    "usable_source_day_fraction": source_day_coverage,
                    "minimum_usable_source_day_fraction": rules[
                        "source_day_coverage_min"
                    ],
                    "active_job_interval_count": active_job_intervals,
                    "monitoring_energy_reconstruction_interval_count": monitored_job_intervals,
                    "monitoring_energy_reconstruction_interval_fraction": monitoring_energy_interval_fraction,
                    "missing_intervals_imputed": False,
                    "interpretation": "source-support missingness is evaluated before complete-case physical metrics; NLR monitoring gaps remain missing and are not treated as zero",
                },
                "uncertainty": {
                    "nominal_coverage": float(uncertainty["nominal_coverage"]),
                    "locked_test_interval_coverage": float(
                        uncertainty["locked_test_interval_coverage"]
                    ),
                    "absolute_coverage_error": uncertainty_error,
                },
                "checks": checks,
                "source_artifacts": {
                    name: {"path": str(paths[name]), "sha256": observed_hashes[name]}
                    for name in required
                },
            }
        )
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None and np.isfinite(float(row[field]))]
    return float(np.mean(values)) if values else float("nan")


def aggregate_hypotheses(
    optimization_rows: Iterable[Mapping[str, Any]],
    *,
    physical_metrics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    energy_forecast_metrics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    workload_forecast_metrics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    forecast_decision_records: Iterable[Mapping[str, Any]] = (),
    statistical_inference_rows: Iterable[Mapping[str, Any]] = (),
    expected_primary_days: int | None = None,
    expected_seed_count: int | None = None,
    holm_alpha: float = 0.05,
    acceptable_expected_cost_premium_fraction: float = 0.05,
    acceptable_sla_deterioration_work_units: float = 0.0,
) -> dict[str, Any]:
    """Aggregate H0--H6 without converting missing evidence into support."""

    rows = [dict(row) for row in optimization_rows if row.get("termination_condition")]
    inference_rows = [
        dict(row) for row in statistical_inference_rows if isinstance(row, Mapping)
    ]
    inference_index = {
        (str(row.get("comparison_id")), str(row.get("metric"))): row
        for row in inference_rows
        if row.get("comparison_id") and row.get("metric")
    }

    def inference_record(
        comparison_id: str, metric: str, *, direction: str | None = None
    ) -> tuple[dict[str, Any] | None, bool]:
        record = inference_index.get((comparison_id, metric))
        if record is None:
            return None, False
        day_count_ok = expected_primary_days is None or int(
            record.get("paired_day_count", -1)
        ) == int(expected_primary_days)
        raw = record.get("p_value_holm")
        p_value = float(raw) if raw is not None and np.isfinite(float(raw)) else 1.0
        lower = float(record.get("block_bootstrap_lower", np.nan))
        upper = float(record.get("block_bootstrap_upper", np.nan))
        directional = True
        if direction == "negative":
            directional = np.isfinite(upper) and upper < 0.0 and p_value <= holm_alpha
        elif direction == "positive":
            directional = np.isfinite(lower) and lower > 0.0 and p_value <= holm_alpha
        return record, bool(day_count_ok and directional)
    output: dict[str, Any] = {
        "H0": _upstream_hypothesis(
            "H0",
            physical_metrics,
            ("evaluated", "hypothesis_status", "checks", "locked_test"),
            "On a chronologically held-out NLR period, the calibrated physical twin improves core physical errors while satisfying declared tolerances.",
        ),
        "H1": _forecast_hypothesis(
            "H1",
            energy_forecast_metrics,
            expected_tasks=("nyiso_da", "nyiso_rt"),
            required_fields=(
                "evaluated",
                "support",
                "advanced_value",
                "baseline_value",
                "absolute_calibration_error_change",
                "tail_f1_advanced",
                "tail_f1_baseline",
            ),
            claim="DGPT-E improves locked-test DA and RT proper scores and tail detection without material calibration harm.",
        ),
        "H2": _forecast_hypothesis(
            "H2",
            workload_forecast_metrics,
            expected_tasks=("nlr_workload",),
            required_fields=(
                "evaluated",
                "support",
                "advanced_value",
                "baseline_value",
                "interval_calibration_improvement",
            ),
            claim="CfC-W improves locked-test workload distribution error and at least one interval-calibration aggregate.",
        ),
    }

    nominal20 = [
        row
        for row in rows
        if row.get("regime") == "independent"
        and int(row.get("reduced_scenario_count", -1)) == 20
        and row.get("constraint_variant", "base") == "base"
    ]
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nominal20:
        by_policy[str(row["policy_id"])].append(row)
    comparators = ("B0", "B1", "B2", "B4")
    if not by_policy.get("B6") or any(not by_policy.get(policy) for policy in comparators):
        output["H3"] = {
            "hypothesis_id": "H3",
            "status": "not_evaluable",
            "reason": "nominal 20-scenario B6 and all B0/B1/B2/B4 comparator cells are required",
        }
    else:
        b6_cost = _mean(by_policy["B6"], "expected_total_cost_usd")
        b6_cvar = _mean(by_policy["B6"], "cvar_total_cost_usd")
        b6_sla = _mean(by_policy["B6"], "expected_sla_shortfall_work_units")
        comparison = {}
        supported = True
        for policy in comparators:
            cost = _mean(by_policy[policy], "expected_total_cost_usd")
            cvar = _mean(by_policy[policy], "cvar_total_cost_usd")
            sla = _mean(by_policy[policy], "expected_sla_shortfall_work_units")
            cost_inference, cost_supported = inference_record(
                f"B6_vs_{policy}", "expected_total_cost_usd", direction="negative"
            )
            cvar_inference, cvar_supported = inference_record(
                f"B6_vs_{policy}", "cvar_total_cost_usd", direction="negative"
            )
            sla_inference, sla_complete = inference_record(
                f"B6_vs_{policy}", "expected_sla_shortfall_work_units"
            )
            comparison[policy] = {
                "expected_cost_delta_usd": b6_cost - cost,
                "cvar_delta_usd": b6_cvar - cvar,
                "sla_delta_work_units": b6_sla - sla,
                "statistical_inference": {
                    "expected_cost": cost_inference,
                    "cvar": cvar_inference,
                    "sla": sla_inference,
                },
            }
            supported &= bool(
                b6_cost < cost
                and b6_cvar < cvar
                and b6_sla <= sla + acceptable_sla_deterioration_work_units
                and cost_supported
                and cvar_supported
                and sla_complete
            )
        output["H3"] = {
            "hypothesis_id": "H3",
            "status": "supported" if supported else "not_supported",
            "comparison": comparison,
            "primary_policy": "B6",
            "holm_alpha": holm_alpha,
            "statistical_inference_consumed": True,
        }

    decision_records = [dict(record) for record in forecast_decision_records]
    complete_h4 = [
        record
        for record in decision_records
        if all(
            key in record
            for key in (
                "operating_date",
                "point_error",
                "tail_calibration_error",
                "scenario_quality_error",
                "decision_value_usd",
                "decision_value_definition",
                "forecast_prediction_evidence",
                "scenario_quality_evidence",
                "optimization_result_evidence",
            )
        )
    ]
    required_h4_count = max(4, int(expected_primary_days or 4))
    seed_complete = all(
        expected_seed_count is None
        or int(record.get("paired_seed_count", -1)) == int(expected_seed_count)
        for record in complete_h4
    )
    if len(complete_h4) != required_h4_count or not seed_complete:
        output["H4"] = {
            "hypothesis_id": "H4",
            "status": "not_evaluable",
            "reason": "one hash-traceable same-date forecast/decision record per primary day and every registered seed is required",
            "required_record_count": required_h4_count,
            "available_record_count": len(complete_h4),
            "expected_seed_count": expected_seed_count,
        }
    else:
        target = np.asarray([float(record["decision_value_usd"]) for record in complete_h4])
        point = np.asarray([float(record["point_error"]) for record in complete_h4])
        tail = np.asarray([float(record["tail_calibration_error"]) for record in complete_h4])
        scenario_quality = np.asarray(
            [float(record["scenario_quality_error"]) for record in complete_h4]
        )

        def r2(*features: np.ndarray) -> float:
            design = np.column_stack([np.ones(len(target)), *features])
            fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
            denominator = np.sum((target - target.mean()) ** 2)
            return float(1.0 - np.sum((target - fitted) ** 2) / denominator) if denominator > 0 else float("nan")

        point_r2 = r2(point)
        tail_r2 = r2(tail)
        scenario_quality_r2 = r2(scenario_quality)
        tail_and_scenario_quality_r2 = r2(tail, scenario_quality)
        output["H4"] = {
            "hypothesis_id": "H4",
            "status": (
                "supported"
                if tail_and_scenario_quality_r2 > point_r2
                else "not_supported"
            ),
            "point_error_r2": point_r2,
            "tail_calibration_error_r2": tail_r2,
            "scenario_quality_error_r2": scenario_quality_r2,
            "tail_and_scenario_quality_r2": tail_and_scenario_quality_r2,
            "record_count": len(complete_h4),
        }

    def row_seed(row: Mapping[str, Any]) -> int:
        return int(row.get("study_seed", row.get("seed", -1)))

    stress_groups: dict[tuple[str, int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("regime") not in {"kappa_0_3", "kappa_0_6"} or row.get("constraint_variant", "base") != "base":
            continue
        if int(row.get("reduced_scenario_count", -1)) != 20:
            continue
        key = (
            str(row["operating_date"]),
            row_seed(row),
            str(row["regime"]),
            int(row["reduced_scenario_count"]),
        )
        if row.get("policy_id") in {"B6", "B7"}:
            stress_groups[key][str(row["policy_id"])] = row
    nominal_b6 = {
        (
            str(row["operating_date"]),
            row_seed(row),
            int(row["reduced_scenario_count"]),
        ): row
        for row in rows
        if row.get("policy_id") == "B6"
        and row.get("regime") == "independent"
        and int(row.get("reduced_scenario_count", -1)) == 20
        and row.get("constraint_variant", "base") == "base"
    }
    paired_h5 = [
        (key, group)
        for key, group in stress_groups.items()
        if {"B6", "B7"}.issubset(group)
        and (key[0], key[1], key[3]) in nominal_b6
    ]
    if not paired_h5:
        output["H5"] = {
            "hypothesis_id": "H5",
            "status": "not_evaluable",
            "reason": "paired nominal B6 and stress-regime B6/B7 cells are required",
        }
    else:
        b6_expected = np.asarray([float(group["B6"]["expected_total_cost_usd"]) for _, group in paired_h5])
        b7_expected = np.asarray([float(group["B7"]["expected_total_cost_usd"]) for _, group in paired_h5])
        b6_tail = np.asarray([float(group["B6"]["cvar_total_cost_usd"]) for _, group in paired_h5])
        b7_tail = np.asarray([float(group["B7"]["cvar_total_cost_usd"]) for _, group in paired_h5])
        nominal_expected = np.asarray(
            [
                float(
                    nominal_b6[(key[0], key[1], key[3])][
                        "expected_total_cost_usd"
                    ]
                )
                for key, _ in paired_h5
            ]
        )
        nominal_tail = np.asarray(
            [
                float(
                    nominal_b6[(key[0], key[1], key[3])][
                        "cvar_total_cost_usd"
                    ]
                )
                for key, _ in paired_h5
            ]
        )
        premium = float((b7_expected.mean() - b6_expected.mean()) / max(abs(b6_expected.mean()), 1e-9))
        expected_erosion = float(b6_expected.mean() - nominal_expected.mean())
        tail_erosion = float(b6_tail.mean() - nominal_tail.mean())
        statistical_evidence: dict[str, Any] = {}
        inference_supported = True
        for regime in ("kappa_0_3", "kappa_0_6"):
            b7_tail_record, b7_tail_ok = inference_record(
                f"B7_vs_B6_{regime}", "cvar_total_cost_usd", direction="negative"
            )
            erosion_cost_record, erosion_cost_ok = inference_record(
                f"B6_{regime}_vs_nominal",
                "expected_total_cost_usd",
                direction="positive",
            )
            erosion_tail_record, erosion_tail_ok = inference_record(
                f"B6_{regime}_vs_nominal",
                "cvar_total_cost_usd",
                direction="positive",
            )
            statistical_evidence[regime] = {
                "B7_vs_B6_cvar": b7_tail_record,
                "B6_vs_nominal_expected_cost": erosion_cost_record,
                "B6_vs_nominal_cvar": erosion_tail_record,
            }
            inference_supported &= b7_tail_ok and erosion_cost_ok and erosion_tail_ok
        supported = bool(
            tail_erosion > 0
            and b7_tail.mean() < b6_tail.mean()
            and premium <= acceptable_expected_cost_premium_fraction
            and inference_supported
        )
        output["H5"] = {
            "hypothesis_id": "H5",
            "status": "supported" if supported else "not_supported",
            "paired_cell_count": len(paired_h5),
            "mean_cvar_delta_usd": float(b7_tail.mean() - b6_tail.mean()),
            "adverse_coupling_expected_cost_erosion_usd": expected_erosion,
            "adverse_coupling_cvar_erosion_usd": tail_erosion,
            "expected_cost_premium_fraction": premium,
            "acceptable_expected_cost_premium_fraction": acceptable_expected_cost_premium_fraction,
            "statistical_inference": statistical_evidence,
            "statistical_inference_consumed": True,
            "holm_alpha": holm_alpha,
        }

    h6_groups: dict[tuple[str, int, str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not (
            row.get("policy_id") == "B6"
            and row.get("regime") == "independent"
            and int(row.get("reduced_scenario_count", -1)) == 20
        ):
            continue
        key = (
            str(row.get("operating_date")),
            row_seed(row),
            str(row.get("regime")),
            int(row.get("reduced_scenario_count", -1)),
            str(row.get("policy_id")),
        )
        h6_groups[key][str(row.get("constraint_variant", "base"))] = row
    paired_h6 = [group for group in h6_groups.values() if {"base", "ramp_concentration"}.issubset(group)]
    paired_h6 = [
        group
        for group in paired_h6
        if group["base"].get("rebound_peak_mw") is not None
        and group["ramp_concentration"].get("rebound_peak_mw") is not None
        and group["base"].get("maximum_zone_concentration_share") is not None
        and group["ramp_concentration"].get("maximum_zone_concentration_share") is not None
    ]
    if not paired_h6:
        output["H6"] = {
            "hypothesis_id": "H6",
            "status": "not_evaluable",
            "reason": "paired base/constrained cells with rebound_peak_mw are required",
        }
    else:
        peak_delta = np.asarray(
            [float(group["ramp_concentration"]["rebound_peak_mw"]) - float(group["base"]["rebound_peak_mw"]) for group in paired_h6]
        )
        cost_premium = np.asarray(
            [
                (float(group["ramp_concentration"]["expected_total_cost_usd"]) - float(group["base"]["expected_total_cost_usd"]))
                / max(abs(float(group["base"]["expected_total_cost_usd"])), 1e-9)
                for group in paired_h6
            ]
        )
        concentration_delta = np.asarray(
            [
                float(group["ramp_concentration"]["maximum_zone_concentration_share"])
                - float(group["base"]["maximum_zone_concentration_share"])
                for group in paired_h6
            ]
        )
        sla_delta = np.asarray(
            [
                float(group["ramp_concentration"]["expected_sla_shortfall_work_units"])
                - float(group["base"]["expected_sla_shortfall_work_units"])
                for group in paired_h6
            ]
        )
        supported = bool(
            peak_delta.mean() < 0
            and concentration_delta.mean() < 0
            and cost_premium.mean() <= acceptable_expected_cost_premium_fraction
            and sla_delta.mean() <= acceptable_sla_deterioration_work_units
        )
        peak_inference, peak_supported = inference_record(
            "B6_H6_vs_B6_base", "rebound_peak_mw", direction="negative"
        )
        concentration_inference, concentration_supported = inference_record(
            "B6_H6_vs_B6_base",
            "maximum_zone_concentration_share",
            direction="negative",
        )
        cost_inference, cost_complete = inference_record(
            "B6_H6_vs_B6_base", "expected_total_cost_usd"
        )
        sla_inference, sla_complete = inference_record(
            "B6_H6_vs_B6_base", "expected_sla_shortfall_work_units"
        )
        supported &= bool(
            peak_supported
            and concentration_supported
            and cost_complete
            and sla_complete
        )
        output["H6"] = {
            "hypothesis_id": "H6",
            "status": "supported" if supported else "not_supported",
            "paired_cell_count": len(paired_h6),
            "mean_rebound_peak_delta_mw": float(peak_delta.mean()),
            "mean_zone_concentration_delta_share": float(concentration_delta.mean()),
            "mean_expected_cost_premium_fraction": float(cost_premium.mean()),
            "mean_sla_delta_work_units": float(sla_delta.mean()),
            "statistical_inference": {
                "rebound_peak": peak_inference,
                "zone_concentration": concentration_inference,
                "expected_cost": cost_inference,
                "sla": sla_inference,
            },
            "statistical_inference_consumed": True,
            "holm_alpha": holm_alpha,
        }

    return {
        "schema_version": "h0_h6_aggregation_v1",
        "evidence_role": "derived",
        "source_relationship": "derived_from_counterfactual_optimization_and_frozen_upstream_metrics",
        "measurement_relationship": "not_measured_by_this_study",
        "hypotheses": output,
    }
