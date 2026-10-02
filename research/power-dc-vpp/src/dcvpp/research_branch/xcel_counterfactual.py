from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_xcel_counterfactual_outputs(root: Path) -> dict[str, str]:
    """Materialize documented rates only; no event series is synthesized."""

    root = Path(root).resolve()
    extraction_path = root / "data" / "raw" / "xcel" / "extraction_status.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if extraction.get("observed_xcel_event_calls", {}).get("record_count") != 0:
        raise ValueError("Xcel branch requires zero observed event-call records")
    processed_path = root / "data" / "processed" / "xcel_counterfactual_inputs.parquet"
    table_path = root / "artifacts" / "tables" / "xcel_local_counterfactual.csv"
    manifest_path = root / "artifacts" / "manifests" / "xcel_counterfactual_manifest.json"
    if processed_path.exists() and table_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_hashes = existing.get("outputs", {})
        if (
            existing.get("schema_version") == "xcel_counterfactual_manifest_v1"
            and existing.get("event_calls") == 0
            and existing.get("input", {}).get("sha256") == _sha256(extraction_path)
            and output_hashes.get("parquet", {}).get("sha256") == _sha256(processed_path)
            and output_hashes.get("csv", {}).get("sha256") == _sha256(table_path)
        ):
            return {
                "parquet": str(processed_path),
                "manifest": str(manifest_path),
                "csv": str(table_path),
            }
    rows: list[dict[str, Any]] = []
    for item in extraction["large_business_schedule_candidates"]:
        charges = dict(item["charges"])
        base_schedule = item["schedule"].removesuffix("-CPP")
        adjustments = extraction["overlap_era_2024_adjustments"]
        rows.append(
            {
                "case_role": "primary_overlap_era_2024",
                "applicability_scope": "overlap_era_2024_only",
                "effective_date": item["base_rate_sheet_effective_date"],
                "schedule": item["schedule"],
                "service_voltage": item["service_voltage"],
                "service_and_facility_usd_per_month": charges.get("service_and_facility_usd_per_month"),
                "non_cpp_energy_usd_per_kwh": charges["non_cpp_energy_usd_per_kwh"],
                "cpp_energy_usd_per_kwh": charges["cpp_energy_usd_per_kwh"],
                "distribution_demand_usd_per_kw_month": charges.get("distribution_demand_usd_per_kw_month"),
                "generation_transmission_demand_usd_per_kw_month": charges[
                    "generation_transmission_demand_usd_per_kw_month"
                ],
                "demand_side_management_usd_per_kw_month": adjustments[
                    "demand_side_management_usd_per_kw_month"
                ][base_schedule],
                "purchased_capacity_usd_per_kw_month": adjustments[
                    "purchased_capacity_usd_per_kw_month"
                ][base_schedule],
                "transmission_cost_usd_per_kw_month": adjustments[
                    "transmission_cost_usd_per_kw_month"
                ][base_schedule],
                "transportation_electrification_usd_per_kw_month": adjustments[
                    "transportation_electrification_usd_per_kw_month"
                ][item["schedule"]],
                "minimum_annual_measured_demand_kw": item["minimum_annual_measured_demand_kw"],
                "minimum_notice_hours": extraction["cpp_program_rules"]["minimum_notice_hours"],
                "event_duration_hours": extraction["cpp_program_rules"]["event_duration_hours"],
                "maximum_events_per_calendar_year": extraction["cpp_program_rules"][
                    "maximum_events_per_calendar_year"
                ],
                "event_calls": 0,
                "tariff_defined_counterfactual": True,
                "training_target": False,
                "realized_response_evidence": False,
                "evidence_role": "counterfactual",
                "source_relationship": "tariff_defined_counterfactual",
                "measurement_relationship": "not_measured_by_this_study",
                "extraction_status": item["extraction_status"],
                "source_anchor": json.dumps(item["source_anchor"], sort_keys=True),
            }
        )
    for item in extraction["current_2026_large_business_reference_rates"]:
        rates = dict(item["total_rates_including_listed_adjustments"])
        rows.append(
            {
                "case_role": "current_2026_sensitivity_context_only",
                "applicability_scope": "current_2026_document_snapshot_not_primary_2024_case",
                "effective_date": item["effective_date"],
                "schedule": item["schedule"],
                "service_voltage": None,
                "service_and_facility_usd_per_month": rates.get("service_and_facility_usd_per_month"),
                "non_cpp_energy_usd_per_kwh": rates.get("energy_usd_per_kwh", rates.get("off_peak_energy_usd_per_kwh")),
                "cpp_energy_usd_per_kwh": None,
                "distribution_demand_usd_per_kw_month": rates.get("distribution_demand_usd_per_kw_month"),
                "generation_transmission_demand_usd_per_kw_month": rates.get(
                    "summer_generation_transmission_demand_usd_per_kw_month"
                ),
                "demand_side_management_usd_per_kw_month": None,
                "purchased_capacity_usd_per_kw_month": None,
                "transmission_cost_usd_per_kw_month": None,
                "transportation_electrification_usd_per_kw_month": None,
                "minimum_annual_measured_demand_kw": None,
                "minimum_notice_hours": None,
                "event_duration_hours": None,
                "maximum_events_per_calendar_year": None,
                "event_calls": 0,
                "tariff_defined_counterfactual": True,
                "training_target": False,
                "realized_response_evidence": False,
                "evidence_role": "counterfactual",
                "source_relationship": "tariff_defined_counterfactual",
                "measurement_relationship": "not_measured_by_this_study",
                "extraction_status": item["extraction_status"],
                "source_anchor": json.dumps(item["source_anchor"], sort_keys=True),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["case_role", "schedule"]).reset_index(drop=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_path, index=False)
    frame.to_csv(table_path, index=False)
    manifest = {
        "schema_version": "xcel_counterfactual_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_case": "2024-05-01 overlap-era SG-CPP/PG-CPP/TG-CPP published charges",
        "sensitivity_context": "2026-07-01 current SG/PG/TG reference rates, separate from primary case",
        "event_calls": 0,
        "tariff_defined_counterfactual": True,
        "training_target": False,
        "realized_response_evidence": False,
        "evidence_role": "counterfactual",
        "source_relationship": "tariff_defined_counterfactual",
        "measurement_relationship": "not_measured_by_this_study",
        "input": {"path": str(extraction_path.relative_to(root)), "sha256": _sha256(extraction_path)},
        "official_document_hashes": {
            name: document["sha256"] for name, document in extraction["documents"].items()
        },
        "row_counts": {
            "primary_overlap_era_2024": int((frame["case_role"] == "primary_overlap_era_2024").sum()),
            "current_2026_sensitivity_context_only": int(
                (frame["case_role"] == "current_2026_sensitivity_context_only").sum()
            ),
        },
        "outputs": {},
    }
    manifest["outputs"] = {
        "parquet": {"path": str(processed_path.relative_to(root)), "sha256": _sha256(processed_path)},
        "csv": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)
    return {
        "parquet": str(processed_path),
        "manifest": str(manifest_path),
        "csv": str(table_path),
    }


def _rate(value: Any) -> float:
    return 0.0 if value is None or pd.isna(value) else float(value)


def build_xcel_local_tariff_replay(
    root: Path,
    nlr300_facility_path: Path,
    *,
    expected_nlr300_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay the referenced 2024 NLR load under 2024 Xcel rate rules.

    This is deliberately separate from NYISO settlement.  With no observed CPP
    event calls, it computes only no-event, observed-interval tariff charges;
    it does not invent event timestamps, response, elasticity, or savings.
    """

    root = Path(root).resolve()
    nlr300_facility_path = Path(nlr300_facility_path).resolve()
    if not nlr300_facility_path.is_file():
        raise FileNotFoundError(f"NLR300 facility panel is missing: {nlr300_facility_path}")
    load_sha256 = _sha256(nlr300_facility_path)
    implementation_sha256 = _sha256(Path(__file__))
    if expected_nlr300_sha256 and load_sha256 != expected_nlr300_sha256:
        raise ValueError("NLR300 facility panel hash disagrees with the frozen physical result")

    rate_paths = build_xcel_counterfactual_outputs(root)
    rate_path = Path(rate_paths["parquet"])
    rate_sha256 = _sha256(rate_path)
    output_parquet = root / "data/processed/xcel_local_tariff_replay.parquet"
    output_csv = root / "artifacts/tables/xcel_local_tariff_replay.csv"
    manifest_path = root / "artifacts/manifests/xcel_local_replay_manifest.json"
    if output_parquet.is_file() and output_csv.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == "xcel_local_tariff_replay_v2"
            and existing.get("implementation_sha256") == implementation_sha256
            and existing.get("inputs", {}).get("nlr300_facility_sha256") == load_sha256
            and existing.get("inputs", {}).get("xcel_rate_input_sha256") == rate_sha256
            and existing.get("outputs", {}).get("parquet_sha256") == _sha256(output_parquet)
            and existing.get("outputs", {}).get("csv_sha256") == _sha256(output_csv)
        ):
            return {
                "status": existing["status"],
                "manifest": str(manifest_path),
                "parquet": str(output_parquet),
                "csv": str(output_csv),
                "coverage": existing["coverage"],
            }

    facility = pd.read_parquet(
        nlr300_facility_path,
        columns=[
            "timestamp_local",
            "facility_power_kw",
            "source_relationship",
            "measurement_relationship",
            "synthetic",
        ],
    )
    if facility["synthetic"].astype(bool).any():
        raise ValueError("Xcel local replay refuses synthetic NLR300 load")
    if not facility["source_relationship"].astype(str).eq(
        "third_party_referenced_public_data"
    ).all():
        raise ValueError("NLR300 load is missing the referenced-source relationship")
    if not facility["measurement_relationship"].astype(str).eq(
        "not_measured_by_this_study"
    ).all():
        raise ValueError("NLR300 load is missing the study-measurement boundary")
    facility["timestamp_local"] = pd.to_datetime(
        facility["timestamp_local"], errors="raise"
    )
    facility["facility_power_kw"] = pd.to_numeric(
        facility["facility_power_kw"], errors="raise"
    )
    facility = facility.loc[
        facility["timestamp_local"].map(lambda value: value.year == 2024)
    ].copy()
    if facility.empty or (facility["facility_power_kw"] < 0).any():
        raise ValueError("NLR300 has no valid nonnegative 2024 facility-load intervals")
    if facility["timestamp_local"].duplicated().any():
        raise ValueError("NLR300 2024 facility load contains duplicate local instants")
    facility["month"] = facility["timestamp_local"].map(
        lambda value: f"{value.year:04d}-{value.month:02d}"
    )
    annual_peak_kw = float(facility["facility_power_kw"].max())

    rates = pd.read_parquet(rate_path)
    rates = rates.loc[rates["case_role"].eq("primary_overlap_era_2024")].copy()
    if set(rates["schedule"]) != {"SG-CPP", "PG-CPP", "TG-CPP"}:
        raise ValueError("the Xcel primary replay requires SG-CPP, PG-CPP, and TG-CPP")

    rows: list[dict[str, Any]] = []
    for month in pd.period_range("2024-01", "2024-12", freq="M"):
        month_key = str(month)
        start = pd.Timestamp(month.start_time, tz="America/Denver")
        end = pd.Timestamp((month + 1).start_time, tz="America/Denver")
        expected_intervals = len(
            pd.date_range(start, end, freq="15min", inclusive="left")
        )
        observed = facility.loc[facility["month"].eq(month_key)]
        interval_count = len(observed)
        coverage_fraction = interval_count / expected_intervals
        energy_kwh = (
            float(observed["facility_power_kw"].sum() * 0.25)
            if interval_count
            else None
        )
        peak_kw = (
            float(observed["facility_power_kw"].max()) if interval_count else None
        )
        for _, rate in rates.sort_values("schedule").iterrows():
            effective_date = pd.Timestamp(str(rate["effective_date"])).date()
            tariff_applicable = month.start_time.date() >= effective_date
            minimum_demand = _rate(rate.get("minimum_annual_measured_demand_kw"))
            schedule_eligible = annual_peak_kw >= minimum_demand
            fixed = _rate(rate["service_and_facility_usd_per_month"])
            energy_rate = _rate(rate["non_cpp_energy_usd_per_kwh"])
            demand_rates = {
                name: _rate(rate[name])
                for name in (
                    "distribution_demand_usd_per_kw_month",
                    "generation_transmission_demand_usd_per_kw_month",
                    "demand_side_management_usd_per_kw_month",
                    "purchased_capacity_usd_per_kw_month",
                    "transmission_cost_usd_per_kw_month",
                    "transportation_electrification_usd_per_kw_month",
                )
            }
            energy_charge = (
                energy_rate * energy_kwh
                if tariff_applicable and schedule_eligible and energy_kwh is not None
                else None
            )
            demand_charge = (
                sum(demand_rates.values()) * peak_kw
                if tariff_applicable and schedule_eligible and peak_kw is not None
                else None
            )
            total = (
                fixed + float(energy_charge) + float(demand_charge)
                if energy_charge is not None and demand_charge is not None
                else None
            )
            rows.append(
                {
                    "month": month_key,
                    "schedule": rate["schedule"],
                    "effective_date": str(rate["effective_date"]),
                    "tariff_applicable": tariff_applicable,
                    "minimum_annual_measured_demand_kw": minimum_demand,
                    "observed_annual_peak_demand_kw": annual_peak_kw,
                    "schedule_eligible": schedule_eligible,
                    "observed_interval_count": interval_count,
                    "expected_interval_count": expected_intervals,
                    "source_coverage_fraction": coverage_fraction,
                    "complete_month": tariff_applicable and coverage_fraction == 1.0,
                    "observed_interval_energy_kwh": energy_kwh,
                    "observed_peak_demand_kw": peak_kw,
                    "fixed_charge_usd": (
                        fixed
                        if tariff_applicable and schedule_eligible and interval_count
                        else None
                    ),
                    "observed_interval_energy_charge_usd": energy_charge,
                    "observed_peak_demand_charge_usd": demand_charge,
                    "observed_interval_tariff_cost_usd": total,
                    "cpp_event_charge_usd": None,
                    "event_calls": 0,
                    "tariff_defined_counterfactual": True,
                    "realized_response_evidence": False,
                    "realized_savings_claimed": False,
                    "counterfactual": True,
                    "evidence_role": "counterfactual",
                    "source_relationship": "tariff_defined_counterfactual",
                    "load_source_relationship": "third_party_referenced_public_data",
                    "measurement_relationship": "not_measured_by_this_study",
                }
            )

    replay = pd.DataFrame(rows)
    applicable = replay.loc[replay["tariff_applicable"]]
    applicable_by_month = applicable.groupby("month", sort=True).first()
    supported_overlap_months = applicable_by_month.index[
        applicable_by_month["observed_interval_count"].gt(0)
    ].tolist()
    complete_overlap_months = applicable_by_month.index[
        applicable_by_month["complete_month"].astype(bool)
    ].tolist()
    source_unsupported_months = applicable_by_month.index[
        applicable_by_month["observed_interval_count"].eq(0)
    ].tolist()
    applicable_facility = facility.loc[
        facility["month"].isin(applicable_by_month.index)
    ]
    observed_interval_count = int(
        applicable_facility[["timestamp_local"]].drop_duplicates().shape[0]
    )
    reported_observed_interval_count = int(
        applicable_by_month["observed_interval_count"].sum()
    )
    all_available_source_intervals_replayed = bool(
        reported_observed_interval_count == observed_interval_count
    )
    execution_complete = bool(
        supported_overlap_months
        and applicable["schedule_eligible"].all()
        and all_available_source_intervals_replayed
    )
    source_coverage_complete = bool(
        execution_complete
        and not source_unsupported_months
        and applicable["complete_month"].all()
    )
    if not execution_complete:
        status = "NO_SUPPORTED_OVERLAP"
    elif source_coverage_complete:
        status = "COMPLETE"
    else:
        # This is an executed experiment with an observed-source limitation,
        # not an unfinished replay.  Missing NLR intervals stay missing.
        status = "PARTIAL_SOURCE_COVERAGE"
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp = output_parquet.with_suffix(".tmp.parquet")
    csv_tmp = output_csv.with_suffix(".tmp.csv")
    replay.to_parquet(parquet_tmp, index=False)
    replay.to_csv(csv_tmp, index=False)
    parquet_tmp.replace(output_parquet)
    csv_tmp.replace(output_csv)
    expected_interval_count = int(applicable_by_month["expected_interval_count"].sum())
    coverage = {
        "observed_interval_count": observed_interval_count,
        "expected_interval_count": expected_interval_count,
        "source_missing_interval_count": expected_interval_count
        - observed_interval_count,
        "source_coverage_fraction": (
            observed_interval_count / expected_interval_count
            if expected_interval_count
            else 0.0
        ),
        "tariff_effective_month_count": int(len(applicable_by_month)),
        "supported_overlap_month_count": int(len(supported_overlap_months)),
        "complete_overlap_month_count": int(len(complete_overlap_months)),
        "source_unsupported_month_count": int(len(source_unsupported_months)),
        "supported_overlap_months": supported_overlap_months,
        "complete_overlap_months": complete_overlap_months,
        "source_unsupported_months": source_unsupported_months,
        # Retained aliases for old readers.  They describe tariff-effective
        # months, not an imputation or a completeness requirement.
        "complete_month_count": int(len(complete_overlap_months)),
        "expected_month_count": int(len(applicable_by_month)),
        "out_of_effective_period_month_count": int(
            replay.loc[~replay["tariff_applicable"], "month"].nunique()
        ),
        "available_source_start": facility["timestamp_local"].min().isoformat(),
        "available_source_end": facility["timestamp_local"].max().isoformat(),
    }
    manifest = {
        "schema_version": "xcel_local_tariff_replay_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "execution_complete": execution_complete,
        "source_coverage_complete": source_coverage_complete,
        "all_available_source_intervals_replayed": all_available_source_intervals_replayed,
        "missing_intervals_imputed": False,
        "implementation_sha256": implementation_sha256,
        "primary_rate_snapshot": "2024-05-01",
        "current_2026_rates_used": False,
        "event_calls": 0,
        "event_timestamps_synthesized": False,
        "price_elasticity_estimated": False,
        "tariff_defined_counterfactual": True,
        "realized_response_evidence": False,
        "realized_savings_claimed": False,
        "coverage": coverage,
        "inputs": {
            "nlr300_facility_path": str(nlr300_facility_path),
            "nlr300_facility_sha256": load_sha256,
            "xcel_rate_input_path": str(rate_path),
            "xcel_rate_input_sha256": rate_sha256,
        },
        "outputs": {
            "parquet": str(output_parquet),
            "parquet_sha256": _sha256(output_parquet),
            "csv": str(output_csv),
            "csv_sha256": _sha256(output_csv),
        },
        "source_relationship": "tariff_defined_counterfactual",
        "load_source_relationship": "third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
        "caveat": (
            "Costs cover only retained source intervals. Missing intervals are not imputed; "
            "no CPP event, response, or realized savings series is fabricated."
        ),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)
    return {
        "status": status,
        "manifest": str(manifest_path),
        "parquet": str(output_parquet),
        "csv": str(output_csv),
        "coverage": coverage,
    }
