from __future__ import annotations

import csv
import math
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PACKAGE = TASK.parent / "ExecutableServiceCommitments_Revision_20260928"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[key])
            if math.isfinite(value):
                values.append(value)
        except (KeyError, ValueError, TypeError):
            pass
    return sum(values) / len(values) if values else math.nan


def main() -> None:
    source = read_csv(PACKAGE / "source_data/fig3_ablation.csv")
    grouped = {}
    for row in source:
        grouped.setdefault(row["variant"], []).append(row)
    full = numeric(grouped["full"], "saving_usd")
    signal_map = [
        ("Full WorldMoE", "full", "Frozen CdfR2 valuation pathway"),
        ("No opportunity score", "no_opportunity", "Frozen checkpoint; opportunity score set to zero"),
        ("No task cost bias", "no_cost_bias", "Frozen checkpoint; task-conditioned price-cost bias suppressed"),
        ("Market-mean signal", "price_only", "Queue-independent price-mean guidance in the common physical allocator"),
    ]
    rows = []
    for label, variant, source_model in signal_map:
        variants = grouped[variant]
        saving = numeric(variants, "saving_usd")
        rows.append({
            "signal": label,
            "source_model": source_model,
            "same_executor": "true",
            "saving_usd": saving,
            "saving_pct": numeric(variants, "saving_pct"),
            "shifted_it_mwh": numeric(variants, "shifted_IT_mwh"),
            "late_it_mwh": numeric(variants, "late_IT_mwh"),
            "closing_it_mwh": numeric(variants, "closing_it_mwh"),
            "cvar95": numeric(variants, "cvar95_usd"),
            "saving_difference_vs_full": saving-full,
            "saving_retention_vs_full": saving/full if full else math.nan,
            "notes": "Three frozen seeds; same workload, deadlines, physical allocator, settlement and matched reference. Controlled fixed-weight signal intervention, not a retrained module deletion.",
        })

    for label in ["Price-only neural signal", "RASC signal", "Seasonal-ME signal", "LightGBM signal"]:
        rows.append({
            "signal": label,
            "source_model": "no separate compatible signal row in current frozen experiment",
            "same_executor": "false",
            "saving_usd": "NA", "saving_pct": "NA",
            "shifted_it_mwh": "NA", "late_it_mwh": "NA",
            "closing_it_mwh": "NA", "cvar95": "NA",
            "saving_difference_vs_full": "NA",
            "saving_retention_vs_full": "NA",
            "notes": "Not run through the identical queue and physical executor. Existing benchmark pipelines use different comparators or no distinct signal adapter; marked incompatible_with_common_executor rather than altering their logic.",
        })

    out = TASK / "outputs/g6_common_executor/common_executor_signal_comparison.csv"
    fields = [
        "signal", "source_model", "same_executor", "saving_usd",
        "saving_pct", "shifted_it_mwh", "late_it_mwh", "closing_it_mwh",
        "cvar95", "saving_difference_vs_full", "saving_retention_vs_full",
        "notes",
    ]
    write_csv(out, rows, fields)

    matrix_path = TASK / "outputs/combined/transportability_robustness_matrix.csv"
    matrix = read_csv(matrix_path)
    for row in rows:
        matrix.append({
            "experiment_family": "common_executor_signal",
            "condition": row["signal"], "market": "NYISO",
            "zone_scope": "4 NYISO zones", "workload_source": "Eagle",
            "workload_type": "aggregate divisible workload",
            "service_window_h": 24, "utilization": 0.0519,
            "flexible_share": 1.0, "pv_scale": 1.0, "import_scale": 1.0,
            "ramp_limit": "NA", "pipeline": row["source_model"],
            "saving_usd": row["saving_usd"], "saving_pct": row["saving_pct"],
            "flexible_it_mwh": 2981.6073925,
            "shifted_it_mwh": row["shifted_it_mwh"],
            "late_it_mwh": row["late_it_mwh"],
            "closing_it_mwh": row["closing_it_mwh"],
            "cvar95_usd": row["cvar95"], "certificate_usd": "NA",
            "capture_fraction": "NA", "saving_per_flexible_mwh": "NA",
            "saving_per_shifted_mwh": "NA", "constraint_retention": "NA",
            "feasible": "NA" if row["same_executor"] == "false" else "true",
            "source_file": "ExecutableServiceCommitments_Revision_20260928/source_data/fig3_ablation.csv",
            "notes": row["notes"],
        })

    apic_rows = read_csv(PACKAGE / "source_data/apic_r2_postda_identity.csv")
    for row in apic_rows:
        matrix.append({
            "experiment_family": "APIC_diagnostic",
            "condition": "R2 post-DA identity",
            "market": "NYISO", "zone_scope": "4 NYISO zones",
            "workload_source": "Eagle", "workload_type": "forecast diagnostic",
            "service_window_h": 24, "utilization": 0.0519,
            "flexible_share": 1.0, "pv_scale": 1.0,
            "import_scale": 1.0, "ramp_limit": "NA",
            "pipeline": f"R2 seed {row['seed']}",
            "saving_usd": "NA", "saving_pct": "NA",
            "flexible_it_mwh": "NA", "shifted_it_mwh": "NA",
            "late_it_mwh": "NA", "closing_it_mwh": "NA",
            "cvar95_usd": "NA", "certificate_usd": "NA",
            "capture_fraction": "NA", "saving_per_flexible_mwh": "NA",
            "saving_per_shifted_mwh": "NA", "constraint_retention": "NA",
            "feasible": "NA",
            "source_file": "ExecutableServiceCommitments_Revision_20260928/source_data/apic_r2_postda_identity.csv",
            "notes": f"Forecast identity diagnostic only; max residual {row['max_abs_identity_residual_usd_mwh']} US$/MWh. Not an APIC on/off economic ablation.",
        })
    write_csv(matrix_path, matrix, list(matrix[0].keys()))
    print(f"Wrote {len(rows)} common-executor signal rows and APIC diagnostic rows.")


if __name__ == "__main__":
    main()
