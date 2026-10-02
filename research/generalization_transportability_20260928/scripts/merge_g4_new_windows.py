from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
TASK = WORKSPACE / "generalization_transportability_20260928"
PACKAGE = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def val(row: dict, *names: str, default=math.nan):
    for key in names:
        try:
            v = float(row.get(key, ""))
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            continue
    return default


def json_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> None:
    src = PACKAGE / "source_data/fig3_service_windows.csv"
    prior = read_csv(src)
    workload = read_csv(PACKAGE / "experiments/E03_heterogeneous_deadlines/mixed_deadline_results.csv")
    total_eagle = val(workload[0], "total_it_mwh")
    primary_rows = []
    for window in (8, 12, 16, 24, 48):
        if window in (12, 16, 24):
            source = next(r for r in prior if int(float(r["window_h"])) == window and r["seed"] == "20260920")
            row = {
                "service_window_h": window, "seed": "20260920",
                "reference_cost_usd": val(source, "reference_cost_usd"),
                "policy_cost_usd": val(source, "adjusted_cost_usd"),
                "saving_usd": val(source, "saving_usd"),
                "saving_pct": val(source, "saving_pct"),
                "flexible_it_mwh": total_eagle,
                "shifted_it_mwh": val(source, "shifted_IT_mwh"),
                "late_it_mwh": val(source, "late_IT_mwh"),
                "closing_it_mwh": val(source, "closing_it_mwh"),
                "certificate_usd": val(source, "certified_savings_ceiling_usd"),
                "capture_fraction": val(source, "capture_lower_fraction"),
                "saving_per_flexible_mwh": val(source, "saving_usd") / total_eagle,
                "marginal_value_usd_per_added_hour": "NA",
                "feasible": "true" if val(source, "late_IT_mwh") < 1e-8 else "false",
                "status": "REUSED_EXISTING_PRIMARY_SEED",
                "cvar95_usd": val(source, "cvar95_usd"),
                "source_file": "ExecutableServiceCommitments_Revision_20260928/source_data/fig3_service_windows.csv",
                "notes": "Frozen 2025 primary seed, exact same selected checkpoint.",
            }
        else:
            path = TASK / f"outputs/g4_service_window/runs/CdfR2_Eagle_2025_s20260920_ep06_coupled_cdf_w{window}.json"
            result = json_row(path)
            saving = val(result, "saving_usd", "cash_value_usd")
            reference = val(result, "reference_cost_usd")
            late = val(result, "late_IT_mwh", "late_it_mwh")
            closing = val(result, "closing_it_mwh", "final_queue_IT_mwh")
            row = {
                "service_window_h": window, "seed": "20260920",
                "reference_cost_usd": reference,
                "policy_cost_usd": val(result, "adjusted_cost_usd", default=reference-saving),
                "saving_usd": saving,
                "saving_pct": val(result, "saving_pct", default=saving/reference*100 if reference else math.nan),
                "flexible_it_mwh": total_eagle,
                "shifted_it_mwh": val(result, "shifted_IT_mwh"),
                "late_it_mwh": late,
                "closing_it_mwh": closing,
                "certificate_usd": "NA",
                "capture_fraction": "NA",
                "saving_per_flexible_mwh": saving / total_eagle,
                "marginal_value_usd_per_added_hour": "NA",
                "feasible": "true" if late < 1e-8 else "false",
                "status": "NEW_PRIMARY_SEED_RUN",
                "cvar95_usd": val(result, "mtm_daily_loss_cvar95_usd", "daily_loss_cvar95_usd"),
                "source_file": path.relative_to(WORKSPACE).as_posix(),
                "notes": (
                    "New frozen-checkpoint annual replay; no certificate available for this window."
                    if window == 8 else
                    "Zero deadline lateness; year-end closing work remains and is valued in the matched settlement."
                ),
            }
        primary_rows.append(row)

    primary_rows.sort(key=lambda r: int(r["service_window_h"]))
    for i, row in enumerate(primary_rows):
        if i:
            prior_row = primary_rows[i - 1]
            row["marginal_value_usd_per_added_hour"] = (
                (float(row["saving_usd"]) - float(prior_row["saving_usd"]))
                / (int(row["service_window_h"]) - int(prior_row["service_window_h"]))
            )

    existing_three = read_csv(TASK / "outputs/g4_service_window/service_window_results.csv")
    mean_rows = [r for r in existing_three if r.get("status") == "REUSED_EXISTING"]
    for row in mean_rows:
        row["seed"] = "three_seed_mean"
        row["status"] = "REUSED_EXISTING_THREE_SEED_MEAN"
        row["marginal_value_usd_per_added_hour"] = "NA"

    table_rows = primary_rows + mean_rows
    out_fields = [
        "service_window_h", "seed", "reference_cost_usd", "policy_cost_usd",
        "saving_usd", "saving_pct", "flexible_it_mwh", "shifted_it_mwh",
        "late_it_mwh", "closing_it_mwh", "certificate_usd", "capture_fraction",
        "saving_per_flexible_mwh", "marginal_value_usd_per_added_hour",
        "feasible", "status", "cvar95_usd", "source_file", "notes",
    ]
    write_csv(TASK / "outputs/g4_service_window/service_window_results.csv", table_rows, out_fields)
    write_csv(TASK / "tables/TABLE_G4_SERVICE_WINDOW.csv", table_rows, out_fields)

    # Add only the new annual 8 h and 48 h rows to the normalized and robustness tables.
    norm_path = TASK / "outputs/g5_normalized/normalized_metrics.csv"
    norm_rows = read_csv(norm_path)
    matrix_path = TASK / "outputs/combined/transportability_robustness_matrix.csv"
    matrix_rows = read_csv(matrix_path)
    manifest_path = TASK / "experiment_manifest.csv"
    manifest_rows = read_csv(manifest_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    for row in primary_rows:
        window = int(row["service_window_h"])
        if window not in (8, 48):
            continue
        saving = float(row["saving_usd"])
        reference = float(row["reference_cost_usd"])
        shifted = val(row, "shifted_it_mwh")
        late = val(row, "late_it_mwh")
        closing = val(row, "closing_it_mwh")
        norm_rows.append({
            "experiment_family": "service_window",
            "condition": f"{window}h primary seed",
            "market": "NYISO", "zone_scope": "4 NYISO zones",
            "workload_source": "Eagle",
            "workload_type": "aggregate divisible workload",
            "service_window_h": window, "seed": "20260920",
            "reference_cost_usd": reference,
            "policy_cost_usd": row["policy_cost_usd"],
            "saving_usd": saving, "saving_pct": row["saving_pct"],
            "total_it_mwh": total_eagle, "flexible_it_mwh": total_eagle,
            "shifted_it_mwh": shifted,
            "saving_per_total_it_mwh": saving / total_eagle,
            "saving_per_flexible_mwh": saving / total_eagle,
            "saving_per_shifted_mwh": saving / shifted if math.isfinite(shifted) and shifted > 1e-8 else "NA",
            "certificate_usd": "NA", "capture_fraction": "NA",
            "constraint_retention": "NA", "late_work_it_mwh": late,
            "closing_it_mwh": closing, "feasible": row["feasible"],
            "source_file": row["source_file"], "notes": row["notes"],
        })
        matrix_rows.append({
            "experiment_family": "service_window",
            "condition": f"{window}h primary seed",
            "market": "NYISO", "zone_scope": "4 NYISO zones",
            "workload_source": "Eagle",
            "workload_type": "aggregate divisible workload",
            "service_window_h": window, "utilization": 0.0519,
            "flexible_share": 1.0, "pv_scale": 1.0,
            "import_scale": 1.0, "ramp_limit": "NA",
            "pipeline": "WorldMoE frozen primary",
            "saving_usd": saving, "saving_pct": row["saving_pct"],
            "flexible_it_mwh": total_eagle, "shifted_it_mwh": shifted,
            "late_it_mwh": late, "closing_it_mwh": closing,
            "cvar95_usd": row["cvar95_usd"], "certificate_usd": "NA",
            "capture_fraction": "NA", "saving_per_flexible_mwh": saving / total_eagle,
            "saving_per_shifted_mwh": saving / shifted if math.isfinite(shifted) and shifted > 1e-8 else "NA",
            "constraint_retention": "NA", "feasible": row["feasible"],
            "source_file": row["source_file"], "notes": row["notes"],
        })
        manifest_rows.append({
            "experiment_id": f"G4_window{window}_seed20260920",
            "family": "G4_service_window", "timestamp": timestamp,
            "seed": "20260920",
            "checkpoint": "ExecutableServiceCommitments_Revision_20260928/engine/checkpoints/CdfR2_seed20260920_epoch06.pt",
            "zone": "all", "workload": "Eagle",
            "utilization": 0.0519, "flexible_share": 1.0,
            "service_window_h": window, "pv_scale": 1.0,
            "import_scale": 1.0, "ramp_limit": "NA",
            "reference_type": "matched earliest service",
            "status": row["status"], "feasible": row["feasible"],
            "runtime_seconds": "NA", "output_file": row["source_file"],
            "notes": row["notes"],
        })
    write_csv(norm_path, norm_rows, list(norm_rows[0].keys()))
    write_csv(matrix_path, matrix_rows, list(matrix_rows[0].keys()))
    write_csv(manifest_path, manifest_rows, list(manifest_rows[0].keys()))
    print(json.dumps({
        "primary_window_rows": len(primary_rows),
        "three_seed_mean_rows": len(mean_rows),
        "new_normalized_rows": 2,
        "new_manifest_rows": 2,
        "output": "outputs/g4_service_window/service_window_results.csv",
    }, indent=2))


if __name__ == "__main__":
    main()
