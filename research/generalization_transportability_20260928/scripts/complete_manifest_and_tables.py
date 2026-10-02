from __future__ import annotations

import csv
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


def fnum(row, key, default="NA"):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def main() -> None:
    manifest_path = TASK / "experiment_manifest.csv"
    manifest = read_csv(manifest_path)
    fields = list(manifest[0].keys())
    seen = {row.get("experiment_id") for row in manifest}
    cdf = {20260920: "06", 20260921: "02", 20260922: "00"}
    checkpoint = lambda seed: (
        f"ExecutableServiceCommitments_Revision_20260928/engine/checkpoints/"
        f"CdfR2_seed{seed}_epoch{cdf.get(int(seed), '06')}.pt"
    )

    def append(row: dict) -> None:
        if row["experiment_id"] in seen:
            return
        manifest.append(row)
        seen.add(row["experiment_id"])

    # Existing E02 same-checkpoint signal interventions.
    e02_path = PACKAGE / "experiments/E02_component_attribution/per_seed_results.csv"
    for row in read_csv(e02_path):
        seed = int(row["seed"])
        variant = row.get("variant", "unknown")
        append({
            "experiment_id": f"E02_signal_{variant}_seed{seed}",
            "family": "G6_common_executor_signal",
            "timestamp": "2026-09-28", "seed": seed,
            "checkpoint": checkpoint(seed), "zone": "all", "workload": "Eagle",
            "utilization": 0.0519, "flexible_share": 1.0,
            "service_window_h": fnum(row, "window_h"),
            "pv_scale": 1.0, "import_scale": 1.0, "ramp_limit": "NA",
            "reference_type": "matched earliest service",
            "status": "REUSED_EXISTING", "feasible": "true",
            "runtime_seconds": fnum(row, "seconds"),
            "output_file": "ExecutableServiceCommitments_Revision_20260928/experiments/E02_component_attribution/per_seed_results.csv",
            "notes": f"Frozen-weight {variant} intervention under common physical allocator; not a retrained deletion.",
        })

    # Existing heterogeneous-deadline runs.
    mixed_path = PACKAGE / "experiments/E03_heterogeneous_deadlines/mixed_deadline_results.csv"
    for i, row in enumerate(read_csv(mixed_path)):
        seed = row.get("seed", "NA")
        pathway = row.get("pathway", "unknown")
        priority = row.get("priority", "unknown")
        append({
            "experiment_id": f"E03_mixed_{i+1}_seed{seed}",
            "family": "G6_mixed_deadlines", "timestamp": "2026-09-28",
            "seed": seed, "checkpoint": checkpoint(seed), "zone": "all",
            "workload": "Eagle mixed deadline", "utilization": 0.0519,
            "flexible_share": 1.0, "service_window_h": 24, "pv_scale": 1.0,
            "import_scale": 1.0, "ramp_limit": "NA",
            "reference_type": "matched 25/50/25 deadline reference",
            "status": "REUSED_EXISTING",
            "feasible": "true" if fnum(row, "late_work_it_mwh", 0) < 1e-8 else "false",
            "runtime_seconds": "NA",
            "output_file": "ExecutableServiceCommitments_Revision_20260928/experiments/E03_heterogeneous_deadlines/mixed_deadline_results.csv",
            "notes": f"{pathway}; priority={priority}; retains late and closing work.",
        })

    # Existing physical PV cases.
    physical_path = PACKAGE / "experiments/E06_physical_sensitivity/results.csv"
    for row in read_csv(physical_path):
        pv = fnum(row, "pv_fraction_of_nominal")
        append({
            "experiment_id": f"E06_pv_{pv}_primary",
            "family": "G6_physical_PV", "timestamp": "2026-09-28",
            "seed": "20260920", "checkpoint": checkpoint(20260920),
            "zone": "all", "workload": "Eagle", "utilization": 0.0519,
            "flexible_share": 1.0, "service_window_h": 24,
            "pv_scale": pv, "import_scale": 1.0, "ramp_limit": "NA",
            "reference_type": "condition-matched earliest service",
            "status": "REUSED_EXISTING",
            "feasible": "true" if fnum(row, "late_it_mwh", 0) < 1e-8 else "false",
            "runtime_seconds": "NA",
            "output_file": "ExecutableServiceCommitments_Revision_20260928/experiments/E06_physical_sensitivity/results.csv",
            "notes": "Primary-seed PV sensitivity; matched reference recomputed for each case.",
        })

    # Existing numerical action-grid robustness.
    numerical_path = PACKAGE / "experiments/E08_numerical_convergence/results.csv"
    for row in read_csv(numerical_path):
        grid = row.get("action_grid_work_units", "unknown")
        append({
            "experiment_id": f"E08_action_grid_{grid}_primary",
            "family": "G6_numerical_convergence", "timestamp": "2026-09-28",
            "seed": "20260920", "checkpoint": checkpoint(20260920),
            "zone": "all", "workload": "Eagle", "utilization": 0.0519,
            "flexible_share": 1.0, "service_window_h": 24,
            "pv_scale": 1.0, "import_scale": 1.0, "ramp_limit": "NA",
            "reference_type": "matched earliest service",
            "status": "REUSED_EXISTING",
            "feasible": "true" if fnum(row, "late_IT_mwh", 0) < 1e-8 else "false",
            "runtime_seconds": fnum(row, "seconds"),
            "output_file": "ExecutableServiceCommitments_Revision_20260928/experiments/E08_numerical_convergence/results.csv",
            "notes": "Current-action resolution diagnostic; no checkpoint change.",
        })

    # Existing three-seed electrical stress summaries.
    for row in read_csv(PACKAGE / "source_data/fig5_electrical_constraints.csv"):
        condition = row.get("scenario", "unknown")
        append({
            "experiment_id": f"G6_electrical_{row.get('experiment')}_{condition}_three_seed_summary",
            "family": "G6_electrical_constraint", "timestamp": "2026-09-28",
            "seed": "three_seed_summary", "checkpoint": "three frozen CdfR2 seeds",
            "zone": "all", "workload": "Eagle", "utilization": 0.0519,
            "flexible_share": 1.0, "service_window_h": 24,
            "pv_scale": 1.0,
            "import_scale": 0.75 if "cap75" in condition else (0.5 if "cap50" in condition else 1.0),
            "ramp_limit": 0.25 if "ramp25" in condition else (0.125 if "ramp125" in condition else "NA"),
            "reference_type": "condition-matched earliest service",
            "status": "REUSED_THREE_SEED_SUMMARY",
            "feasible": "true" if fnum(row, "reference_late_max", 0) < 1e-8 else "false",
            "runtime_seconds": "NA",
            "output_file": "ExecutableServiceCommitments_Revision_20260928/source_data/fig5_electrical_constraints.csv",
            "notes": "Aggregated existing three-seed summary; individual seed rows are not in this CSV.",
        })

    write_csv(manifest_path, manifest, fields)

    norm_path = TASK / "outputs/g5_normalized/normalized_metrics.csv"
    norm = read_csv(norm_path)
    write_csv(TASK / "tables/TABLE_G5_NORMALIZED_TRANSPORTABILITY.csv", norm, list(norm[0].keys()))
    print(f"Manifest rows: {len(manifest)}; normalized/table rows: {len(norm)}")


if __name__ == "__main__":
    main()
