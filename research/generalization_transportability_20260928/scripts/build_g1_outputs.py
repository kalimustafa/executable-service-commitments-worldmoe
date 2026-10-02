from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
TASK = WORKSPACE / "generalization_transportability_20260928"
PACKAGE = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928"
ZONE_NAMES = ["WEST", "CAPITL", "N.Y.C.", "LONGIL"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict, *names: str, default=math.nan):
    for name in names:
        try:
            v = float(row.get(name, ""))
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            continue
    return default


def main() -> None:
    forcing = WORKSPACE / "external_validation/data/forcing_2025.npz"
    with np.load(forcing, allow_pickle=False) as data:
        arrivals = data["eagle_arrivals"].astype(np.float64)
        k = data["k"].astype(np.float64)
    dt = 0.25
    zone_work = dt * np.sum(arrivals * k[None, :], axis=0)
    raw_rows = []
    for idx, zone in enumerate(ZONE_NAMES):
        path = (
            TASK / "outputs/g1_spatial/runs" /
            f"CdfR2_Eagle_2025_s20260920_ep06_coupled_cdf_w24_zone{idx}_single_zone.json"
        )
        with path.open("r", encoding="utf-8-sig") as handle:
            source = json.load(handle)
        saving = value(source, "saving_usd", "cash_value_usd")
        reference = value(source, "reference_cost_usd")
        policy = value(source, "adjusted_cost_usd")
        flexible = float(zone_work[idx])
        raw_rows.append({
            "fold": f"single-zone frozen replay: {zone}",
            "train_zones": "WEST;CAPITL;N.Y.C.;LONGIL (same frozen checkpoint)",
            "heldout_zone": "none",
            "zone_evaluated": zone,
            "validation_type": "single_zone_frozen_replay_not_held_out",
            "checkpoint": "CdfR2 seed 20260920 epoch 6",
            "seed": 20260920,
            "seed_scope": "primary_only",
            "annual_saving_usd": saving,
            "saving_pct": saving / reference * 100 if reference > 0 else "NA",
            "reference_cost_usd": reference,
            "policy_cost_usd": policy,
            "flexible_it_mwh": flexible,
            "shifted_it_mwh": value(source, "shifted_IT_mwh"),
            "late_it_mwh": value(source, "late_IT_mwh"),
            "closing_it_mwh": value(source, "closing_it_mwh", "final_queue_IT_mwh"),
            "cvar95": value(source, "cvar95_usd", "mtm_daily_loss_cvar95_usd"),
            "saving_per_flexible_mwh": saving / flexible if flexible > 1e-8 else "NA",
            "saving_per_shifted_mwh": (
                saving / value(source, "shifted_IT_mwh")
                if value(source, "shifted_IT_mwh") > 1e-8 else "NA"
            ),
            "certificate_ceiling_usd": "NA",
            "capture_fraction": "NA",
            "feasible": "true" if value(source, "late_IT_mwh") < 1e-8 else "false",
            "source_file": path.relative_to(WORKSPACE).as_posix(),
            "notes": (
                "Single-zone frozen replay. Zone embedding was trained on all four zones; "
                "this is not held-out or zero-shot transport. Zone-specific certificate is unavailable. "
                "Year-end closing work is included in the settlement liability."
            ),
        })

    g1_fields = [
        "fold", "train_zones", "heldout_zone", "zone_evaluated",
        "validation_type", "checkpoint", "seed", "seed_scope",
        "annual_saving_usd", "saving_pct", "reference_cost_usd",
        "policy_cost_usd", "flexible_it_mwh", "shifted_it_mwh",
        "late_it_mwh", "closing_it_mwh", "cvar95",
        "saving_per_flexible_mwh", "saving_per_shifted_mwh",
        "certificate_ceiling_usd", "capture_fraction", "feasible",
        "source_file", "notes",
    ]
    out = TASK / "outputs/g1_spatial/spatial_holdout_summary.csv"
    write_csv(out, raw_rows, g1_fields)
    write_csv(TASK / "tables/TABLE_G1_SPATIAL_GENERALIZATION.csv", raw_rows, g1_fields)

    (TASK / "reports/G1_ZONE_ARCHITECTURE_CHECK.md").write_text(
        """# G1 zone architecture check

zone representation:
    learned fixed-size zone/site embeddings: nn.Embedding(4, 16) in
    ExecutableServiceCommitments_Revision_20260928/engine/src/models/market_query.py;
    nn.Embedding(4, width) in engine/src/models/causal_physical_encoder.py;
    zone-indexed calibration in engine/vendor/r2_market.py.

unseen zone directly supported:
    no

reason:
    the frozen model and physical encoder accept four trained integer indices,
    not a feature-only representation for an unseen zone. A held-out zone fold
    would either reuse an embedding trained on that zone or require changing
    the selected architecture/training protocol. The four annual replays are
    therefore single-zone frozen robustness runs, not spatial holdout or
    zero-shot evaluation.
""",
        encoding="utf-8",
    )
    (TASK / "reports/G1_NOT_SUPPORTED_REPORT.md").write_text(
        """# G1 strict held-out result not supported by the frozen architecture

Strict unseen-zone generalization is not supported by this model: the checkpoint
contains fixed four-zone learned embeddings. The study therefore uses a
single-zone frozen replay for WEST, CAPITL, N.Y.C. and LONGIL. These runs
characterize within-NYISO zone-wise robustness under the existing trained
embeddings; they do not establish held-out-zone transfer.

Measured rows are in outputs/g1_spatial/spatial_holdout_summary.csv. Each
zone uses the same primary seed and frozen 2024-selected checkpoint. The
zone-specific certified savings ceiling is unavailable, so capture fractions
are NA. A separate independent-zone refit is outside this experiment because
the current architecture does not support unseen indices without changing
the frozen design.
""",
        encoding="utf-8",
    )

    norm_path = TASK / "outputs/g5_normalized/normalized_metrics.csv"
    norm = read_csv(norm_path)
    matrix_path = TASK / "outputs/combined/transportability_robustness_matrix.csv"
    matrix = read_csv(matrix_path)
    manifest_path = TASK / "experiment_manifest.csv"
    manifest = read_csv(manifest_path)
    for row in raw_rows:
        norm.append({
            "experiment_family": "spatial_zonewise_robustness",
            "condition": row["zone_evaluated"],
            "market": "NYISO", "zone_scope": row["zone_evaluated"],
            "workload_source": "Eagle",
            "workload_type": "aggregate divisible workload",
            "service_window_h": 24, "seed": 20260920,
            "reference_cost_usd": row["reference_cost_usd"],
            "policy_cost_usd": row["policy_cost_usd"],
            "saving_usd": row["annual_saving_usd"],
            "saving_pct": row["saving_pct"],
            "total_it_mwh": row["flexible_it_mwh"],
            "flexible_it_mwh": row["flexible_it_mwh"],
            "shifted_it_mwh": row["shifted_it_mwh"],
            "saving_per_total_it_mwh": row["saving_per_flexible_mwh"],
            "saving_per_flexible_mwh": row["saving_per_flexible_mwh"],
            "saving_per_shifted_mwh": row["saving_per_shifted_mwh"],
            "certificate_usd": "NA", "capture_fraction": "NA",
            "constraint_retention": "NA",
            "late_work_it_mwh": row["late_it_mwh"],
            "closing_it_mwh": row["closing_it_mwh"],
            "feasible": row["feasible"],
            "source_file": row["source_file"], "notes": row["notes"],
        })
        matrix.append({
            "experiment_family": "spatial_zonewise_robustness",
            "condition": row["zone_evaluated"], "market": "NYISO",
            "zone_scope": row["zone_evaluated"], "workload_source": "Eagle",
            "workload_type": "aggregate divisible workload",
            "service_window_h": 24, "utilization": 0.0519,
            "flexible_share": 1.0, "pv_scale": 1.0,
            "import_scale": 1.0, "ramp_limit": "NA",
            "pipeline": "WorldMoE frozen primary single-zone replay",
            "saving_usd": row["annual_saving_usd"],
            "saving_pct": row["saving_pct"],
            "flexible_it_mwh": row["flexible_it_mwh"],
            "shifted_it_mwh": row["shifted_it_mwh"],
            "late_it_mwh": row["late_it_mwh"],
            "closing_it_mwh": row["closing_it_mwh"],
            "cvar95_usd": row["cvar95"],
            "certificate_usd": "NA", "capture_fraction": "NA",
            "saving_per_flexible_mwh": row["saving_per_flexible_mwh"],
            "saving_per_shifted_mwh": row["saving_per_shifted_mwh"],
            "constraint_retention": "NA", "feasible": row["feasible"],
            "source_file": row["source_file"], "notes": row["notes"],
        })
        manifest.append({
            "experiment_id": f"G1_single_zone_{row['zone_evaluated']}_seed20260920",
            "family": "G1_spatial_zonewise_robustness",
            "timestamp": "2026-09-28", "seed": 20260920,
            "checkpoint": "ExecutableServiceCommitments_Revision_20260928/engine/checkpoints/CdfR2_seed20260920_epoch06.pt",
            "zone": row["zone_evaluated"], "workload": "Eagle",
            "utilization": 0.0519, "flexible_share": 1.0,
            "service_window_h": 24, "pv_scale": 1.0,
            "import_scale": 1.0, "ramp_limit": "NA",
            "reference_type": "zone-conditioned matched earliest service",
            "status": "NEW_PRIMARY_SEED_RUN",
            "feasible": row["feasible"],
            "runtime_seconds": "NA", "output_file": row["source_file"],
            "notes": "Frozen zone-wise replay; not a held-out or zero-shot fold.",
        })
    write_csv(norm_path, norm, list(norm[0].keys()))
    write_csv(matrix_path, matrix, list(matrix[0].keys()))
    write_csv(manifest_path, manifest, list(manifest[0].keys()))
    print(json.dumps({
        "zones": [r["zone_evaluated"] for r in raw_rows],
        "savings_usd": {r["zone_evaluated"]: round(float(r["annual_saving_usd"]), 2) for r in raw_rows},
        "rows": len(raw_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
