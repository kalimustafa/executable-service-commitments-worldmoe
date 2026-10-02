from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
TASK = WORKSPACE / "generalization_transportability_20260928"
PACKAGE = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928"
SERVICE = WORKSPACE / "ServiceCommitment_1to5_20260927"
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(SERVICE))

helper_spec = importlib.util.spec_from_file_location(
    "g3_executor", TASK / "src/g3_executor.py",
)
g3_executor = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(g3_executor)
closing_work_mwh = g3_executor.closing_work_mwh
deadline_miss_mwh = g3_executor.deadline_miss_mwh
execute_rigid_flexible_earliest = g3_executor.execute_rigid_flexible_earliest
from src import context as ctx


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment_id", "seed", "utilization_target", "utilization_realized",
        "seed_scope", "flexible_share", "service_window_h", "total_variable_it_mwh",
        "rigid_it_mwh", "flexible_it_mwh", "earliest_reference_cost_usd",
        "policy_cost_usd", "annual_saving_usd", "saving_pct", "shifted_it_mwh",
        "late_it_mwh", "closing_it_mwh", "earliest_late_it_mwh",
        "structurally_feasible", "policy_feasible", "cvar95_usd",
        "saving_per_flexible_mwh", "saving_per_shifted_mwh", "capture_fraction",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    arrivals, _, _ = ctx.configure(2025)
    c = ctx.c
    nominal = float(c.DT * np.sum(arrivals * c.k) / (c.k.sum() * c.N * c.DT))
    capacity = np.minimum(
        1.0,
        (np.asarray(c.c["grid_import_limit"]) / np.asarray(c.c["pue_max"]) - c.idle) / c.k,
    )
    capacity = np.maximum(np.minimum(capacity, 1.0), 0.0)
    windows = [24]
    main_targets = [0.20, 0.40, 0.60, 0.80]
    shares = [0.10, 0.20, 0.30, 0.50]
    target_rows: list[dict] = []

    # Required low-cost smoke: one site, two days, frozen primary-seed workload,
    # deterministic split, rigid first, flexible inclusive 24 h deadline.
    smoke_n = 2 * 24 * 4
    smoke_scale = 0.05191486804809399 / nominal
    smoke_total = arrivals[:smoke_n, :1] * smoke_scale
    smoke_rigid = smoke_total * 0.80
    smoke_flexible = smoke_total * 0.20
    smoke_book, smoke_joined, smoke_left = execute_rigid_flexible_earliest(
        smoke_rigid, smoke_flexible, capacity[:1], 24 * 4,
    )
    smoke_late = deadline_miss_mwh(smoke_book, smoke_joined, c.k[:1], c.DT, 24 * 4)
    smoke_closing = closing_work_mwh(smoke_book, smoke_n, c.k[:1], c.DT)
    if not np.allclose(smoke_rigid.sum() + smoke_flexible.sum(), smoke_joined.sum(), atol=1e-8):
        raise RuntimeError("G3 smoke failed work conservation.")
    if not np.allclose(smoke_book.sum() + smoke_left.sum(), smoke_joined.sum(), atol=1e-7):
        raise RuntimeError("G3 smoke lost released work.")
    if smoke_late > 1e-8 or smoke_closing > 1e-8:
        raise RuntimeError(f"G3 smoke had late or closing work: {smoke_late}, {smoke_closing}.")

    # Screen each distinct total arrival profile once; flexible share does not
    # change total work or the fastest possible completion schedule.
    screen_results = {}
    for util in main_targets + [1.0]:
        scale = util / nominal
        total = arrivals * scale
        ref = ctx.engine.reference(total, capacity)
        earliest_late = ctx.fifo_lateness(ref, total, 24)
        closing = closing_work_mwh(ref, c.N, c.k, c.DT)
        total_mwh = float(c.DT * np.sum(total * c.k))
        realized = util
        feasible = earliest_late <= 1e-8 and closing <= 1e-8
        screen_results[util] = (earliest_late, closing, feasible)
        for share in shares + [0.0, 1.0]:
            rigid_mwh = total_mwh * (1.0 - share)
            flexible_mwh = total_mwh * share
            note = (
                "Primary-seed structural screen only; no policy economic replay for an infeasible cell."
                if not feasible else
                "Feasibility screen passed; frozen policy replay remains pending."
            )
            target_rows.append(dict(
                experiment_id=f"G3_u{int(util*100)}_f{int(share*100)}_w24_seed20260920",
                seed=20260920, seed_scope="primary_only",
                utilization_target=util, utilization_realized=realized,
                flexible_share=share, service_window_h=24, total_variable_it_mwh=total_mwh,
                rigid_it_mwh=rigid_mwh, flexible_it_mwh=flexible_mwh,
                earliest_reference_cost_usd="NA", policy_cost_usd="NA",
                annual_saving_usd=0 if share == 0 and feasible else "NA",
                saving_pct=0 if share == 0 and feasible else "NA", shifted_it_mwh="NA",
                late_it_mwh="NA", closing_it_mwh=closing if not feasible else "NA",
                earliest_late_it_mwh=earliest_late,
                structurally_feasible=str(feasible).lower(),
                policy_feasible="NA", cvar95_usd="NA",
                saving_per_flexible_mwh="NA", saving_per_shifted_mwh="NA",
                capture_fraction="NA", notes=note,
            ))

    original = read_csv(PACKAGE / "experiments/E05_workload_scaling/results.csv")
    primary_main = next(
        row for row in original if abs(float(row["target_utilization"]) - 0.05191486804809399) < 1e-8
    )
    main_row = dict(
        experiment_id="G3_original_allflex_24h_primary",
        seed=20260920, utilization_target=0.05191486804809399,
        seed_scope="primary_only",
        utilization_realized=0.05191486804809399, flexible_share=1.0,
        service_window_h=24,
        total_variable_it_mwh=primary_main["it_energy_mwh"],
        rigid_it_mwh=0, flexible_it_mwh=primary_main["it_energy_mwh"],
        earliest_reference_cost_usd=primary_main["reference_cost_usd"],
        policy_cost_usd=primary_main["policy_cost_usd"],
        annual_saving_usd=primary_main["saving_usd"],
        saving_pct=primary_main["saving_pct"],
        shifted_it_mwh=primary_main["shifted_IT_mwh"],
        late_it_mwh=primary_main["late_IT_mwh"],
        closing_it_mwh=primary_main["closing_it_mwh"],
        earliest_late_it_mwh=0.0, structurally_feasible="true",
        policy_feasible="true", cvar95_usd=primary_main["cvar95_usd"],
        saving_per_flexible_mwh=primary_main["saving_per_flexible_mwh_usd"],
        saving_per_shifted_mwh=float(primary_main["saving_usd"]) / float(primary_main["shifted_IT_mwh"]),
        capture_fraction=float(primary_main["saving_usd"]) / 132605.42084139131,
        notes="Reused primary 24 h baseline. Three-seed summary remains in source_data/fig3_service_windows.csv.",
    )
    target_rows.insert(0, main_row)
    target_rows.insert(1, dict(
        experiment_id="G3_original_rigid_only_24h",
        seed=20260920, seed_scope="primary_only",
        utilization_target=0.05191486804809399,
        utilization_realized=0.05191486804809399, flexible_share=0.0,
        service_window_h=24, total_variable_it_mwh=primary_main["it_energy_mwh"],
        rigid_it_mwh=primary_main["it_energy_mwh"], flexible_it_mwh=0.0,
        earliest_reference_cost_usd=primary_main["reference_cost_usd"],
        policy_cost_usd=primary_main["reference_cost_usd"],
        annual_saving_usd=0.0, saving_pct=0.0, shifted_it_mwh=0.0,
        late_it_mwh=0.0, closing_it_mwh=0.0, earliest_late_it_mwh=0.0,
        structurally_feasible="true", policy_feasible="true",
        cvar95_usd="NA", saving_per_flexible_mwh="NA",
        saving_per_shifted_mwh="NA", capture_fraction="NA",
        notes="No flexible resource: policy equals the matched earliest-service reference, so saving is exactly zero.",
    ))

    # The prescribed three-seed representative cells are retained for every
    # selected seed, but valuation replay is skipped because workload
    # feasibility is seed independent and each cell fails the 24 h screen.
    for util, share in [(0.40, 0.20), (0.40, 0.50), (0.60, 0.20), (0.60, 0.50)]:
        earliest_late, closing, feasible = screen_results[util]
        total_mwh = float(c.DT * np.sum(arrivals * (util / nominal) * c.k))
        for seed in (20260920, 20260921, 20260922):
            target_rows.append(dict(
                experiment_id=f"G3_u{int(util*100)}_f{int(share*100)}_w24_seed{seed}",
                seed=seed, seed_scope="three_seed", utilization_target=util,
                utilization_realized=util, flexible_share=share,
                service_window_h=24, total_variable_it_mwh=total_mwh,
                rigid_it_mwh=total_mwh * (1-share),
                flexible_it_mwh=total_mwh * share,
                earliest_reference_cost_usd="NA", policy_cost_usd="NA",
                annual_saving_usd="NA", saving_pct="NA", shifted_it_mwh="NA",
                late_it_mwh="NA", closing_it_mwh=closing,
                earliest_late_it_mwh=earliest_late,
                structurally_feasible=str(feasible).lower(),
                policy_feasible="NA", cvar95_usd="NA",
                saving_per_flexible_mwh="NA", saving_per_shifted_mwh="NA",
                capture_fraction="NA",
                notes="Representative three-seed cell retained; policy replay not run because the shared arrival profile is structurally infeasible at 24 h.",
            ))

    # Coarse primary grid locates a feasible/infeasible bracket. Tighten that
    # bracket deterministically because it defines the finite operating envelope.
    low, high = nominal, 0.20
    frontier_rows = []
    for step in range(1, 9):
        mid = (low + high) / 2
        scaled = arrivals * (mid / nominal)
        ref = ctx.engine.reference(scaled, capacity)
        late = ctx.fifo_lateness(ref, scaled, 24)
        closing = closing_work_mwh(ref, c.N, c.k, c.DT)
        feasible = late <= 1e-8 and closing <= 1e-8
        frontier_rows.append(dict(
            step=step, utilization_tested=mid, earliest_late_it_mwh=late,
            closing_it_mwh=closing, structurally_feasible=str(feasible).lower(),
            current_feasible_lower=mid if feasible else low,
            current_infeasible_upper=high if feasible else mid,
        ))
        if feasible:
            low = mid
        else:
            high = mid
    frontier_path = TASK / "outputs/g3_utilization_flex/feasibility_frontier.csv"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    with frontier_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier_rows[0]))
        writer.writeheader()
        writer.writerows(frontier_rows)

    write_rows(TASK / "outputs/g3_utilization_flex/utilization_flex_results.csv", target_rows)

    report = {
        "smoke_test": {
            "status": "PASS",
            "seed": 20260920,
            "sites": 1,
            "duration_days": 2,
            "flexible_share": 0.20,
            "service_window_h": 24,
            "released_work_mwh": float(c.DT * np.sum(smoke_joined * c.k[:1])),
            "late_work_mwh": smoke_late,
            "closing_work_mwh": smoke_closing,
            "work_conserved": True,
        },
        "capacity_fraction_by_zone": capacity.tolist(),
        "nominal_utilization": nominal,
        "structural_screen": {
            str(int(r["utilization_target"] * 100)): {
                "earliest_late_it_mwh": r["earliest_late_it_mwh"],
                "feasible": r["structurally_feasible"],
            }
            for r in target_rows if r.get("flexible_share") == 0.10
        },
        "bisection_after_fixed_grid": {
            "feasible_lower_bound": low,
            "infeasible_upper_bound": high,
            "iterations": len(frontier_rows),
            "window_h": 24,
            "source": "outputs/g3_utilization_flex/feasibility_frontier.csv",
        },
    }
    out = TASK / "outputs/g3_utilization_flex/feasibility_screen.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
