from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


TASK = Path(__file__).resolve().parents[1]
WORKSPACE = TASK.parent
ENGINE = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928" / "engine"
ARCHIVE = WORKSPACE / "R2_PAPER_INTEGRATION_20260923"
EXTENSION = WORKSPACE / "ServiceCommitment_1to5_20260927"
sys.path[:0] = [str(ENGINE), str(ARCHIVE), str(EXTENSION), str(TASK / "scripts")]

import src  # noqa: E402
import vendor  # noqa: E402

src.__path__.append(str(EXTENSION / "src"))
vendor.__path__.append(str(EXTENSION / "vendor"))
from src import context as ctx  # noqa: E402
from isone_context_adapter import install as install_isone_context  # noqa: E402
from isone_context_adapter import install as install_isone_context  # noqa: E402


def slot_cost(x, rt, pue, pv, idle, k, refgrid):
    load = pue * (idle + k * x)
    grid = max(load - pv, 0.0)
    spill = max(pv - load, 0.0)
    return ctx.c.DT * (rt * grid + spill + ctx.c.FEE * abs(grid - refgrid))


def lower_hull(points):
    hull = []
    for point in points:
        while len(hull) >= 2:
            x0, y0 = hull[-2]
            x1, y1 = hull[-1]
            x2, y2 = point
            if (y1 - y0) * (x2 - x1) >= (y2 - y1) * (x1 - x0):
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def zone_bound(zone: int, arrivals: np.ndarray, reference_grid: np.ndarray, window: int, name: str) -> dict:
    c = ctx.c
    n = c.N
    slopes, widths, slots, constants = [], [], [], []
    convexified = 0
    for t in range(n):
        cap = min(1.0, max(0.0, ((c.c["grid_import_limit"][zone] + c.pv[t, zone]) / c.pue[t, zone] - c.idle[zone]) / c.k[zone]))
        points_x = [0.0, cap]
        for candidate in ((c.pv[t, zone] / c.pue[t, zone] - c.idle[zone]) / c.k[zone],
                          ((c.pv[t, zone] + reference_grid[t, zone]) / c.pue[t, zone] - c.idle[zone]) / c.k[zone]):
            if 0 < candidate < cap:
                points_x.append(float(candidate))
        points = [(x, slot_cost(x, c.rt[t, zone], c.pue[t, zone], c.pv[t, zone], c.idle[zone], c.k[zone], reference_grid[t, zone])) for x in sorted(set(points_x))]
        hull = lower_hull(points)
        convexified += len(hull) < len(points)
        constants.append(points[0][1])
        for (x0, y0), (x1, y1) in zip(hull[:-1], hull[1:]):
            slopes.append((y1 - y0) / (x1 - x0))
            widths.append(x1 - x0)
            slots.append(t)

    cumulative = np.cumsum(arrivals[:, zone])
    lag = int(window * 4) - 1
    expired = np.r_[np.zeros(lag), cumulative[:-lag]]
    queue_upper = np.maximum(cumulative - expired, 0.0)
    m = len(slopes)
    objective = np.r_[slopes, np.zeros(n)]
    objective[-1] = c.DT * c.k[zone] * max(c.c["pue_max"]) * c.train_rt[zone]
    upper = np.r_[widths, queue_upper]
    matrix = coo_matrix((np.r_[np.ones(m + n), -np.ones(n - 1)],
                         (np.r_[slots, np.arange(n), np.arange(1, n)],
                          np.r_[np.arange(m), m + np.arange(n), m + np.arange(n - 1)])),
                        shape=(n, m + n)).tocsr()
    result = linprog(objective, A_eq=matrix, b_eq=arrivals[:, zone],
                     bounds=list(zip(np.zeros_like(upper), upper)), method="highs",
                     options={"primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    multipliers = result.eqlin.marginals
    reduced = objective - matrix.T @ multipliers
    dual = float(multipliers @ arrivals[:, zone] + np.minimum(reduced, 0) @ upper + np.sum(constants))
    primal = float(result.fun + np.sum(constants))
    schedule = np.bincount(np.asarray(slots), weights=result.x[:m], minlength=n)
    actual = float(sum(slot_cost(schedule[t], c.rt[t, zone], c.pue[t, zone], c.pv[t, zone], c.idle[zone], c.k[zone], reference_grid[t, zone]) for t in range(n)) + objective[-1] * result.x[-1])
    residual = float(np.max(np.abs(matrix @ result.x - arrivals[:, zone])))
    if residual > 1e-6 or primal < dual - 1e-5 or actual < primal - 1e-5:
        raise RuntimeError("Invalid condition-matched ISO-NE certificate.")
    output = TASK / "outputs" / "certificates" / f"{name}_zone{zone}_certificate.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, multipliers=multipliers, reduced_cost=reduced, upper=upper,
                        objective=objective, arrivals=arrivals[:, zone], constant=constants,
                        slot=slots, segment_width=widths, relaxed_schedule=schedule, queue=result.x[m:])
    return {"zone_index": zone, "dual_usd": dual, "conservative_lower_bound_usd": math.floor(dual) - 1,
            "primal_usd": primal, "true_relaxed_schedule_cost_usd": actual,
            "duality_gap_usd": primal - dual, "balance_residual": residual,
            "convexified_slots": convexified}


def main() -> None:
    output_root = TASK / "outputs" / "certificates"
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for portfolio in ("ISO-NE-SOUTH", "ISO-NE-NORTH"):
        install_isone_context(ctx, TASK, portfolio, "neutral_mean")
        arrivals, reference, shared = ctx.configure(2025, "Eagle")
        _, reference_grid = ctx.c.cash(reference)
        reference_cost = float(ctx.c.cash(reference)[0].sum() + shared.sum() + ctx.c.DT * np.sum(reference[ctx.c.N:] * ctx.c.k * max(ctx.c.c["pue_max"]) * ctx.c.train_rt))
        started = time.perf_counter()
        zone_rows = [zone_bound(z, arrivals, reference_grid, 24, portfolio.replace("-", "_")) for z in range(4)]
        lower_bound = sum(row["conservative_lower_bound_usd"] for row in zone_rows) + float(shared.sum())
        ceiling = reference_cost - lower_bound
        result = {
            "market": "ISO-NE", "portfolio": portfolio, "workload": "Eagle",
            "window_h": 24, "reference_cost_rt_basis_usd": reference_cost,
            "certified_policy_cost_lower_bound_usd": lower_bound,
            "certified_saving_ceiling_usd": ceiling,
            "shared_da_component_usd": float(shared.sum()),
            "cash_basis": "frozen executor RT grid-import settlement plus PV curtailment and frozen terminal liability; no external day-ahead nomination series supplied",
            "seconds": time.perf_counter() - started,
            "zones": zone_rows,
        }
        (output_root / f"{portfolio.replace('-', '_')}_w24_certificate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_rows.append({k: v for k, v in result.items() if k != "zones"})
        for row in zone_rows:
            all_rows.append({"market": "ISO-NE", "portfolio": portfolio, **row})
        print({"portfolio": portfolio, "certified_saving_ceiling_usd": round(ceiling, 2), "seconds": round(result["seconds"], 1)})
    pd.DataFrame(all_rows).to_csv(output_root / "ISO-NE_w24_certificate_summary.csv", index=False)


if __name__ == "__main__":
    main()
