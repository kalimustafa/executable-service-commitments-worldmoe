from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

TASK = Path(__file__).resolve().parents[1]
FIG = TASK / "figures"
TABLES = TASK / "tables"
OUT = TASK / "outputs"
BLUE = "#274C77"
MID_BLUE = "#6096BA"
LIGHT_BLUE = "#A3CEF1"
PALE = "#D9E2EC"
GRAY = "#B7BDC5"
DARK = "#30343B"
RED = "#B23A48"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict, key: str, default=math.nan) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def save_figure(fig, stem: str, aliases: tuple[str, ...] = ()) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for name in (stem, *aliases):
        fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB2BD")
    ax.spines["bottom"].set_color("#AAB2BD")
    ax.tick_params(colors=DARK, labelsize=9)
    ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
    ax.set_axisbelow(True)


def write_source(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def figure_g1() -> None:
    rows = read_csv(OUT / "g1_spatial/spatial_holdout_summary.csv")
    names = [r["zone_evaluated"] for r in rows]
    normalized = [as_float(r, "saving_per_flexible_mwh") for r in rows]
    saving = [as_float(r, "annual_saving_usd") for r in rows]
    late = [as_float(r, "late_it_mwh", default=0) for r in rows]
    closing = [as_float(r, "closing_it_mwh", default=0) for r in rows]
    source = [{
        "zone": r["zone_evaluated"],
        "saving_usd": r["annual_saving_usd"],
        "saving_per_flexible_mwh": r["saving_per_flexible_mwh"],
        "late_it_mwh": r["late_it_mwh"],
        "closing_it_mwh": r["closing_it_mwh"],
        "validation_type": r["validation_type"],
    } for r in rows]
    write_source(FIG / "FIG_G1_spatial_holdout_source.csv", source, list(source[0]))
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.1), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(names))
    bars = axes[0].bar(x, normalized, color=[BLUE, MID_BLUE, LIGHT_BLUE, "#7F9DB9"], width=.66)
    axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_ylabel("Saving per flexible IT MWh (US$/MWh)")
    axes[0].set_xticks(x, names)
    axes[0].set_title("Within-NYISO zone-wise replay", loc="left", fontsize=11, color=DARK)
    style(axes[0])
    axes[1].bar(x-.18, late, width=.36, color=RED, label="Late work")
    axes[1].bar(x+.18, closing, width=.36, color=GRAY, label="Closing work")
    axes[1].set_ylabel("Work (MWh)")
    axes[1].set_xticks(x, names)
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1])
    fig.suptitle("Frozen single-zone replays (not held-out or zero-shot)", x=.06, ha="left", fontsize=13, color=DARK)
    fig.text(.06, .01, "The selected checkpoint uses learned four-zone embeddings; certificate ceilings are unavailable by zone.", fontsize=8, color="#59636E")
    fig.tight_layout(rect=(0, .04, 1, .92))
    save_figure(fig, "FIG_G1_spatial_holdout")


def figure_g2() -> None:
    rows = read_csv(TABLES / "TABLE_G2_CROSS_WORKLOAD.csv")
    names, values, late, closing, source = [], [], [], [], []
    for row in rows:
        label = f"{row['workload']} {row['service_window_h']}h"
        names.append(label)
        values.append(as_float(row, "saving_per_it_mwh"))
        late.append(as_float(row, "late_work_it_mwh", default=0))
        closing.append(as_float(row, "closing_work_it_mwh", default=0))
        source.append({
            "workload": row["workload"], "resource_model": row["resource_model"],
            "service_window_h": row["service_window_h"],
            "saving_per_it_mwh": values[-1], "saving_per_shifted_mwh": row["saving_per_shifted_mwh"],
            "late_work_it_mwh": late[-1], "closing_work_it_mwh": closing[-1],
            "service_feasible": row["service_feasible"],
        })
    write_source(FIG / "FIG_G2_cross_workload_source.csv", source, list(source[0]))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(names))
    axes[0].bar(x, values, color=[BLUE] + [MID_BLUE] * (len(names)-1), width=0.68)
    axes[0].set_ylabel("Saving per total IT MWh (US$/MWh)")
    axes[0].set_xticks(x, names, rotation=24, ha="right")
    axes[0].set_title("Workload-normalized value", loc="left", fontsize=11, color=DARK)
    style(axes[0])
    axes[1].bar(x - .18, late, width=.36, color=RED, label="Late work")
    axes[1].bar(x + .18, closing, width=.36, color=GRAY, label="Closing work")
    axes[1].set_ylabel("Work (MWh)")
    axes[1].set_xticks(x, names, rotation=24, ha="right")
    axes[1].set_title("Job-level delivery limits", loc="left", fontsize=11, color=DARK)
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1])
    fig.suptitle("Cross-workload transfer uses normalized value and completion", x=.06, ha="left", fontsize=13, color=DARK)
    fig.text(.06, .01, "Eagle and Frontier dollar totals are not directly comparable; Frontier is nonpreemptive job execution.", fontsize=8, color="#59636E")
    fig.tight_layout(rect=(0, .05, 1, .93))
    save_figure(fig, "FIG_G2_cross_workload")


def figure_g3() -> None:
    rows = read_csv(OUT / "g3_utilization_flex/utilization_flex_results.csv")
    grid = {}
    for row in rows:
        u = as_float(row, "utilization_target")
        share = as_float(row, "flexible_share")
        if u in (.2, .4, .6, .8) and share in (.1, .2, .3, .5):
            grid[(u, share)] = row
    utils = [.2, .4, .6, .8]
    shares = [.1, .2, .3, .5]
    source = []
    for u in utils:
        for share in shares:
            row = grid[(u, share)]
            source.append({
                "utilization_target": u, "flexible_share": share,
                "saving_per_flexible_mwh": row.get("saving_per_flexible_mwh", "NA"),
                "earliest_late_it_mwh": row.get("earliest_late_it_mwh", "NA"),
                "structurally_feasible": row.get("structurally_feasible", "NA"),
                "policy_replay": "not run for structurally infeasible cell",
            })
    write_source(FIG / "FIG_G3_flexibility_regime_source.csv", source, list(source[0]))
    values = np.full((len(utils), len(shares)), np.nan)
    feasible = np.zeros_like(values, dtype=bool)
    for iy, u in enumerate(utils):
        for ix, share in enumerate(shares):
            row = grid[(u, share)]
            values[iy, ix] = as_float(row, "saving_per_flexible_mwh")
            feasible[iy, ix] = row.get("structurally_feasible", "").lower() == "true"
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    ax.imshow(np.where(feasible, np.nan_to_num(values, nan=0), 0), cmap=ListedColormap([GRAY, LIGHT_BLUE, MID_BLUE, BLUE]), aspect="auto", vmin=0)
    for iy, u in enumerate(utils):
        for ix, share in enumerate(shares):
            miss = as_float(grid[(u, share)], "earliest_late_it_mwh")
            ax.add_patch(plt.Rectangle((ix-.5, iy-.5), 1, 1, facecolor="#D5D8DC", edgecolor="white", linewidth=1))
            ax.text(ix, iy-.07, "×", ha="center", va="center", fontsize=19, color=DARK, weight="bold")
            ax.text(ix, iy+.22, f"{miss:,.0f} MWh late", ha="center", va="center", fontsize=7.5, color=DARK)
    ax.set_xticks(np.arange(len(shares)), [f"{v:.0%}" for v in shares])
    ax.set_yticks(np.arange(len(utils)), [f"{v:.0%}" for v in utils])
    ax.set_xlabel("Flexible workload share")
    ax.set_ylabel("Total variable IT utilization")
    ax.set_title("24 h earliest-service feasibility screen", loc="left", fontsize=12, color=DARK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(.12, .01, "All prespecified 20–80% utilization cells fail the matched earliest-service deadline screen; no delivered savings are assigned.", fontsize=8, color="#59636E")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save_figure(fig, "FIG_G3A_value_regime_map", aliases=("FIG_G3_flexibility_regime",))
    fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
    labels = ["5.19% / 0% flex", "5.19% / 100% flex"]
    vals = [0.0, 105565.66546172019]
    write_source(
        FIG / "FIG_G3B_absolute_savings_source.csv",
        [
            {"utilization_target": 0.051914868, "flexible_share": 0.0, "annual_saving_usd": 0.0, "status": "matched rigid-only reference"},
            {"utilization_target": 0.051914868, "flexible_share": 1.0, "annual_saving_usd": 105565.66546172019, "status": "reused primary 24 h result"},
        ],
        ["utilization_target", "flexible_share", "annual_saving_usd", "status"],
    )
    bars = ax2.bar(labels, vals, color=[GRAY, BLUE], width=.5)
    ax2.set_ylabel("Annual saving (US$)")
    ax2.set_title("Observed main case and rigid-only diagnostic", loc="left", fontsize=12, color=DARK)
    ax2.bar_label(bars, labels=["$0", "$105,566"], padding=3, fontsize=9)
    style(ax2)
    fig2.tight_layout()
    save_figure(fig2, "FIG_G3B_absolute_savings")
    fig3, ax3 = plt.subplots(figsize=(6.7, 4.6))
    ax3.imshow(np.zeros((len(utils), len(shares))), cmap=ListedColormap(["#C7CCD2"]), aspect="auto")
    for iy, u in enumerate(utils):
        for ix, share in enumerate(shares):
            ax3.text(ix, iy, "INFEASIBLE", ha="center", va="center", fontsize=7.2, color=DARK, rotation=0)
    ax3.set_xticks(np.arange(len(shares)), [f"{v:.0%}" for v in shares])
    ax3.set_yticks(np.arange(len(utils)), [f"{v:.0%}" for v in utils])
    ax3.set_xlabel("Flexible workload share")
    ax3.set_ylabel("Total variable IT utilization")
    ax3.set_title("Structural feasibility at a 24 h service window", loc="left", fontsize=12, color=DARK)
    ax3.grid(False)
    for s in ax3.spines.values():
        s.set_visible(False)
    fig3.tight_layout()
    save_figure(fig3, "FIG_G3C_feasibility_map")


def figure_g4() -> None:
    rows = read_csv(OUT / "g4_service_window/service_window_results.csv")
    primary = sorted([r for r in rows if r["seed"] == "20260920"], key=lambda r: int(r["service_window_h"]))
    means = [r for r in rows if r["seed"] == "three_seed_mean"]
    source = primary + means
    write_source(FIG / "FIG_G4_service_window_source.csv", source, list(source[0]))
    x = np.array([int(r["service_window_h"]) for r in primary])
    saving = np.array([as_float(r, "saving_usd") for r in primary])
    late = np.array([as_float(r, "late_it_mwh", default=0) for r in primary])
    closing = np.array([as_float(r, "closing_it_mwh", default=0) for r in primary])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.1), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    axes[0].plot(x, saving/1000, "-o", color=BLUE, linewidth=2, markersize=5, label="Primary seed")
    for row in means:
        axes[0].scatter(int(row["service_window_h"]), as_float(row, "saving_usd")/1000,
                        s=55, facecolors="white", edgecolors=MID_BLUE, linewidth=1.5, zorder=4)
    axes[0].set_ylabel("Annual saving (US$ thousands)")
    axes[0].set_title("Matched-reference savings by service window", loc="left", fontsize=11, color=DARK)
    axes[0].legend(["Primary seed", "Three-seed mean (12/16/24 h)"], frameon=False, fontsize=8)
    style(axes[0])
    axes[1].bar(x-.16, late, width=.32, color=RED, label="Late work")
    axes[1].bar(x+.16, closing, width=.32, color=GRAY, label="Closing work")
    axes[1].set_ylabel("Work (MWh)")
    axes[1].set_xlabel("Service window (h)")
    axes[1].set_xticks(x)
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1])
    fig.text(.13, .01, "8 h and 48 h certificates are unavailable; 48 h has zero late work but retains valued year-end closing work.", fontsize=8, color="#59636E")
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, "FIG_G4_service_window")


def figure_g5() -> None:
    rows = read_csv(OUT / "g5_normalized/normalized_metrics.csv")
    selected = []
    eagle = [r for r in rows if r["experiment_family"] == "service_window" and r["condition"] == "24h"]
    if eagle:
        selected.append(("Eagle 24 h", float(np.nanmean([as_float(r, "saving_per_total_it_mwh") for r in eagle]))))
    frontier = [r for r in rows if r["experiment_family"] == "cross_workload" and ("48h" in r["condition"] or "24h" in r["condition"])]
    for r in frontier:
        selected.append((r["condition"].replace(" nonpreemptive", ""), as_float(r, "saving_per_total_it_mwh")))
    pv = [r for r in rows if r["experiment_family"] == "physical_PV" and r["condition"] in ("PV 0% of nominal", "PV 100% of nominal")]
    for r in pv:
        selected.append((r["condition"], as_float(r, "saving_per_total_it_mwh")))
    spatial = [r for r in rows if r["experiment_family"] == "spatial_zonewise_robustness"]
    for r in spatial:
        selected.append((r["condition"] + " zone", as_float(r, "saving_per_total_it_mwh")))
    selected = [(label, value) for label, value in selected if math.isfinite(value)]
    source = [{"condition": label, "saving_per_total_it_mwh": value} for label, value in selected]
    write_source(FIG / "FIG_G5_transportability_summary_source.csv", source, list(source[0]))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(selected))
    colors = [BLUE if "Eagle" in label else (MID_BLUE if "Frontier" in label else LIGHT_BLUE) for label, _ in selected]
    axes[0].bar(x, [v for _, v in selected], color=colors, width=.72)
    axes[0].set_ylabel("Saving per total IT MWh (US$/MWh)")
    axes[0].set_xticks(x, [k for k, _ in selected], rotation=32, ha="right")
    axes[0].set_title("Normalized value across resource conditions", loc="left", fontsize=11, color=DARK)
    style(axes[0])
    grid = read_csv(OUT / "g3_utilization_flex/feasibility_frontier.csv")
    lo = max(as_float(r, "current_feasible_lower") for r in grid)
    hi = min(as_float(r, "current_infeasible_upper") for r in grid)
    axes[1].barh([0], [lo*100], color=BLUE, height=.36, label="Feasible lower bound")
    axes[1].barh([0], [(hi-lo)*100], left=[lo*100], color=GRAY, height=.36, label="Infeasible bracket")
    axes[1].set_xlim(0, 20)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Variable IT utilization (%)")
    axes[1].set_title("24 h operating envelope", loc="left", fontsize=11, color=DARK)
    axes[1].text(lo*100/2, 0, f"≤ {lo*100:.2f}%\nfeasible", ha="center", va="center", color="white", fontsize=8)
    axes[1].text((lo+hi)*50, 0, f"{lo*100:.2f}–{hi*100:.2f}%", ha="center", va="center", color=DARK, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    axes[1].grid(axis="x", color="#E6E9ED", linewidth=.7)
    fig.text(.06, .01, "Zone rows are frozen single-zone replays, not held-out folds. Frontier uses nonpreemptive jobs; absolute dollars are not compared.", fontsize=8, color="#59636E")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save_figure(fig, "FIG_G5_transportability_summary")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    figure_g1()
    figure_g2()
    figure_g3()
    figure_g4()
    figure_g5()
    print("Wrote G2/G3/G4/G5 PNG, PDF, and figure-source CSV files.")


if __name__ == "__main__":
    main()
