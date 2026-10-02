from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
WORKSPACE = TASK.parent
DATA = TASK / "data_processed"
OUT = TASK / "outputs"
TZ = "America/New_York"
PORTFOLIOS = {
    "ISO-NE-SOUTH": ["CONNECTICUT", "NEMASSBOST", "SEMASS", "WCMASS"],
    "ISO-NE-NORTH": ["MAINE", "NEWHAMPSHIRE", "VERMONT", "RHODEISLAND"],
}
SEEDS = [20260920, 20260921, 20260922]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_forecast(cache_path: Path, zones: list[str], features: pd.DataFrame, portfolio: str, embedding: str) -> dict:
    with np.load(cache_path, allow_pickle=False) as cache:
        cache_zones = cache["zone_codes"].astype(str).tolist()
        selected = [cache_zones.index(zone) for zone in zones]
        forecasts = cache["rt_1"][:, selected, :]
    frame = features[features.zone_code.astype(str).isin(zones)].copy()
    order = {zone: i for i, zone in enumerate(zones)}
    frame["site_order"] = frame.zone_code.astype(str).map(order)
    frame = frame.sort_values(["timestamp_utc", "site_order"], kind="stable")
    target = frame.rt_integrated.to_numpy(np.float32).reshape(forecasts.shape[:2])
    tau = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32)
    error = target[:, :, None] - forecasts
    pinball = np.maximum(tau * error, (tau - 1) * error).mean(-1)
    covered = (target >= forecasts[:, :, 0]) & (target <= forecasts[:, :, -1])
    return {
        "market": "ISO-NE", "portfolio": portfolio, "embedding_rule": embedding,
        "forecast_pinball_rt": float(pinball.mean()),
        "coverage90_pct": float(100 * covered.mean()),
        "forecast_rows": int(target.size),
    }


def daily_execution(result: dict, result_path: Path, daily_path: Path, trajectory_path: Path,
                    workload: dict, market15: pd.DataFrame, portfolio: str, embedding: str) -> pd.DataFrame:
    zones = PORTFOLIOS[portfolio]
    times = workload["times"]
    n = len(times)
    book = np.load(trajectory_path, allow_pickle=False)["compute"].astype(np.float64)
    reference = np.load(trajectory_path, allow_pickle=False)["reference"].astype(np.float64)
    da = market15.pivot(index="timestamp_utc", columns="zone_code", values="da_lmp_usd_mwh").reindex(index=times, columns=zones).to_numpy(np.float64)
    rt = market15.pivot(index="timestamp_utc", columns="zone_code", values="rt_lmp_final_usd_mwh").reindex(index=times, columns=zones).to_numpy(np.float64)
    pv, pue = workload["pv"], workload["pue"]
    idle, k = workload["idle"], workload["k"]
    dt = 0.25
    policy_load = pue * (idle + k * book[:n])
    reference_load = pue * (idle + k * reference[:n])
    policy_grid = np.maximum(policy_load - pv, 0.0)
    reference_grid = np.maximum(reference_load - pv, 0.0)
    policy_cost = dt * (rt * policy_grid + np.maximum(pv - policy_load, 0.0)).sum(1)
    reference_cost = dt * (rt * reference_grid + np.maximum(pv - reference_load, 0.0)).sum(1)
    fee = dt * np.abs(policy_grid - reference_grid).sum(1)
    cash_saving = reference_cost - policy_cost - fee

    released = np.cumsum(workload["arrivals"] * k[None, :], axis=0)
    served = np.cumsum(book[:n] * k[None, :], axis=0)
    outstanding = dt * np.maximum(released - served, 0.0).sum(1)
    due = np.arange(n) + 24 * 4 - 1
    booked_cum = np.cumsum(book, axis=0)
    due_service = booked_cum[np.minimum(due, len(booked_cum) - 1)]
    released_cum = np.cumsum(workload["arrivals"], axis=0)
    previous_release = released_cum - workload["arrivals"]
    late_units = np.maximum(released_cum - np.maximum(previous_release, due_service), 0.0)
    late_by_slot = dt * late_units.dot(k)

    local_dates = times.tz_convert(TZ).strftime("%Y-%m-%d")
    daily_core = pd.DataFrame({
        "date": local_dates,
        "cash_saving_usd": cash_saving,
        "price_spread_mean_usd_mwh": (da - rt).mean(1),
        "price_spread_abs_p95_usd_mwh": np.nan,
        "outstanding_work_mwh": outstanding,
        "late_it_mwh": late_by_slot,
    })
    daily_core["price_spread_abs_p95_usd_mwh"] = np.abs(da - rt).mean(1)
    daily = daily_core.groupby("date", sort=True).agg(
        cash_saving_usd=("cash_saving_usd", "sum"),
        price_spread_mean_usd_mwh=("price_spread_mean_usd_mwh", "mean"),
        price_spread_abs_p95_usd_mwh=("price_spread_abs_p95_usd_mwh", lambda x: float(np.quantile(x, 0.95))),
        outstanding_work_mwh=("outstanding_work_mwh", "last"),
        late_it_mwh=("late_it_mwh", "sum"),
    ).reset_index()
    source_daily = pd.read_csv(daily_path)
    source_daily["date"] = source_daily["date"].astype(str)
    daily = daily.merge(source_daily[["date", "value_usd", "mtm_value_usd"]].rename(columns={
        "value_usd": "service_adjusted_saving_usd", "mtm_value_usd": "service_adjusted_mtm_saving_usd",
    }), on="date", how="left", validate="one_to_one")
    daily["market"] = "ISO-NE"
    daily["portfolio"] = portfolio
    daily["seed"] = result.get("seed", 0)
    daily["embedding_rule"] = embedding
    daily["policy"] = "Frozen WorldMoE zero-shot" if result.get("seed", 0) else result.get("policy", "baseline")
    daily["outstanding_liability_usd"] = daily.outstanding_work_mwh * workload["liability_rate_usd_per_mwh"]
    daily["closing_work_mwh"] = 0.0
    daily.loc[daily.index[-1], "closing_work_mwh"] = float(result.get("closing_it_mwh", result.get("final_queue_IT_mwh", 0.0)))
    daily["certificate_usd"] = result.get("certified_saving_ceiling_usd", np.nan)
    daily["cumulative_service_adjusted_saving_usd"] = daily.service_adjusted_saving_usd.cumsum()
    return daily


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "market_only" / "runs"
    trajectory_dir = run_dir / "trajectories"
    baseline_dir = OUT / "baselines" / "runs"
    baseline_trajectory = baseline_dir / "trajectories"

    native = pd.read_csv(WORKSPACE / "ExecutableServiceCommitments_Revision_20260928" / "source_data" / "worldmoe_three_seed_results.csv")
    native_scores = pd.read_csv(WORKSPACE / "ExecutableServiceCommitments_Revision_20260928" / "source_data" / "worldmoe_forecast_scores.csv")
    native = native.merge(native_scores[["seed", "post_rt_pinball_usd_mwh", "post_rt_coverage90_pct"]], on="seed", validate="one_to_one")
    native["reference_cost_usd_rt_basis"] = native.reference_cost_usd - native.common_da_component_usd
    native["saving_rate_pct_rt_basis"] = 100 * native.saving_usd / native.reference_cost_usd_rt_basis
    native["certificate_ceiling_usd"] = native.saving_usd / native.capture_lower_fraction

    features = pd.read_csv(DATA / "isone_r2_features_2025.csv", dtype={"zone_code": "string", "hour_ending_label": "string"})
    features["timestamp_utc"] = pd.to_datetime(features.timestamp_utc, utc=True)
    forecasts = {}
    for portfolio, zones in PORTFOLIOS.items():
        forecasts[(portfolio, "neutral_mean")] = score_forecast(
            DATA / "FORECAST_CACHE_ALL_neutral_mean.npz", zones, features, portfolio, "neutral_mean",
        )
    for embedding in ("P1", "P2", "P3", "P4"):
        forecasts[("ISO-NE-SOUTH", embedding)] = score_forecast(
            DATA / f"FORECAST_CACHE_ISO-NE-SOUTH_{embedding}.npz",
            PORTFOLIOS["ISO-NE-SOUTH"], features, "ISO-NE-SOUTH", embedding,
        )

    certificates = {}
    for portfolio in PORTFOLIOS:
        cert_path = OUT / "certificates" / f"{portfolio.replace('-', '_')}_w24_certificate.json"
        certificates[portfolio] = load_json(cert_path)["certified_saving_ceiling_usd"]

    forcing_path = WORKSPACE / "external_validation" / "data" / "forcing_2025.npz"
    with np.load(forcing_path, allow_pickle=False) as forcing:
        workload = {
            "times": pd.DatetimeIndex(pd.to_datetime(forcing["times"], utc=True)),
            "pv": forcing["pv"].astype(np.float64),
            "pue": forcing["pue"].astype(np.float64),
            "arrivals": forcing["eagle_arrivals"].astype(np.float64),
            "idle": forcing["idle"].astype(np.float64),
            "k": forcing["k"].astype(np.float64),
        }
    constraints = load_json(WORKSPACE / "R2_PAPER_INTEGRATION_20260923" / "configs" / "economic_constraints.json")["constraints"]
    normalization = load_json(TASK / "configs" / "frozen_nyiso_r2_normalization.json")
    neutral_rt = float(np.asarray(normalization["target_center"], dtype=float)[:, 2].mean())
    liability_rate = float(np.max(constraints["pue_max"]) * neutral_rt)
    workload["liability_rate_usd_per_mwh"] = liability_rate
    market15 = pd.read_csv(DATA / "isone_market_15min.csv", dtype={"zone_code": "string"})
    market15["timestamp_utc"] = pd.to_datetime(market15.timestamp_utc, utc=True)

    output_rows = []
    daily_tables = []
    native_rows = []
    for _, row in native.iterrows():
        seed = int(row.seed)
        native_rows.append({
            "market": "NYISO", "portfolio": "NYISO-Eagle native", "zone_set": "WEST;CAPITL;N.Y.C.;LONGIL",
            "seed": seed, "embedding_rule": "native", "policy": "Frozen WorldMoE native",
            "reference_cost_usd": row.reference_cost_usd_rt_basis,
            "policy_cost_usd": row.reference_cost_usd_rt_basis - row.saving_usd,
            "saving_usd": row.saving_usd, "saving_pct": row.saving_rate_pct_rt_basis,
            "certificate_usd": row.certificate_ceiling_usd,
            "capture_pct": 100 * row.capture_lower_fraction,
            "shifted_it_mwh": row.shifted_IT_mwh, "late_it_mwh": row.late_IT_mwh,
            "closing_it_mwh": row.closing_it_mwh, "cvar95_usd": row.cvar95_usd,
            "maximum_recovery_import_mw": row.max_fleet_recovery_mw,
            "forecast_pinball_rt": row.post_rt_pinball_usd_mwh,
            "coverage90_pct": row.post_rt_coverage90_pct,
            "feasible": bool(row.late_IT_mwh <= 1e-8 and row.closing_it_mwh <= 1e-8),
            "saving_rate_retention_pct": 100.0,
            "opportunity_capture_retention_pct": 100.0,
            "shifted_work_retention_pct": 100.0,
            "forecast_degradation_ratio": 1.0,
            "world_context": "NYISO-native frozen context",
        })

    for result_path in sorted(run_dir.glob("*.json")):
        result = load_json(result_path)
        portfolio = result["portfolio"]
        embedding = result["embedding_rule"]
        forecast_metric = forecasts[(portfolio, embedding)]
        ref_cost = float(result["reference_cost_usd"])
        saving = float(result["saving_usd"])
        capture = 100 * saving / certificates[portfolio]
        native_seed = native[native.seed.eq(int(result["seed"]))].iloc[0]
        native_rate = float(native_seed.saving_rate_pct_rt_basis)
        native_capture = 100 * float(native_seed.capture_lower_fraction)
        native_shift = float(native_seed.shifted_IT_mwh)
        native_pinball = float(native_seed.post_rt_pinball_usd_mwh)
        row = {
            "market": "ISO-NE", "portfolio": portfolio,
            "zone_set": ";".join(PORTFOLIOS[portfolio]), "seed": int(result["seed"]),
            "embedding_rule": embedding, "policy": "Frozen WorldMoE zero-shot",
            "reference_cost_usd": ref_cost, "policy_cost_usd": float(result["adjusted_cost_usd"]),
            "saving_usd": saving, "saving_pct": 100 * saving / ref_cost,
            "certificate_usd": certificates[portfolio], "capture_pct": capture,
            "shifted_it_mwh": float(result["shifted_IT_mwh"]),
            "late_it_mwh": float(result["late_IT_mwh"]),
            "closing_it_mwh": float(result["closing_it_mwh"]),
            "cvar95_usd": float(result["cvar95_usd"]),
            "maximum_recovery_import_mw": float(result.get("max_fleet_recovery_mw", np.nan)),
            "forecast_pinball_rt": forecast_metric["forecast_pinball_rt"],
            "coverage90_pct": forecast_metric["coverage90_pct"],
            "feasible": bool(result["late_IT_mwh"] <= 1e-8 and result["closing_it_mwh"] <= 1e-8),
            "saving_rate_retention_pct": 100 * (100 * saving / ref_cost) / native_rate,
            "opportunity_capture_retention_pct": 100 * capture / native_capture,
            "shifted_work_retention_pct": 100 * float(result["shifted_IT_mwh"]) / native_shift,
            "forecast_degradation_ratio": forecast_metric["forecast_pinball_rt"] / native_pinball,
            "world_context": "frozen zero-world-context fallback; mean NYISO MarketQuery zone embedding",
            "result_name": result["name"],
        }
        output_rows.append(row)
        daily_tables.append(daily_execution(
            result, result_path,
            trajectory_dir / f"{result['name']}_daily.csv",
            trajectory_dir / f"{result['name']}_complete_book.npz",
            workload, market15, portfolio, embedding,
        ))

    # Keep sensitivity outputs separate from the primary market table.
    sensitivity_rows = [row for row in output_rows if row["market"] == "ISO-NE" and row["embedding_rule"] in {"P1", "P2", "P3", "P4"}]
    primary_rows = native_rows + [row for row in output_rows if row["market"] == "ISO-NE" and row["embedding_rule"] == "neutral_mean"]
    baseline_file = OUT / "baselines" / "zero_shot_isone_direct_baselines.csv"
    baseline_df = pd.read_csv(baseline_file) if baseline_file.exists() else pd.DataFrame()
    if not baseline_df.empty:
        for _, result in baseline_df.iterrows():
            portfolio = str(result["portfolio"])
            ref_cost = float(result["reference_cost_usd"])
            saving = float(result["saving_usd"])
            row = {
                "market": "ISO-NE", "portfolio": portfolio,
                "zone_set": ";".join(PORTFOLIOS[portfolio]), "seed": 0,
                "embedding_rule": "neutral_mean", "policy": result["policy"],
                "reference_cost_usd": ref_cost, "policy_cost_usd": float(result["adjusted_cost_usd"]),
                "saving_usd": saving, "saving_pct": 100 * saving / ref_cost,
                "certificate_usd": certificates[portfolio], "capture_pct": 100 * saving / certificates[portfolio],
                "shifted_it_mwh": float(result["shifted_IT_mwh"]), "late_it_mwh": float(result["late_IT_mwh"]),
                "closing_it_mwh": float(result["closing_it_mwh"]), "cvar95_usd": float(result["cvar95_usd"]),
                "maximum_recovery_import_mw": float(result.get("max_fleet_recovery_mw", np.nan)),
                "forecast_pinball_rt": np.nan, "coverage90_pct": np.nan,
                "feasible": bool(result["late_IT_mwh"] <= 1e-8 and result["closing_it_mwh"] <= 1e-8),
                "saving_rate_retention_pct": np.nan, "opportunity_capture_retention_pct": np.nan,
                "shifted_work_retention_pct": np.nan, "forecast_degradation_ratio": np.nan,
                "world_context": "non-trained direct market baseline",
                "result_name": result["name"],
            }
            primary_rows.append(row)
            daily_tables.append(daily_execution(
                result, baseline_dir / f"{result['name']}.json",
                baseline_trajectory / f"{result['name']}_daily.csv",
                baseline_trajectory / f"{result['name']}_complete_book.npz",
                workload, market15, portfolio, "neutral_mean",
            ))

    # Earliest service is the matched 24 h reference baseline for both portfolios.
    for portfolio in PORTFOLIOS:
        row = next(r for r in output_rows if r["portfolio"] == portfolio and r["embedding_rule"] == "neutral_mean")
        primary_rows.append({
            "market": "ISO-NE", "portfolio": portfolio, "zone_set": ";".join(PORTFOLIOS[portfolio]),
            "seed": 0, "embedding_rule": "neutral_mean", "policy": "earliest_service",
            "reference_cost_usd": row["reference_cost_usd"], "policy_cost_usd": row["reference_cost_usd"],
            "saving_usd": 0.0, "saving_pct": 0.0, "certificate_usd": certificates[portfolio],
            "capture_pct": 0.0, "shifted_it_mwh": 0.0, "late_it_mwh": 0.0,
            "closing_it_mwh": 0.0, "cvar95_usd": 0.0,
            "maximum_recovery_import_mw": 0.0, "forecast_pinball_rt": np.nan,
            "coverage90_pct": np.nan, "feasible": True,
            "saving_rate_retention_pct": np.nan, "opportunity_capture_retention_pct": np.nan,
            "shifted_work_retention_pct": np.nan, "forecast_degradation_ratio": np.nan,
            "world_context": "matched causal earliest-service reference",
        })

    result_frame = pd.DataFrame(primary_rows + sensitivity_rows)
    results_path = OUT / "zero_shot_isone_seed_results.csv"
    result_frame.to_csv(results_path, index=False)
    daily = pd.concat(daily_tables, ignore_index=True) if daily_tables else pd.DataFrame()
    daily_path = OUT / "zero_shot_isone_daily.csv"
    daily.to_csv(daily_path, index=False)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_summary = []
    if not sensitivity.empty:
        for metric in ("saving_pct", "capture_pct", "forecast_pinball_rt", "coverage90_pct", "shifted_it_mwh", "late_it_mwh", "closing_it_mwh"):
            sensitivity_summary.append({"metric": metric, "mean": sensitivity[metric].mean(), "median": sensitivity[metric].median(), "min": sensitivity[metric].min(), "max": sensitivity[metric].max(), "permutations": "P1;P2;P3;P4", "selection_forbidden": True})
    pd.DataFrame(sensitivity_summary).to_csv(OUT / "embedding_sensitivity" / "embedding_sensitivity_summary.csv", index=False)
    OUT.joinpath("embedding_sensitivity").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for market_label, frame in (
        ("NYISO-Eagle native", pd.DataFrame(native_rows)),
        ("ISO-NE-SOUTH zero-shot", result_frame[(result_frame.market == "ISO-NE") & (result_frame.portfolio == "ISO-NE-SOUTH") & (result_frame.embedding_rule == "neutral_mean") & (result_frame.seed != 0)]),
        ("ISO-NE-NORTH zero-shot", result_frame[(result_frame.market == "ISO-NE") & (result_frame.portfolio == "ISO-NE-NORTH") & (result_frame.embedding_rule == "neutral_mean") & (result_frame.seed != 0)]),
    ):
        summary_rows.append({
            "market_portfolio": market_label,
            "Saving %": frame.saving_pct.mean(),
            "Opportunity capture %": frame.capture_pct.mean(),
            "Pinball loss": frame.forecast_pinball_rt.mean(),
            "90pct coverage %": frame.coverage90_pct.mean(),
            "Shifted IT MWh": frame.shifted_it_mwh.mean(),
            "Late IT MWh": frame.late_it_mwh.mean(),
            "Closing IT MWh": frame.closing_it_mwh.mean(),
            "CVaR95 USD": frame.cvar95_usd.mean(),
            "Saving-rate retention %": frame.saving_rate_retention_pct.mean(),
            "Opportunity-capture retention %": frame.opportunity_capture_retention_pct.mean(),
            "Saving-rate retention SD pp": frame.saving_pct.std(ddof=1) if len(frame) > 1 else np.nan,
        })
    table = pd.DataFrame(summary_rows)
    (TASK / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(TASK / "tables" / "TABLE_ZERO_SHOT_EXTERNAL_MARKET.csv", index=False)

    native_daily = pd.read_csv(WORKSPACE / "ExecutableServiceCommitments_Revision_20260928" / "source_data" / "worldmoe_daily_by_seed.csv")
    if "date" in native_daily.columns:
        native_long = native_daily.melt(id_vars=["date"], var_name="seed", value_name="service_adjusted_saving_usd")
        native_long["market"] = "NYISO"
        native_long["portfolio"] = "NYISO-Eagle native"
        native_long["embedding_rule"] = "native"
        native_long["policy"] = "Frozen WorldMoE native"
        native_long["cash_saving_usd"] = native_long.service_adjusted_saving_usd
        native_long["outstanding_work_mwh"] = np.nan
        native_long["outstanding_liability_usd"] = np.nan
        native_long["late_it_mwh"] = np.nan
        native_long["closing_work_mwh"] = np.nan
        native_long["price_spread_mean_usd_mwh"] = np.nan
        native_long["price_spread_abs_p95_usd_mwh"] = np.nan
        daily = pd.concat([daily, native_long], ignore_index=True, sort=False)
        daily.to_csv(daily_path, index=False)

    make_figures(table, daily, result_frame)
    print({"seed_rows": len(result_frame), "daily_rows": len(daily), "table": str(TASK / "tables" / "TABLE_ZERO_SHOT_EXTERNAL_MARKET.csv")})


def make_figures(table: pd.DataFrame, daily: pd.DataFrame, results: pd.DataFrame) -> None:
    figdir = TASK / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    labels = table.market_portfolio.tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x, table["Saving %"], color=["#55738a", "#4c9a8a", "#8a72ad"])
    for i, label in enumerate(labels):
        rows = results[(results.market == "NYISO") & (results.policy == "Frozen WorldMoE native")] if i == 0 else results[(results.market == "ISO-NE") & (results.portfolio == ("ISO-NE-SOUTH" if i == 1 else "ISO-NE-NORTH")) & (results.embedding_rule == "neutral_mean") & (results.seed != 0)]
        if not rows.empty: ax.scatter(np.full(len(rows), i), rows.saving_pct, color="black", s=18, zorder=3)
    ax.set_ylabel("Saving rate (%)");ax.set_xticks(x, labels, rotation=12, ha="right");ax.set_title("a  Saving rate")

    ax = axes[0, 1]
    ax.bar(x, table["Opportunity capture %"], color=["#55738a", "#4c9a8a", "#8a72ad"])
    for i, label in enumerate(labels):
        rows = results[(results.market == "NYISO") & (results.policy == "Frozen WorldMoE native")] if i == 0 else results[(results.market == "ISO-NE") & (results.portfolio == ("ISO-NE-SOUTH" if i == 1 else "ISO-NE-NORTH")) & (results.embedding_rule == "neutral_mean") & (results.seed != 0)]
        if not rows.empty: ax.scatter(np.full(len(rows), i), rows.capture_pct, color="black", s=18, zorder=3)
    ax.set_ylabel("Saving / certified ceiling (%)");ax.set_xticks(x, labels, rotation=12, ha="right");ax.set_title("b  Opportunity capture")

    ax = axes[1, 0]
    nyiso_pinball = float(table.loc[table.market_portfolio.eq("NYISO-Eagle native"), "Pinball loss"].iloc[0])
    pinball_ratio = table["Pinball loss"] / nyiso_pinball
    ax.bar(x, pinball_ratio, color=["#55738a", "#4c9a8a", "#8a72ad"])
    ax.axhline(1.0, color="#555555", alpha=.4, linewidth=1)
    ax.set_ylabel("RT pinball ratio vs NYISO");ax.set_xticks(x, labels, rotation=12, ha="right");ax.set_title("c  Forecast transfer")
    ax2 = ax.twinx();ax2.plot(x, table["90pct coverage %"], color="#b14a43", marker="o", linestyle="--");ax2.axhline(90, color="#b14a43", alpha=.35, linewidth=1);ax2.set_ylabel("90% interval coverage (%)")

    ax = axes[1, 1]
    ax.bar(x - .16, table["Late IT MWh"], width=.32, color="#c37465", label="Late")
    ax.bar(x + .16, table["Closing IT MWh"], width=.32, color="#d6a54a", label="Closing")
    ax.set_ylabel("IT work (MWh)");ax.set_xticks(x, labels, rotation=12, ha="right");ax.set_title("d  Service feasibility");ax.legend(frameon=False)
    fig.savefig(figdir / "FIG_ZERO_SHOT_EXTERNAL_TRANSFER.png", dpi=220)
    fig.savefig(figdir / "FIG_ZERO_SHOT_EXTERNAL_TRANSFER.pdf")
    plt.close(fig)

    native = daily[daily.market.eq("NYISO")].copy()
    iso = daily[(daily.market.eq("ISO-NE")) & (daily.embedding_rule.eq("neutral_mean")) & daily.policy.eq("Frozen WorldMoE zero-shot")].copy()
    if not native.empty and not iso.empty:
        native["date"] = pd.to_datetime(native["date"])
        iso["date"] = pd.to_datetime(iso["date"])
        native_curve = native.groupby("date").service_adjusted_saving_usd.mean().sort_index().cumsum()
        iso_curves = iso.groupby(["portfolio", "date"]).service_adjusted_saving_usd.mean().unstack(0).sort_index().cumsum()
        fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
        ax.plot(native_curve.index, native_curve.values, label="NYISO-Eagle native", color="#55738a")
        for portfolio, color in (("ISO-NE-SOUTH", "#4c9a8a"), ("ISO-NE-NORTH", "#8a72ad")):
            if portfolio in iso_curves: ax.plot(iso_curves.index, iso_curves[portfolio], label=f"{portfolio} zero-shot", color=color)
        ax.set_ylabel("Cumulative service-adjusted saving (USD)");ax.set_xlabel("2025 calendar day");ax.legend(frameon=False)
        ax.set_title("Cumulative value over the full continuous year")
        fig.savefig(figdir / "FIG_ZERO_SHOT_CUMULATIVE_VALUE.png", dpi=220)
        fig.savefig(figdir / "FIG_ZERO_SHOT_CUMULATIVE_VALUE.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
