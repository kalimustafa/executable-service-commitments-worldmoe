from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
WORKSPACE = TASK.parent
DATA = TASK / "data_processed"
OUT = TASK / "outputs"
REPORT = TASK / "report"
TZ = "America/New_York"
PORTFOLIOS = ("ISO-NE-SOUTH", "ISO-NE-NORTH")


def summarize_price(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return {
        "rows": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
        "std": float(np.std(x)), "p05": float(np.quantile(x, .05)),
        "p95": float(np.quantile(x, .95)), "min": float(np.min(x)),
        "max": float(np.max(x)), "negative_pct": float(100 * np.mean(x < 0)),
    }


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    seed_results = pd.read_csv(OUT / "zero_shot_isone_seed_results.csv")
    table = pd.read_csv(TASK / "tables" / "TABLE_ZERO_SHOT_EXTERNAL_MARKET.csv")
    daily = pd.read_csv(OUT / "zero_shot_isone_daily.csv", dtype={"date": "string"})
    sens = pd.read_csv(OUT / "embedding_sensitivity" / "embedding_sensitivity_summary.csv")
    price = pd.read_csv(DATA / "isone_hourly_canonical.csv", dtype={"zone_code": "string", "hour_ending_label": "string"})
    price["interval_start_utc"] = pd.to_datetime(price.interval_start_utc, utc=True)
    price_2025 = price[price.market_date_local.str.startswith("2025")].copy()
    price_2025["spread"] = price_2025.da_lmp_usd_mwh - price_2025.rt_lmp_final_usd_mwh

    with np.load(WORKSPACE / "external_validation" / "data" / "forcing_2025.npz", allow_pickle=False) as forcing:
        ny_times = pd.DatetimeIndex(pd.to_datetime(forcing["times"], utc=True))
        ny_da = forcing["da"].astype(float)
        ny_rt = forcing["rt"].astype(float)
        local_dates = ny_times.tz_convert(TZ).strftime("%Y-%m-%d")
    ny_spread = ny_da - ny_rt
    iso_da = price_2025.da_lmp_usd_mwh.to_numpy(float)
    iso_rt = price_2025.rt_lmp_final_usd_mwh.to_numpy(float)
    iso_spread = price_2025.spread.to_numpy(float)
    price_stats = {
        "NYISO_2025_DA": summarize_price(ny_da),
        "ISO_NE_2025_DA": summarize_price(iso_da),
        "NYISO_2025_final_RT": summarize_price(ny_rt),
        "ISO_NE_2025_final_RT": summarize_price(iso_rt),
        "NYISO_2025_DA_minus_RT": summarize_price(ny_spread),
        "ISO_NE_2025_DA_minus_RT": summarize_price(iso_spread),
    }
    tail_metrics = []
    for label, ny_values, iso_values in (
        ("DA", ny_da, iso_da), ("Final RT", ny_rt, iso_rt), ("DA-RT spread", ny_spread, iso_spread),
    ):
        threshold = float(np.quantile(ny_values, .95))
        tail_metrics.append({
            "series": label, "nyiso_p95_threshold": threshold,
            "nyiso_above_its_p95_pct": float(100 * np.mean(ny_values > threshold)),
            "isone_above_nyiso_p95_pct": float(100 * np.mean(iso_values > threshold)),
            "isone_p95": float(np.quantile(iso_values, .95)),
        })
    pd.DataFrame(tail_metrics).to_csv(OUT / "market_tail_frequency_diagnostics.csv", index=False)

    normal = json.loads((TASK / "configs" / "frozen_nyiso_r2_normalization.json").read_text(encoding="utf-8"))
    features = pd.read_csv(DATA / "isone_r2_features_2025.csv", dtype={"zone_code": "string", "hour_ending_label": "string"})
    names = normal["feature_names"]
    raw = features[names].to_numpy(np.float64)
    center = np.asarray(normal["feature_center"], dtype=np.float64)
    scale = np.asarray(normal["feature_scale"], dtype=np.float64)
    standardized = (raw - center[None, :]) / scale[None, :]
    finite = np.isfinite(standardized)
    clip = float(normal["feature_clip"])
    clip_rate = float(np.mean(finite & (np.abs(standardized) > clip)))
    nan_rate = float(np.mean(~finite))
    price_features = ["da_lbmp_lag2", "rt_integrated_lag2", "spread_lag2"]
    price_ood = {}
    for feature in price_features:
        index = names.index(feature)
        z = standardized[:, index]
        price_ood[feature] = {
            "abs_z_gt_3_pct": float(100 * np.nanmean(np.abs(z) > 3)),
            "abs_z_gt_5_pct": float(100 * np.nanmean(np.abs(z) > 5)),
            "abs_z_gt_clip_pct": float(100 * np.nanmean(np.abs(z) > clip)),
            "abs_z_p95": float(np.nanquantile(np.abs(z), .95)),
        }
    feature_ood = {"feature_clip": clip, "all_feature_clip_rate_pct": 100 * clip_rate,
                   "nonfinite_frozen_feature_rate_pct": 100 * nan_rate, "price_features": price_ood}

    # Monthly value concentration and within-market extreme-day influence.
    primary_daily = daily[(daily.market == "ISO-NE") & (daily.embedding_rule == "neutral_mean") & (daily.policy == "Frozen WorldMoE zero-shot")].copy()
    primary_daily["date"] = pd.to_datetime(primary_daily.date)
    monthly = primary_daily.groupby(["portfolio", primary_daily.date.dt.to_period("M")]).service_adjusted_saving_usd.sum().reset_index()
    monthly["date"] = monthly.date.astype(str)
    monthly["share_of_annual_saving_pct"] = monthly.groupby("portfolio").service_adjusted_saving_usd.transform(lambda x: 100 * x / x.sum())
    monthly.to_csv(OUT / "monthly_saving_concentration.csv", index=False)
    spread_by_day = price_2025.groupby("market_date_local").spread.agg(lambda x: float(x.max() - x.min())).rename("spread_range").reset_index()
    threshold_iso = float(spread_by_day.spread_range.quantile(.95))
    spread_by_day["extreme_iso_ne_p95"] = spread_by_day.spread_range >= threshold_iso
    extreme_rows = []
    for portfolio, group in primary_daily.groupby("portfolio"):
        merged = group.merge(spread_by_day, left_on=group.date.dt.strftime("%Y-%m-%d"), right_on="market_date_local", how="left")
        extreme = merged[merged.extreme_iso_ne_p95.fillna(False)]
        ordinary = merged[~merged.extreme_iso_ne_p95.fillna(False)]
        extreme_rows.append({
            "portfolio": portfolio, "extreme_definition": "daily 8-zone DA-RT spread range at or above ISO-NE 2025 p95",
            "threshold_usd_mwh": threshold_iso, "extreme_date_count": int(extreme.date.nunique()),
            "extreme_saving_usd": float(extreme.service_adjusted_saving_usd.sum()),
            "ordinary_saving_usd": float(ordinary.service_adjusted_saving_usd.sum()),
            "extreme_share_of_saving_pct": float(100 * extreme.service_adjusted_saving_usd.sum() / merged.service_adjusted_saving_usd.sum()),
        })
    extreme_df = pd.DataFrame(extreme_rows)
    extreme_df.to_csv(OUT / "extreme_event_dependence.csv", index=False)
    monthly_summary = []
    for portfolio, block in monthly.groupby("portfolio"):
        ordered = block.sort_values("share_of_annual_saving_pct", ascending=False)
        monthly_summary.append({
            "portfolio": portfolio, "largest_month": ordered.iloc[0]["date"],
            "largest_month_share_pct": float(ordered.iloc[0]["share_of_annual_saving_pct"]),
            "top_three_month_share_pct": float(ordered.head(3)["share_of_annual_saving_pct"].sum()),
        })
    monthly_summary_df = pd.DataFrame(monthly_summary).merge(extreme_df[["portfolio", "extreme_date_count", "extreme_share_of_saving_pct"]], on="portfolio", how="left")

    results = seed_results
    neutral = results[(results.market == "ISO-NE") & (results.embedding_rule == "neutral_mean") & (results.seed != 0)]
    baselines = results[(results.market == "ISO-NE") & (results.seed == 0)]
    ny = results[(results.market == "NYISO") & (results.policy == "Frozen WorldMoE native")]
    base = table.iloc[0]
    south = table.iloc[1]
    north = table.iloc[2]

    # Build compact failure diagnostics from the measured forecasts, execution
    # ledger, market prices and frozen scaler. No diagnostic changes a policy.
    sensitivity_summary = sens.set_index("metric").to_dict(orient="index")
    diag_lines = [
        "# ZERO-SHOT FAILURE DIAGNOSTICS", "",
        "## Market and feature distribution shift", "",
        "Official ISO-NE prices are hourly values copied to four executor quarter-hours. Distribution statistics below use the retained 2025 rows; no rows were trimmed.", "",
        "| Series | NYISO mean | ISO-NE mean | NYISO p95 | ISO-NE p95 | ISO-NE negative (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [("DA", "Day-Ahead LMP"), ("final_RT", "Final Real-Time LMP"), ("DA_minus_RT", "DA–RT spread")]:
        nys = price_stats[f"NYISO_2025_{key}"]
        iso = price_stats[f"ISO_NE_2025_{key}"]
        diag_lines.append(f"| {label} | {nys['mean']:.2f} | {iso['mean']:.2f} | {nys['p95']:.2f} | {iso['p95']:.2f} | {iso['negative_pct']:.2f} |")
    diag_lines += [
        "", f"The fraction of frozen NYISO-normalized input entries beyond the unchanged ±{clip:g} clip is {feature_ood['all_feature_clip_rate_pct']:.5f}%; {feature_ood['nonfinite_frozen_feature_rate_pct']:.5f}% of entries are nonfinite before the model's existing frozen-center fill. The nonfinite lag rows come from the missing same-local-hour market interval on the spring DST day; the frozen NYISO feature center is used by the existing input path.",
        "", "| Price series | NYISO p95 threshold | NYISO above own p95 (%) | ISO-NE above NYISO p95 (%) | ISO-NE p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in tail_metrics:
        diag_lines.append(f"| {row['series']} | {row['nyiso_p95_threshold']:.2f} | {row['nyiso_above_its_p95_pct']:.2f} | {row['isone_above_nyiso_p95_pct']:.2f} | {row['isone_p95']:.2f} |")
    diag_lines += [
        "", "| Frozen price feature | |z|>3 (%) | |z|>5 (%) | |z|>clip (%) | p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in price_ood.items():
        diag_lines.append(f"| {name} | {value['abs_z_gt_3_pct']:.2f} | {value['abs_z_gt_5_pct']:.2f} | {value['abs_z_gt_clip_pct']:.5f} | {value['abs_z_p95']:.2f} |")
    diag_lines += ["", "## Forecast and execution attribution", ""]
    for row in (south, north):
        diag_lines.append(f"- {row['market_portfolio']}: RT pinball {row['Pinball loss']:.3f} USD/MWh; 90% coverage {row['90pct coverage %']:.2f}%; saving-rate retention {row['Saving-rate retention %']:.2f}%; opportunity-capture retention {row['Opportunity-capture retention %']:.2f}%; late IT {row['Late IT MWh']:.3e} MWh; closing IT {row['Closing IT MWh']:.3f} MWh; CVaR95 ${row['CVaR95 USD']:,.2f}.")
    diag_lines += [
        "", "Forecast-domain failure: post-publication RT forecast error did not worsen against the native NYISO score in either portfolio; the central 90% coverage is near nominal. Feature-space diagnostics show whether individual lag features reached the frozen clip.",
        "Valuation failure: despite retained forecast scores and substantial saving-rate retention, ISO-NE saving rates and capture are lower than the native NYISO reference. Price scale and market spread changes alter the value of each shift.",
        "Execution failure: no material late IT was recorded, but both zero-shot portfolios ended 2025 with about 2.1 MWh of closing IT. The annual 24 h service contract is therefore not fully completed; the saved policy was not repaired.",
        "Settlement difference: the transferred cash account uses the existing RT grid-import and PV-curtailment settlement; the closing-work liability keeps the original frozen NYISO RT-median fallback. The NYISO-only common DA schedule adjustment was set to zero because the supplied ISO-NE files do not include a day-ahead nomination schedule. DA LMPs remain in the forecast and publication-gate path.",
        "", "## Monthly value and extreme-event dependence", "",
        f"Using the ISO-NE daily eight-zone DA–RT spread range, the p95 threshold is {threshold_iso:.2f} USD/MWh. Monthly concentration and extreme-day contributions are summarized below.",
        "", monthly_summary_df.to_markdown(index=False, floatfmt=".2f"),
        "", "## Embedding sensitivity", "",
        f"For the four predeclared cyclic assignments on ISO-NE-SOUTH / primary seed, saving-rate median is {sensitivity_summary['saving_pct']['median']:.3f}% (range {sensitivity_summary['saving_pct']['min']:.3f}–{sensitivity_summary['saving_pct']['max']:.3f}%). No permutation was selected as primary.",
        "", "The frozen WorldMoE/R2 inference graph uses its existing explicit missing-world-context path: zone-level ISO-NE inputs cannot be mapped to the checkpoint's NYISO node-level W0/W1 packet without inventing node data. The four learned NYISO MarketQuery zone embeddings are pooled by their arithmetic mean for the primary result. This limitation narrows the claim to the defined frozen missing-world fallback path.",
    ]
    (REPORT / "ZERO_SHOT_FAILURE_DIAGNOSTICS.md").write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    seed_lines = []
    for row in table.to_dict(orient="records"):
        seed_lines.append(row)
    table_md = table.to_markdown(index=False, floatfmt=".3f")
    direct_md = baselines[["portfolio", "policy", "saving_usd", "saving_pct", "late_it_mwh", "closing_it_mwh", "cvar95_usd"]].to_markdown(index=False, floatfmt=".3f") if not baselines.empty else "No direct baseline rows."
    mean_pinball = float(ny.forecast_pinball_rt.mean())
    neutral_model_rows = seed_results[(seed_results.market == "ISO-NE") & (seed_results.embedding_rule == "neutral_mean") & (seed_results.seed != 0)]
    mean_costs = neutral_model_rows.groupby("portfolio")[["reference_cost_usd", "policy_cost_usd", "saving_usd"]].mean().to_dict(orient="index")
    certificates = {
        portfolio: json.loads((OUT / "certificates" / f"{portfolio.replace('-', '_')}_w24_certificate.json").read_text(encoding="utf-8"))["certified_saving_ceiling_usd"]
        for portfolio in PORTFOLIOS
    }
    forecast_ratio_south = float(south["Pinball loss"] / mean_pinball)
    forecast_ratio_north = float(north["Pinball loss"] / mean_pinball)
    certificate_text = "\n".join(
        f"- {portfolio}: ${certificates[portfolio]:,.2f} certified saving ceiling."
        for portfolio in PORTFOLIOS
    )
    sensitivity_text = f"P1–P4 median saving rate {sensitivity_summary['saving_pct']['median']:.3f}% (min {sensitivity_summary['saving_pct']['min']:.3f}%, max {sensitivity_summary['saving_pct']['max']:.3f}%); opportunity capture median {sensitivity_summary['capture_pct']['median']:.3f}% (min {sensitivity_summary['capture_pct']['min']:.3f}%, max {sensitivity_summary['capture_pct']['max']:.3f}%)."
    report_lines = [
        "# ZERO-SHOT EXTERNAL TRANSFER REPORT", "",
        "## 1. Question", "",
        "Can a NYISO-trained frozen WorldMoE retain executable electricity-market value when transferred without adaptation to an unseen ISO-NE market?", "",
        "## 2. Zero-shot definition", "",
        "No ISO-NE observation was used for training, gradient updates, parameter selection, recalibration, or normalization fitting. The six primary annual replays used the three existing frozen CdfR2 checkpoints for both fixed portfolios. The primary unseen-zone mapping is the arithmetic mean of the four frozen NYISO zone-query embeddings; the inference graph's explicit missing-world-context mask is used because ISO-NE load-zone prices do not provide the NYISO node-level W0/W1 packet. The four predeclared cyclic embedding assignments are sensitivity only.", "",
        "## 3. Frozen model", "",
        "W0/W1 are the frozen NYISO lineage listed in `report/FROZEN_MODEL_MANIFEST.md`. The shared frozen StructuredR2 base produces the publication-aware market quantiles. Final decision checkpoints were seed 20260920/epoch 06, 20260921/epoch 02, and 20260922/epoch 00. They were not reselected. The final CdfR2 inference call used `eval()`, `requires_grad=False`, and `torch.inference_mode()`; no optimizer or backward call was used. Per user instruction, no post-run hash values were rechecked.", "",
        "## 4. External market", "",
        "ISO-NE 2025 official hourly Day-Ahead LMPs and Final Real-Time LMPs cover all eight load zones. The single official hourly value is copied to H:00, H:15, H:30, and H:45 in the execution frame; there is no interpolation or added noise. Prices remain on America/New_York market dates with DST occurrence preserved. The 2024-12 files are used only for initial lag history.", "",
        "## 5. External portfolios", "",
        "- ISO-NE-SOUTH: CONNECTICUT, NEMASSBOST, SEMASS, WCMASS.\n- ISO-NE-NORTH: MAINE, NEWHAMPSHIRE, VERMONT, RHODEISLAND.", "",
        "The Eagle workload, annual IT energy, 24 h service contract, capacity definition, deadline logic, idle-IT assumptions, physical priors, executor, and closing-work account follow the frozen NYISO case. The market price arrays and DA availability gate are ISO-NE specific.", "",
        "## 6. Forecast transfer", "",
        f"NYISO post-publication RT pinball mean is {mean_pinball:.3f} USD/MWh. ISO-NE-SOUTH is {south['Pinball loss']:.3f} (ratio {forecast_ratio_south:.3f}); ISO-NE-NORTH is {north['Pinball loss']:.3f} (ratio {forecast_ratio_north:.3f}). 90% coverage is {south['90pct coverage %']:.2f}% and {north['90pct coverage %']:.2f}%, respectively. Official DA notice timestamps control the post-publication branch; a fixed noon gate was not used.", "",
        "## 7. Execution transfer", "",
        f"Both portfolios have late IT below 1e-8 MWh across all three seeds. Mean closing IT is {south['Closing IT MWh']:.3f} MWh (SOUTH) and {north['Closing IT MWh']:.3f} MWh (NORTH), so the 24 h contract is not fully complete at year end. The recorded policy is unchanged; no repair was applied. Mean shifted IT is {south['Shifted IT MWh']:.1f} and {north['Shifted IT MWh']:.1f} MWh.", "",
        "## 8. Economic transfer", "",
        table_md, "",
        f"Mean annual service-adjusted saving is {money(mean_costs['ISO-NE-SOUTH']['saving_usd'])} for SOUTH and {money(mean_costs['ISO-NE-NORTH']['saving_usd'])} for NORTH. Mean RT-basis reference/policy costs are {money(mean_costs['ISO-NE-SOUTH']['reference_cost_usd'])}/{money(mean_costs['ISO-NE-SOUTH']['policy_cost_usd'])} SOUTH and {money(mean_costs['ISO-NE-NORTH']['reference_cost_usd'])}/{money(mean_costs['ISO-NE-NORTH']['policy_cost_usd'])} NORTH.", "",
        certificate_text, "",
        "Opportunity capture is policy saving divided by the condition-matched certified saving ceiling. Cash costs use the existing RT grid-import and PV-curtailment settlement; the closing-work liability keeps the original frozen NYISO RT-median fallback. No ISO-NE DA nomination schedule was supplied, so the NYISO-only shared DA schedule adjustment is not carried into ISO-NE costs.", "",
        "Direct non-trained controls:", "",
        direct_md, "",
        "## 9. Transfer retention", "",
        f"Saving-rate retention versus the native NYISO RT-normalized reference is {south['Saving-rate retention %']:.2f}% for SOUTH and {north['Saving-rate retention %']:.2f}% for NORTH. Opportunity-capture retention is {south['Opportunity-capture retention %']:.2f}% and {north['Opportunity-capture retention %']:.2f}%. These rates are reported on a matched RT-settlement basis; the NYISO common DA schedule component was removed from its reference denominator because ISO-NE nomination quantities were not provided. Pinball ratios are {forecast_ratio_south:.3f} and {forecast_ratio_north:.3f}.", "",
        "## 10. Embedding sensitivity", "",
        sensitivity_text, "",
        "The sensitivity summary reports the median, minimum, and maximum over P1–P4. No best assignment was selected. Because the frozen missing-world mask suppresses the W0/W1 side context, forecast scores are invariant to these four query-embedding assignments in this run.", "",
        "## 11. Failure analysis", "",
        "The forecast diagnostics show no pinball degradation against the NYISO native score, while saving-rate retention is about 83%. The dominant execution shortfall is the retained year-end closing queue, about 2.1 MWh per portfolio. The world-context branch was unavailable for ISO-NE's zone-level market data and remained on the model's preexisting zero/unavailable fallback. Market-value differences therefore cannot be attributed solely to predictive error.", "",
        "See `report/ZERO_SHOT_FAILURE_DIAGNOSTICS.md` for price distribution, frozen-feature OOD rates, tail frequency, monthly concentration, and extreme-event dependence.", "",
        "## 12. Interpretation", "",
        "The results support **partial zero-shot transportability** for the frozen R2/CdfR2 path under its documented missing-world-context fallback. They do not establish full W0/W1 node-context transfer to ISO-NE. Forecast quality transferred, and savings remained positive, but the nonzero closing IT means the 24 h executable service contract was not fully met.", "",
        "## Acceptance answers", "",
        "- Q1 new parameter training: NO.\n- Q2 ISO-NE scaler fit: NO.\n- Q3 ISO-NE checkpoint selection: NO.\n- Q4 zone selection from 2025 performance: NO.\n- Q5 full-year continuous replay: YES, 2025 local calendar, 35,040 quarter-hour intervals, continuous queue.\n- Q6 ISO-NE-SOUTH mean saving rate: " + pct(float(south["Saving %"])) + ".\n- Q7 ISO-NE-NORTH mean saving rate: " + pct(float(north["Saving %"])) + ".\n- Q8 saving-rate retention: " + pct(float(south["Saving-rate retention %"])) + " SOUTH; " + pct(float(north["Saving-rate retention %"])) + " NORTH.\n- Q9 opportunity-capture retention: " + pct(float(south["Opportunity-capture retention %"])) + " SOUTH; " + pct(float(north["Opportunity-capture retention %"])) + " NORTH.\n- Q10 service: late IT is effectively zero; closing IT remains about 2.1 MWh in both portfolios, so year-end completion fails.\n- Q11 forecast pinball ratio: " + f"{forecast_ratio_south:.3f} SOUTH; {forecast_ratio_north:.3f} NORTH." + "\n- Q12 OOD: market prices are outside the NYISO domain; frozen-feature clipping/OOD rates are reported in the diagnostics. W0/W1 context is masked unavailable under the existing fallback.", "",
        "The manuscript was not changed. See `report/MANUSCRIPT_IMPLICATIONS.md`.",
    ]
    (REPORT / "ZERO_SHOT_EXTERNAL_TRANSFER_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    mean_results = results[(results.market == "ISO-NE") & (results.embedding_rule == "neutral_mean") & (results.seed != 0)]
    implications = [
        "# MANUSCRIPT IMPLICATIONS", "",
        "## Claims supported by this experiment", "",
        "- A frozen NYISO-trained R2/CdfR2 market-decision path produced positive 2025 savings in both predeclared ISO-NE four-zone portfolios without ISO-NE training, recalibration, or scaler fitting.",
        "- The result can be described as a zero-shot external-market evaluation **under the existing missing-world-context fallback**, with official ISO-NE DA publication times gating information use.",
        f"- The observed saving-rate retention is {south['Saving-rate retention %']:.2f}% SOUTH and {north['Saving-rate retention %']:.2f}% NORTH on the matched RT cash basis; the experiment supports the phrase **partial zero-shot transportability**.",
        "", "## Claims not supported", "",
        "- Do not claim full W0/W1 node-context transfer: the ISO-NE source is load-zone LMP data and the inference path set the unsupported NYISO node-context input to its preexisting unavailable/zero state.",
        "- Do not claim complete execution of the 24 h service contract: closing IT remains nonzero at year end.",
        "- Do not claim ISO-NE recalibration, adaptation, or a best embedding assignment.",
        "- Do not use these outputs to compare absolute NYISO and ISO-NE dollar savings without accounting for different market size and settlement bases.",
        "", "## Cross-market wording", "",
        "A qualified claim of zero-shot cross-market transfer is supported only when the text identifies the masked W0/W1 context and the closing-work failure. A stronger end-to-end WorldMoE transportability claim requires a compatible official ISO-NE context interface; no node prices were synthesized for this experiment.",
        "", "No manuscript was edited.",
    ]
    (REPORT / "MANUSCRIPT_IMPLICATIONS.md").write_text("\n".join(implications) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
