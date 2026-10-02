from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def install(ctx, task_root: Path, portfolio: str, embedding: str):
    """Replace only the market price adapter; keep Eagle and physics unchanged."""
    original_configure = ctx.configure
    portfolio_map = pd.read_csv(task_root / "configs" / "zone_portfolios.csv")
    zones = (
        portfolio_map[portfolio_map.portfolio.eq(portfolio)]
        .sort_values("site_order")["load_zone"].tolist()
    )
    forecast_path = task_root / "data_processed" / f"FORECAST_CACHE_{portfolio}_{embedding}.npz"
    if not forecast_path.exists() and embedding == "neutral_mean":
        forecast_path = task_root / "data_processed" / "FORECAST_CACHE_ALL_neutral_mean.npz"
    market_path = task_root / "data_processed" / "isone_market_15min.csv"

    with np.load(forecast_path, allow_pickle=False) as cached:
        forecast_zones = cached["zone_codes"].astype(str).tolist()
        historical = cached["historical_rt"].astype(np.float64)
    selected = [forecast_zones.index(zone) for zone in zones]
    historical = historical[selected]

    market = pd.read_csv(market_path, dtype={"zone_code": "string", "hour_ending_label": "string"})
    market["timestamp_utc"] = pd.to_datetime(market["timestamp_utc"], utc=True, errors="raise")
    da = market.pivot(index="timestamp_utc", columns="zone_code", values="da_lmp_usd_mwh")
    rt = market.pivot(index="timestamp_utc", columns="zone_code", values="rt_lmp_final_usd_mwh")

    def configure(year: int, trace: str = "Eagle"):
        if year != 2025 or trace != "Eagle":
            raise ValueError("ISO-NE transfer adapter is fixed to the 2025 Eagle primary experiment.")
        arrivals, _, _ = original_configure(year, trace)
        c = ctx.c
        if not da.index.equals(rt.index):
            raise ValueError("ISO-NE DA and final RT timestamps differ.")
        prices_da = da.reindex(index=c.times, columns=zones)
        prices_rt = rt.reindex(index=c.times, columns=zones)
        if prices_da.isna().any().any() or prices_rt.isna().any().any():
            raise ValueError("ISO-NE official hourly prices do not align to the frozen Eagle executor calendar.")
        c.da = prices_da.to_numpy(dtype=np.float64)
        c.rt = prices_rt.to_numpy(dtype=np.float64)
        c.train_rt = historical.copy()
        c.day_index, c.days = pd.factorize(c.times.tz_convert("America/New_York").date, sort=True)
        shared_da = np.zeros(c.N, dtype=np.float64)
        reference = ctx.engine.reference(arrivals, 1.0)
        return arrivals, reference, shared_da

    ctx.configure = configure
    return zones
