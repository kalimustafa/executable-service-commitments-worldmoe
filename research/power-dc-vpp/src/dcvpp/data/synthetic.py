from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def generate_demo_domains(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    """Create deterministic, deliberately non-synchronized demo domains."""
    seed = int(config.get("seed", 20260809))
    rng = np.random.default_rng(seed)
    days = int(config.get("days", 14))
    sites = list(config.get("sites", ["DC1", "DC3"]))
    n_steps = days * 96
    out = root / "data/processed/demo"

    # Market clock is 2025. Physical and workload clocks are intentionally different.
    market_times = pd.date_range("2025-01-01", periods=n_steps, freq="15min", tz="UTC")
    rows: list[dict[str, Any]] = []
    for site_idx, site in enumerate(sites):
        slot = np.arange(n_steps)
        daily = np.sin(2 * np.pi * (slot % 96) / 96 - 1.1)
        weekly = np.sin(2 * np.pi * slot / (96 * 7))
        da = 32 + 11 * daily + 4 * weekly + 3 * site_idx + rng.normal(0, 1.5, n_steps)
        rt = da + rng.normal(0, 5.0, n_steps)
        rt[48 + site_idx] = -25 - 5 * site_idx
        temp = 2 + 8 * np.sin(2 * np.pi * (slot % 96) / 96 - 1.5) + 0.5 * site_idx
        irradiance = np.maximum(0, 750 * np.sin(np.pi * ((slot % 96) - 24) / 48))
        load = 17000 + 1500 * daily + 300 * site_idx + rng.normal(0, 120, n_steps)
        for j, ts in enumerate(market_times):
            rows.append(
                {
                    "timestamp_utc": ts,
                    "timestamp_local": ts.tz_convert("America/New_York"),
                    "zone": site,
                    "da_lbmp": float(da[j]),
                    "rt_lbmp": float(rt[j]),
                    "rt_spread": float(rt[j] - da[j]),
                    "system_load_mw": float(load[j]),
                    "temperature_c": float(temp[j]),
                    "irradiance_wm2": float(irradiance[j]),
                    "profile": "demo",
                    "synthetic": True,
                    "evidence_role": "counterfactual",
                    "source_domain": "synthetic_nyiso_like",
                }
            )
    market = pd.DataFrame(rows)

    physical_times = pd.date_range("2023-03-01", periods=n_steps, freq="15min", tz="UTC")
    slot = np.arange(n_steps)
    temp = 8 + 10 * np.sin(2 * np.pi * (slot % 96) / 96 - 1.4) + rng.normal(0, 0.5, n_steps)
    solar = np.maximum(0, 800 * np.sin(np.pi * ((slot % 96) - 24) / 48))
    utilization = np.clip(0.48 + 0.23 * np.sin(2 * np.pi * (slot % 96) / 96) + rng.normal(0, 0.08, n_steps), 0.05, 0.98)
    it_power = 900 + 2100 * utilization + rng.normal(0, 25, n_steps)
    pue = np.clip(1.16 + 0.009 * np.maximum(temp - 12, 0) + rng.normal(0, 0.012, n_steps), 1.10, 1.55)
    cooling = np.maximum(0, it_power * (pue - 1.0) - 55) + rng.normal(0, 8, n_steps)
    facility = it_power + cooling + 55 + rng.normal(0, 12, n_steps)
    physical = pd.DataFrame(
        {
            "timestamp_utc": physical_times,
            "it_power_kw": it_power,
            "cooling_kw": cooling,
            "facility_power_kw": facility,
            "pue": facility / it_power,
            "outside_air_temp_c": temp,
            "solar_wm2": solar,
            "profile": "demo",
            "synthetic": True,
            "evidence_role": "observed",
            "source_domain": "synthetic_nlr_like",
        }
    )

    workload_times = pd.date_range("2024-02-01", periods=n_steps, freq="15min", tz="UTC")
    rate = 6 + 4 * np.maximum(0, np.sin(2 * np.pi * (slot % 96) / 96 - 0.5))
    counts = rng.poisson(rate)
    work = np.maximum(0.1, counts * rng.lognormal(mean=0.1, sigma=0.35, size=n_steps))
    workload = pd.DataFrame(
        {
            "timestamp_utc": workload_times,
            "arrival_count": counts,
            "requested_work": work,
            "cohort": np.where(slot % 4 == 0, "C_preemptible_training", "D_elastic_batch"),
            "delta_t_minutes": 15.0,
            "profile": "demo",
            "synthetic": True,
            "evidence_role": "observed",
            "source_domain": "synthetic_nlr_workload_like",
        }
    )

    return {
        "market": _write(market, out / "nyiso_market_15m.parquet"),
        "physical": _write(physical, out / "nlr_physical_15m.parquet"),
        "workload": _write(workload, out / "nlr_workload_15m.parquet"),
    }

