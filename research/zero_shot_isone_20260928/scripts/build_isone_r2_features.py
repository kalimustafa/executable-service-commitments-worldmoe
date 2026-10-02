from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"
TZ = "America/New_York"
START = "2024-12-01"
END = "2025-12-31"
ZONE_DUMMIES = ["zone_CAPITL", "zone_LONGIL", "zone_N.Y.C.", "zone_WEST"]


def add_frozen_r2_features(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame["timestamp_utc"].dt.tz_convert(TZ)
    frame["date"] = local.dt.strftime("%Y-%m-%d")
    frame["hour"] = local.dt.hour
    frame["dow"] = local.dt.dayofweek
    frame["month"] = local.dt.month
    frame["doy"] = local.dt.dayofyear
    frame["utc_offset_hours"] = [x.utcoffset().total_seconds() / 3600 for x in local]
    frame["spread"] = frame["da_lbmp" ] - frame["rt_integrated"]

    # Repeated wall-clock hours are averaged only when used as historical
    # same-hour lag inputs; the raw fall-back intervals stay separate.
    hourly = frame.groupby(["zone_code", "date", "hour"], sort=False)[
        ["da_lbmp", "rt_integrated", "spread"]
    ].mean().reset_index()
    hourly["date"] = pd.to_datetime(hourly["date"])
    calendar = pd.date_range(START, END, freq="D")

    parts = []
    for (zone, hour), block in hourly.groupby(["zone_code", "hour"], sort=False):
        b = block.set_index("date").sort_index().reindex(calendar)
        f = pd.DataFrame(index=b.index)
        for field in ("da_lbmp", "rt_integrated", "spread"):
            for lag in (2, 3, 7, 14, 28):
                f[f"{field}_lag{lag}"] = b[field].shift(lag)
            for window in (7, 28):
                f[f"{field}_mean{window}"] = b[field].shift(2).rolling(
                    window, min_periods=window // 2,
                ).mean()
        f["spread_hour_std28"] = b["spread"].shift(2).rolling(28, min_periods=14).std()
        f["zone_code"] = zone
        f["hour"] = hour
        f["date"] = f.index.strftime("%Y-%m-%d")
        parts.append(f.reset_index(drop=True))
    frame = frame.merge(pd.concat(parts, ignore_index=True), on=["zone_code", "date", "hour"], validate="many_to_one")

    daily = frame.groupby(["zone_code", "date"], sort=False).agg(
        day_da=("da_lbmp", "mean"),
        day_rt=("rt_integrated", "mean"),
        day_spread=("spread", "mean"),
        day_std=("spread", "std"),
        day_abs=("spread", lambda x: np.mean(np.abs(x))),
        day_min=("spread", "min"),
        day_max=("spread", "max"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    parts = []
    for zone, block in daily.groupby("zone_code", sort=False):
        b = block.set_index("date").sort_index().reindex(calendar)
        f = pd.DataFrame(index=b.index)
        for field in ("day_da", "day_rt", "day_spread", "day_std", "day_abs", "day_min", "day_max"):
            f[f"{field}_lag2"] = b[field].shift(2)
            for window in (7, 28):
                f[f"{field}_mean{window}"] = b[field].shift(2).rolling(
                    window, min_periods=window // 2,
                ).mean()
        f["zone_code"] = zone
        f["date"] = f.index.strftime("%Y-%m-%d")
        parts.append(f.reset_index(drop=True))
    frame = frame.merge(pd.concat(parts, ignore_index=True), on=["zone_code", "date"], validate="many_to_one")

    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["doy"] / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["doy"] / 365.25)
    frame["is_weekend"] = (frame["dow"] >= 5).astype(float)

    # Primary OOD zone rule: the external site indicator is the deterministic
    # mean of the four frozen NYISO indicators, never a learned ISO-NE vector.
    for column in ZONE_DUMMIES:
        frame[column] = 0.25

    local_day = pd.to_datetime(frame["date"])
    issue_local = local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=4, minutes=55)
    frame["issue_utc"] = issue_local.dt.tz_localize(TZ).dt.tz_convert("UTC")
    frame["feature_last_interval_end_utc"] = (
        (local_day - pd.Timedelta(days=1)).dt.tz_localize(TZ).dt.tz_convert("UTC")
    )
    frame["lead_hours"] = (
        frame["timestamp_utc"] - frame["issue_utc"]
    ).dt.total_seconds() / 3600
    return frame


def main() -> None:
    source = DATA / "isone_hourly_canonical.csv"
    frame = pd.read_csv(source, dtype={"zone_code": "string", "hour_ending_label": "string"})
    frame["timestamp_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="raise")
    frame["zone_code"] = frame["zone_code"].astype(str)
    frame["da_lbmp"] = frame["da_lmp_usd_mwh"].astype(float)
    frame["rt_integrated"] = frame["rt_lmp_final_usd_mwh"].astype(float)
    frame = frame.sort_values(["timestamp_utc", "zone_code"], kind="stable").reset_index(drop=True)
    features = add_frozen_r2_features(frame)
    eval_frame = features[features["date"].between("2025-01-01", "2025-12-31")].copy()
    normalization = json.loads((TASK / "configs" / "frozen_nyiso_r2_normalization.json").read_text(encoding="utf-8"))
    names = normalization["feature_names"]
    missing = set(names) - set(eval_frame.columns)
    if missing:
        raise ValueError(f"Frozen feature interface is missing fields: {sorted(missing)}")
    # Keep raw features and labels for the read-only scoring adapter. Any
    # normalization is applied later from the copied NYISO scaler only.
    output_cols = [
        "timestamp_utc", "date", "zone_code", "hour_ending_label", "occurrence",
        "issue_utc", "feature_last_interval_end_utc", "da_lbmp", "spread", "rt_integrated",
        *names,
    ]
    eval_frame[output_cols].to_csv(DATA / "isone_r2_features_2025.csv", index=False)
    print({
        "rows": len(eval_frame),
        "zones": sorted(eval_frame.zone_code.unique().tolist()),
        "min_history_cutoff_utc": eval_frame.feature_last_interval_end_utc.min().isoformat(),
        "max_history_cutoff_utc": eval_frame.feature_last_interval_end_utc.max().isoformat(),
        "nonfinite_frozen_features": int(eval_frame[names].isna().sum().sum()),
        "nyiso_scaler_loaded": True,
        "isone_normalization_fit": False,
    })


if __name__ == "__main__":
    main()
