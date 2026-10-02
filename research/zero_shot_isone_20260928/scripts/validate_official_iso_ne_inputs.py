"""Validate manually supplied official data after raw headers are mapped."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"
EXPECTED_ZONES = {
    "CONNECTICUT", "MAINE", "NEMASSBOST", "NEWHAMPSHIRE",
    "RHODEISLAND", "SEMASS", "VERMONT", "WCMASS",
}
START_LOCAL = date(2024, 12, 1)
END_LOCAL = date(2025, 12, 31)


def main() -> None:
    market_path = DATA / "isone_hourly_canonical.csv"
    notice_path = DATA / "isone_da_publication_notices.csv"
    if not market_path.exists() or not notice_path.exists():
        raise FileNotFoundError(
            "Official ISO-NE downloads are not present. See report/MANUAL_DOWNLOAD_REQUIRED.md; "
            "do not substitute synthetic or mirrored data."
        )

    frame = pd.read_csv(market_path, dtype={"zone_code": "string", "hour_ending_label": "string"})
    required = {
        "market_date_local", "hour_ending_label", "occurrence", "interval_start_utc",
        "zone_code", "da_lmp_usd_mwh", "rt_lmp_final_usd_mwh",
        "da_source_file", "rt_source_file", "source_row_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical ISO-NE market file is missing columns: {sorted(missing)}")

    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="raise")
    frame["market_date_local"] = pd.to_datetime(frame["market_date_local"], errors="raise").dt.date
    if frame["zone_code"].isna().any() or set(frame["zone_code"].unique()) != EXPECTED_ZONES:
        raise ValueError("The canonical file must contain exactly the eight fixed ISO-NE load zones.")
    if frame["da_lmp_usd_mwh"].isna().any() or frame["rt_lmp_final_usd_mwh"].isna().any():
        raise ValueError("Missing official DA or final RT LMP; do not impute.")
    if frame.duplicated(["interval_start_utc", "zone_code"]).any():
        raise ValueError("Duplicate UTC hour/zone rows; resolve DST/source row mapping.")
    expected_local_dates = set(pd.date_range("2024-12-01", "2025-12-31", freq="D").date)
    if set(frame["market_date_local"]) != expected_local_dates:
        raise ValueError("Canonical prices must cover every local operating date from 2024-12-01 through 2025-12-31.")
    local_start = frame["interval_start_utc"].dt.tz_convert("America/New_York").dt.date
    if not (local_start == frame["market_date_local"]).all():
        raise ValueError("UTC interval starts do not map back to their ISO-NE local operating date.")

    notices = pd.read_csv(notice_path)
    notice_required = {"operating_date", "notice_posted_at_local", "notice_posted_at_utc", "official_notice_url"}
    if notice_required - set(notices.columns):
        raise ValueError(f"Publication-notice file is missing columns: {sorted(notice_required-set(notices.columns))}")
    notices["operating_date"] = pd.to_datetime(notices["operating_date"], errors="raise").dt.date
    notices["notice_posted_at_utc"] = pd.to_datetime(notices["notice_posted_at_utc"], format="mixed", utc=True, errors="raise")
    notices["notice_posted_at_local"] = pd.to_datetime(notices["notice_posted_at_local"], format="mixed", utc=True, errors="raise")
    if not (notices["notice_posted_at_local"] == notices["notice_posted_at_utc"]).all():
        raise ValueError("Notice local/UTC timestamps disagree.")
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").date
    if set(notices["operating_date"]) != set(expected_dates):
        raise ValueError("Need one official DA result publication notice for every 2025 operating date.")
    if notices.duplicated("operating_date").any():
        raise ValueError("Duplicate publication notices by operating date; resolve source notices.")

    evaluation = frame[frame["market_date_local"].between(date(2025, 1, 1), date(2025, 12, 31))].copy()
    counts = evaluation.groupby("zone_code").size().to_dict()
    if set(counts.values()) != {8760}:
        raise ValueError(f"Expected 8,760 local-calendar 2025 hourly intervals per zone; got {counts}")
    for zone, group in evaluation.groupby("zone_code"):
        stamps = group.sort_values("interval_start_utc")["interval_start_utc"]
        if not (stamps.diff().dropna() == pd.Timedelta(hours=1)).all():
            raise ValueError(f"Non-continuous UTC hourly series for {zone} in local-calendar 2025.")
    print({
        "status": "PASS",
        "evaluation_hours_per_zone": counts,
        "all_eight_zones": True,
        "publication_notice_dates": len(notices),
        "external_scaler_fit": False,
        "imputation": False,
    })


if __name__ == "__main__":
    main()
