"""Replicate official hourly LMP values across four quarter-hour slots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"


def main() -> None:
    source = DATA / "isone_hourly_canonical.csv"
    if not source.exists():
        raise FileNotFoundError("Canonical official ISO-NE data are missing; see MANUAL_DOWNLOAD_REQUIRED.md.")
    frame = pd.read_csv(source, dtype={"hour_ending_label": "string", "zone_code": "string"})
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="raise")
    pieces = []
    for quarter in (0, 15, 30, 45):
        part = frame.copy()
        part["timestamp_utc"] = part["interval_start_utc"] + pd.Timedelta(minutes=quarter)
        part["quarter_minute"] = quarter
        part["source_hour_start_utc"] = part["interval_start_utc"]
        pieces.append(part)
    output = pd.concat(pieces, ignore_index=True).sort_values(["timestamp_utc", "zone_code"])
    if output.duplicated(["timestamp_utc", "zone_code"]).any():
        raise ValueError("Duplicate market interval/site after hourly-to-quarter-hour replication.")
    if not (output["da_lmp_usd_mwh"].groupby([output["source_hour_start_utc"], output["zone_code"]]).nunique() == 1).all():
        raise ValueError("DA hourly values changed during replication.")
    if not (output["rt_lmp_final_usd_mwh"].groupby([output["source_hour_start_utc"], output["zone_code"]]).nunique() == 1).all():
        raise ValueError("Final RT hourly values changed during replication.")
    output.to_csv(DATA / "isone_market_15min.csv", index=False)
    print({
        "rows": len(output),
        "hourly_values_replicated": True,
        "interpolation": False,
        "noise_added": False,
    })


if __name__ == "__main__":
    main()
