from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"
TZ = ZoneInfo("America/New_York")
UTC = timezone.utc
ZONES = {
    "CONNECTICUT", "MAINE", "NEMASSBOST", "NEWHAMPSHIRE",
    "RHODEISLAND", "SEMASS", "VERMONT", "WCMASS",
}
START = date(2024, 12, 1)
END = date(2025, 12, 31)
FIELDS = [
    "market_date_local", "hour_ending_label", "occurrence", "interval_start_utc",
    "zone_code", "da_lmp_usd_mwh", "rt_lmp_final_usd_mwh", "da_source_file",
    "rt_source_file", "source_row_id",
]


def hour_start(day_text: str, label: str) -> tuple[int, int, datetime]:
    day = date.fromisoformat(day_text)
    label = label.strip().upper()
    repeated = label.endswith("X")
    he_text = label[:-1] if repeated else label
    he = int(he_text)
    if not 1 <= he <= 24:
        raise ValueError(f"Invalid Hour Ending label {label!r} on {day_text}")
    local_hour = he - 1
    local = datetime.combine(day, time(local_hour)).replace(tzinfo=TZ, fold=1 if repeated else 0)
    return he, 2 if repeated else 1, local.astimezone(UTC)


def read_prices(market: str) -> list[dict[str, str]]:
    path = DATA / f"{market}_load_zone_prices_partial.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_market_rows(rows: list[dict[str, str]], market: str) -> dict[tuple, dict[str, str]]:
    output: dict[tuple, dict[str, str]] = {}
    for row in rows:
        day = date.fromisoformat(row["market_date_local"])
        if not START <= day <= END:
            continue
        if row["zone_code"] not in ZONES:
            raise ValueError(f"Unexpected ISO-NE zone {row['zone_code']!r}")
        key = (row["market_date_local"], row["hour_ending_label"].strip().upper(), row["zone_code"])
        value = row["lmp_usd_per_mwh_raw"].strip()
        if key in output:
            if output[key]["lmp_usd_per_mwh_raw"].strip() != value:
                raise ValueError(f"Conflicting copied {market} prices for {key}")
            continue
        output[key] = row
    return output


def main() -> None:
    da_rows = unique_market_rows(read_prices("DA"), "DA")
    rt_rows = unique_market_rows(read_prices("RT"), "RT")
    if da_rows.keys() != rt_rows.keys():
        da_only = sorted(da_rows.keys() - rt_rows.keys())[:5]
        rt_only = sorted(rt_rows.keys() - da_rows.keys())[:5]
        raise ValueError(f"DA/RT hour-key mismatch; DA-only={da_only}, RT-only={rt_only}")

    canonical = []
    per_day: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for key in sorted(da_rows):
        day_text, he_label, zone = key
        he, occurrence, start_utc = hour_start(day_text, he_label)
        day = date.fromisoformat(day_text)
        local = start_utc.astimezone(TZ)
        if local.date() != day or local.hour != he - 1:
            raise ValueError(f"DST/hour-ending mapping failed for {key}")
        if occurrence == 2 and (he != 2 or day.month != 11):
            raise ValueError(f"Unexpected repeated-hour label {key}")
        da = da_rows[key]
        rt = rt_rows[key]
        canonical.append({
            "market_date_local": day_text,
            "hour_ending_label": he_label,
            "occurrence": occurrence,
            "interval_start_utc": start_utc.isoformat(),
            "zone_code": zone,
            "da_lmp_usd_mwh": float(da["lmp_usd_per_mwh_raw"]),
            "rt_lmp_final_usd_mwh": float(rt["lmp_usd_per_mwh_raw"]),
            "da_source_file": da["source_file"],
            "rt_source_file": rt["source_file"],
            "source_row_id": f"{day_text}|HE{he_label}|{zone}",
        })
        per_day[day_text].add((he_label, occurrence))

    expected_day = START
    expected_dates = 0
    while expected_day <= END:
        text = expected_day.isoformat()
        hours = int((
            datetime.combine(expected_day + timedelta(days=1), time.min, tzinfo=TZ).astimezone(UTC)
            - datetime.combine(expected_day, time.min, tzinfo=TZ).astimezone(UTC)
        ).total_seconds() // 3600)
        expected_labels = 23 if hours == 23 else 25 if hours == 25 else 24
        if text not in per_day or len(per_day[text]) != expected_labels:
            raise ValueError(f"Incomplete or malformed hourly data for {text}: {len(per_day.get(text, set()))} intervals")
        day_rows = [row for row in canonical if row["market_date_local"] == text]
        if len(day_rows) != hours * len(ZONES) or {row["zone_code"] for row in day_rows} != ZONES:
            raise ValueError(f"Eight-zone hour coverage failed for {text}")
        expected_dates += 1
        expected_day += timedelta(days=1)

    frame = pd.DataFrame(canonical, columns=FIELDS)
    frame = frame.sort_values(["interval_start_utc", "zone_code"], kind="stable")
    frame.to_csv(DATA / "isone_hourly_canonical.csv", index=False)

    notices_in = DATA / "OFFICIAL_DA_PUBLICATION_NOTICES_2025.csv"
    notices = pd.read_csv(notices_in)
    notice_columns = {
        "operating_date": "operating_date",
        "availability_gate_timestamp_eastern": "notice_posted_at_local",
        "availability_gate_timestamp_utc": "notice_posted_at_utc",
        "event_page_url": "official_notice_url",
    }
    notices[list(notice_columns)].rename(columns=notice_columns).to_csv(
        DATA / "isone_da_publication_notices.csv", index=False,
    )
    print({
        "status": "CANONICAL_INPUTS_BUILT",
        "local_dates": expected_dates,
        "canonical_hour_zone_rows": len(frame),
        "expected_hours_per_zone_2025": int(len(frame[frame.market_date_local.str.startswith("2025")]) / 8),
        "notice_dates": int(notices.operating_date.nunique()),
        "dst_02x_rows": int((frame.hour_ending_label == "02X").sum()),
        "duplicate_source_rows_removed_after_conflict_check": len(da_rows) + len(rt_rows) - 2 * len(canonical),
    })


if __name__ == "__main__":
    main()
