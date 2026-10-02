from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile

TASK = Path(__file__).resolve().parents[1]
RAW = TASK / "data_raw/official_iso_ne/RT/raw_archives"
EXTRACTED = TASK / "data_raw/official_iso_ne/RT/unpacked_by_archive"
PROCESSED = TASK / "data_processed"
EXPECTED_ZONES = {
    "MAINE", "NEWHAMPSHIRE", "VERMONT", "CONNECTICUT",
    "RHODEISLAND", "SEMASS", "WCMASS", "NEMASSBOST",
}
START = date(2024, 12, 1)
END = date(2025, 12, 31)
DATE_RE = re.compile(r"lmp_rt_final_(\d{8})\.csv$", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_day_hours(day: date) -> int:
    tz = ZoneInfo("America/New_York")
    start = datetime.combine(day, time.min, tzinfo=tz)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz)
    return int((end.astimezone(ZoneInfo("UTC")) - start.astimezone(ZoneInfo("UTC"))).total_seconds() / 3600)


def inspect_daily_csv(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    date_match = DATE_RE.search(path.name)
    if not date_match:
        return {"file": path.name, "date": "", "zone_rows": 0, "status": "invalid_filename"}
    day = datetime.strptime(date_match.group(1), "%Y%m%d").date()
    if "No data exists for this period." in text:
        return {
            "file": path.name, "date": day.isoformat(), "zone_rows": 0,
            "expected_hours_per_zone": expected_day_hours(day),
            "zones_found": "", "status": "NO_DATA",
        }

    zone_hours = defaultdict(set)
    duplicate_zone_hour = 0
    invalid_lmp = 0
    header_seen = False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip().upper() == "H" and len(row) > 2 and row[1].strip().lower() == "date":
                header_seen = True
                continue
            if row[0].strip().upper() != "D" or len(row) < 7:
                continue
            if row[5].strip().upper() != "LOAD ZONE":
                continue
            zone_name = row[4].strip().upper()
            if zone_name.startswith(".Z."):
                zone_name = zone_name[3:]
            he = row[2].strip().upper()
            key = (zone_name, he)
            if he in zone_hours[zone_name]:
                duplicate_zone_hour += 1
            zone_hours[zone_name].add(he)
            try:
                float(row[6])
            except (TypeError, ValueError):
                invalid_lmp += 1

    hours = expected_day_hours(day)
    found_zones = set(zone_hours)
    missing_zones = sorted(EXPECTED_ZONES - found_zones)
    wrong_hours = {
        zone: len(zone_hours.get(zone, set()))
        for zone in sorted(EXPECTED_ZONES)
        if len(zone_hours.get(zone, set())) != hours
    }
    status = "PASS" if header_seen and not missing_zones and not wrong_hours and not duplicate_zone_hour and not invalid_lmp else "FAIL"
    return {
        "file": path.name,
        "date": day.isoformat(),
        "zone_rows": sum(len(v) for v in zone_hours.values()),
        "expected_hours_per_zone": hours,
        "zones_found": ";".join(sorted(found_zones)),
        "missing_zones": ";".join(missing_zones),
        "zone_hour_counts": ";".join(f"{z}:{len(zone_hours.get(z, set()))}" for z in sorted(EXPECTED_ZONES)),
        "duplicate_zone_hour_rows": duplicate_zone_hour,
        "invalid_lmp_rows": invalid_lmp,
        "status": status,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    archives = sorted(RAW.glob("*.zip"))
    csv_files = sorted(EXTRACTED.rglob("lmp_rt_final_*.csv"))
    daily_rows = [inspect_daily_csv(p) for p in csv_files]

    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in daily_rows:
        if row.get("date"):
            by_date[row["date"]].append(row)
    coverage = []
    day = START
    while day <= END:
        key = day.isoformat()
        hits = by_date.get(key, [])
        coverage.append({
            "market_date": key,
            "found_archive_copies": len(hits),
            "source_csvs": ";".join(r["file"] for r in hits),
            "daily_zone_hour_status": ";".join(r["status"] for r in hits) if hits else "MISSING",
            "status": "MISSING" if not hits else ("DUPLICATE_SOURCE_COPY" if len(hits) > 1 else hits[0]["status"]),
        })
        day += timedelta(days=1)

    zip_index = []
    for archive in archives:
        with ZipFile(archive) as zf:
            names = [n.rsplit("/", 1)[-1] for n in zf.namelist() if n.lower().endswith(".csv")]
        dates = sorted(
            datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
            for name in names if (m := DATE_RE.search(name))
        )
        zip_index.append({
            "archive": archive.name,
            "bytes": archive.stat().st_size,
            "daily_csv_count": len(names),
            "first_market_date": dates[0] if dates else "",
            "last_market_date": dates[-1] if dates else "",
            "out_of_2024-12_to_2025-12_range": sum(d < "2024-12-01" or d > "2025-12-31" for d in dates),
            "sha256": sha256(archive),
        })

    duplicate_dates = []
    for key, hits in sorted(by_date.items()):
        if len(hits) > 1:
            duplicate_dates.append({
                "market_date": key,
                "copies": len(hits),
                "files": ";".join(r["file"] for r in hits),
                "source_archives": ";".join(sorted({p.parent.name for p in csv_files if p.name in {r["file"] for r in hits}})),
            })
    out_of_range = [r for r in daily_rows if r.get("date") and not (START.isoformat() <= r["date"] <= END.isoformat())]
    in_range_unique_dates = sum(START.isoformat() <= d <= END.isoformat() for d in by_date)

    write_csv(PROCESSED / "RT_DAILY_ZONE_HOUR_AUDIT.csv", daily_rows)
    write_csv(PROCESSED / "RT_EXPECTED_DATE_COVERAGE.csv", coverage)
    write_csv(PROCESSED / "RT_MISSING_DATES.csv", [r for r in coverage if r["status"] == "MISSING"] or [{"market_date": "", "found_archive_copies": 0, "source_csvs": "", "daily_zone_hour_status": "", "status": "none"}])
    write_csv(PROCESSED / "RT_DUPLICATE_DATE_COPIES.csv", duplicate_dates or [{"market_date": "", "copies": 0, "files": "", "source_archives": ""}])
    write_csv(PROCESSED / "RT_OUT_OF_RANGE_FILES.csv", out_of_range or [{"file": "", "date": "", "zone_rows": 0, "expected_hours_per_zone": "", "zones_found": "", "missing_zones": "", "zone_hour_counts": "", "duplicate_zone_hour_rows": 0, "invalid_lmp_rows": 0, "status": "none"}])
    write_csv(PROCESSED / "RT_ZIP_CONTENT_INDEX.csv", zip_index)

    missing = [r["market_date"] for r in coverage if r["status"] == "MISSING"]
    failed = [r for r in daily_rows if r["status"] != "PASS" and START.isoformat() <= r.get("date", "") <= END.isoformat()]
    report = [
        "# ISO-NE Final Real-Time archive audit",
        "",
        f"- ZIPs copied and extracted: {len(archives)}",
        f"- Daily CSV files found: {len(csv_files)}",
        f"- Unique dates inside the required 2024-12-01 through 2025-12-31 span: {in_range_unique_dates}",
        f"- Unique out-of-range dates: {len(set(r['date'] for r in out_of_range))}",
        f"- Required calendar dates missing: {len(missing)}",
        f"- Duplicate daily source copies: {len(duplicate_dates)}",
        f"- Per-day eight-zone/hour checks failing within required span: {len(failed)}",
        f"- Out-of-range files: {len(out_of_range)}",
        "",
        "This is filename/CSV-row coverage only; raw CSVs were not modified. RT files provide the full 2024-12 lag month and all 2025 dates except any gaps listed below.",
        "",
        "## Missing ranges",
        "",
        "; ".join(missing) if missing else "None",
        "",
        "## Duplicate dates",
        "",
        "; ".join(r["market_date"] for r in duplicate_dates) if duplicate_dates else "None",
        "",
        "## Out-of-range dates",
        "",
        "; ".join(r["date"] for r in out_of_range) if out_of_range else "None",
    ]
    (TASK / "report/RT_ARCHIVE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print({
        "zip_count": len(archives),
        "daily_csv_count": len(csv_files),
        "missing_required_dates": len(missing),
        "duplicate_dates": [r["market_date"] for r in duplicate_dates],
        "failing_zone_hour_files": len(failed),
        "out_of_range_files": len(out_of_range),
    })


if __name__ == "__main__":
    main()
