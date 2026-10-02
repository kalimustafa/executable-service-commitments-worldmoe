from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path


TASK = Path(__file__).resolve().parents[1]
EXPECTED_ZONES = {
    "MAINE", "NEWHAMPSHIRE", "VERMONT", "CONNECTICUT",
    "RHODEISLAND", "SEMASS", "WCMASS", "NEMASSBOST",
}
START = date(2024, 12, 1)
END = date(2025, 12, 31)
MARKETS = {
    "DA": re.compile(r"WW_DALMP_ISO_(\d{8})\.csv$", re.I),
    "RT": re.compile(r"lmp_rt_final_(\d{8})\.csv$", re.I),
}
OUTPUT_FIELDS = [
    "market", "market_date_local", "hour_ending_label", "zone_code",
    "lmp_usd_per_mwh_raw", "location_id", "source_location_name",
    "source_archive", "source_file", "source_record_number",
]


def normalized_zone(value: str) -> str:
    zone = value.strip().upper()
    return zone[3:] if zone.startswith(".Z.") else zone


def extract_market(market: str) -> dict:
    source_root = TASK / "data_raw" / "official_iso_ne" / market / "unpacked_by_archive"
    output_path = TASK / "data_processed" / f"{market}_load_zone_prices_partial.csv"
    date_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    row_count = 0
    skipped_out_of_scope = 0
    unknown_zone_rows = 0
    filename_date_mismatches = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for source_path in sorted(source_root.rglob("*.csv")):
            match = MARKETS[market].search(source_path.name)
            if not match:
                continue
            filename_day = date.fromisoformat(
                f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
            )
            if not START <= filename_day <= END:
                skipped_out_of_scope += 1
                continue

            archive_dir = source_path.parent.name
            archive_name = f"{archive_dir}.zip"
            relative_source = source_path.relative_to(TASK).as_posix()
            with source_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
                reader = csv.reader(source_handle)
                for record_number, row in enumerate(reader, start=1):
                    if not row or row[0].strip().upper() != "D" or len(row) < 7:
                        continue
                    if row[5].strip().upper() != "LOAD ZONE":
                        continue
                    zone = normalized_zone(row[4])
                    if zone not in EXPECTED_ZONES:
                        unknown_zone_rows += 1
                        continue
                    try:
                        row_day = datetime.strptime(row[1].strip(), "%m/%d/%Y").date()
                    except ValueError:
                        continue
                    if row_day != filename_day:
                        filename_date_mismatches += 1
                    writer.writerow({
                        "market": market,
                        "market_date_local": row_day.isoformat(),
                        "hour_ending_label": row[2].strip(),
                        "zone_code": zone,
                        "lmp_usd_per_mwh_raw": row[6].strip(),
                        "location_id": row[3].strip(),
                        "source_location_name": row[4].strip(),
                        "source_archive": archive_name,
                        "source_file": relative_source,
                        "source_record_number": record_number,
                    })
                    row_count += 1
                    date_counts[row_day.isoformat()] += 1
                    file_counts[relative_source] += 1

    return {
        "market": market,
        "output": output_path,
        "source_files_with_rows": len(file_counts),
        "unique_dates_with_rows": len(date_counts),
        "output_rows": row_count,
        "duplicate_dates": sum(1 for n in date_counts.values() if n > 8 * 25),
        "skipped_out_of_scope_files": skipped_out_of_scope,
        "unknown_zone_rows": unknown_zone_rows,
        "filename_date_mismatches": filename_date_mismatches,
    }


def main() -> None:
    results = [extract_market(market) for market in ("DA", "RT")]
    for result in results:
        print({key: str(value) if isinstance(value, Path) else value for key, value in result.items()})


if __name__ == "__main__":
    main()
