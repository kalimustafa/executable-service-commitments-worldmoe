from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"
MANIFEST = TASK / "data_raw/official_download_manifest.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def date_map(path: Path) -> dict[str, dict[str, str]]:
    return {r["market_date"]: r for r in rows(path)}


def main() -> None:
    da = date_map(DATA / "DA_EXPECTED_DATE_COVERAGE.csv")
    rt = date_map(DATA / "RT_EXPECTED_DATE_COVERAGE.csv")
    notice_path = DATA / "OFFICIAL_DA_PUBLICATION_NOTICES_2025.csv"
    notice_rows = rows(notice_path) if notice_path.exists() else []
    notices = {r["operating_date"]: r for r in notice_rows}
    manifest = rows(MANIFEST)
    updated = []
    for row in manifest:
        family = row["data_family"]
        start = date.fromisoformat(row["from_date_inclusive"])
        end = date.fromisoformat(row["to_date_inclusive"])
        if family == "hourly_da_lmp":
            coverage = da
        elif family == "hourly_rt_lmp_final":
            coverage = rt
        elif family == "da_publication_notices":
            start = date.fromisoformat(row["from_date_inclusive"])
            end = date.fromisoformat(row["to_date_inclusive"])
            missing = []
            received_count = 0
            cursor = start
            while cursor <= end:
                key = cursor.isoformat()
                if key in notices:
                    received_count += 1
                else:
                    missing.append(key)
                cursor += timedelta(days=1)
            row["received_day_count"] = received_count
            row["missing_day_count"] = len(missing)
            row["missing_dates"] = ";".join(missing)
            row["received"] = "yes" if not missing else ("partial" if received_count else "no")
            row["status"] = "RECEIVED_FULL" if not missing else ("RECEIVED_PARTIAL" if received_count else "MISSING")
            row["official_page"] = "https://www.iso-ne.com/calendar/?eventTypeId=2"
            row["manual_action"] = "Collected from the official ISO-NE Market Notices calendar; see data_processed/OFFICIAL_DA_PUBLICATION_NOTICES_2025.csv."
            updated.append(row)
            continue
        else:
            row.update(received="no", status="MISSING_OFFICIAL_NOTICE_FILE",
                       received_day_count=0, missing_day_count=(end-start).days+1,
                       missing_dates="all 2025 operating dates")
            updated.append(row)
            continue
        missing = []
        received_count = 0
        cursor = start
        while cursor <= end:
            key = cursor.isoformat()
            info = coverage.get(key)
            if info and int(info.get("found_archive_copies", "0")) > 0:
                received_count += 1
            else:
                missing.append(key)
            cursor += timedelta(days=1)
        row["received_day_count"] = received_count
        row["missing_day_count"] = len(missing)
        row["missing_dates"] = ";".join(missing)
        row["received"] = "yes" if not missing else ("partial" if received_count else "no")
        row["status"] = "RECEIVED_FULL" if not missing else ("RECEIVED_PARTIAL" if received_count else "MISSING")
        updated.append(row)
    fields = list(updated[0])
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated)
    print({"manifest_rows": len(updated), "DA_chunks": sum(r["data_family"] == "hourly_da_lmp" for r in updated),
           "RT_chunks": sum(r["data_family"] == "hourly_rt_lmp_final" for r in updated),
           "notice_rows": sum(r["data_family"] == "da_publication_notices" for r in updated)})


if __name__ == "__main__":
    main()
