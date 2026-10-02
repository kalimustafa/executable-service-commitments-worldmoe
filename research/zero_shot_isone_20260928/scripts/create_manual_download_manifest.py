from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "data_raw/official_download_manifest.csv"
START = date(2024, 12, 1)
END = date(2025, 12, 31)
MAX_DAYS = 15

SOURCES = {
    "hourly_da_lmp": "https://www.iso-ne.com/isoexpress/web/reports/pricing/-/tree/lmps-da-hourly",
    "hourly_rt_lmp_final": "https://www.iso-ne.com/isoexpress/web/reports/pricing/-/tree/lmps-rt-hourly-final",
}


def main() -> None:
    rows = []
    start = START
    sequence = 1
    while start <= END:
        finish = min(start + timedelta(days=MAX_DAYS - 1), END)
        for data_family, official_url in SOURCES.items():
            rows.append({
                "sequence": sequence,
                "data_family": data_family,
                "from_date_inclusive": start.isoformat(),
                "to_date_inclusive": finish.isoformat(),
                "load_zones": "all eight ISO-NE load zones",
                "official_page": official_url,
                "manual_action": "Download official raw CSV/ZIP; CAPTCHA must be completed by the user.",
                "received": "no",
            })
            sequence += 1
        start = finish + timedelta(days=1)
    rows.append({
        "sequence": sequence,
        "data_family": "da_publication_notices",
        "from_date_inclusive": "2025-01-01",
        "to_date_inclusive": "2025-12-31",
        "load_zones": "one official result-publication timestamp per operating date",
        "official_page": "https://www.iso-ne.com/isoexpress/",
        "manual_action": "Provide official Day Ahead Market Results notice timestamps and URLs for each operating date.",
        "received": "no",
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} official download requests; no market data were downloaded.")


if __name__ == "__main__":
    main()
