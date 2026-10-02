from __future__ import annotations

import csv
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


TASK = Path(__file__).resolve().parents[1]
OUTPUT = TASK / "data_processed" / "OFFICIAL_DA_PUBLICATION_NOTICES_2025.csv"
API = "https://www.iso-ne.com/api/1/services/events.json"
EVENT_URL = "https://www.iso-ne.com/event-details?eventId={}"
EASTERN = ZoneInfo("America/New_York")
TITLE_RE = re.compile(
    r"Day\s+Ahead\s+Market\s+Results\s+for\s+(\d{8})\s+are\s+now\s+available",
    re.IGNORECASE,
)
FIELDS = [
    "operating_date", "event_id", "event_title", "notice_event_timestamp_eastern",
    "notice_event_timestamp_utc", "calendar_publish_timestamp_eastern",
    "calendar_publish_timestamp_utc", "availability_gate_timestamp_eastern",
    "availability_gate_timestamp_utc", "event_timezone", "event_page_url",
    "source_api_url",
]


def utc_iso(epoch_ms: int | None) -> str:
    if not epoch_ms:
        return ""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


def eastern_iso(epoch_ms: int | None) -> str:
    if not epoch_ms:
        return ""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).astimezone(EASTERN).isoformat()


def fetch_events(start: datetime, end: datetime, depth: int = 0) -> list[dict]:
    query = urlencode({
        "eventTypeId": "2",       # Notices
        "eventSubTypeId": "13",  # Market Notices
        "fromDate": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "toDate": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": "1000",
    })
    url = f"{API}?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=45) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"ISO-NE event API returned an error for {start}–{end}: {payload}")
    events = payload.get("events", [])
    if len(events) >= 1000:
        if depth >= 12 or end - start <= timedelta(seconds=1):
            raise RuntimeError(f"ISO-NE event API limit reached for {start}–{end}")
        midpoint = start + (end - start) / 2
        left = fetch_events(start, midpoint, depth + 1)
        time.sleep(0.1)
        right = fetch_events(midpoint + timedelta(seconds=1), end, depth + 1)
        return left + right
    return events


def main() -> None:
    # Notices for 2025-01-01 can be posted on 2024-12-31. Include a small
    # margin on both ends, then retain only 2025 operating dates from titles.
    start = datetime(2024, 12, 30, 0, 0, 0)
    end = datetime(2026, 1, 2, 23, 59, 59)
    all_events: dict[str, dict] = {}
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=29, hours=23, minutes=59, seconds=59), end)
        for event in fetch_events(cursor, chunk_end):
            if event.get("event_id") is not None:
                all_events[str(event["event_id"])] = event
        cursor = chunk_end + timedelta(seconds=1)
        time.sleep(0.1)

    rows = []
    for event in all_events.values():
        title = (event.get("event_title") or "").replace("\xa0", " ")
        match = TITLE_RE.search(title)
        if not match:
            continue
        operating_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        if not (date(2025, 1, 1) <= operating_date <= date(2025, 12, 31)):
            continue
        event_id = str(event["event_id"])
        start_ms = event.get("event_start_date_gmt")
        publish_ms = event.get("event_publish_date_gmt")
        gate_ms = max(value for value in (start_ms, publish_ms) if value)
        api_url = f"{API}?eventTypeId=2&eventSubTypeId=13&count=1000"
        rows.append({
            "operating_date": operating_date.isoformat(),
            "event_id": event_id,
            "event_title": title,
            "notice_event_timestamp_eastern": eastern_iso(start_ms),
            "notice_event_timestamp_utc": utc_iso(start_ms),
            "calendar_publish_timestamp_eastern": eastern_iso(publish_ms),
            "calendar_publish_timestamp_utc": utc_iso(publish_ms),
            "availability_gate_timestamp_eastern": eastern_iso(gate_ms),
            "availability_gate_timestamp_utc": utc_iso(gate_ms),
            "event_timezone": event.get("event_start_date_tz") or "",
            "event_page_url": EVENT_URL.format(event_id),
            "source_api_url": api_url,
        })

    rows.sort(key=lambda row: (row["operating_date"], row["availability_gate_timestamp_utc"], row["event_id"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_date: dict[str, list[dict]] = {}
    for row in rows:
        by_date.setdefault(row["operating_date"], []).append(row)
    missing = [
        (date(2025, 1, 1) + timedelta(days=offset)).isoformat()
        for offset in range(365)
        if (date(2025, 1, 1) + timedelta(days=offset)).isoformat() not in by_date
    ]
    multiple = {day: entries for day, entries in by_date.items() if len(entries) > 1}
    print({
        "output": str(OUTPUT),
        "notice_rows": len(rows),
        "unique_2025_operating_dates": len(by_date),
        "missing_operating_dates": missing,
        "multiple_notice_dates": sorted(multiple),
    })


if __name__ == "__main__":
    main()
