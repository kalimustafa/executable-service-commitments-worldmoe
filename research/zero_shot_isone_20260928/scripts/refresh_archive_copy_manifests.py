from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from zipfile import ZipFile


TASK = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
SPECS = {
    "DA": {
        "archive_dir": TASK / "data_raw/official_iso_ne/DA/raw_archives",
        "extract_dir": TASK / "data_raw/official_iso_ne/DA/unpacked_by_archive",
        "pattern": re.compile(r"WW_DALMP_ISO_(\d{8})\.csv$", re.I),
    },
    "RT": {
        "archive_dir": TASK / "data_raw/official_iso_ne/RT/raw_archives",
        "extract_dir": TASK / "data_raw/official_iso_ne/RT/unpacked_by_archive",
        "pattern": re.compile(r"lmp_rt_final_(\d{8})\.csv$", re.I),
    },
}
FIELDS = [
    "source_path", "copied_path", "bytes", "sha256", "daily_csv_count",
    "first_market_date", "last_market_date", "extracted_csv_count", "status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for market, spec in SPECS.items():
        output_rows = []
        for archive in sorted(spec["archive_dir"].glob("*.zip")):
            with ZipFile(archive) as zip_file:
                csv_members = [
                    name for name in zip_file.namelist()
                    if name.lower().endswith(".csv")
                ]
                member_dates = sorted(
                    match.group(1)
                    for name in csv_members
                    if (match := spec["pattern"].search(Path(name).name))
                )
            extracted_dir = spec["extract_dir"] / archive.stem
            extracted_count = sum(1 for _ in extracted_dir.rglob("*.csv")) if extracted_dir.exists() else 0
            source = DOWNLOADS / archive.name
            source_path = str(source) if source.exists() else ""
            copied_hash = sha256(archive)
            same_as_download = source.exists() and sha256(source) == copied_hash
            extraction_ok = extracted_count == len(csv_members)
            if same_as_download and extraction_ok:
                status = "COPIED_AND_EXTRACTED"
            elif source.exists() and not same_as_download:
                status = "SOURCE_COPY_MISMATCH"
            elif not source.exists() and extraction_ok:
                status = "COPIED_AND_EXTRACTED_SOURCE_NOT_PRESENT"
            else:
                status = "EXTRACTION_COUNT_MISMATCH"
            output_rows.append({
                "source_path": source_path,
                "copied_path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": copied_hash,
                "daily_csv_count": len(csv_members),
                "first_market_date": f"{member_dates[0][:4]}-{member_dates[0][4:6]}-{member_dates[0][6:8]}" if member_dates else "",
                "last_market_date": f"{member_dates[-1][:4]}-{member_dates[-1][4:6]}-{member_dates[-1][6:8]}" if member_dates else "",
                "extracted_csv_count": extracted_count,
                "status": status,
            })
        output_path = TASK / "data_processed" / f"{market}_ARCHIVE_COPY_MANIFEST.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        print({
            "market": market,
            "archives": len(output_rows),
            "statuses": {status: sum(row["status"] == status for row in output_rows) for status in sorted({row["status"] for row in output_rows})},
        })


if __name__ == "__main__":
    main()
