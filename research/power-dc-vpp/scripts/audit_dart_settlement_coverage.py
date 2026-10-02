from __future__ import annotations

from pathlib import Path

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_real_input_preparation import (
    audit_referenced_nyiso_settlement_coverage,
    exact_phase_dates,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/dart_value_select/real_input_preparation/v1"
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for phase in ("selection", "locked_test"):
        payload = audit_referenced_nyiso_settlement_coverage(
            repository_root=root,
            phase=phase,
            dates=exact_phase_dates(phase),
            referenced_nyiso_data_root="data/processed/referenced/nyiso",
        )
        path = output / f"{phase}_nyiso_settlement_coverage.json"
        atomic_json(path, payload)
        reports[phase] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "status": payload["status"],
            "expected_operating_date_count": payload[
                "expected_operating_date_count"
            ],
            "retained_operating_date_count": payload[
                "retained_operating_date_count"
            ],
            "settlement_evaluable_date_count": payload[
                "settlement_evaluable_date_count"
            ],
            "settlement_invalid_date_count": payload[
                "settlement_invalid_date_count"
            ],
            "missing_settlement_interval_count": payload[
                "missing_settlement_interval_count"
            ],
            "date_drop_count": payload["date_drop_count"],
        }
    manifest = {
        "schema_version": "dcvpp.nyiso_settlement_coverage_index.v1",
        "status": (
            "COMPLETE"
            if all(value["status"] == "COMPLETE" for value in reports.values())
            else "PARTIAL_SOURCE_COVERAGE"
        ),
        "created_at": utc_now(),
        "missing_settlement_imputed": False,
        "missing_dates_silently_dropped": False,
        "formal_a1_a2_mcs_lcb_winner_complete": all(
            value["status"] == "COMPLETE" for value in reports.values()
        ),
        "terminal_paper_cap": (
            None
            if all(value["status"] == "COMPLETE" for value in reports.values())
            else "PARTIAL_PAPER"
        ),
        "phases": reports,
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    path = output / "nyiso_settlement_coverage_index.json"
    atomic_json(path, manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
