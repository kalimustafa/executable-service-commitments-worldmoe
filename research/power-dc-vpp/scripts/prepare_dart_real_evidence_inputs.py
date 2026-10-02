from __future__ import annotations

import json
from pathlib import Path

from dcvpp.experiments.dart_real_input_preparation import (
    exact_phase_dates,
    prepare_dart_real_evidence_inputs,
    write_counterfactual_input_design_manifest,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    locked_index = (
        "artifacts/real_forecasting_recovery/run_id=paper-forecast-v1-recovery-v1/"
        "forecast_days/index.json"
    )
    coverage_index = (
        "artifacts/dart_value_select/real_input_preparation/v1/"
        "nyiso_settlement_coverage_index.json"
    )
    coverage_path = root / coverage_index
    if coverage_path.is_file():
        result = write_counterfactual_input_design_manifest(
            repository_root=root,
            output_root=root / "artifacts/dart_value_select/real_input_preparation/v1",
            settlement_coverage_index=coverage_index,
            nlr295_workload_path=(
                "artifacts/real_forecasting_recovery/run_id=paper-forecast-v1-recovery-v1/"
                "data/nlr295_15m_workload.parquet"
            ),
            nasa_derived_pv_path="data/processed/nyiso_zone_pv_15m.parquet",
            frozen_physical_artifact_path=(
                "artifacts/research/physical/run_id=paper-physical-v1/"
                "frozen_physical_artifact.json"
            ),
            frozen_twin_path=(
                "data/processed/research/physical/run_id=paper-physical-v1/"
                "nyiso_zone_twins.parquet"
            ),
            locked_issue_index_path=locked_index,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "settlement_coverage": result["settlement_coverage"],
                    "efficiency_stop": result["efficiency_stop"],
                    "manifest_path": result["manifest_path"],
                },
                indent=2,
            )
        )
        return 2 if result["status"] == "BLOCKED_BY_SETTLEMENT_COVERAGE" else 0
    result = prepare_dart_real_evidence_inputs(
        repository_root=root,
        output_root=root / "artifacts/dart_value_select/real_input_preparation/v1",
        phase_dates={
            "selection": exact_phase_dates("selection"),
            "locked_test": exact_phase_dates("locked_test"),
        },
        issue_time_index_by_phase={"selection": None, "locked_test": locked_index},
        replay_forcing_index_by_phase={"selection": None, "locked_test": None},
        actionability_index_by_phase={"selection": None, "locked_test": None},
        source_inventory={
            "nasa_derived_pv_15m": "data/processed/nyiso_zone_pv_15m.parquet",
            "nlr295_workload_15m": (
                "artifacts/real_forecasting_recovery/run_id=paper-forecast-v1-recovery-v1/"
                "data/nlr295_15m_workload.parquet"
            ),
            "frozen_physical_artifact": (
                "artifacts/research/physical/run_id=paper-physical-v1/"
                "frozen_physical_artifact.json"
            ),
            "locked_2025_issue_index": locked_index,
        },
    )
    print(json.dumps({"status": result["status"], "phases": result["phases"]}, indent=2))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
