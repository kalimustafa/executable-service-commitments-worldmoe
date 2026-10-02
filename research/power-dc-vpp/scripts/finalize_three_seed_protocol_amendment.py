from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    relative_repository_files = [
        "configs/paper.yaml",
        "configs/dart_value_select.yaml",
        "src/dcvpp/experiments/v1_dart_readiness.py",
        "src/dcvpp/experiments/dart_forecast_oos.py",
        "src/dcvpp/experiments/dart_replay.py",
        "src/dcvpp/experiments/paper_forecast_extensions.py",
        "src/dcvpp/experiments/forecast_extension_bridge.py",
        "src/dcvpp/experiments/dart_training.py",
        "src/dcvpp/experiments/dart_real_evidence.py",
        "src/dcvpp/experiments/dart_value_select.py",
        "src/dcvpp/experiments/dart_pipeline.py",
        "src/dcvpp/paper_artifacts.py",
        "src/dcvpp/paper_finalization.py",
        "src/dcvpp/research_branch/sensitivities.py",
        "docs/dart_real_evidence_bridge.md",
        "scripts/apply_three_seed_protocol_docx_patch.py",
    ]
    workspace_files = [
        "CODEX_Executable_Experiment_Specification_English_NLR_NYISO_Revised.md",
        "Data_Center_VPP_English_Manuscript_and_Experimental_Protocol_NLR_NYISO_Revised.docx",
        "CODEX_ONE_SHOT_GOAL_EN.md",
        "EXECUTABLE_PLAN_CN.md",
    ]
    files = []
    for relative in relative_repository_files:
        path = repository / relative
        files.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    for relative in workspace_files:
        path = workspace / relative
        files.append(
            {
                "path": f"../{relative}",
                "sha256": sha256(path),
                "byte_count": path.stat().st_size,
            }
        )

    payload = {
        "schema_version": "dcvpp.three_seed_runtime_protocol_amendment/v1",
        "status": "COMPLETE",
        "effective_date": "2026-08-11",
        "confirmatory_study_seeds": [20260809, 20260819, 20260829],
        "excluded_legacy_seeds": [20260908, 20260918],
        "completed_v1_four_seed_recovery_preserved_immutable": True,
        "excluded_seed_20260908_policy": "retained_as_auxiliary_v1_evidence_only",
        "applies_to": [
            "forecast_extensions",
            "dart_annual_training",
            "dart_a1_a2_replay",
            "dart_selection_and_locked_2025",
            "primary_96_day_optimization",
            "full_year_extension",
            "sensitivities",
            "bootstrap_and_paired_inference",
        ],
        "counts": {
            "forecast_extension_seed_runs": 63,
            "dart_annual_neural_fits": 144,
            "dart_annual_lightgbm_fits": 8,
            "dart_annual_deterministic_units": 16,
            "dart_annual_total_units": 168,
            "dart_selection_replay_cells": 59184,
            "dart_locked_2025_replay_cells": 19710,
            "dart_shared_baseline_cells": 4383,
            "dart_perfect_information_cells": 1461,
            "dart_top_level_replay_total": 84738,
        },
        "count_formulas": {
            "forecast_extensions": "21*3",
            "dart_annual_neural": "4_origins*2_targets*6_neural_models*3_seeds",
            "dart_selection": "1096_dates*9_candidates*2_experiments*3_seeds",
            "dart_locked_2025": "365_dates*9_candidates*2_experiments*3_seeds",
            "dart_shared_baseline": "1461_dates*3_seeds",
            "dart_perfect_information": "1461_dates*1",
        },
        "focused_test": {
            "status": "PASS",
            "passed": 49,
            "command": ".venv/Scripts/python.exe -m pytest -q tests/test_dart_forecast_oos.py tests/test_dart_training.py tests/test_dart_replay.py tests/test_dart_value_select_metrics.py tests/test_forecast_extension_bridge.py tests/test_v1_dart_readiness.py",
        },
        "docx_render_or_layout_check_performed": False,
        "docx_render_or_layout_check_reason": "user_explicitly_requested_no_format_checks",
        "files": files,
    }
    destination = repository / "artifacts/manifests/three_seed_runtime_amendment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


if __name__ == "__main__":
    main()
