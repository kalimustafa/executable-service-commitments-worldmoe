"""Finalize the persisted DART selection without rerunning any model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcvpp.experiments.dart_forecast_oos import DARTForecastOOSProducer
from dcvpp.experiments.dart_partial_finalize_fast import fast_finalize_partial_selection
from dcvpp.experiments.dart_training import ValidationResidualGaussianCopulaProvider


def main() -> int:
    stage_root = ROOT / "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos"
    raw = json.loads((stage_root / "selection_stage_manifest.json").read_text(encoding="utf-8"))
    producer = DARTForecastOOSProducer(
        stage_root,
        input_fingerprints=raw["input_fingerprints"],
        joint_draw_provider=ValidationResidualGaussianCopulaProvider(residual_root=stage_root),
    )
    producer.initialize()
    result = fast_finalize_partial_selection(
        producer=producer,
        stage_manifest_path=stage_root / "selection_stage_manifest.json",
        scheduler_manifest_path=stage_root / "checkpoint_scheduler/hybrid_scheduler_manifest.json",
        max_workers=8,
    )
    print(json.dumps({
        "status": result["status"],
        "completed": result["completed_unit_count"],
        "failed": result["failed_unit_count"],
        "eligible": result["eligible_completed_unit_count"],
        "date_drop_count": result["date_drop_count"],
        "manifest_fingerprint": result["manifest_fingerprint"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
