from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcvpp.experiments.dart_conditional_forecast_metrics import (
    build_conditional_forecast_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--forecast-root",
        type=Path,
        default=Path(
            "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos"
        ),
    )
    parser.add_argument(
        "--settlement-index",
        type=Path,
        default=Path(
            "artifacts/dart_value_select/real_input_preparation/v1/official_rt_recovery/selection_nyiso_replay_by_date_index.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/dart_value_select/conditional_forecast_metrics/v1"),
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    result = build_conditional_forecast_metrics(
        repository_root=root,
        forecast_root=resolved(args.forecast_root),
        settlement_index=resolved(args.settlement_index),
        output_root=resolved(args.output_root),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "evidence_fingerprint": result["evidence_fingerprint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
