from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcvpp.experiments.dart_conditional_settlement_protocol import (  # noqa: E402
    write_conditional_settlement_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the pre-result common valid-settlement protocol; does not run replay."
    )
    parser.add_argument("--coverage-index", required=True)
    parser.add_argument("--official-rt-recovery-manifest", required=True)
    parser.add_argument(
        "--selection-manifest",
        default=(
            "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos/"
            "selection_stage_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1",
    )
    args = parser.parse_args()
    selection_path = ROOT / args.selection_manifest
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = write_conditional_settlement_protocol(
        repository_root=ROOT,
        output_root=ROOT / args.output_root,
        settlement_coverage_index=args.coverage_index,
        official_rt_recovery_manifest=args.official_rt_recovery_manifest,
        candidate_pool_contract=selection["candidate_pool_contract"],
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
