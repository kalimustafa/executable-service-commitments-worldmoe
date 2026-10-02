"""Cold-build the hash-pinned formal selection preflight without a solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcvpp.core import load_yaml  # noqa: E402
from dcvpp.experiments.dart_conditional_settlement_protocol import (  # noqa: E402
    apply_conditional_settlement_overlay,
    load_conditional_settlement_protocol,
)
from dcvpp.experiments.dart_formal_preflight_runtime import (  # noqa: E402
    prepare_formal_preflight,
)


READY_PATH = ROOT / "artifacts/dart_value_select/counterfactual_inputs/v1/selection_config_ready_input_block.json"
SELECTION_PATH = ROOT / "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos/selection_stage_manifest.json"
PROTOCOL_PATH = ROOT / "artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1/conditional_settlement_protocol_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts/dart_value_select/run_control/conditional_selection_replay_v1/formal_preflight"
PILOT_OUTPUT = DEFAULT_OUTPUT / "pilot_lstm_jan2022"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not a mapping: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope", choices=("full", "lstm-jan2022"), default="full"
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    ready = _read(READY_PATH)
    selection = _read(SELECTION_PATH)
    config = load_yaml(ROOT / "configs/dart_value_select.yaml")
    input_key = "dart_real_evidence_inputs"
    config[input_key] = dict(ready[input_key])
    protocol = load_conditional_settlement_protocol(
        PROTOCOL_PATH, repository_root=ROOT
    )
    config = apply_conditional_settlement_overlay(
        config, protocol, repository_root=ROOT
    )
    if args.scope == "lstm-jan2022":
        candidate_names = ("lstm",)
        shared_baseline_dependency_name = "seasonal_naive"
        operating_dates = tuple(
            day
            for day in config["real_evidence_contract"][
                "operating_dates_by_phase"
            ]["selection"]
            if str(day).startswith("2022-01-")
        )
        if len(operating_dates) != 31:
            raise RuntimeError("frozen conditional panel lacks all 31 January 2022 dates")
        scope_id = "selection_lstm_jan2022_v1"
        output_root = args.output_root or PILOT_OUTPUT
    else:
        candidate_names = None
        shared_baseline_dependency_name = None
        operating_dates = None
        scope_id = "selection_full_phase_v1"
        output_root = args.output_root or DEFAULT_OUTPUT
    result = prepare_formal_preflight(
        repository_root=ROOT,
        stage_root=Path(selection["stage_root"]),
        phase="selection",
        forecast_result={input_key: ready[input_key]},
        config=config,
        candidate_pool_contract=selection["candidate_pool_contract"],
        output_root=output_root,
        candidate_names=candidate_names,
        operating_dates=operating_dates,
        scope_id=scope_id,
        shared_baseline_dependency_name=shared_baseline_dependency_name,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
