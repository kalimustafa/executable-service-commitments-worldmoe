from __future__ import annotations

import json
from pathlib import Path

from dcvpp.experiments.dart_counterfactual_input_materialization import (
    finalize_selection_input_block,
    materialize_selection_counterfactual_inputs,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = materialize_selection_counterfactual_inputs(repository_root=root, workers=4)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selection": result["selection"],
                "locked_test": result["locked_test"],
                "causal_audit": result["causal_audit"],
                "workload_unit_bridge_contract_sha256": result[
                    "workload_unit_bridge_contract_sha256"
                ],
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0


def finalize_only() -> int:
    root = Path(__file__).resolve().parents[1]
    result = finalize_selection_input_block(repository_root=root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
