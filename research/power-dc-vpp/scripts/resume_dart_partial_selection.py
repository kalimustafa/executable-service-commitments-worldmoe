"""Resume only the six failed LightGBM OOS units and finalize the strict pool.

This launcher keeps the frozen trainer/adapter untouched and stops before
replay, optimization, selection freeze, and the once-only 2025 test.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import sys
from typing import Any, ContextManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from dcvpp.core import load_yaml, sha256_file  # noqa: E402
from dcvpp.experiments import dart_checkpoint_training as frozen  # noqa: E402
from dcvpp.experiments.dart_amp_overflow_compat import (  # noqa: E402
    amp_overflow_compatibility,
)
from dcvpp.experiments.dart_lightgbm_path_compat import (  # noqa: E402
    lightgbm_unicode_path_compatibility,
)
from dcvpp.experiments.dart_partial_candidate import (  # noqa: E402
    PARTIAL_CANDIDATE_STATUS,
    run_partial_candidate_selection_stage,
)
from dcvpp.experiments.dart_pipeline import (  # noqa: E402
    audit_v1_dart_readiness_evidence,
)
from dcvpp.experiments.dart_scheduler_failure_compat import (  # noqa: E402
    isolated_hybrid_scheduler_failures,
)
from dcvpp.experiments.dart_training import (  # noqa: E402
    run_dart_forecast_training_stage,
)
from run_dart_primary_selection import (  # noqa: E402
    _configured_run_ids,
    _ensure_v3_anchor_snapshot,
)


def _optional_fast_npz_context(stage_root: Path) -> ContextManager[Any]:
    """Use the optimization context when installed; absence changes no result."""

    try:
        module = importlib.import_module(
            "dcvpp.experiments.dart_npz_efficiency_compat"
        )
    except ModuleNotFoundError as exc:
        if exc.name != "dcvpp.experiments.dart_npz_efficiency_compat":
            raise
        return nullcontext({"status": "NOT_INSTALLED"})
    factory = getattr(module, "npz_io_efficiency_compatibility", None)
    if not callable(factory):
        raise RuntimeError(
            "dart_npz_efficiency_compat lacks npz_io_efficiency_compatibility"
        )
    return factory(
        ledger_path=stage_root
        / "recovery_overlay/npz_io_efficiency_compatibility.json"
    )


def main() -> int:
    root = REPOSITORY_ROOT
    artifacts = root / "artifacts"
    v1_manifest = (
        artifacts
        / "real_forecasting_recovery/run_id=paper-forecast-v1-recovery-v1"
        / "run_manifest.json"
    )
    dart_config = root / "configs/dart_value_select.yaml"
    readiness_run_id, selection_run_id = _configured_run_ids(root)
    policy = load_yaml(dart_config)
    amendment = policy["pre_dart_v1_readiness"][
        "anchor_only_checkpoint_amendment"
    ]
    if selection_run_id != amendment["destination_run_id"]:
        raise RuntimeError("configured DART selection run is not the frozen v4 run")
    _ensure_v3_anchor_snapshot(root, dart_config)

    readiness_path = (
        artifacts
        / "dart_value_select/v1_readiness"
        / f"run_id={readiness_run_id}"
        / "v1_dart_readiness_manifest.json"
    )
    if not readiness_path.is_file():
        raise FileNotFoundError("completed v1 DART-readiness manifest is absent")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness_audit = audit_v1_dart_readiness_evidence(
        root,
        {
            "manifest_path": readiness_path.relative_to(root).as_posix(),
            "manifest_file_sha256": sha256_file(readiness_path),
        },
        expected_v1_manifest=v1_manifest,
    )
    if readiness_audit.get("passed") is not True:
        raise RuntimeError(
            "v1 DART-readiness audit failed: "
            + "; ".join(map(str, readiness_audit.get("failures", ())))
        )
    stage_root = (
        artifacts
        / "dart_value_select"
        / f"run_id={selection_run_id}"
        / "forecast_oos"
    )
    ledger_root = stage_root / "recovery_overlay"
    with ExitStack() as stack:
        stack.enter_context(
            amp_overflow_compatibility(
                ledger_path=ledger_root / "amp_overflow_compatibility.json"
            )
        )
        stack.enter_context(isolated_hybrid_scheduler_failures(frozen))
        stack.enter_context(
            lightgbm_unicode_path_compatibility(
                ledger_path=ledger_root
                / "lightgbm_unicode_path_compatibility.json"
            )
        )
        stack.enter_context(_optional_fast_npz_context(stage_root))
        selection = run_partial_candidate_selection_stage(
            root,
            artifacts,
            dart_config,
            selection_run_id,
            phase="selection",
            resume=True,
            stage_runner=run_dart_forecast_training_stage,
            input_fingerprints={
                "immutable_v1_manifest": sha256_file(v1_manifest),
                "v1_readiness_manifest": str(
                    readiness_audit["manifest_file_sha256"]
                ),
                "v1_testing_selection_policy": str(
                    readiness["frozen_policy_sha256"]
                ),
            },
        )

    print(json.dumps({"stage": "selection", "result": selection}, indent=2))
    strict_success = (
        selection.get("status") == PARTIAL_CANDIDATE_STATUS
        and selection.get("partial_candidate_downstream_ready") is True
        and selection.get("downstream_ready") is False
        and int(selection.get("completed_unit_count", -1)) == 123
        and int(selection.get("failed_unit_count", -1)) == 3
        and int(selection.get("eligible_completed_unit_count", -1)) == 108
        and int(selection.get("date_drop_count", -1)) == 0
    )
    return 0 if strict_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
