"""Continue the paper run only after the strict eight-candidate DART gate.

This is intentionally a thin launcher.  The scientific sequence remains owned
by :mod:`dcvpp.paper_execution` and :mod:`dcvpp.experiments.dart_pipeline`:
forecast-extension resume, DART freeze, the once-only locked 2025 stage, real
replay/value selection, optimization, and paper finalization.  This file only
adds the verified runtime compatibility contexts needed by the long DART run.

It never starts or repairs selection OOS training.  Until the finalized
``PARTIAL_WITH_FAILED_CANDIDATE`` manifest proves the exact preregistered
eight-candidate contract, it exits before entering the paper pipeline.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dcvpp.core import load_yaml, sha256_file, sha256_json  # noqa: E402
from dcvpp.experiments import dart_checkpoint_training as frozen  # noqa: E402
from dcvpp.experiments.dart_amp_overflow_compat import (  # noqa: E402
    amp_overflow_compatibility,
)
from dcvpp.experiments.dart_conditional_settlement_protocol import (  # noqa: E402
    conditional_settlement_runtime,
)
from dcvpp.experiments.dart_lightgbm_path_compat import (  # noqa: E402
    lightgbm_unicode_path_compatibility,
)
from dcvpp.experiments.dart_npz_efficiency_compat import (  # noqa: E402
    npz_io_efficiency_compatibility,
)
from dcvpp.experiments.dart_partial_candidate import (  # noqa: E402
    PARTIAL_CANDIDATE_STATUS,
    validate_partial_candidate_pool_contract,
)
from dcvpp.experiments.dart_replay_parallel_runtime import (  # noqa: E402
    parallel_dart_replay_runtime,
)
from dcvpp.experiments.dart_scheduler_failure_compat import (  # noqa: E402
    isolated_hybrid_scheduler_failures,
)
from dcvpp.paper_status import TERMINAL_STATUSES  # noqa: E402
from dcvpp.pipeline import run_pipeline  # noqa: E402


def _selection_manifest_path(root: Path, paper_config: Mapping[str, Any]) -> Path:
    execution = paper_config.get("paper_execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("paper_execution must be a mapping")
    run_id = str(execution.get("dart_forecast_run_id", ""))
    if not run_id:
        raise ValueError("paper_execution.dart_forecast_run_id is absent")
    return (
        root
        / "artifacts"
        / "dart_value_select"
        / f"run_id={run_id}"
        / "forecast_oos"
        / "selection_stage_manifest.json"
    )


def _require_strict_partial_selection(
    root: Path, paper_config_path: Path
) -> tuple[dict[str, Any], Path]:
    """Validate the sole admissible partial pool before any downstream write."""

    paper_config = load_yaml(paper_config_path)
    manifest_path = _selection_manifest_path(root, paper_config)
    if not manifest_path.is_file():
        raise RuntimeError(
            "strict partial selection manifest is absent; selection must finish first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != PARTIAL_CANDIDATE_STATUS:
        raise RuntimeError(
            "selection is not finalized as PARTIAL_WITH_FAILED_CANDIDATE; "
            "this continuation launcher does not run or repair selection"
        )
    if (
        manifest.get("partial_candidate_downstream_ready") is not True
        or manifest.get("downstream_ready") is not False
        or int(manifest.get("completed_unit_count", -1)) != 123
        or int(manifest.get("failed_unit_count", -1)) != 3
        or int(manifest.get("eligible_completed_unit_count", -1)) != 108
        or int(manifest.get("date_drop_count", -1)) != 0
    ):
        raise RuntimeError("strict partial selection completion counts are invalid")
    validate_partial_candidate_pool_contract(
        manifest.get("candidate_pool_contract"),
        failed_units=manifest.get("failed_units"),
        require_failed_units=True,
    )
    manifest_core = {
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    }
    if manifest.get("manifest_fingerprint") != sha256_json(manifest_core):
        raise RuntimeError("strict partial selection manifest self-hash is invalid")

    execution = paper_config.get("paper_execution", {})
    dart_config_value = execution.get(
        "dart_config_path", "configs/dart_value_select.yaml"
    )
    dart_config_path = Path(str(dart_config_value))
    if not dart_config_path.is_absolute():
        dart_config_path = root / dart_config_path
    if manifest.get("config_sha256") != sha256_file(dart_config_path):
        raise RuntimeError("strict selection and configured DART protocol differ")
    return manifest, manifest_path


def _runtime_ledger_root(root: Path, selection: Mapping[str, Any]) -> Path:
    stage_root_value = selection.get("stage_root")
    if stage_root_value in (None, ""):
        raise RuntimeError("strict selection manifest has no stage_root")
    stage_root = Path(str(stage_root_value))
    if not stage_root.is_absolute():
        stage_root = root / stage_root
    stage_root = stage_root.resolve()
    expected_root = (root / "artifacts" / "dart_value_select").resolve()
    if expected_root not in stage_root.parents:
        raise RuntimeError("selection stage_root is outside the DART artifact tree")
    return stage_root / "continuation_runtime"


def main() -> int:
    root = REPOSITORY_ROOT.resolve()
    paper_config_path = root / "configs" / "paper.yaml"
    selection, selection_path = _require_strict_partial_selection(
        root, paper_config_path
    )
    ledger_root = _runtime_ledger_root(root, selection)

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
        stack.enter_context(
            npz_io_efficiency_compatibility(
                ledger_path=ledger_root / "npz_io_efficiency_compatibility.json"
            )
        )
        # This protocol must exist before the continuation is admitted.  It
        # injects one hash-pinned, candidate-independent set of fully observed
        # settlement dates immediately before real-evidence preflight.  The
        # full-calendar INVALID_MISSING_SETTLEMENT ledger remains outside the
        # solver panel and no fallback to silent deletion is allowed.
        stack.enter_context(
            conditional_settlement_runtime(
                repository_root=root,
                protocol_manifest=(
                    root
                    / "artifacts"
                    / "dart_value_select"
                    / "protocol_amendments"
                    / "conditional_settlement_v1"
                    / "conditional_settlement_protocol_manifest.json"
                ),
                runtime_ledger_path=(
                    ledger_root / "conditional_settlement_runtime.json"
                ),
            )
        )
        stack.enter_context(
            parallel_dart_replay_runtime(
                max_day_workers=4,
                solver_threads_per_cell=1,
                ledger_path=ledger_root / "parallel_dart_replay_runtime.json",
            )
        )
        result = run_pipeline(
            root,
            paper_config_path,
            target="paper",
            resume=True,
        )

    status = str(result.get("status", ""))
    summary = {
        "launcher": "continue_dart_partial_paper",
        "selection_manifest": selection_path.as_posix(),
        "candidate_pool_contract_sha256": selection[
            "candidate_pool_contract"
        ]["contract_sha256"],
        "paper_status": status,
        "terminal": bool(result.get("terminal", status in TERMINAL_STATUSES)),
        "result": result,
    }
    print(json.dumps(summary, indent=2, default=str))
    if status in TERMINAL_STATUSES or status == "PAPER_EXECUTION_RUNNING":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
