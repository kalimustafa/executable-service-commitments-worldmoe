"""Run or finalize the formal 2022--2024 conditional DART selection only."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcvpp.core import atomic_json  # noqa: E402
from dcvpp.experiments import dart_real_evidence as real  # noqa: E402
from dcvpp.experiments.dart_formal_preflight_runtime import (  # noqa: E402
    formal_preflight_runtime,
)
from dcvpp.experiments.dart_replay_candidate_shard_runtime import (  # noqa: E402
    candidate_shard_dart_replay_runtime,
)
from dcvpp.experiments.dart_replay_equivalent_reuse_runtime import (  # noqa: E402
    equivalent_baseline_reuse_runtime,
)
from dcvpp.experiments.dart_selection_only import (  # noqa: E402
    FORMAL_REPLAY_RELATIVE,
    FORMAL_SELECTOR_RELATIVE,
    finalize_completed_selection_replay,
    load_selection_only_inputs,
    run_formal_selection_only,
)


@contextmanager
def _formal_runtime(
    bundle: Mapping[str, Any], control_root: Path, *, workers: int, initial_workers: int
) -> Iterator[dict[str, Any]]:
    """Compose performance-only seams around the unchanged replay adapter."""

    with ExitStack() as stack:
        formal = stack.enter_context(
            formal_preflight_runtime(
                repository_root=bundle["root"],
                manifest_path=bundle["preflight_manifest_path"],
                runtime_ledger_path=control_root / "formal_preflight_runtime.json",
                expected_candidate_ids=bundle["candidate_ids"],
                expected_operating_dates=bundle["operating_dates"],
            )
        )
        shard = stack.enter_context(
            candidate_shard_dart_replay_runtime(
                max_candidate_workers=workers,
                initial_candidate_workers=initial_workers,
                solver_threads_per_process=1,
                progress_path=control_root / "progress.json",
                ledger_path=control_root / "candidate_shard_runtime.json",
            )
        )
        reuse = stack.enter_context(
            equivalent_baseline_reuse_runtime(
                ledger_path=control_root / "equivalent_baseline_runtime.json"
            )
        )
        yield {"formal_preflight": formal, "candidate_shards": shard, "reuse": reuse}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=(4, 6, 8), default=8)
    parser.add_argument("--initial-workers", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--finalize-existing", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.initial_workers > args.workers:
        raise ValueError("--initial-workers cannot exceed --workers")
    bundle = load_selection_only_inputs(repository_root=ROOT)
    dry = {
        "schema_version": "dcvpp.dart_selection_only_dry_run.v1",
        "status": "DRY_RUN_COMPLETE_NO_SOLVER",
        "selection_only": True,
        "candidate_ids": list(bundle["candidate_ids"]),
        "operating_date_count": len(bundle["operating_dates"]),
        "study_seeds": [20260809, 20260819, 20260829],
        "expected_replay_cell_count": len(bundle["operating_dates"])
        * len(bundle["candidate_ids"])
        * 2
        * 3,
        "bounded_gap_protocol": bundle["protocol_pins"],
        "formal_preflight": bundle["source_pins"]["formal_preflight"],
        "requested_workers": args.workers,
        "initial_workers": args.initial_workers,
        "locked_2025_opened": False,
        "main_optimization_started": False,
        "solver_invocation_count": 0,
    }
    if args.dry_run:
        path = (
            ROOT
            / "artifacts/dart_value_select/run_control/conditional_selection_replay_v1/"
            "formal_run/dry_run.json"
        )
        atomic_json(path, dry)
        print(json.dumps(dry, indent=2))
        return 0
    if args.finalize_existing is not None:
        value = json.loads(args.finalize_existing.read_text(encoding="utf-8"))
        result = finalize_completed_selection_replay(
            repository_root=ROOT,
            replay_result=value,
            output_root=ROOT / FORMAL_SELECTOR_RELATIVE,
        )
    else:
        result = run_formal_selection_only(
            repository_root=ROOT,
            replay_builder=real.run_real_dart_phase_evidence,
            runtime_context_factory=lambda loaded, control: _formal_runtime(
                loaded,
                control,
                workers=args.workers,
                initial_workers=args.initial_workers,
            ),
            replay_output_root=ROOT / FORMAL_REPLAY_RELATIVE,
            selector_output_root=ROOT / FORMAL_SELECTOR_RELATIVE,
            resume=True,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "PARTIAL_PAPER" else 2


if __name__ == "__main__":
    raise SystemExit(main())
