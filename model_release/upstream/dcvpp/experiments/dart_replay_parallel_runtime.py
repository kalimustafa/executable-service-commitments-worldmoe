"""Runtime patch that parallelizes frozen DART replay by disjoint dates."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from dcvpp.core import atomic_json, sha256_file, utc_now
from dcvpp.experiments import dart_real_evidence as real_evidence
from dcvpp.experiments.dart_replay_parallel_compat import (
    run_dart_replay_panel_parallel_by_day,
)


@contextmanager
def parallel_dart_replay_runtime(
    *,
    max_day_workers: int = 4,
    solver_threads_per_cell: int = 1,
    ledger_path: Path | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Patch only the imported replay callable for one orchestrator process."""

    if max_day_workers < 1 or solver_threads_per_cell < 1:
        raise ValueError("parallel replay worker and solver-thread counts must be positive")
    if max_day_workers * solver_threads_per_cell > 16:
        raise ValueError("parallel replay CPU budget exceeds 16 physical cores")
    original = real_evidence.run_dart_replay_panel
    calls = 0

    def parallel(panel: Mapping[str, Any], checkpoint_root: Path, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if solver_threads_per_cell != 1:
            raise ValueError(
                "the frozen real-evidence adapter currently permits one HiGHS thread per concurrent day"
            )
        result = run_dart_replay_panel_parallel_by_day(
            panel,
            checkpoint_root,
            max_day_workers=max_day_workers,
            solver_threads_per_cell=solver_threads_per_cell,
            progress_path=Path(checkpoint_root) / "parallel_progress.json",
            **kwargs,
        )
        return result

    real_evidence.run_dart_replay_panel = parallel
    record: dict[str, Any] = {
        "schema_version": "dart_replay_parallel_runtime_v1",
        "status": "ACTIVE",
        "activated_at_utc": utc_now(),
        "max_day_workers": max_day_workers,
        "solver_threads_per_cell": solver_threads_per_cell,
        "formal_result_semantics_changed": False,
        "frozen_modules_modified": False,
        "compatibility_module_sha256": sha256_file(Path(__file__).resolve()),
    }
    if ledger_path is not None:
        atomic_json(ledger_path, record)
    try:
        yield record
    finally:
        real_evidence.run_dart_replay_panel = original
        record.update(
            {
                "status": "INACTIVE",
                "deactivated_at_utc": utc_now(),
                "replay_panel_call_count": calls,
            }
        )
        if ledger_path is not None:
            atomic_json(ledger_path, record)


__all__ = ["parallel_dart_replay_runtime"]
