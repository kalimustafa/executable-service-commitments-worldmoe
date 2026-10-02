"""Bounded day-shard concurrency for the frozen DART replay implementation.

The formal replay implementation is hash-frozen, so throughput-only changes
live in this opt-in compatibility layer.  A day is the concurrency boundary:
all candidates, A1/A2 cells, and three seeds for that day remain under the
unchanged serial runner.  Different day shards have disjoint checkpoint roots,
which preserves atomic cell checkpoints and smallest-unit resume semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from joblib import Parallel, delayed

from dcvpp.core import atomic_json, sha256_json, utc_now
from dcvpp.experiments.dart_replay import ReplayCell, run_dart_replay_panel


ParallelProgressHook = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ReplayParallelPolicy:
    """Resource policy for a single-workstation 16C/22T replay run."""

    max_day_workers: int = 4
    solver_threads_per_cell: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_day_workers, bool) or self.max_day_workers < 1:
            raise ValueError("max_day_workers must be a positive integer")
        if (
            isinstance(self.solver_threads_per_cell, bool)
            or self.solver_threads_per_cell < 1
        ):
            raise ValueError("solver_threads_per_cell must be a positive integer")
        if self.max_day_workers * self.solver_threads_per_cell > 16:
            raise ValueError(
                "worker x solver-thread budget may not exceed the 16 physical cores"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "max_day_workers": int(self.max_day_workers),
            "solver_threads_per_cell": int(self.solver_threads_per_cell),
            "concurrency_boundary": "operating_date",
            "within_day_execution": "frozen_serial_panel_runner",
            "shared_checkpoint_collision_possible": False,
            "supported_worker_backends": ["loky", "threading"],
        }


def _validate_panel(panel: Mapping[str, Any]) -> tuple[list[ReplayCell], tuple[str, ...]]:
    raw_cells = panel.get("cells")
    if not isinstance(raw_cells, Sequence) or not raw_cells:
        raise ValueError("panel contains no replay cells")
    cells = list(raw_cells)
    if not all(isinstance(cell, ReplayCell) for cell in cells):
        raise TypeError("panel cells must be ReplayCell instances")
    if panel.get("panel_sha256") != sha256_json(
        [cell.to_mapping() for cell in cells]
    ):
        raise ValueError("panel SHA-256 does not match its cells")
    dates = tuple(dict.fromkeys(cell.operating_date for cell in cells))
    for operating_date in dates:
        positions = [
            index
            for index, cell in enumerate(cells)
            if cell.operating_date == operating_date
        ]
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("each operating date must occupy one contiguous panel shard")
    return cells, dates


def _day_panel(cells: Sequence[ReplayCell]) -> dict[str, Any]:
    return {
        "schema_version": "dart_value_select_replay_panel_day_shard_v1",
        "operating_dates": [cells[0].operating_date],
        "candidate_ids": sorted({cell.candidate_id for cell in cells}),
        "experiment_ids": sorted({cell.experiment_id for cell in cells}),
        "seeds": sorted({int(cell.seed) for cell in cells}),
        "cell_count": len(cells),
        "panel_sha256": sha256_json([cell.to_mapping() for cell in cells]),
        "cells": list(cells),
    }


def run_dart_replay_panel_parallel_by_day(
    panel: Mapping[str, Any],
    checkpoint_root: Path,
    *,
    candidate_runner: Callable[[ReplayCell], Mapping[str, Any]],
    baseline_runner: Callable[[str, str, int], Mapping[str, Any]],
    perfect_information_runner: Callable[[str], Mapping[str, Any]],
    max_day_workers: int = 4,
    solver_threads_per_cell: int = 1,
    worker_backend: str = "loky",
    progress_path: Path | None = None,
    progress_hook: ParallelProgressHook | None = None,
) -> dict[str, Any]:
    """Run exact replay day shards concurrently and merge in panel order.

    Any failed shard is re-raised after already completed shards have written
    their normal atomic cell checkpoints.  A resume therefore revisits only
    unfinished cells through the unchanged frozen runner.
    """

    cells, dates = _validate_panel(panel)
    policy = ReplayParallelPolicy(
        max_day_workers=int(max_day_workers),
        solver_threads_per_cell=int(solver_threads_per_cell),
    )
    worker_count = min(policy.max_day_workers, len(dates))
    if worker_backend not in {"loky", "threading"}:
        raise ValueError("worker_backend must be 'loky' or 'threading'")
    by_date = {
        operating_date: [
            cell for cell in cells if cell.operating_date == operating_date
        ]
        for operating_date in dates
    }
    started = perf_counter()
    completed_dates: set[str] = set()

    # The real adapter's runner closures already own the full solver config,
    # so resource accounting is recorded here and enforced in production by
    # setting optimization_runtime.solver_threads to this same value.

    def emit_progress(*, last_completed_date: str | None = None) -> None:
        payload = {
            "schema_version": "dart_replay_parallel_progress_v1",
            "status": (
                "COMPLETE" if len(completed_dates) == len(dates) else "IN_PROGRESS"
            ),
            "updated_at_utc": utc_now(),
            "panel_sha256": str(panel["panel_sha256"]),
            "date_count": len(dates),
            "completed_date_count": len(completed_dates),
            "remaining_date_count": len(dates) - len(completed_dates),
            "completed_dates": [day for day in dates if day in completed_dates],
            "last_completed_date": last_completed_date,
            "policy": policy.to_mapping(),
            "effective_day_workers": worker_count,
            "worker_backend": worker_backend,
        }
        if progress_path is not None:
            atomic_json(progress_path, payload)
        if progress_hook is not None:
            progress_hook(payload)

    def run_day(operating_date: str) -> dict[str, Any]:
        return run_dart_replay_panel(
            _day_panel(by_date[operating_date]),
            Path(checkpoint_root) / "day_shards" / operating_date,
            candidate_runner=candidate_runner,
            baseline_runner=baseline_runner,
            perfect_information_runner=perfect_information_runner,
        )

    def run_process_shard(
        worker_index: int,
        operating_dates: Sequence[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Run one long-lived spawned worker without repeated closure copies."""

        shard_progress_path = (
            Path(checkpoint_root)
            / "parallel_worker_progress"
            / f"worker_{worker_index:02d}.json"
        )
        shard_outputs: list[tuple[str, dict[str, Any]]] = []
        for position, operating_date in enumerate(operating_dates, start=1):
            try:
                result = run_day(operating_date)
            except BaseException as exc:
                atomic_json(
                    shard_progress_path,
                    {
                        "schema_version": "dart_replay_parallel_worker_progress_v1",
                        "status": "FAILED",
                        "updated_at_utc": utc_now(),
                        "panel_sha256": str(panel["panel_sha256"]),
                        "worker_index": worker_index,
                        "assigned_dates": list(operating_dates),
                        "completed_dates": [day for day, _result in shard_outputs],
                        "failed_date": operating_date,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                raise
            shard_outputs.append((operating_date, result))
            atomic_json(
                shard_progress_path,
                {
                    "schema_version": "dart_replay_parallel_worker_progress_v1",
                    "status": (
                        "COMPLETE"
                        if position == len(operating_dates)
                        else "IN_PROGRESS"
                    ),
                    "updated_at_utc": utc_now(),
                    "panel_sha256": str(panel["panel_sha256"]),
                    "worker_index": worker_index,
                    "assigned_dates": list(operating_dates),
                    "completed_dates": [day for day, _result in shard_outputs],
                    "completed_date_count": position,
                    "remaining_date_count": len(operating_dates) - position,
                },
            )
        return shard_outputs

    emit_progress()
    results_by_date: dict[str, dict[str, Any]] = {}
    if worker_count == 1:
        for operating_date in dates:
            results_by_date[operating_date] = run_day(operating_date)
            completed_dates.add(operating_date)
            emit_progress(last_completed_date=operating_date)
    elif worker_backend == "threading":
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="dart-replay-day",
        ) as executor:
            futures: dict[Future[dict[str, Any]], str] = {
                executor.submit(run_day, operating_date): operating_date
                for operating_date in dates
            }
            try:
                for future in as_completed(futures):
                    operating_date = futures[future]
                    results_by_date[operating_date] = future.result()
                    completed_dates.add(operating_date)
                    emit_progress(last_completed_date=operating_date)
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
    else:
        # joblib/loky uses reusable spawned processes and cloudpickle.  This
        # both isolates HiGHS instances and carries the real adapter closures.
        # Send one long-lived date shard per process rather than one task per
        # day: the real closure owns multi-year scenario stores, so serializing
        # it O(number of days) times would dominate useful solver work.
        date_shards = [
            tuple(dates[worker_index::worker_count])
            for worker_index in range(worker_count)
        ]
        completed_shards = Parallel(
            n_jobs=worker_count,
            backend="loky",
            batch_size=1,
            pre_dispatch=worker_count,
        )(
            delayed(run_process_shard)(worker_index, operating_dates)
            for worker_index, operating_dates in enumerate(date_shards)
        )
        for shard_outputs in completed_shards:
            for operating_date, result in shard_outputs:
                results_by_date[operating_date] = result
                completed_dates.add(operating_date)
        emit_progress(last_completed_date=dates[-1])

    outputs = [
        result
        for operating_date in dates
        for result in results_by_date[operating_date]["results"]
    ]
    if len(outputs) != len(cells):
        raise RuntimeError("parallel replay merge changed the exact panel cell count")
    expected_identity = [cell.cell_sha256 for cell in cells]
    observed_identity = [str(result.get("cell_sha256")) for result in outputs]
    if observed_identity != expected_identity:
        raise RuntimeError("parallel replay merge changed the exact panel order")
    return {
        "schema_version": str(panel.get("schema_version", "unknown")),
        "status": "COMPLETE",
        "panel_sha256": str(panel["panel_sha256"]),
        "cell_count": len(outputs),
        "resumed_cell_count": sum(
            int(results_by_date[day].get("resumed_cell_count", 0)) for day in dates
        ),
        "shared_resume_count": sum(
            int(results_by_date[day].get("shared_resume_count", 0)) for day in dates
        ),
        "results": outputs,
        "results_sha256": sha256_json(outputs),
        "execution": {
            "mode": (
                "serial"
                if worker_count == 1
                else f"bounded_day_{worker_backend}_pool"
            ),
            "policy": policy.to_mapping(),
            "effective_day_workers": worker_count,
            "worker_backend": worker_backend,
            "process_task_count": (
                worker_count if worker_backend == "loky" else None
            ),
            "process_assignment": (
                "panel_date_position_mod_worker_count"
                if worker_backend == "loky"
                else None
            ),
            "elapsed_seconds": max(perf_counter() - started, 0.0),
            "result_order": "exact_original_panel_order",
            "failure_isolation": "completed_cell_checkpoints_preserved_for_resume",
        },
    }


def run_dart_replay_panel_process_shard(
    panel: Mapping[str, Any],
    checkpoint_root: Path,
    *,
    shard_index: int,
    shard_count: int,
    candidate_runner: Callable[[ReplayCell], Mapping[str, Any]],
    baseline_runner: Callable[[str, str, int], Mapping[str, Any]],
    perfect_information_runner: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one deterministic disjoint date shard in an external process.

    This is the preferred production boundary for Pyomo/HiGHS.  A launcher
    starts ``shard_count`` normal Python processes, each with a distinct
    ``shard_index``.  It avoids unsafe shared-process solver threads and needs
    no serialization of closures or large input stores.
    """

    cells, dates = _validate_panel(panel)
    if isinstance(shard_count, bool) or int(shard_count) != shard_count:
        raise ValueError("shard_count must be an integer")
    if isinstance(shard_index, bool) or int(shard_index) != shard_index:
        raise ValueError("shard_index must be an integer")
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("require shard_count >= 1 and 0 <= shard_index < shard_count")
    selected_dates = tuple(
        operating_date
        for position, operating_date in enumerate(dates)
        if position % shard_count == shard_index
    )
    if not selected_dates:
        return {
            "schema_version": "dart_replay_external_process_shard_v1",
            "status": "COMPLETE",
            "panel_sha256": str(panel["panel_sha256"]),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "operating_dates": [],
            "cell_count": 0,
            "results": [],
            "results_sha256": sha256_json([]),
        }
    selected_set = set(selected_dates)
    shard_cells = [cell for cell in cells if cell.operating_date in selected_set]
    result = run_dart_replay_panel(
        {
            "schema_version": "dart_value_select_replay_process_shard_v1",
            "operating_dates": list(selected_dates),
            "candidate_ids": sorted({cell.candidate_id for cell in shard_cells}),
            "experiment_ids": sorted({cell.experiment_id for cell in shard_cells}),
            "seeds": sorted({int(cell.seed) for cell in shard_cells}),
            "cell_count": len(shard_cells),
            "panel_sha256": sha256_json(
                [cell.to_mapping() for cell in shard_cells]
            ),
            "cells": shard_cells,
        },
        Path(checkpoint_root) / "process_shards" / f"shard_{shard_index:02d}",
        candidate_runner=candidate_runner,
        baseline_runner=baseline_runner,
        perfect_information_runner=perfect_information_runner,
    )
    return {
        **result,
        "schema_version": "dart_replay_external_process_shard_v1",
        "parent_panel_sha256": str(panel["panel_sha256"]),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "operating_dates": list(selected_dates),
        "sharding_rule": "panel_date_position_mod_shard_count",
        "worker_backend": "external_python_process",
    }


def merge_dart_replay_process_shards(
    panel: Mapping[str, Any],
    shard_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge complete external-process shards into the exact panel order."""

    cells, dates = _validate_panel(panel)
    if not shard_results:
        raise ValueError("no replay process shards were supplied")
    shard_counts = {int(result.get("shard_count", -1)) for result in shard_results}
    if len(shard_counts) != 1:
        raise ValueError("process shard_count values differ")
    shard_count = shard_counts.pop()
    by_index = {int(result.get("shard_index", -1)): result for result in shard_results}
    if set(by_index) != set(range(shard_count)):
        raise ValueError("process shard index panel is incomplete or duplicated")
    rows_by_cell: dict[str, Mapping[str, Any]] = {}
    for index, result in sorted(by_index.items()):
        if result.get("status") != "COMPLETE":
            raise ValueError(f"process shard {index} is not COMPLETE")
        if result.get("parent_panel_sha256", result.get("panel_sha256")) != panel[
            "panel_sha256"
        ]:
            raise ValueError(f"process shard {index} parent panel differs")
        for row in result.get("results", ()):
            cell_sha = str(row.get("cell_sha256"))
            if cell_sha in rows_by_cell:
                raise ValueError(f"duplicate replay cell across process shards: {cell_sha}")
            rows_by_cell[cell_sha] = row
    expected = [cell.cell_sha256 for cell in cells]
    if set(rows_by_cell) != set(expected):
        raise ValueError("process shard merge does not cover the exact common panel")
    outputs = [dict(rows_by_cell[cell_sha]) for cell_sha in expected]
    return {
        "schema_version": str(panel.get("schema_version", "unknown")),
        "status": "COMPLETE",
        "panel_sha256": str(panel["panel_sha256"]),
        "cell_count": len(outputs),
        "operating_date_count": len(dates),
        "resumed_cell_count": sum(
            int(result.get("resumed_cell_count", 0)) for result in shard_results
        ),
        "shared_resume_count": sum(
            int(result.get("shared_resume_count", 0)) for result in shard_results
        ),
        "results": outputs,
        "results_sha256": sha256_json(outputs),
        "execution": {
            "mode": "external_process_date_shards",
            "process_count": shard_count,
            "sharding_rule": "panel_date_position_mod_shard_count",
            "result_order": "exact_original_panel_order",
            "duplicate_cell_writes": 0,
        },
    }


__all__ = [
    "ReplayParallelPolicy",
    "merge_dart_replay_process_shards",
    "run_dart_replay_panel_parallel_by_day",
    "run_dart_replay_panel_process_shard",
]
