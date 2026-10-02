"""Candidate-by-date process sharding for exact, resumable DART replay.

The production order is deliberately strict:

1. materialize one baseline per date/seed and one perfect-information result
   per date in date-disjoint spawned processes;
2. execute disjoint candidate/date units in spawned ``loky`` processes;
3. merge rows back into the exact frozen panel order.

The stable checkpoint boundary is a candidate/date pair, not a worker number.
Changing the effective worker count from 8 to 6 or 4 therefore cannot change
scientific identity or invalidate completed work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from joblib import Parallel, delayed, parallel_config
import psutil

from dcvpp.core import atomic_json, sha256_json, utc_now
from dcvpp.experiments.dart_replay import (
    CANDIDATE_INDEPENDENT_HASHES,
    DART_VALUE_SELECT_SEEDS,
    ReplayCell,
    decision_value_metrics,
)


SCHEMA_VERSION = "dart_replay_candidate_date_shards_v1"
RAM_CAP_GB = 26.0
ALLOWED_WORKER_COUNTS = (4, 6, 8)


def _validate_panel(
    panel: Mapping[str, Any],
) -> tuple[list[ReplayCell], tuple[str, ...], tuple[str, ...]]:
    raw = panel.get("cells")
    if not isinstance(raw, Sequence) or not raw:
        raise ValueError("panel contains no replay cells")
    cells = list(raw)
    if not all(isinstance(cell, ReplayCell) for cell in cells):
        raise TypeError("panel cells must be ReplayCell instances")
    if panel.get("panel_sha256") != sha256_json(
        [cell.to_mapping() for cell in cells]
    ):
        raise ValueError("panel SHA-256 does not match its cells")
    dates = tuple(dict.fromkeys(cell.operating_date for cell in cells))
    candidates = tuple(dict.fromkeys(cell.candidate_id for cell in cells))
    expected = {
        (day, candidate, experiment, seed)
        for day in dates
        for candidate in candidates
        for experiment in ("A1", "A2")
        for seed in DART_VALUE_SELECT_SEEDS
    }
    observed = {
        (cell.operating_date, cell.candidate_id, cell.experiment_id, cell.seed)
        for cell in cells
    }
    if observed != expected or len(cells) != len(expected):
        raise ValueError("candidate/date panel is not exact A1/A2 x three seeds")
    return cells, dates, candidates


def _shared_hashes(cells: Sequence[ReplayCell], day: str) -> dict[str, str]:
    values: dict[str, set[str]] = {
        key: {
            cell.evidence_hashes[key]
            for cell in cells
            if cell.operating_date == day
        }
        for key in sorted(CANDIDATE_INDEPENDENT_HASHES)
    }
    changed = [key for key, digests in values.items() if len(digests) != 1]
    if changed:
        raise ValueError(f"candidate-independent hashes changed: {changed}")
    return {key: next(iter(digests)) for key, digests in values.items()}


def _complete(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("input_sha256") != fingerprint:
        raise ValueError(f"shared checkpoint fingerprint changed: {path}")
    if payload.get("status") != "COMPLETE":
        return None
    result = payload.get("result")
    if payload.get("output_sha256") != sha256_json(result):
        raise ValueError(f"shared checkpoint result hash changed: {path}")
    return dict(result)


def _run_shared(
    path: Path,
    *,
    fingerprint: str,
    role: str,
    runner: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    existing = _complete(path, fingerprint)
    if existing is not None:
        return existing, True
    result = dict(runner())
    cost = float(result.get("simulated_total_cost_usd", float("nan")))
    if not (-float("inf") < cost < float("inf")):
        raise ValueError(f"{role} result lacks a finite simulated_total_cost_usd")
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "role": role,
            "input_sha256": fingerprint,
            "result": result,
            "output_sha256": sha256_json(result),
            "completed_at_utc": utc_now(),
        },
    )
    return result, False


def _unit_panel(cells: Sequence[ReplayCell]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operating_dates": [cells[0].operating_date],
        "candidate_ids": [cells[0].candidate_id],
        "experiments": ["A1", "A2"],
        "seeds": list(DART_VALUE_SELECT_SEEDS),
        "cell_count": len(cells),
        "cells": list(cells),
        "panel_sha256": sha256_json([cell.to_mapping() for cell in cells]),
    }


def _read_shared(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("output_sha256") != sha256_json(payload.get("result"))
    ):
        raise ValueError(f"shared replay input is incomplete or corrupt: {path}")
    return dict(payload["result"])


def _requested_workers(value: int) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError("max_candidate_workers must be an integer")
    value = int(value)
    if value not in ALLOWED_WORKER_COUNTS:
        raise ValueError("max_candidate_workers must be one of 4, 6, or 8")
    return value


def _initial_workers(requested: int, initial: int, unit_count: int) -> int:
    used_gb = psutil.virtual_memory().used / 1024**3
    if used_gb >= RAM_CAP_GB:
        admitted = 4
    elif used_gb >= RAM_CAP_GB - 2.0:
        admitted = 6
    else:
        admitted = initial
    return max(1, min(admitted, requested, unit_count))


def _downshift(workers: int, peak_system_ram_gb: float) -> int:
    if peak_system_ram_gb <= RAM_CAP_GB:
        return workers
    if workers > 6:
        return 6
    if workers > 4:
        return 4
    return workers


def _post_gate_workers(
    *,
    initial: int,
    requested: int,
    peak_system_ram_gb: float,
    peak_process_tree_rss_gb: float,
    system_cpu_percent: float,
) -> int:
    downshifted = _downshift(initial, peak_system_ram_gb)
    if downshifted < initial:
        return downshifted
    if (
        requested >= 8
        and initial <= 4
        and peak_system_ram_gb < RAM_CAP_GB
        and peak_process_tree_rss_gb < 20.0
        and system_cpu_percent < 85.0
    ):
        return 8
    if requested >= 6 and initial < 6 and peak_system_ram_gb < RAM_CAP_GB - 1.0:
        return 6
    return min(initial, requested)


def _write_progress(
    path: Path | None,
    *,
    panel_sha256: str,
    status: str,
    total_cells: int,
    completed_cells: int,
    total_units: int,
    completed_units: int,
    requested_workers: int,
    effective_workers: int,
    stage: str,
    active_worker_root: Path | None = None,
    shared_total_units: int = 0,
    shared_completed_units: int = 0,
    active_shared_root: Path | None = None,
) -> None:
    if path is None:
        return
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "updated_at_utc": utc_now(),
            "panel_sha256": panel_sha256,
            "stage": stage,
            "total_cell_count": total_cells,
            "completed_cell_count": completed_cells,
            "total_unit_count": total_units,
            "completed_unit_count": completed_units,
            "requested_workers": requested_workers,
            "effective_workers": effective_workers,
            "solver_threads_per_process": 1,
            "ram_cap_gb": RAM_CAP_GB,
            "active_worker_progress_root": (
                None if active_worker_root is None else active_worker_root.as_posix()
            ),
            "shared_total_unit_count": int(shared_total_units),
            "shared_completed_unit_count": int(shared_completed_units),
            "active_shared_progress_root": (
                None if active_shared_root is None else active_shared_root.as_posix()
            ),
            "locked_2025_opened": False,
        },
    )


def run_dart_replay_panel_parallel_by_candidate_date(
    panel: Mapping[str, Any],
    checkpoint_root: Path,
    *,
    candidate_runner: Callable[[ReplayCell], Mapping[str, Any]],
    baseline_runner: Callable[[str, str, int], Mapping[str, Any]],
    perfect_information_runner: Callable[[str], Mapping[str, Any]],
    serial_panel_runner: Callable[..., Mapping[str, Any]] | None = None,
    max_candidate_workers: int = 8,
    initial_candidate_workers: int | None = None,
    solver_threads_per_process: int = 1,
    worker_backend: str = "loky",
    progress_path: Path | None = None,
    resource_snapshot: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Warm shared solves, execute disjoint units, and merge exact order."""

    if solver_threads_per_process != 1:
        raise ValueError("formal candidate shards require one solver thread per process")
    if worker_backend not in {"loky", "threading"}:
        raise ValueError("worker_backend must be loky or threading")
    requested = _requested_workers(max_candidate_workers)
    initial_requested = _requested_workers(
        requested if initial_candidate_workers is None else initial_candidate_workers
    )
    if initial_requested > requested:
        raise ValueError("initial_candidate_workers cannot exceed max_candidate_workers")
    cells, dates, candidates = _validate_panel(panel)
    del serial_panel_runner  # candidate units use the dedicated no-shared-solve runner below
    root = Path(checkpoint_root) / "candidate_date_parallel"
    shared_root = root / "shared"
    unit_root = root / "units"
    worker_root = root / "worker_progress"
    shared_progress_root = (
        root / "shared_progress" / str(panel["panel_sha256"])[:16]
    )
    units = [
        (day, candidate)
        for day in dates
        for candidate in candidates
    ]
    total_cells = len(cells)
    started = perf_counter()
    effective = _initial_workers(requested, initial_requested, len(units))
    _write_progress(
        progress_path,
        panel_sha256=str(panel["panel_sha256"]),
        status="IN_PROGRESS",
        total_cells=total_cells,
        completed_cells=0,
        total_units=len(units),
        completed_units=0,
        requested_workers=requested,
        effective_workers=effective,
        stage="WARM_SHARED_BASELINE_AND_PI",
        shared_total_units=len(dates),
        shared_completed_units=0,
        active_shared_root=shared_progress_root,
    )

    def warm_shared_day(day: str) -> dict[str, int]:
        hashes = _shared_hashes(cells, day)
        resumed_count = 0
        for seed in DART_VALUE_SELECT_SEEDS:
            fingerprint = sha256_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "role": "strict_common_baseline",
                    "operating_date": day,
                    "seed": seed,
                    "shared_hashes": hashes,
                }
            )
            _result, resumed = _run_shared(
                shared_root / "baseline" / day / f"seed_{seed}.json",
                fingerprint=fingerprint,
                role="strict_common_baseline",
                runner=lambda day=day, seed=seed: baseline_runner(day, "A2", seed),
            )
            resumed_count += int(resumed)
        pi_fingerprint = sha256_json(
            {
                "schema_version": SCHEMA_VERSION,
                "role": "per_day_perfect_information",
                "operating_date": day,
                "shared_hashes": hashes,
            }
        )
        _result, resumed = _run_shared(
            shared_root / "perfect_information" / f"{day}.json",
            fingerprint=pi_fingerprint,
            role="per_day_perfect_information",
            runner=lambda day=day: perfect_information_runner(day),
        )
        resumed_count += int(resumed)
        atomic_json(
            shared_progress_root / f"date_{day}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE",
                "panel_sha256": str(panel["panel_sha256"]),
                "operating_date": day,
                "completed_shared_unit_count": 1,
                "completed_shared_solve_count": len(DART_VALUE_SELECT_SEEDS) + 1,
                "resumed_shared_solve_count": resumed_count,
                "pid": psutil.Process().pid,
                "process_rss_gb": psutil.Process().memory_info().rss / 1024**3,
                "completed_at_utc": utc_now(),
            },
        )
        return {"resumed_shared_solve_count": resumed_count}

    shared_workers = max(1, min(effective, len(dates)))
    if shared_workers == 1:
        shared_rows = [warm_shared_day(day) for day in dates]
    elif worker_backend == "threading":
        shared_rows = Parallel(n_jobs=shared_workers, backend="threading")(
            delayed(warm_shared_day)(day) for day in dates
        )
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            shared_rows = Parallel(
                n_jobs=shared_workers,
                batch_size=1,
                pre_dispatch=shared_workers,
            )(delayed(warm_shared_day)(day) for day in dates)
    shared_resume_count = sum(
        int(row["resumed_shared_solve_count"]) for row in shared_rows
    )

    def run_worker(
        batch_name: str,
        worker_index: int,
        assigned: Sequence[tuple[str, str]],
    ) -> dict[str, Any]:
        baseline_cache: dict[tuple[str, int], dict[str, Any]] = {}
        pi_cache: dict[str, dict[str, Any]] = {}

        def baseline(day: str, _experiment: str, seed: int) -> Mapping[str, Any]:
            key = (day, seed)
            if key not in baseline_cache:
                baseline_cache[key] = _read_shared(
                    shared_root / "baseline" / day / f"seed_{seed}.json"
                )
            return baseline_cache[key]

        def pi(day: str) -> Mapping[str, Any]:
            if day not in pi_cache:
                pi_cache[day] = _read_shared(
                    shared_root / "perfect_information" / f"{day}.json"
                )
            return pi_cache[day]

        output: list[dict[str, Any]] = []
        completed = 0
        resumed = 0
        progress_file = worker_root / batch_name / f"worker_{worker_index:02d}.json"
        for day, candidate in assigned:
            unit_cells = [
                cell
                for cell in cells
                if cell.operating_date == day and cell.candidate_id == candidate
            ]
            candidate_checkpoint_root = unit_root / candidate / day / "cells"
            for cell in unit_cells:
                baseline_result = dict(baseline(day, cell.experiment_id, cell.seed))
                pi_result = dict(pi(day))
                cell_fingerprint = sha256_json(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "cell_sha256": cell.cell_sha256,
                        "baseline_sha256": sha256_json(baseline_result),
                        "perfect_information_sha256": sha256_json(pi_result),
                    }
                )
                cell_path = (
                    candidate_checkpoint_root
                    / cell.experiment_id
                    / f"seed_{cell.seed}.json"
                )
                cached = _complete(cell_path, cell_fingerprint)
                if cached is not None:
                    output.append(cached)
                    resumed += 1
                    continue
                candidate_result = dict(candidate_runner(cell))
                candidate_cost = float(
                    candidate_result.get("simulated_total_cost_usd", float("nan"))
                )
                if not (-float("inf") < candidate_cost < float("inf")):
                    raise ValueError("candidate lacks finite simulated_total_cost_usd")
                row = {
                    "operating_date": cell.operating_date,
                    "candidate_id": cell.candidate_id,
                    "experiment_id": cell.experiment_id,
                    "seed": cell.seed,
                    "cell_sha256": cell.cell_sha256,
                    "identity_sha256": cell.identity_sha256,
                    "evidence_hashes": dict(cell.evidence_hashes),
                    "candidate_result": candidate_result,
                    "baseline_result": baseline_result,
                    "perfect_information_result": pi_result,
                    "decision_value": decision_value_metrics(
                        candidate_result, baseline_result, pi_result
                    ),
                }
                atomic_json(
                    cell_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "COMPLETE",
                        "role": "candidate_replay_cell",
                        "input_sha256": cell_fingerprint,
                        "result": row,
                        "output_sha256": sha256_json(row),
                        "completed_at_utc": utc_now(),
                    },
                )
                output.append(row)
            completed += len(unit_cells)
            atomic_json(
                progress_file,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "COMPLETE" if completed == len(assigned) * 6 else "IN_PROGRESS",
                    "updated_at_utc": utc_now(),
                    "batch_name": batch_name,
                    "worker_index": worker_index,
                    "assigned_unit_count": len(assigned),
                    "completed_cell_count": completed,
                    "last_operating_date": day,
                    "last_candidate_id": candidate,
                    "pid": psutil.Process().pid,
                    "process_rss_gb": psutil.Process().memory_info().rss / 1024**3,
                },
            )
        return {"rows": output, "resumed_cell_count": resumed}

    def execute_batch(
        batch_name: str,
        batch_units: Sequence[tuple[str, str]],
        workers: int,
        already_completed_units: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not batch_units:
            return [], 0
        workers = max(1, min(workers, len(batch_units)))
        assignments = [tuple(batch_units[index::workers]) for index in range(workers)]
        active_root = worker_root / batch_name
        _write_progress(
            progress_path,
            panel_sha256=str(panel["panel_sha256"]),
            status="IN_PROGRESS",
            total_cells=total_cells,
            completed_cells=already_completed_units * 6,
            total_units=len(units),
            completed_units=already_completed_units,
            requested_workers=requested,
            effective_workers=workers,
            stage=f"CANDIDATE_DATE_SHARDS_{batch_name.upper()}",
            active_worker_root=active_root,
            shared_total_units=len(dates),
            shared_completed_units=len(dates),
        )
        if workers == 1:
            shard = run_worker(batch_name, 0, assignments[0])
            return list(shard["rows"]), int(shard["resumed_cell_count"])
        if worker_backend == "threading":
            shards = Parallel(n_jobs=workers, backend="threading")(
                    delayed(run_worker)(batch_name, index, assigned)
                    for index, assigned in enumerate(assignments)
                )
            return (
                [row for shard in shards for row in shard["rows"]],
                sum(int(shard["resumed_cell_count"]) for shard in shards),
            )
        with parallel_config(backend="loky", inner_max_num_threads=1):
            shards = Parallel(
                n_jobs=workers,
                batch_size=1,
                pre_dispatch=workers,
            )(
                delayed(run_worker)(batch_name, index, assigned)
                for index, assigned in enumerate(assignments)
            )
        return (
            [row for shard in shards for row in shard["rows"]],
            sum(int(shard["resumed_cell_count"]) for shard in shards),
        )

    # A bounded first wave provides a measured memory gate before admitting
    # all remaining candidate/date units at the requested concurrency.
    probe_count = min(effective, len(units))
    probe_units = units[:probe_count]
    remaining_units = units[probe_count:]
    rows, resumed_cell_count = execute_batch("memory_gate", probe_units, effective, 0)
    resources = dict(resource_snapshot() if resource_snapshot is not None else {})
    peak_ram = float(
        resources.get("peak_system_ram_gb", psutil.virtual_memory().used / 1024**3)
    )
    peak_tree_rss = float(resources.get("peak_process_tree_rss_gb", 0.0))
    cpu_percent = float(resources.get("system_cpu_percent", 100.0))
    admitted = min(
        _post_gate_workers(
            initial=effective,
            requested=requested,
            peak_system_ram_gb=peak_ram,
            peak_process_tree_rss_gb=peak_tree_rss,
            system_cpu_percent=cpu_percent,
        ),
        len(remaining_units) if remaining_units else effective,
    )
    if remaining_units:
        main_rows, main_resumed = execute_batch(
                "main",
                remaining_units,
                admitted,
                len(probe_units),
            )
        rows.extend(main_rows)
        resumed_cell_count += main_resumed

    by_sha: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest = str(row.get("cell_sha256"))
        if digest in by_sha:
            raise RuntimeError(f"duplicate candidate cell write: {digest}")
        by_sha[digest] = row
    expected = [cell.cell_sha256 for cell in cells]
    if set(by_sha) != set(expected):
        raise RuntimeError("candidate shard merge does not cover the exact panel")
    ordered = [by_sha[digest] for digest in expected]
    _write_progress(
        progress_path,
        panel_sha256=str(panel["panel_sha256"]),
        status="COMPLETE",
        total_cells=total_cells,
        completed_cells=total_cells,
        total_units=len(units),
        completed_units=len(units),
        requested_workers=requested,
        effective_workers=admitted,
        stage="COMPLETE",
        shared_total_units=len(dates),
        shared_completed_units=len(dates),
    )
    return {
        "schema_version": str(panel.get("schema_version", SCHEMA_VERSION)),
        "status": "COMPLETE",
        "panel_sha256": str(panel["panel_sha256"]),
        "cell_count": len(ordered),
        "resumed_cell_count": resumed_cell_count,
        "shared_resume_count": shared_resume_count,
        "results": ordered,
        "results_sha256": sha256_json(ordered),
        "execution": {
            "mode": "candidate_date_process_shards",
            "requested_workers": requested,
            "initial_effective_workers": effective,
            "main_effective_workers": admitted,
            "solver_threads_per_process": 1,
            "worker_backend": worker_backend,
            "shared_warmup_mode": "operating_date_process_shards",
            "shared_warmup_effective_workers": shared_workers,
            "shared_unit_identity": "operating_date",
            "shared_completed_unit_count": len(dates),
            "unit_identity": "operating_date_x_candidate",
            "stable_checkpoint_identity_across_worker_downshift": True,
            "strict_common_baseline_solve_count": len(dates) * 3,
            "perfect_information_solve_count": len(dates),
            "peak_system_ram_gb_at_memory_gate": peak_ram,
            "peak_process_tree_rss_gb_at_memory_gate": peak_tree_rss,
            "system_cpu_percent_at_memory_gate": cpu_percent,
            "ram_cap_gb": RAM_CAP_GB,
            "gpu_accelerates_highs_milp": False,
            "elapsed_seconds": max(perf_counter() - started, 0.0),
            "locked_2025_opened": False,
        },
    }


@contextmanager
def candidate_shard_dart_replay_runtime(
    *,
    max_candidate_workers: int = 8,
    initial_candidate_workers: int | None = None,
    solver_threads_per_process: int = 1,
    progress_path: Path | None = None,
    ledger_path: Path | None = None,
    resource_snapshot: Callable[[], Mapping[str, Any]] | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Patch the real replay panel seam for one selection-only launcher."""

    from dcvpp.experiments import dart_real_evidence as real

    requested = _requested_workers(max_candidate_workers)
    initial = _requested_workers(
        requested if initial_candidate_workers is None else initial_candidate_workers
    )
    if initial > requested:
        raise ValueError("initial_candidate_workers cannot exceed max_candidate_workers")
    if solver_threads_per_process != 1:
        raise ValueError("formal HiGHS shards require exactly one solver thread")
    original = real.run_dart_replay_panel
    calls = 0

    def candidate_parallel(
        panel: Mapping[str, Any], checkpoint_root: Path, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return run_dart_replay_panel_parallel_by_candidate_date(
            panel,
            checkpoint_root,
            max_candidate_workers=requested,
            initial_candidate_workers=initial,
            solver_threads_per_process=solver_threads_per_process,
            progress_path=progress_path,
            resource_snapshot=resource_snapshot,
            **kwargs,
        )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ACTIVE",
        "activated_at_utc": utc_now(),
        "requested_workers": requested,
        "initial_workers": initial,
        "solver_threads_per_process": 1,
        "ram_cap_gb": RAM_CAP_GB,
        "selection_only": True,
        "locked_2025_opened": False,
        "gpu_accelerates_highs_milp": False,
    }
    real.run_dart_replay_panel = candidate_parallel
    if ledger_path is not None:
        atomic_json(ledger_path, record)
    try:
        yield record
    finally:
        real.run_dart_replay_panel = original
        record.update(
            {
                "status": "INACTIVE",
                "deactivated_at_utc": utc_now(),
                "replay_panel_call_count": calls,
            }
        )
        if ledger_path is not None:
            atomic_json(ledger_path, record)


__all__ = [
    "ALLOWED_WORKER_COUNTS",
    "RAM_CAP_GB",
    "SCHEMA_VERSION",
    "candidate_shard_dart_replay_runtime",
    "run_dart_replay_panel_parallel_by_candidate_date",
]
