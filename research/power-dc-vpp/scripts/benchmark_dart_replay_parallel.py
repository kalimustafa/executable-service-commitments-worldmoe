"""Benchmark the result-neutral DART day-shard scheduler without MILP work."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter, sleep
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcvpp.core import sha256_file  # noqa: E402
from dcvpp.experiments.dart_replay import (  # noqa: E402
    A1_ALLOWED_TO_DIFFER,
    A2_ALLOWED_TO_DIFFER,
    REQUIRED_EVIDENCE_HASHES,
    CandidateEvidence,
    build_common_replay_panel,
)
from dcvpp.experiments.dart_replay_parallel_compat import (  # noqa: E402
    run_dart_replay_panel_parallel_by_day,
)


def _hash(index: int) -> str:
    return f"{index:064x}"


def _candidate(experiment: str, name: str) -> CandidateEvidence:
    hashes = {
        key: _hash(position + 100)
        for position, key in enumerate(sorted(REQUIRED_EVIDENCE_HASHES))
    }
    allowed = A1_ALLOWED_TO_DIFFER if experiment == "A1" else A2_ALLOWED_TO_DIFFER
    if name == "model-b":
        for position, key in enumerate(sorted(allowed), start=500):
            hashes[key] = _hash(position)
    if experiment == "A2":
        for position, key in enumerate(sorted(allowed), start=700):
            hashes[key] = _hash(position + (100 if name == "model-b" else 0))
    return CandidateEvidence(name, experiment, hashes)


def _run(root: Path, *, workers: int, delay: float) -> dict[str, Any]:
    dates = [f"2025-01-{day:02d}" for day in range(2, 10)]
    panel = build_common_replay_panel(
        dates,
        {
            experiment: [
                _candidate(experiment, name) for name in ("model-a", "model-b")
            ]
            for experiment in ("A1", "A2")
        },
    )
    calls: Counter[str] = Counter()

    def candidate(cell: Any) -> Mapping[str, Any]:
        calls[cell.cell_sha256] += 1
        sleep(delay)
        return {"simulated_total_cost_usd": 80.0}

    def baseline(*_args: Any) -> Mapping[str, Any]:
        return {"simulated_total_cost_usd": 100.0}

    def pi(*_args: Any) -> Mapping[str, Any]:
        return {"simulated_total_cost_usd": 70.0}

    started = perf_counter()
    result = run_dart_replay_panel_parallel_by_day(
        panel,
        root,
        candidate_runner=candidate,
        baseline_runner=baseline,
        perfect_information_runner=pi,
        max_day_workers=workers,
        worker_backend="threading",
    )
    elapsed = perf_counter() - started
    return {
        "workers": workers,
        "elapsed_seconds": elapsed,
        "cell_count": result["cell_count"],
        "results_sha256": result["results_sha256"],
        "candidate_call_count": sum(calls.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-seconds", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/dart_value_select/benchmarks/dart_replay_parallel.json",
    )
    args = parser.parse_args()
    scratch = args.output.parent / "scratch"
    if scratch.is_dir():
        shutil.rmtree(scratch)
    serial = _run(scratch / "serial", workers=1, delay=args.delay_seconds)
    parallel = _run(
        scratch / "parallel", workers=args.workers, delay=args.delay_seconds
    )
    payload = {
        "schema_version": "dart_replay_parallel_scheduler_benchmark_v1",
        "scope": "scheduler_only_no_formal_replay_no_milp",
        "formal_result_claim": False,
        "serial": serial,
        "parallel": parallel,
        "measured_speedup": serial["elapsed_seconds"]
        / parallel["elapsed_seconds"],
        "result_hashes_equal": serial["results_sha256"]
        == parallel["results_sha256"],
        "compatibility_module": {
            "path": "src/dcvpp/experiments/dart_replay_parallel_compat.py",
            "sha256": sha256_file(
                ROOT / "src/dcvpp/experiments/dart_replay_parallel_compat.py"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0 if payload["result_hashes_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
