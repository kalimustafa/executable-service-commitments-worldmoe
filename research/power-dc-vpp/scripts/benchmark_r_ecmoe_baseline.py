"""Benchmark one eligible Dense Direct R or Independent R-MoE best."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.dataset  # noqa: F401 - preload before CUDA on Windows
import torch

from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe_baseline_runtime import (
    LATENCY_BATCH_SIZE,
    THROUGHPUT_BATCH_SIZE,
    load_baseline_benchmark_runtime,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _validation_batch(
    corpus: WorldMoECorpus,
    *,
    batch_size: int,
    device: torch.device,
) -> Mapping[str, Any]:
    batch = next(
        iter(
            corpus.batches(
                "validation",
                batch_size,
                device=device,
                max_days=batch_size,
                origin=0,
                require_h96=False,
                include_next_day=False,
            )
        )
    )
    dates = tuple(str(value) for value in batch.get("operating_dates", ()))
    if len(dates) != batch_size or any(date.fromisoformat(value).year != 2024 for value in dates):
        raise SystemExit("official baseline benchmark requires 2024 validation rows")
    return batch


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    runtime = load_baseline_benchmark_runtime(
        args.best_checkpoint, device=device
    )
    corpus = WorldMoECorpus(
        runtime.loaded.world_mapping,
        runtime.loaded.repository_root,
        weather_coverage_gate=False,
    )
    batch1 = _validation_batch(
        corpus, batch_size=LATENCY_BATCH_SIZE, device=device
    )
    batch16 = _validation_batch(
        corpus, batch_size=THROUGHPUT_BATCH_SIZE, device=device
    )
    result = runtime.benchmark_suite(batch1, batch16)
    output = runtime.write_official(result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "baseline_type": result["baseline_type"],
                "batch1_p50_ms": result["full_inference_latency_ms"]["p50"],
                "batch1_p95_ms": result["full_inference_latency_ms"]["p95"],
                "batch16_windows_per_second": result[
                    "throughput_windows_per_second"
                ],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
