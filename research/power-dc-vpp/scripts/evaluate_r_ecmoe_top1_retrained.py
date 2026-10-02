"""Evaluate a separately trained Top-1+conditional-tail R model on 2024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset  # noqa: F401 - preload before CUDA/PyTorch on Windows
import torch

from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe_top1_evaluation import (
    RetrainedTop12024Evaluator,
    load_retrained_top1_evaluation_runtime,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top1-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--primary-top2-per-day", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20240813)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime = load_retrained_top1_evaluation_runtime(
        args.top1_checkpoint,
        device=torch.device(args.device),
    )
    corpus = WorldMoECorpus(
        runtime.world_mapping,
        runtime.repository_root,
        weather_coverage_gate=False,
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else runtime.run_root / "evaluation_2024"
    )
    result = RetrainedTop12024Evaluator(
        runtime,
        corpus,
        output_root,
        batch_size=args.batch_size,
        expected_validation_days=364,
        primary_top2_per_day=args.primary_top2_per_day,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        block_days=args.block_days,
    ).run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
