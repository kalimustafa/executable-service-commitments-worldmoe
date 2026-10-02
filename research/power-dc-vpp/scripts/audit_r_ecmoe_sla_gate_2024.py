"""Read-only 2024 SLA audit of primary latest Top-2 versus zero-raw safe tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset  # noqa: F401 - preload before CUDA/PyTorch on Windows
import torch

from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe_sla_audit import (
    PrimarySLA2024Audit,
    load_primary_latest_for_sla_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime = load_primary_latest_for_sla_audit(
        args.latest_checkpoint,
        args.run_config,
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
        else runtime.run_root / "audits" / "sla_gate_2024"
    )
    result = PrimarySLA2024Audit(
        runtime,
        corpus,
        output_root,
        batch_size=args.batch_size,
        expected_validation_days=364,
    ).run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
