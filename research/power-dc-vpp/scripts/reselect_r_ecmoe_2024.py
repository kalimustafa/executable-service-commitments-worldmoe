"""Revalidate source latest on 2024 and write a selector-repaired best lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset  # noqa: F401 - preload before CUDA/PyTorch on Windows
import torch

from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe_reselection import revalidate_and_reselect
from dcvpp.vpp_world_moe.r_ecmoe_sla_audit import (
    load_primary_latest_for_sla_audit,
)

try:
    from run_r_ecmoe import _load_extended_yaml, _runner_config  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    if exc.name != "run_r_ecmoe":
        raise
    from scripts.run_r_ecmoe import _load_extended_yaml, _runner_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-latest", type=Path, required=True)
    parser.add_argument("--source-run-config", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--audit-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = load_primary_latest_for_sla_audit(
        args.source_latest,
        args.source_run_config,
        device=torch.device(args.device),
    )
    selector_path = args.selector_config.resolve()
    selector_resolved = _load_extended_yaml(selector_path)
    runner_config = _runner_config(selector_resolved)
    corpus = WorldMoECorpus(
        source.world_mapping,
        source.repository_root,
        weather_coverage_gate=False,
    )
    result = revalidate_and_reselect(
        source,
        corpus,
        selector_resolved=selector_resolved,
        selector_config_path=selector_path,
        runner_config=runner_config,
        audit_artifact=args.audit_artifact,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
