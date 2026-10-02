"""Aggregate already-measured R-ECMoE evidence into paper result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcvpp.vpp_world_moe.r_ecmoe_results import aggregate_r_ecmoe_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-best", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inference-latency", type=Path)
    parser.add_argument("--evaluation-2024", type=Path)
    parser.add_argument("--sla-audit", type=Path)
    parser.add_argument("--top1-evaluation", type=Path)
    parser.add_argument("--dense-evaluation", type=Path)
    parser.add_argument("--independent-evaluation", type=Path)
    parser.add_argument("--top1-inference-latency", type=Path)
    parser.add_argument("--dense-inference-latency", type=Path)
    parser.add_argument("--independent-inference-latency", type=Path)
    # Optional for backward compatibility, but strictly validated when supplied.
    parser.add_argument("--selector-candidate-audit", type=Path)
    # This option only consumes a completed one-shot summary.  Omitting it
    # neither searches for nor opens any 2025 artifact.
    parser.add_argument("--one-shot-2025", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = aggregate_r_ecmoe_results(
        primary_best=args.primary_best,
        output_root=args.output_root,
        inference_latency=args.inference_latency,
        evaluation_2024=args.evaluation_2024,
        sla_audit=args.sla_audit,
        top1_evaluation=args.top1_evaluation,
        dense_evaluation=args.dense_evaluation,
        independent_evaluation=args.independent_evaluation,
        top1_inference_latency=args.top1_inference_latency,
        dense_inference_latency=args.dense_inference_latency,
        independent_inference_latency=args.independent_inference_latency,
        selector_candidate_audit=args.selector_candidate_audit,
        one_shot_2025=args.one_shot_2025,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "primary_status": result["primary"]["status"],
                "missing_experiments": result["missing_experiments"],
                "failed_experiments": result["failed_experiments"],
                "selector_candidate_audit": result["selector_candidate_audit"],
                "output_root": str(args.output_root.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
