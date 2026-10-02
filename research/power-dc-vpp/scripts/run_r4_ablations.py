"""Prepare, execute, resume, or accept measured R4 ablation results.

Execution is limited to the fixed train/2023-H2-validation corpus.  The command
has no 2024 execution mode and rejects every 2025 path before opening it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import argparse
import json
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.e2e_moe_r4.ablations import (
    ALL_ABLATION_METHODS,
    FORMAL_SEEDS,
    AblationContractError,
    AblationResultReceiver,
    build_execution_plan,
    default_preregistration,
    write_execution_plan_once,
)
from dcvpp.e2e_moe_r4.ablation_runner import (  # noqa: E402
    AblationExecutionError,
    ablation_status,
    run_ablation_method,
)


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e2e_moe_dart_vpp_R4_hybrid_solver_free_formal.yaml"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/e2e_moe_hybrid_r4_formal/ablations"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="write immutable preregistration and plan")
    prepare.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    prepare.add_argument("--methods", nargs="+", default=list(ALL_ABLATION_METHODS))
    prepare.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    accept = subparsers.add_parser("accept-result", help="validate and store measured evidence")
    accept.add_argument("result", type=Path)
    accept.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    for command, help_text in (
        ("run", "start one fresh preregistered train/validation ablation"),
        ("continue", "resume one preregistered ablation exactly"),
    ):
        execute = subparsers.add_parser(command, help=help_text)
        execute.add_argument("method", choices=ALL_ABLATION_METHODS)
        execute.add_argument("--seed", type=int, choices=FORMAL_SEEDS, default=FORMAL_SEEDS[0])
        execute.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    status = subparsers.add_parser("status", help="show method/seed execution states")
    status.add_argument("--method", choices=ALL_ABLATION_METHODS)
    status.add_argument("--seed", type=int, choices=FORMAL_SEEDS)
    status.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def _prepare(output_root: Path, methods: Sequence[str], seeds: Sequence[int]) -> dict:
    registration = default_preregistration()
    root = output_root.resolve()
    registration_path = registration.write_once(root / "preregistration.json")
    specs = build_execution_plan(
        registration=registration,
        methods=tuple(methods),
        seeds=tuple(seeds),
    )
    plan_path = write_execution_plan_once(
        root / "execution_plan.json", specs, registration=registration
    )
    return {
        "status": "PLANNED_NO_RESULTS",
        "preregistration": str(registration_path),
        "preregistration_sha256": registration.digest,
        "execution_plan": str(plan_path),
        "run_count": len(specs),
    }


def _accept(output_root: Path, result_path: Path) -> dict:
    payload = json.loads(result_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("measured result JSON root must be an object")
    specs = build_execution_plan()
    destination = AblationResultReceiver(
        output_root.resolve() / "results", specs
    ).accept(payload)
    return {"status": "ACCEPTED_MEASURED_RESULT", "path": str(destination)}


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            output = _prepare(args.output_root, args.methods, args.seeds)
        elif args.command == "accept-result":
            output = _accept(args.output_root, args.result)
        elif args.command in {"run", "continue"}:
            output = dict(
                run_ablation_method(
                    args.config,
                    args.method,
                    args.seed,
                    resume=args.command == "continue",
                )
            )
        else:
            output = dict(ablation_status(args.config, args.method, args.seed))
    except (AblationExecutionError, AblationContractError, ValueError) as exc:
        output = {
            "status": "BLOCKED",
            "reason": str(exc),
            "formal_selection_2024_opened": False,
            "locked_test_2025_opened": False,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
