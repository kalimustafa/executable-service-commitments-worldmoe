from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the evidence-limited paper run as terminal PARTIAL_PAPER. "
            "The command refuses to run until optimization has a terminal direct-execution manifest."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", default="paper-terminal-partial-v1")
    parser.add_argument("--optimization-execution", type=Path)
    parser.add_argument("--selection-replay-terminal", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from dcvpp.terminal_partial_finalization import finalize_terminal_partial_paper

    paths = {}
    if args.optimization_execution is not None:
        paths["optimization_execution"] = args.optimization_execution
    if args.selection_replay_terminal is not None:
        paths["selection_replay_terminal"] = args.selection_replay_terminal
    result = finalize_terminal_partial_paper(root, run_id=args.run_id, paths=paths)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
