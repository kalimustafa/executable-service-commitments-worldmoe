from __future__ import annotations

import argparse
from pathlib import Path

from dcvpp.experiments.nyiso_rt_official_recovery import (
    recover_official_rt_settlement,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover missing RT blocks from official NYISO 5-minute archives."
    )
    parser.add_argument(
        "--offline-cache",
        action="store_true",
        help="Use already downloaded official archives instead of live retrieval.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = (
        root
        / "artifacts/dart_value_select/real_input_preparation/v1/official_rt_recovery"
    )
    result = recover_official_rt_settlement(
        repository_root=root,
        destination=destination,
        refresh_official=not args.offline_cache,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

