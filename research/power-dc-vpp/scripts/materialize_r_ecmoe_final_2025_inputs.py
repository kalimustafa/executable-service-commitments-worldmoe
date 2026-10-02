"""Materialize the authorized 362-day R-ECMoE final input panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcvpp.vpp_world_moe.r_ecmoe_final_2025_inputs import (
    materialize_final_2025_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = materialize_final_2025_inputs(root, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
