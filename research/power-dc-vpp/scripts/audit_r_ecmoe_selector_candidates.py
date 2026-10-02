#!/usr/bin/env python
"""Recompute the corrected selector over every recorded full-2024 candidate.

This is a read-only development-evidence audit.  It does not load data, run a
model, update a checkpoint, or open the locked final split.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


SCHEMA = "dcvpp.r_ecmoe.corrected_selector_candidate_audit.v1"


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric")
    return result


def _truth(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be boolean")


def audit_candidates(
    *,
    validation_csv: str | Path,
    resolved_config: str | Path,
    selected_checkpoint: str | Path,
) -> Mapping[str, Any]:
    validation_path = Path(validation_csv).expanduser().resolve()
    config_path = Path(resolved_config).expanduser().resolve()
    checkpoint_path = Path(selected_checkpoint).expanduser().resolve()
    with validation_path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(config, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("selector inputs must be mappings")
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("resolved config omitted selection")
    if (
        checkpoint.get("schema") != "dcvpp.r_ecmoe.runner.v1"
        or checkpoint.get("lineage_operation")
        != "read_only_2024_revalidation_and_reselection"
        or checkpoint.get("training_not_repeated") is not True
        or checkpoint.get("locked_2025_opened") is not False
    ):
        raise ValueError("selected checkpoint is not the frozen read-only reselection")
    selected_epoch = int(checkpoint.get("epoch", -1))
    if selected_epoch < 1:
        raise ValueError("selected checkpoint omitted epoch")

    risk_weight = _number(selection["decision_risk_weight"], "risk weight")
    sla_weight = _number(selection["decision_sla_weight"], "SLA weight")
    constraint_weight = _number(
        selection["decision_constraint_weight"], "constraint weight"
    )
    latency_weight = _number(
        selection["decision_latency_penalty"], "latency weight"
    )
    latency_limit = _number(
        selection["latency_limit_ms_per_sample"], "latency limit"
    )
    sla_tolerance = _number(selection["sla_tolerance"], "SLA tolerance")
    residual_tolerance = _number(
        selection["hard_residual_tolerance"], "hard residual tolerance"
    )
    if latency_limit <= 0.0:
        raise ValueError("latency limit must be positive")

    candidates: list[dict[str, Any]] = []
    for raw in raw_rows:
        if str(raw.get("validation_scope")) != "full":
            continue
        cost = _number(raw["realized_daily_mean_cost"], "realized cost")
        cvar = _number(raw["empirical_cvar95"], "empirical CVaR95")
        sla = _number(raw["worst_origin_sla"], "worst-origin SLA")
        constraint = _number(raw["constraint_residual"], "constraint objective")
        hard_residual = _number(raw["hard_residual_max"], "hard residual max")
        latency = _number(raw["latency_p95_ms_per_sample"], "latency p95")
        finite = _truth(raw["finite"], "finite")
        rb_ready = _truth(raw["rb_stage_ready_for_selection"], "R-B ready")
        eligible = (
            finite
            and rb_ready
            and hard_residual <= residual_tolerance
            and sla <= sla_tolerance
            and latency <= latency_limit
        )
        latency_excess = max(0.0, latency / latency_limit - 1.0)
        decision_value = (
            cost
            + risk_weight * cvar
            + sla_weight * sla
            + constraint_weight * constraint
            + latency_weight * latency_excess
        )
        candidates.append(
            {
                "epoch": int(raw["epoch"]),
                "eligible_under_corrected_selector": eligible,
                "corrected_decision_value": decision_value,
                "realized_daily_mean_cost": cost,
                "empirical_cvar95": cvar,
                "worst_origin_sla": sla,
                "constraint_residual_objective": constraint,
                "hard_residual_max": hard_residual,
                "latency_p95_ms_per_sample": latency,
                "latency_excess_penalty_input": latency_excess,
            }
        )
    if not candidates:
        raise ValueError("training record contains no full-2024 candidates")
    eligible = [row for row in candidates if row["eligible_under_corrected_selector"]]
    if not eligible:
        raise ValueError("corrected selector has no eligible full-2024 candidate")
    ordered = sorted(
        eligible,
        key=lambda row: (float(row["corrected_decision_value"]), int(row["epoch"])),
    )
    best = ordered[0]
    if int(best["epoch"]) != selected_epoch:
        raise ValueError(
            "frozen selected checkpoint is not the lowest corrected full candidate"
        )
    return {
        "schema": SCHEMA,
        "status": "PROVEN_SELECTED_EPOCH_MINIMUM_AMONG_RECORDED_FULL_CANDIDATES",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_split": "2024_full_registered_364_days",
        "locked_2025_opened": False,
        "training_performed": False,
        "checkpoint_updated": False,
        "hash_computed": False,
        "selected_checkpoint": str(checkpoint_path),
        "selected_epoch": selected_epoch,
        "recorded_full_candidate_count": len(candidates),
        "eligible_full_candidate_count": len(eligible),
        "minimum_corrected_decision_value": float(best["corrected_decision_value"]),
        "selector_formula": (
            "cost + risk_weight*cvar95 + sla_weight*worst_origin_sla + "
            "constraint_weight*constraint_objective + "
            "latency_weight*max(0,p95/limit-1)"
        ),
        "thresholds": {
            "sla_tolerance": sla_tolerance,
            "hard_residual_tolerance": residual_tolerance,
            "latency_limit_ms_per_sample": latency_limit,
        },
        "candidates": sorted(candidates, key=lambda row: int(row["epoch"])),
        "source_validation_csv": str(validation_path),
        "source_resolved_config": str(config_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = audit_candidates(
        validation_csv=args.validation_csv,
        resolved_config=args.resolved_config,
        selected_checkpoint=args.selected_checkpoint,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
