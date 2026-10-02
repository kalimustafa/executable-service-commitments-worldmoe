"""Prepare an isolated, optimizer-faithful R-ECMoE continuation run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from run_r_ecmoe import (
    _load_extended_yaml,
    _load_w1,
    _resolve,
    _runner_config,
    _write_run_contract,
)
from dcvpp.vpp_world_moe.config import WorldMoEConfig


ALLOWED_RUNNER_CONFIG_CHANGES = frozenset(
    {
        "epochs",
        "early_stop_patience",
        "early_stop_min_relative_improvement",
        "sla_tolerance",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-latest", type=Path, required=True)
    parser.add_argument("--selection-best", type=Path, required=True)
    return parser


def _load(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"checkpoint does not exist: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SystemExit(f"checkpoint payload is malformed: {resolved}")
    return payload


def _tensor_state_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if set(left) != set(right):
        return False
    for name in left:
        left_value = left[name]
        right_value = right[name]
        if isinstance(left_value, torch.Tensor) and isinstance(right_value, torch.Tensor):
            if not torch.equal(left_value, right_value):
                return False
        elif left_value != right_value:
            return False
    return True


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare(
    *,
    config_path: Path,
    run_id: str,
    source_latest_path: Path,
    selection_best_path: Path,
) -> Path:
    config_path = config_path.resolve()
    repository_root = config_path.parent.parent.resolve()
    config = _load_extended_yaml(config_path)
    if str(config.get("phase", "")).lower() != "r_ecmoe":
        raise SystemExit("continuation config must use phase=r_ecmoe")
    source_mapping = config.get("source")
    if not isinstance(source_mapping, Mapping):
        raise SystemExit("continuation config omitted source")
    if source_mapping.get("locked_test_enabled") is not False:
        raise SystemExit("continuation must keep the locked test disabled")

    source_latest_path = source_latest_path.resolve()
    selection_best_path = selection_best_path.resolve()
    source_latest = _load(source_latest_path)
    selection_best = _load(selection_best_path)
    for label, payload in (("source", source_latest), ("selection", selection_best)):
        if payload.get("schema") != "dcvpp.r_ecmoe.runner.v1":
            raise SystemExit(f"{label} checkpoint schema mismatch")
        if str(payload.get("phase", "")).lower() != "r_ecmoe":
            raise SystemExit(f"{label} checkpoint phase mismatch")
        if payload.get("locked_2025_opened") is not False:
            raise SystemExit(f"{label} checkpoint opened locked 2025")
        if payload.get("external_solver") is not False:
            raise SystemExit(f"{label} checkpoint used an external solver")
        if payload.get("candidate_search") is not False:
            raise SystemExit(f"{label} checkpoint used candidate search")

    if source_latest.get("optimizer") is None:
        raise SystemExit("source latest omitted optimizer state")
    if source_latest.get("scheduler") is None or source_latest.get("scaler") is None:
        raise SystemExit("source latest omitted scheduler/scaler state")
    if selection_best.get("training_not_repeated") is not True:
        raise SystemExit("selection checkpoint is not an audited reselection")
    if selection_best.get("resume_for_training_allowed") is not False:
        raise SystemExit("selection checkpoint contract is malformed")
    if Path(str(selection_best.get("source_checkpoint", ""))).resolve() != source_latest_path:
        raise SystemExit("selection checkpoint does not reference source latest")
    if int(selection_best.get("epoch", -1)) != int(source_latest.get("epoch", -2)):
        raise SystemExit("selection/source epochs differ")
    if int(selection_best.get("global_update", -1)) != int(
        source_latest.get("global_update", -2)
    ):
        raise SystemExit("selection/source updates differ")
    if not _tensor_state_equal(
        source_latest.get("r_state"), selection_best.get("r_state")
    ):
        raise SystemExit("selection/source R weights differ")
    if not _tensor_state_equal(
        source_latest.get("tuned_dynamics_state", {}),
        selection_best.get("tuned_dynamics_state", {}),
    ):
        raise SystemExit("selection/source tuned dynamics differ")

    metrics = selection_best.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SystemExit("selection checkpoint omitted metrics")
    if metrics.get("validation_scope") != "full" or metrics.get(
        "selection_eligible"
    ) is not True:
        raise SystemExit("selection seed is not a full eligible validation")
    decision_value = float(metrics.get("decision_value", math.inf))
    if not math.isfinite(decision_value):
        raise SystemExit("selection decision value is not finite")

    runner_config = asdict(_runner_config(config))
    source_runner_config = source_latest.get("resolved_config")
    if not isinstance(source_runner_config, Mapping):
        raise SystemExit("source latest omitted resolved runner config")
    changed = {
        key
        for key in set(source_runner_config) | set(runner_config)
        if source_runner_config.get(key) != runner_config.get(key)
    }
    if not changed.issubset(ALLOWED_RUNNER_CONFIG_CHANGES):
        raise SystemExit(
            "continuation changed non-approved runner settings: "
            + ", ".join(sorted(changed - ALLOWED_RUNNER_CONFIG_CHANGES))
        )
    source_epoch = int(source_latest["epoch"])
    if int(runner_config["epochs"]) <= source_epoch:
        raise SystemExit("continuation epoch limit must exceed the source epoch")
    revision = selection_best.get("selection_formula_revision")
    if not isinstance(revision, Mapping) or float(
        revision.get("sla_tolerance", math.nan)
    ) != float(runner_config["sla_tolerance"]):
        raise SystemExit("continuation SLA tolerance differs from audited reselection")

    world_config_path = _resolve(repository_root, str(source_mapping["world_config"]))
    world_mapping = _load_extended_yaml(world_config_path)
    world_config = WorldMoEConfig.from_mapping(world_mapping["model"])
    w1_checkpoint = _resolve(repository_root, str(source_mapping["w1_checkpoint"]))
    _, source_meta = _load_w1(world_config, w1_checkpoint)
    output_mapping = config.get("outputs", {})
    if not isinstance(output_mapping, Mapping):
        raise SystemExit("continuation config outputs must be a mapping")
    output_root = _resolve(repository_root, str(output_mapping["root"]))
    run_root = output_root / run_id
    if run_root.exists():
        raise SystemExit(f"continuation run already exists: {run_root}")
    _write_run_contract(
        run_root,
        run_id=run_id,
        resolved_config=config,
        source_meta=source_meta,
        config_path=config_path,
        world_config_path=world_config_path,
        resume=False,
    )

    prepared = dict(source_latest)
    prepared.update(
        {
            "run_id": run_id,
            "resolved_config": runner_config,
            "metrics": dict(metrics),
            "best_decision_value": decision_value,
            "best_epoch": source_epoch,
            "best_speed_ms": float(
                metrics.get("latency_p95_ms_per_sample", math.inf)
            ),
            "early_stop_reference_value": decision_value,
            "early_stop_bad_epochs": 0,
            "epoch_completed": True,
            "batch_in_epoch": 0,
            "saved_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        }
    )
    checkpoint_root = run_root / "checkpoints"
    _atomic_torch(checkpoint_root / "latest.pt", prepared)
    _atomic_torch(checkpoint_root / "best_decision_value.pt", prepared)
    lineage = {
        "schema": "dcvpp.r_ecmoe.training_continuation.v1",
        "run_id": run_id,
        "source_latest": str(source_latest_path),
        "selection_seed": str(selection_best_path),
        "source_epoch": source_epoch,
        "source_global_update": int(source_latest["global_update"]),
        "initial_best_decision_value": decision_value,
        "maximum_epoch": int(runner_config["epochs"]),
        "early_stop_patience_full_evaluations": int(
            runner_config["early_stop_patience"]
        ),
        "early_stop_min_relative_improvement": float(
            runner_config["early_stop_min_relative_improvement"]
        ),
        "full_validation_interval_epochs": int(
            runner_config["full_validation_interval_epochs"]
        ),
        "locked_2025_opened": False,
        "external_solver": False,
        "hash_computed": False,
    }
    _atomic_json(run_root / "continuation_lineage.json", lineage)
    return run_root


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = prepare(
        config_path=args.config,
        run_id=args.run_id,
        source_latest_path=args.source_latest,
        selection_best_path=args.selection_best,
    )
    print(json.dumps({"run_root": str(run_root)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
