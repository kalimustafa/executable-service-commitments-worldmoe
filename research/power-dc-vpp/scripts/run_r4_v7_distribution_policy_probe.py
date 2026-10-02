from __future__ import annotations

"""Train a diagnostic Policy on the frozen Cycle-31 V7 Distribution."""

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcvpp.e2e_moe_r4.hashing import sha256_file  # noqa: E402
from dcvpp.e2e_moe_r4.losses import normalize_weights, weighted_cvar  # noqa: E402
from dcvpp.e2e_moe_r4.training_data import PreloadedR4Corpus  # noqa: E402
from dcvpp.e2e_moe_r4_v7.config import load_v7_config  # noqa: E402
from dcvpp.e2e_moe_r4_v7.data import load_frozen_residual_anchor  # noqa: E402
from dcvpp.e2e_moe_r4_v7.model import (  # noqa: E402
    BaselineAnchoredJointWorldPolicyMoEV7,
)
from dcvpp.e2e_moe_r4_v7.rollout import (  # noqa: E402
    differentiable_training_rollout,
    make_training_rollout,
)
from dcvpp.e2e_moe_r4_v7.trainer import (  # noqa: E402
    V7JointTrainer,
    build_v7_optimizers,
    training_contract_from_config,
)


PROBE_SCHEMA = "dcvpp.r4.v7.distribution-policy-probe.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20240813)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--train-days-per-update", type=int, default=8)
    parser.add_argument("--eval-days", type=int, default=2)
    parser.add_argument("--origin-budget", type=int, default=8)
    parser.add_argument("--run-id", default="pilot")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _slice_days(value: Any, indices: Tensor, total_days: int) -> Any:
    if isinstance(value, Tensor):
        if value.ndim >= 1 and value.shape[0] == total_days:
            return value.index_select(0, indices.to(value.device))
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_days(item, indices, total_days)
            for key, item in value.items()
        }
    if isinstance(value, list) and len(value) == total_days:
        return [value[int(index)] for index in indices.tolist()]
    if isinstance(value, tuple) and len(value) == total_days:
        return tuple(value[int(index)] for index in indices.tolist())
    return value


def _one_batch(
    corpus: PreloadedR4Corpus,
    split: str,
    days: int,
    *,
    device: torch.device,
    seed: int | None,
) -> dict[str, Any]:
    return dict(
        next(
            corpus.batches(
                split,
                days,
                device=device,
                max_days=days,
                origin=0,
                shuffle_seed=seed,
            )
        )
    )


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (normalize_weights(weights).to(values) * values).sum(1)


def _evaluate(
    model: BaselineAnchoredJointWorldPolicyMoEV7,
    batch: Mapping[str, Any],
    *,
    seed: int,
    origin_budget: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        distribution = model.forward_distribution(
            batch, sample_count=64, seed=int(seed)
        )["distribution"]
        rollout = differentiable_training_rollout(
            model,
            batch,
            distribution,
            horizon=96,
            phase="p0_distribution_probe_eval",
            origin_budget=origin_budget,
        )
        costs = rollout["world_costs"].float()
        weights = rollout["weights"].float()
        expected = _weighted_mean(costs, weights)
        cvar = weighted_cvar(costs, weights, alpha=0.95)
        normalized_sla = _weighted_mean(
            rollout["normalized_sla"].float(), weights
        )
        intervention = _weighted_mean(
            rollout["shield_intervention"].float(), weights
        )
        feasible = (
            normalize_weights(weights).to(costs)
            * rollout["feasible"].float()
        ).sum(1)
        baseline_costs = rollout.get("baseline_world_costs")
        if isinstance(baseline_costs, Tensor):
            baseline_costs = baseline_costs.float()
            baseline_expected = _weighted_mean(baseline_costs, weights)
            baseline_cvar = weighted_cvar(baseline_costs, weights, alpha=0.95)
        else:
            baseline_expected = rollout["baseline_cost"].float()
            baseline_cvar = rollout["baseline_cvar"].float()
    if was_training:
        model.train()
    elapsed = time.perf_counter() - started
    return {
        "days": int(costs.shape[0]),
        "worlds": int(costs.shape[1]),
        "horizon": 96,
        "expected_cost": float(expected.mean().cpu()),
        "baseline_expected_cost": float(baseline_expected.mean().cpu()),
        "expected_cost_delta": float((expected - baseline_expected).mean().cpu()),
        "cvar95": float(cvar.mean().cpu()),
        "baseline_cvar95": float(baseline_cvar.mean().cpu()),
        "cvar95_delta": float((cvar - baseline_cvar).mean().cpu()),
        "normalized_sla": float(normalized_sla.mean().cpu()),
        "feasibility_fraction": float(feasible.mean().cpu()),
        "shield_intervention": float(intervention.mean().cpu()),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    arguments = _arguments()
    updates = int(arguments.cycles) * 4
    total_days = updates * int(arguments.train_days_per_update)
    if arguments.cycles <= 0 or not 1 <= total_days <= 543:
        raise ValueError("probe training must use between 1 and 543 train days")
    if arguments.eval_days <= 0:
        raise ValueError("eval-days must be positive")
    run_id = str(arguments.run_id).strip()
    if not run_id or any(not (character.isalnum() or character in "-_") for character in run_id):
        raise ValueError("run-id may contain only letters, numbers, '-' and '_'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Policy probe")

    config_path = ROOT / "configs" / "e2e_moe_dart_vpp_R4_joint_world_policy_v7.yaml"
    config, repository_root, resolved = load_v7_config(config_path)
    device = torch.device("cuda")
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    anchor = load_frozen_residual_anchor(resolved["residual_anchor"])
    corpus = PreloadedR4Corpus(config, repository_root)
    parent = (
        resolved["output_root"]
        / f"seed={arguments.seed}"
        / "phase=d0"
        / "checkpoints"
        / "candidates"
        / "cycle=0031.pt"
    )
    if not parent.is_file():
        raise FileNotFoundError(f"immutable Cycle-31 checkpoint is missing: {parent}")
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    if (
        parent_payload.get("phase") != "d0"
        or int(parent_payload.get("cycles_completed", -1)) != 31
        or parent_payload.get("locked_2025_opened") is not False
    ):
        raise RuntimeError("Cycle-31 parent lineage is invalid")

    model = BaselineAnchoredJointWorldPolicyMoEV7(
        anchor,
        config,
        physical_twin_core=corpus.physical_core,
    )
    model.load_state_dict(parent_payload["model_state"], strict=True)
    model.to(device)
    training = config["training"]
    optimizers, partition = build_v7_optimizers(
        model,
        distribution_learning_rate=float(training["distribution_learning_rate"]),
        decision_learning_rate=float(training["decision_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    parameters = dict(model.named_parameters())
    for name in partition.distribution_truth:
        parameters[name].requires_grad_(False)
    trainer = V7JointTrainer(
        model,
        optimizers,
        baseline_proper_scales=parent_payload["proper_scales"],
        cost_scale=corpus.fixed_training_cost_scale,
        channel_scale=anchor.channel_scale.to(device),
        partition=partition,
        rollout_fn=make_training_rollout(
            model,
            phase="p0_distribution_probe",
            origin_budget=int(arguments.origin_budget),
        ),
        contract=training_contract_from_config(config),
        distribution_loss_weights=training["distribution_loss_weights"],
        policy_loss_weights=training["policy_loss_weights"],
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        decision_microbatch_days=1,
        # Keep the analytic shield, physical state and risk arithmetic in FP32.
        decision_autocast_dtype=None,
        path_seed=arguments.seed,
    )

    train_batch = _one_batch(
        corpus,
        "train",
        total_days,
        device=device,
        seed=arguments.seed,
    )
    train_batch["channel_scale"] = anchor.channel_scale.to(device)
    train_batch["safe_anchor_actions"] = train_batch["rollout_targets"][
        "baseline_actions"
    ]
    evaluation_batch = _one_batch(
        corpus,
        "validation",
        int(arguments.eval_days),
        device=device,
        seed=None,
    )
    evaluation_batch["channel_scale"] = anchor.channel_scale.to(device)

    output_root = (
        repository_root
        / "artifacts"
        / "e2e_moe_joint_world_policy_r4_v7_diagnostic"
        / f"seed={arguments.seed}"
        / "phase=p0_distribution_probe"
        / f"run={run_id}"
    )
    checkpoint_path = output_root / "checkpoints" / "latest.pt"
    state_path = output_root / "state.json"
    before = _evaluate(
        model,
        evaluation_batch,
        seed=arguments.seed + 7001,
        origin_budget=int(arguments.origin_budget),
    )
    history: list[dict[str, Any]] = []
    for update_index in range(updates):
        start = update_index * int(arguments.train_days_per_update)
        stop = start + int(arguments.train_days_per_update)
        piece = _slice_days(
            train_batch, torch.arange(start, stop), total_days
        )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = trainer.update(
            piece,
            phase="p0",
            optimizer_update_index=update_index,
        )
        torch.cuda.synchronize(device)
        record = {
            "update": update_index + 1,
            "decision_horizon": (8, 24, 48, 96)[update_index % 4],
            "train_days": int(arguments.train_days_per_update),
            "loss": result.loss,
            "metrics": result.metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        history.append(record)
        checkpoint = {
            "schema": PROBE_SCHEMA,
            "status": "TRAINING" if update_index + 1 < updates else "TRAINED",
            "architecture_version": model.architecture_version,
            "phase": "p0_distribution_probe",
            "seed": arguments.seed,
            "run_id": run_id,
            "updates_completed": update_index + 1,
            "model_state": model.state_dict(),
            "decision_optimizer_state": optimizers["decision"].state_dict(),
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": sha256_file(parent),
            "distribution_truth_frozen": True,
            "formal_mainline_released": False,
            "diagnostic_only": True,
            "locked_2025_opened": False,
            "history": history,
            "updated_at_utc": _utc_now(),
        }
        _atomic_torch(checkpoint_path, checkpoint)
        _atomic_json(
            state_path,
            {key: value for key, value in checkpoint.items() if not key.endswith("state")}
            | {"checkpoint": str(checkpoint_path), "evaluation_before": before},
        )

    after = _evaluate(
        model,
        evaluation_batch,
        seed=arguments.seed + 7001,
        origin_budget=int(arguments.origin_budget),
    )
    summary = {
        "schema": PROBE_SCHEMA,
        "status": "COMPLETE_DIAGNOSTIC",
        "diagnostic_only": True,
        "formal_mainline_released": False,
        "phase": "p0_distribution_probe",
        "seed": arguments.seed,
        "run_id": run_id,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_file(parent),
        "distribution_truth_frozen": True,
        "train_days": total_days,
        "updates": updates,
        "history": history,
        "evaluation_before": before,
        "evaluation_after": after,
        "policy_change": {
            key: float(after[key]) - float(before[key])
            for key in (
                "expected_cost",
                "expected_cost_delta",
                "cvar95",
                "cvar95_delta",
                "normalized_sla",
                "feasibility_fraction",
                "shield_intervention",
            )
        },
        "mae_used": False,
        "test_2024_opened": False,
        "locked_2025_opened": False,
        "checkpoint": str(checkpoint_path),
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(output_root / "summary.json", summary)
    _atomic_json(state_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
