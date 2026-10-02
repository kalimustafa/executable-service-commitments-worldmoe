from __future__ import annotations

"""Full-backbone Distribution fine-tuning from the immutable Cycle-31 parent."""

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
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcvpp.e2e_moe_r4.hashing import sha256_file  # noqa: E402
from dcvpp.e2e_moe_r4.losses import (  # noqa: E402
    empirical_coverage,
    weighted_energy_score,
    weighted_interval_score,
    weighted_marginal_crps,
    weighted_quantile,
    weighted_variogram_score,
)
from dcvpp.e2e_moe_r4_v7.config import load_v7_config  # noqa: E402
from dcvpp.e2e_moe_r4_v7.corpus import (  # noqa: E402
    PreloadedV7DevelopmentCorpus,
    materialize_dynamic_residual_anchor,
)
from dcvpp.e2e_moe_r4_v7.data import (  # noqa: E402
    issue_center_from_batch,
    load_frozen_residual_anchor,
)
from dcvpp.e2e_moe_r4_v7.preflight import fit_train_only_proper_scales  # noqa: E402
from dcvpp.e2e_moe_r4_v7.model import (  # noqa: E402
    BaselineAnchoredJointWorldPolicyMoEV7,
)
from dcvpp.e2e_moe_r4_v7.trainer import (  # noqa: E402
    ParameterPartition,
    PhaseSpec,
    V7JointTrainer,
    V7TrainingContract,
)


SCHEMA = "dcvpp.r4.v7.full-distribution-training.v1"
LOSS_WEIGHTS = {
    "center_scale": 0.10,
    "energy_score": 2.00,
    "marginal_crps": 1.00,
    "interval": 0.50,
    "tail": 0.50,
    "dependence": 0.50,
    # The N64 soft-coverage surrogate is intentionally excluded because its
    # direction disagreed with the registered N256 hard-coverage audit.
    "coverage90_hinge": 0.0,
    "coverage95_hinge": 0.0,
    "diversity": 0.02,
    "identity": 0.0,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20240813)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--microbatch-days", type=int, default=8)
    parser.add_argument("--eval-days", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2.0e-5)
    parser.add_argument("--truth-lr", type=float, default=1.0e-4)
    parser.add_argument("--run-id", default="full_distribution_r1")
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


def _one_batch(
    corpus: PreloadedV7DevelopmentCorpus,
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


def _close_identity(samples: Tensor) -> Tensor:
    return torch.cat(
        (
            samples[..., 0:4],
            samples[..., 4:8],
            samples[..., 0:4] - samples[..., 4:8],
            samples[..., 12:],
        ),
        dim=-1,
    )


def _tail(samples: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    values: list[Tensor] = []
    for quantile in (0.05, 0.95):
        forecast = weighted_quantile(samples, weights, quantile)
        error = target - forecast
        values.append(
            torch.maximum(quantile * error, (quantile - 1.0) * error)
            .flatten(1)
            .mean(1)
        )
    return torch.stack(values, dim=1).mean(1)


def _metric_vectors(samples: Tensor, weights: Tensor, target: Tensor) -> dict[str, Tensor]:
    coverage90, _ = empirical_coverage(samples, weights, target, 0.90)
    coverage95, _ = empirical_coverage(samples, weights, target, 0.95)
    return {
        "energy_score": weighted_energy_score(samples, weights, target),
        "marginal_crps": weighted_marginal_crps(samples, weights, target),
        "interval_score_90": weighted_interval_score(samples, weights, target, 0.90),
        "interval_score_95": weighted_interval_score(samples, weights, target, 0.95),
        "tail_score": _tail(samples, weights, target),
        "dependence_score": weighted_variogram_score(
            samples, weights, target, lags=(1, 4, 16), power=0.5
        ),
        "coverage_90": coverage90,
        "coverage_95": coverage95,
    }


def _evaluate(
    model: BaselineAnchoredJointWorldPolicyMoEV7,
    batch: Mapping[str, Any],
    anchor: Any,
    *,
    seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        output = model.forward_distribution(batch, sample_count=64, seed=seed)
        distribution = output["distribution"]
        samples = distribution["samples"].float()
        weights = distribution["weights"].float()
        indices = distribution["base_indices"]
        target = batch["scenario_target_zonal"].float()
        center = issue_center_from_batch(batch, anchor).to(samples).float()
        residuals = anchor.residuals.to(samples.device, dtype=torch.float32)[indices]
        baseline = _close_identity(center[:, None] + residuals)
        scale = anchor.channel_scale.to(samples.device, dtype=torch.float32)
        normalized_target = target / scale.view(1, 1, 21)
        model_metrics = _metric_vectors(
            samples / scale.view(1, 1, 1, 21), weights, normalized_target
        )
        baseline_metrics = _metric_vectors(
            baseline / scale.view(1, 1, 1, 21), weights, normalized_target
        )
        identity = (
            samples[..., 0:4] - samples[..., 4:8] - samples[..., 8:12]
        ).abs().max()
    model.train(was_training)
    result: dict[str, float] = {
        "identity_residual": float(identity.cpu()),
        "coverage90_error": abs(float(model_metrics["coverage_90"].mean().cpu()) - 0.90),
        "coverage95_error": abs(float(model_metrics["coverage_95"].mean().cpu()) - 0.95),
        "model_coverage90": float(model_metrics["coverage_90"].mean().cpu()),
        "model_coverage95": float(model_metrics["coverage_95"].mean().cpu()),
    }
    for name in (
        "energy_score",
        "marginal_crps",
        "interval_score_90",
        "interval_score_95",
        "tail_score",
        "dependence_score",
    ):
        model_value = float(model_metrics[name].mean().cpu())
        baseline_value = float(baseline_metrics[name].mean().cpu())
        result[f"model_{name}"] = model_value
        result[f"baseline_{name}"] = baseline_value
        result[f"{name}_delta"] = model_value - baseline_value
        result[f"{name}_ratio"] = model_value / baseline_value
    return result


def _parameter_count(parameters: Mapping[str, nn.Parameter], names: tuple[str, ...]) -> int:
    return sum(parameters[name].numel() for name in names)


def main() -> None:
    arguments = _arguments()
    if arguments.cycles <= 0 or arguments.microbatch_days <= 0 or arguments.eval_days <= 0:
        raise ValueError("cycles, microbatch-days and eval-days must be positive")
    run_id = str(arguments.run_id).strip()
    if not run_id or any(not (character.isalnum() or character in "-_") for character in run_id):
        raise ValueError("run-id may contain only letters, numbers, '-' and '_'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = ROOT / "configs" / "e2e_moe_dart_vpp_R4_joint_world_policy_v7.yaml"
    config, repository_root, resolved = load_v7_config(config_path)
    device = torch.device("cuda")
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    corpus = PreloadedV7DevelopmentCorpus(config, repository_root)
    train_split = config["protocol"]["splits"]["train"]
    if not resolved["residual_anchor"].is_file():
        materialize_dynamic_residual_anchor(corpus, resolved["residual_anchor"])
    anchor = load_frozen_residual_anchor(
        resolved["residual_anchor"],
        expected_start=train_split["start"],
        expected_end=train_split["end"],
        expected_days=int(train_split["days"]),
    )
    parent = resolved["warm_start_checkpoint"]
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    if (
        parent_payload.get("phase") != "d0"
        or int(parent_payload.get("cycles_completed", -1)) != 31
        or parent_payload.get("locked_2025_opened") is not False
    ):
        raise RuntimeError("immutable Cycle-31 parent is invalid")

    model = BaselineAnchoredJointWorldPolicyMoEV7(
        anchor,
        config,
        physical_twin_core=corpus.physical_core,
    )
    model.load_state_dict(parent_payload["model_state"], strict=True)
    model.to(device)
    parameters = dict(model.named_parameters())
    representation_names = model.representation_parameter_names()
    truth_names = model.distribution_truth_parameter_names()
    decision_names = model.decision_parameter_names()
    distribution_names = tuple(dict.fromkeys(representation_names + truth_names))
    allowed = set(distribution_names)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in allowed)
    distribution_optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameters[name] for name in representation_names],
                "lr": float(arguments.backbone_lr),
            },
            {
                "params": [parameters[name] for name in truth_names],
                "lr": float(arguments.truth_lr),
            },
        ],
        weight_decay=0.01,
    )
    # Required by the generic trainer contract but never stepped in this phase.
    decision_optimizer = torch.optim.AdamW(
        [parameters[name] for name in decision_names], lr=1.0e-4, weight_decay=0.01
    )
    partition = ParameterPartition(distribution_names, decision_names)
    phase = PhaseSpec(
        "distribution_full",
        minimum_cycles=1,
        maximum_cycles=int(arguments.cycles),
        distribution_updates=1,
        decision_updates=0,
        sample_count=64,
        early_stopping=False,
    )
    train_days = len(corpus.records("train"))
    contract = V7TrainingContract((phase,), train_days=train_days)
    proper_scales = fit_train_only_proper_scales(anchor, device=device)
    trainer = V7JointTrainer(
        model,
        {"distribution": distribution_optimizer, "decision": decision_optimizer},
        baseline_proper_scales=proper_scales,
        cost_scale=corpus.fixed_training_cost_scale,
        channel_scale=anchor.channel_scale.to(device),
        partition=partition,
        contract=contract,
        distribution_loss_weights=LOSS_WEIGHTS,
        gradient_clip_norm=1.0,
        distribution_microbatch_days=int(arguments.microbatch_days),
        decision_autocast_dtype=None,
        path_seed=arguments.seed,
    )

    train_batch = _one_batch(
        corpus, "train", train_days, device=device, seed=None
    )
    train_batch["channel_scale"] = anchor.channel_scale.to(device)
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
        / "e2e_moe_joint_world_policy_r4_v7_2025final_diagnostic"
        / f"seed={arguments.seed}"
        / "phase=distribution_full"
        / f"run={run_id}"
    )
    before = _evaluate(
        model, evaluation_batch, anchor, seed=arguments.seed + 9001
    )
    history: list[dict[str, Any]] = []
    for cycle in range(int(arguments.cycles)):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        results = trainer.run_coverage_cycle(
            train_batch,
            phase="distribution_full",
            coverage_cycle_index=cycle,
            seed=arguments.seed,
            shuffle=True,
        )
        torch.cuda.synchronize(device)
        validation = _evaluate(
            model,
            evaluation_batch,
            anchor,
            seed=arguments.seed + 9001,
        )
        record = {
            "cycle": cycle + 1,
            "optimizer_updates": 4,
            "coverage_days": train_days,
            "losses": [result.loss for result in results],
            "metrics": [result.metrics for result in results],
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        history.append(record)
        checkpoint = {
            "schema": SCHEMA,
            "status": "TRAINING" if cycle + 1 < arguments.cycles else "TRAINED",
            "architecture_version": model.architecture_version,
            "phase": "distribution_full",
            "seed": arguments.seed,
            "run_id": run_id,
            "cycles_completed": cycle + 1,
            "optimizer_updates_completed": (cycle + 1) * 4,
            "model_state": model.state_dict(),
            "distribution_optimizer_state": distribution_optimizer.state_dict(),
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": sha256_file(parent),
            "representation_parameter_tensors": len(representation_names),
            "truth_parameter_tensors": len(truth_names),
            "representation_parameter_count": _parameter_count(
                parameters, representation_names
            ),
            "truth_parameter_count": _parameter_count(parameters, truth_names),
            "decision_and_policy_frozen": True,
            "physical_core_frozen": True,
            "formal_mainline_released": False,
            "diagnostic_only": True,
            "loss_weights": LOSS_WEIGHTS,
            "history": history,
            "development_validation_2024_opened": True,
            "final_test_2025_opened": False,
            "locked_2025_opened": False,
            "updated_at_utc": _utc_now(),
        }
        cycle_path = output_root / "checkpoints" / f"cycle={cycle + 1:04d}.pt"
        latest_path = output_root / "checkpoints" / "latest.pt"
        _atomic_torch(cycle_path, checkpoint)
        _atomic_torch(latest_path, checkpoint)
        _atomic_json(
            output_root / "state.json",
            {
                key: value
                for key, value in checkpoint.items()
                if key not in {"model_state", "distribution_optimizer_state"}
            }
            | {"checkpoint": str(latest_path)},
        )

    after = history[-1]["validation"]
    summary = {
        "schema": SCHEMA,
        "status": "COMPLETE_DIAGNOSTIC",
        "phase": "distribution_full",
        "seed": arguments.seed,
        "run_id": run_id,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_file(parent),
        "train_days_per_cycle": train_days,
        "train_window": [train_split["start"], train_split["end"]],
        "development_validation_window": ["2024-01-01", "2024-12-31"],
        "development_split_lineage": dict(corpus.split_lineage),
        "cycles": int(arguments.cycles),
        "representation_parameter_tensors": len(representation_names),
        "truth_parameter_tensors": len(truth_names),
        "representation_parameter_count": _parameter_count(
            parameters, representation_names
        ),
        "truth_parameter_count": _parameter_count(parameters, truth_names),
        "decision_and_policy_frozen": True,
        "physical_core_frozen": True,
        "evaluation_before": before,
        "evaluation_after": after,
        "change": {
            key: float(after[key]) - float(before[key])
            for key in (
                "energy_score_delta",
                "marginal_crps_ratio",
                "interval_score_90_ratio",
                "interval_score_95_ratio",
                "tail_score_ratio",
                "dependence_score_ratio",
                "coverage90_error",
                "coverage95_error",
            )
        },
        "history": history,
        "diagnostic_only": True,
        "formal_mainline_released": False,
        "mae_used": False,
        "development_validation_2024_opened": True,
        "final_test_2025_opened": False,
        "locked_2025_opened": False,
        "checkpoint": str(output_root / "checkpoints" / "latest.pt"),
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(output_root / "summary.json", summary)
    _atomic_json(output_root / "state.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
