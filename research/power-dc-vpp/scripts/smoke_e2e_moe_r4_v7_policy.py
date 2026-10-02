from __future__ import annotations

"""One real-data policy update without mutating the formal V7 lineage."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcvpp.e2e_moe_r4.training_data import PreloadedR4Corpus  # noqa: E402
from dcvpp.e2e_moe_r4_v7.config import load_v7_config  # noqa: E402
from dcvpp.e2e_moe_r4_v7.data import load_frozen_residual_anchor  # noqa: E402
from dcvpp.e2e_moe_r4_v7.model import (  # noqa: E402
    BaselineAnchoredJointWorldPolicyMoEV7,
)
from dcvpp.e2e_moe_r4_v7.rollout import make_training_rollout  # noqa: E402
from dcvpp.e2e_moe_r4_v7.trainer import (  # noqa: E402
    V7JointTrainer,
    build_v7_optimizers,
    training_contract_from_config,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--optimizer-update-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.days <= 0:
        raise ValueError("--days must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config_path = ROOT / "configs" / "e2e_moe_dart_vpp_R4_joint_world_policy_v7.yaml"
    config, repository_root, resolved = load_v7_config(config_path)
    device = torch.device("cuda")
    anchor = load_frozen_residual_anchor(resolved["residual_anchor"])
    corpus = PreloadedR4Corpus(config, repository_root)
    model = BaselineAnchoredJointWorldPolicyMoEV7(
        anchor,
        config,
        physical_twin_core=corpus.physical_core,
    )
    checkpoint_path = (
        resolved["output_root"]
        / "seed=20240813"
        / "phase=d0"
        / "checkpoints"
        / "best_cheap.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    training = config["training"]
    optimizers, partition = build_v7_optimizers(
        model,
        distribution_learning_rate=float(training["distribution_learning_rate"]),
        decision_learning_rate=float(training["decision_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch = dict(
        next(
            corpus.batches(
                "train",
                arguments.days,
                device=device,
                max_days=arguments.days,
                origin=0,
                shuffle_seed=20240813,
            )
        )
    )
    rollout_targets = batch["rollout_targets"]
    batch["safe_anchor_actions"] = rollout_targets["baseline_actions"]
    batch["channel_scale"] = anchor.channel_scale.to(device)
    trainer = V7JointTrainer(
        model,
        optimizers,
        baseline_proper_scales=checkpoint["proper_scales"],
        cost_scale=corpus.fixed_training_cost_scale,
        channel_scale=anchor.channel_scale.to(device),
        partition=partition,
        rollout_fn=make_training_rollout(model, phase="p0", origin_budget=8),
        contract=training_contract_from_config(config),
        distribution_loss_weights=training["distribution_loss_weights"],
        policy_loss_weights=training["policy_loss_weights"],
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = trainer.update(
        batch,
        phase="p0",
        optimizer_update_index=arguments.optimizer_update_index,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "status": "PASSED",
                "days": arguments.days,
                "optimizer_update_index": arguments.optimizer_update_index,
                "phase": result.phase,
                "update_kind": result.update_kind,
                "loss": result.loss,
                "metrics": result.metrics,
                "elapsed_seconds": elapsed,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "locked_2025_opened": False,
                "checkpoint_read_only": str(checkpoint_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
