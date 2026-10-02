from __future__ import annotations

"""One real-data, one-update smoke check for R4-V7 before formal D0."""

import json
from pathlib import Path
import sys

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
from dcvpp.e2e_moe_r4_v7.preflight import fit_train_only_proper_scales  # noqa: E402
from dcvpp.e2e_moe_r4_v7.trainer import (  # noqa: E402
    V7JointTrainer,
    build_v7_optimizers,
    training_contract_from_config,
)


def main() -> None:
    config_path = ROOT / "configs" / "e2e_moe_dart_vpp_R4_joint_world_policy_v7.yaml"
    config, root, resolved = load_v7_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    anchor = load_frozen_residual_anchor(resolved["residual_anchor"])
    corpus = PreloadedR4Corpus(config, root)
    model = BaselineAnchoredJointWorldPolicyMoEV7(
        anchor,
        config,
        physical_twin_core=corpus.physical_core,
    )
    predecessor = model.load_representation_predecessor(
        resolved["representation_checkpoint"], expected_epoch=28
    )
    model.to(device)
    training = config["training"]
    optimizers, partition = build_v7_optimizers(
        model,
        distribution_learning_rate=float(training["distribution_learning_rate"]),
        decision_learning_rate=float(training["decision_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scales = fit_train_only_proper_scales(anchor, device=device)
    trainer = V7JointTrainer(
        model,
        optimizers,
        baseline_proper_scales=scales,
        cost_scale=corpus.fixed_training_cost_scale,
        channel_scale=anchor.channel_scale.to(device),
        partition=partition,
        contract=training_contract_from_config(config),
    )
    batch = next(
        corpus.batches(
            "train", 2, device=device, max_days=2, origin=0, shuffle_seed=20240813
        )
    )
    batch["channel_scale"] = anchor.channel_scale.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = trainer.update(batch, phase="d0", optimizer_update_index=0)
    payload = {
        "status": "PASSED",
        "phase": result.phase,
        "update_kind": result.update_kind,
        "loss": result.loss,
        "metrics": result.metrics,
        "parameter_count": model.parameter_count,
        "truth_parameter_count": len(partition.distribution_truth),
        "decision_parameter_count": len(partition.decision),
        "representation_loaded_keys": len(predecessor["loaded_keys"]),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "locked_2025_opened": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
