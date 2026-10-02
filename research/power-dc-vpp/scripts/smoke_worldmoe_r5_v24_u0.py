"""Run one real CUDA U0 update from the registered R5-V2.4 W1 parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset  # noqa: F401
import torch

from run_vpp_world_moe import _construct_world_corpus, _seed_before_construction
from dcvpp.vpp_world_moe.config import WorldMoEConfig, load_world_moe_config
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.model import VPPWorldMoE
from dcvpp.vpp_world_moe.runner import WorldMoERunner, phase_spec_from_config
from dcvpp.vpp_world_moe.trainer import WorldModelTrainer, loss_weights_from_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-gate", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20240813)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config, repository_root, paths = load_world_moe_config(args.config)
    _seed_before_construction(args.seed, device=args.device)
    model = VPPWorldMoE(WorldMoEConfig.from_mapping(config["model"]))
    corpus = _construct_world_corpus(
        WorldMoECorpus,
        config,
        repository_root,
        smoke=False,
    )
    runner = WorldMoERunner(
        config=config,
        model=model,
        corpus=corpus,
        output_root=paths["output_root"],
        device=args.device,
        run_id="r5v24-u0-cuda-one-update-smoke",
        seed=args.seed,
    )
    parent = runner._initialize_from_parent(
        "u0",
        checkpoint_path=args.parent_checkpoint,
        gate_path=args.parent_gate,
        require_release=True,
    )
    spec = phase_spec_from_config(config, "u0")
    trainer = WorldModelTrainer(
        model,
        spec,
        device=args.device,
        weights=loss_weights_from_config(config),
    )
    raw_train = next(runner._batches("train", spec, epoch=1))
    train_batch = runner._prepare_batch(raw_train, phase="u0")
    step = trainer.train_step(train_batch, seed=args.seed + 1_000_003)

    raw_validation = next(runner._batches("validation", spec, epoch=0))
    validation_batch = runner._prepare_batch(raw_validation, phase="u0")
    validation_loss, diagnostics = trainer.evaluate_loss_and_diagnostics(
        validation_batch,
        seed=args.seed + 2_000_003,
    )
    result = {
        "schema": "dcvpp.vpp-world-moe.r5.v2.4.cuda-one-update-smoke.v1",
        "parent_sha256": parent["sha256"],
        "parent_epoch": parent["source_epoch"],
        "parent_update": parent["source_update"],
        "parent_gate_released": parent["gate"]["release_verified"],
        "support_upgrade_missing_keys": len(
            parent["model_upgrade"]["support_atom_missing_keys"]
        ),
        "support_prior_initialization_preserved": parent["model_upgrade"][
            "support_atom_train_prior_initialization_preserved"
        ],
        "train_loss": step.loss,
        "train_gradient_norm": step.gradient_norm,
        "train_atom_brier": step.components.get("atom_brier"),
        "validation_loss": float(validation_loss.total.detach().cpu()),
        "validation_atom_brier": float(
            validation_loss.components["atom_brier"].detach().cpu()
        ),
        "dart_identity_max_abs": diagnostics.get("dart_identity_max_abs"),
        "updated_groups": list(step.updated_groups),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if str(args.device).lower().startswith("cuda")
            else 0
        ),
        "locked_2025_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
