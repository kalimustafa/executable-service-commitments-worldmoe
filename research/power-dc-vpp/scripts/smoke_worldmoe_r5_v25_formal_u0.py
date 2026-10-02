"""Run one real 100-path formal U0 validation batch for R5-V2.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset  # noqa: F401

from run_vpp_world_moe import _construct_world_corpus, _seed_before_construction
from dcvpp.vpp_world_moe.config import WorldMoEConfig, load_world_moe_config
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.model import VPPWorldMoE
from dcvpp.vpp_world_moe.runner import (
    U0_FORMAL_SEED_OFFSETS,
    WorldMoERunner,
    phase_spec_from_config,
)
from dcvpp.vpp_world_moe.trainer import WorldModelTrainer, loss_weights_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-gate", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20240813)
    args = parser.parse_args()

    config, repository_root, paths = load_world_moe_config(args.config)
    _seed_before_construction(args.seed, device=args.device)
    model = VPPWorldMoE(WorldMoEConfig.from_mapping(config["model"]))
    corpus = _construct_world_corpus(
        WorldMoECorpus, config, repository_root, smoke=False
    )
    runner = WorldMoERunner(
        config=config,
        model=model,
        corpus=corpus,
        output_root=paths["output_root"],
        device=args.device,
        run_id="r5v25-formal-u0-one-batch-smoke",
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
    raw = next(runner._batches("validation", spec, epoch=0))
    batch = runner._prepare_batch(raw, phase="u0")
    loss, diagnostics = trainer.evaluate_loss_and_diagnostics(
        batch,
        seed=args.seed + 10_000,
        formal_seed_offsets=U0_FORMAL_SEED_OFFSETS,
    )
    result = {
        "schema": "dcvpp.vpp-world-moe.r5.v2.5.formal-u0-smoke.v1",
        "parent_sha256": parent["sha256"],
        "loss": float(loss.total.detach().cpu()),
        "formal_paths": diagnostics.get("formal_ensemble_paths"),
        "analytic_distribution_coverage": diagnostics.get(
            "analytic_distribution_coverage"
        ),
        "pv_coverage_valid": diagnostics.get("coverage_pv_90_valid_count"),
        "pv_coverage_insufficient": diagnostics.get(
            "coverage_pv_90_insufficient_count"
        ),
        "dart_identity_max_abs": diagnostics.get("dart_identity_max_abs"),
        "locked_2025_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
