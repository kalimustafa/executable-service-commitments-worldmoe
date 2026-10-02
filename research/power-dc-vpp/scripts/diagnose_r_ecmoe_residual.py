"""Print per-family residual maxima for one frozen R checkpoint/origin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_r_ecmoe import (
    _load_extended_yaml,
    _load_w1,
    _resolve,
    _runner_config,
)
from dcvpp.vpp_world_moe.config import WorldMoEConfig
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe import RECMoE, RECMoEConfig
from dcvpp.vpp_world_moe.r_ecmoe_runner import RECMoERunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--origin", type=int, default=95)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = args.config.resolve()
    repository_root = config_path.parent.parent.resolve()
    config = _load_extended_yaml(config_path)
    source = config["source"]
    world_config_path = _resolve(repository_root, source["world_config"])
    world_mapping = _load_extended_yaml(world_config_path)
    world_config = WorldMoEConfig.from_mapping(world_mapping["model"])
    rec_config = RECMoEConfig.from_mapping(dict(config["model"]))
    w1_path = _resolve(repository_root, source["w1_checkpoint"])
    world_model, _ = _load_w1(world_config, w1_path)
    model = RECMoE(world_model, rec_config)
    corpus = WorldMoECorpus(world_mapping, repository_root, weather_coverage_gate=False)
    runner = RECMoERunner(
        model,
        corpus,
        _runner_config(config),
        _resolve(repository_root, config["outputs"]["root"]),
        args.run_id,
        torch.device(args.device),
        (w1_path, config_path, world_config_path),
    )
    if not runner.resume(args.checkpoint.resolve()):
        raise FileNotFoundError(args.checkpoint)
    runner.model.eval()
    batch = next(
        iter(
            runner._batches(  # noqa: SLF001 - explicit diagnostic entrypoint
                "validation",
                epoch=runner.epoch,
                smoke=False,
                origin=int(args.origin),
                validation_scope="sentinel",
            )
        )
    )
    with torch.inference_mode():
        output, result, _ = runner._forward_loss(  # noqa: SLF001
            batch,
            source_split="validation",
            stage_id=1,
            decision_origin=int(args.origin),
            decision_horizon=int(args.horizon),
            progress=1.0,
            action_range=1.0,
            rolling_route_load=None,
            return_diagnostics=True,
        )
    payload = {
        "run_id": args.run_id,
        "epoch": runner.epoch,
        "origin": int(args.origin),
        "horizon": int(args.horizon),
        "hard_residual_max": runner._hard_residual(output),  # noqa: SLF001
        "by_family": {
            name: {
                "max_abs": float(value.detach().float().abs().amax().cpu()),
                "mean_abs": float(value.detach().float().abs().mean().cpu()),
            }
            for name, value in output.constraint_residuals.items()
        },
        "loss_constraint_residual": float(
            result.components["constraint_residual"].detach().cpu()
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
