"""Run one write-free W1/U0 CUDA update or vectorized U0 validation batch."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

from dcvpp.vpp_world_moe.config import WorldMoEConfig, load_world_moe_config
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.model import VPPWorldMoE
from dcvpp.vpp_world_moe.runner import WorldMoERunner, phase_spec_from_config
from dcvpp.vpp_world_moe.trainer import WorldModelTrainer, loss_weights_from_config
from run_vpp_world_moe import (
    _configure_torch_runtime,
    _construct_world_corpus,
    _seed_before_construction,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("w1", "u0"), default="w1")
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-gate", type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20240813)
    parser.add_argument(
        "--formal-u0-eval",
        action="store_true",
        help="Evaluate one 100-path validation batch without an optimizer update.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.formal_u0_eval and args.phase != "u0":
        raise SystemExit("--formal-u0-eval requires --phase u0")
    config, repository_root, paths = load_world_moe_config(args.config)
    _configure_torch_runtime(config, device=args.device)
    _seed_before_construction(args.seed, device=args.device)
    model_config = WorldMoEConfig.from_mapping(config["model"])
    model = VPPWorldMoE(model_config)
    corpus = _construct_world_corpus(
        WorldMoECorpus,
        config,
        repository_root,
        smoke=False,
        paths=paths,
        model_config=model_config,
    )
    with tempfile.TemporaryDirectory(prefix="dcvpp-w1-probe-") as output_root:
        runner = WorldMoERunner(
            config=config,
            model=model,
            corpus=corpus,
            output_root=output_root,
            device=args.device,
            run_id=f"write-free-{args.phase}-batch-probe",
            seed=args.seed,
        )
        parent = runner._initialize_from_parent(  # noqa: SLF001
            args.phase,
            checkpoint_path=args.parent_checkpoint,
            gate_path=args.parent_gate,
            require_release=True,
        )
        spec = replace(
            phase_spec_from_config(config, args.phase),
            batch_size=args.batch_size,
            validation_batch_size=args.batch_size,
        )
        weights = loss_weights_from_config(config)
        trainer = WorldModelTrainer(model, spec, device=args.device, weights=weights)
        split = "validation" if args.formal_u0_eval else "train"
        raw_batch = next(iter(runner._batches(split, spec, epoch=0)))  # noqa: SLF001
        batch = runner._prepare_batch(raw_batch, phase=args.phase)  # noqa: SLF001
        tracked_before = {
            "regime_head": model.dynamics.regime_head.weight.detach().cpu().clone(),
            "residual_log_scale": (
                model.dynamics.residual_log_scale.weight.detach().cpu().clone()
            ),
            "legacy_distribution_scale": (
                model.distribution_scale_head.weight.detach().cpu().clone()
            ),
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        if args.formal_u0_eval:
            result, diagnostics = trainer.evaluate_loss_and_diagnostics(
                batch,
                seed=args.seed,
                formal_seed_offsets=(0, 104729, 209759, 314159, 419431),
            )
        else:
            result = trainer.train_step(batch, seed=args.seed)
            diagnostics = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tracked_after = {
            "regime_head": model.dynamics.regime_head.weight.detach().cpu(),
            "residual_log_scale": model.dynamics.residual_log_scale.weight.detach().cpu(),
            "legacy_distribution_scale": (
                model.distribution_scale_head.weight.detach().cpu()
            ),
        }
        parameter_delta = {
            name: float((tracked_after[name] - before).abs().amax())
            for name, before in tracked_before.items()
        }
        result_loss = (
            float(result.total.detach().float().cpu())
            if args.formal_u0_eval
            else result.loss
        )
        result_components = (
            result.detached()
            if args.formal_u0_eval
            else dict(result.components)
        )
        payload = {
            "status": "PASS",
            "phase": args.phase,
            "mode": "formal_u0_eval" if args.formal_u0_eval else "train_step",
            "batch_size": args.batch_size,
            "loss": result_loss,
            "components": result_components,
            "diagnostics": dict(diagnostics),
            "gradient_norm": None if args.formal_u0_eval else result.gradient_norm,
            "mixture_nll_weight": weights.mixture_nll,
            "parameter_max_abs_delta": parameter_delta,
            "seconds": time.perf_counter() - started,
            "max_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            ),
            "max_cuda_reserved_mib": (
                torch.cuda.max_memory_reserved() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            ),
            "parent_checkpoint_sha256": parent["sha256"],
            "optimizer_parent_loaded": parent["optimizer_loaded"],
            "rng_parent_loaded": parent["rng_loaded"],
            "locked_2025_opened": False,
        }
        if args.phase == "u0" and not args.formal_u0_eval and (
            weights.mixture_nll != 0.0
            or parameter_delta["regime_head"] <= 0.0
            or parameter_delta["residual_log_scale"] <= 0.0
            or parameter_delta["legacy_distribution_scale"] != 0.0
        ):
            raise RuntimeError(
                "U0 repair smoke failed its mixture/gradient/legacy-scale contract"
            )
        if args.formal_u0_eval and (
            diagnostics.get("formal_ensemble_paths") != 100.0
            or diagnostics.get("formal_ensemble_chunks") != 5.0
            or any(value != 0.0 for value in parameter_delta.values())
        ):
            raise RuntimeError(
                "formal U0 evaluation violated its 100-path/no-update contract"
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
