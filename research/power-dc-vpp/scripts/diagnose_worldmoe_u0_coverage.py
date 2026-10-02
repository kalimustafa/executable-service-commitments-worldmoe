"""Read-only family coverage diagnostics for a frozen WorldMoE U0 checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.dataset  # noqa: F401
import torch

from run_vpp_world_moe import (
    _construct_world_corpus,
    _seed_before_construction,
)
from dcvpp.vpp_world_moe.config import WorldMoEConfig, load_world_moe_config
from dcvpp.vpp_world_moe.data import WEATHER_SLOT, WorldMoECorpus
from dcvpp.vpp_world_moe.model import VPPWorldMoE
from dcvpp.vpp_world_moe.runner import WorldMoERunner, phase_spec_from_config
from dcvpp.vpp_world_moe.trainer import WorldModelTrainer


GROUPS: Mapping[str, slice] = {
    "da": slice(0, 4),
    "rt": slice(4, 8),
    "dart": slice(8, 12),
    "workload": slice(12, 13),
    "pv": slice(13, 17),
    "pue": slice(17, 21),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20240813)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config, repository_root, paths = load_world_moe_config(args.config)
    _seed_before_construction(args.seed, device=args.device)
    model_config = WorldMoEConfig.from_mapping(config["model"])
    model = VPPWorldMoE(model_config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    corpus = _construct_world_corpus(
        WorldMoECorpus,
        config,
        repository_root,
        smoke=False,
        paths=paths,
        model_config=model_config,
    )
    runner = WorldMoERunner(
        config=config,
        model=model,
        corpus=corpus,
        output_root=paths["output_root"],
        device=args.device,
        run_id=args.run_id,
        seed=args.seed,
    )
    spec = phase_spec_from_config(config, "u0")
    trainer = WorldModelTrainer(model, spec, device=args.device)
    model.eval()

    levels = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}
    totals: dict[str, dict[str, float]] = {
        name: {
            "count": 0.0,
            "mae_sum": 0.0,
            "bias_sum": 0.0,
            "spread_sum": 0.0,
            "boundary_count": 0.0,
            "interior_count": 0.0,
            "scenario_boundary_count": 0.0,
            "scenario_boundary_total": 0.0,
            **{f"coverage_{nominal}_sum": 0.0 for nominal in levels},
            **{f"width_{nominal}_sum": 0.0 for nominal in levels},
            **{f"boundary_coverage_{nominal}_sum": 0.0 for nominal in levels},
            **{f"interior_coverage_{nominal}_sum": 0.0 for nominal in levels},
        }
        for name in GROUPS
    }
    pv_support = {
        "valid_count": 0.0,
        "inactive_count": 0.0,
        "active_count": 0.0,
        "inactive_target_zero_count": 0.0,
        "active_target_zero_count": 0.0,
        "support_match_count": 0.0,
    }
    batches = 0
    with torch.inference_mode():
        for index, raw_batch in enumerate(
            runner._batches("validation", spec, epoch=0)
        ):
            batch = runner._prepare_batch(raw_batch, phase="u0")
            output = trainer.forward(batch, seed=args.seed + 10_000 + index)
            scenarios = output["scenarios"].float()
            target = batch["stochastic_target"].float()
            packet = batch["world_input"]
            target_pv_zero = target[..., GROUPS["pv"]].abs() <= 1.0e-8
            weather_valid = packet.fast_mask[..., WEATHER_SLOT]
            causal_inactive = (
                packet.fast_values[..., WEATHER_SLOT, 4:8] <= 1.0e-8
            ) & weather_valid[..., None]
            causal_active = (~causal_inactive) & weather_valid[..., None]
            valid_support = weather_valid[..., None].expand_as(target_pv_zero)
            pv_support["valid_count"] += float(valid_support.sum().cpu())
            pv_support["inactive_count"] += float(causal_inactive.sum().cpu())
            pv_support["active_count"] += float(causal_active.sum().cpu())
            pv_support["inactive_target_zero_count"] += float(
                (causal_inactive & target_pv_zero).sum().cpu()
            )
            pv_support["active_target_zero_count"] += float(
                (causal_active & target_pv_zero).sum().cpu()
            )
            pv_support["support_match_count"] += float(
                ((causal_inactive == target_pv_zero) & valid_support).sum().cpu()
            )
            for name, channel_slice in GROUPS.items():
                samples = scenarios[..., channel_slice]
                truth = target[..., channel_slice]
                mean = samples.mean(dim=1)
                count = float(truth.numel())
                item = totals[name]
                item["count"] += count
                item["mae_sum"] += float((mean - truth).abs().sum().cpu())
                item["bias_sum"] += float((mean - truth).sum().cpu())
                item["spread_sum"] += float(
                    samples.std(dim=1, unbiased=False).sum().cpu()
                )
                boundary = None
                scenario_boundary = None
                if name in {"workload", "pv"}:
                    boundary = truth.abs() <= 1.0e-8
                    scenario_boundary = samples.abs() <= 1.0e-8
                elif name == "pue":
                    lower = packet.constraints.pue_min[:, None, :].float()
                    boundary = (truth - lower).abs() <= 1.0e-6
                    scenario_boundary = (
                        samples - lower[:, None]
                    ).abs() <= 1.0e-6
                if boundary is not None and scenario_boundary is not None:
                    item["boundary_count"] += float(boundary.sum().cpu())
                    item["interior_count"] += float((~boundary).sum().cpu())
                    item["scenario_boundary_count"] += float(
                        scenario_boundary.sum().cpu()
                    )
                    item["scenario_boundary_total"] += float(
                        scenario_boundary.numel()
                    )
                for nominal, (lower, upper) in levels.items():
                    low = torch.quantile(samples, lower, dim=1)
                    high = torch.quantile(samples, upper, dim=1)
                    hits = (truth >= low) & (truth <= high)
                    item[f"coverage_{nominal}_sum"] += float(hits.sum().cpu())
                    item[f"width_{nominal}_sum"] += float((high - low).sum().cpu())
                    if boundary is not None:
                        item[f"boundary_coverage_{nominal}_sum"] += float(
                            (hits & boundary).sum().cpu()
                        )
                        item[f"interior_coverage_{nominal}_sum"] += float(
                            (hits & ~boundary).sum().cpu()
                        )
            batches += 1
            if (index + 1) % 20 == 0:
                print(json.dumps({"completed_batches": index + 1}), flush=True)

    result: dict[str, Any] = {
        "schema": "dcvpp.worldmoe.u0-family-coverage-diagnostic.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "source_epoch": payload.get("resume_state", {}).get("epoch"),
        "validation_batches": batches,
        "groups": {},
        "causal_pv_support": {
            **pv_support,
            "inactive_fraction": (
                pv_support["inactive_count"] / pv_support["valid_count"]
            ),
            "target_zero_given_inactive": (
                pv_support["inactive_target_zero_count"]
                / max(pv_support["inactive_count"], 1.0)
            ),
            "target_zero_given_active": (
                pv_support["active_target_zero_count"]
                / max(pv_support["active_count"], 1.0)
            ),
            "support_match_rate": (
                pv_support["support_match_count"] / pv_support["valid_count"]
            ),
        },
    }
    for name, item in totals.items():
        count = item["count"]
        result["groups"][name] = {
            "count": int(count),
            "mean_absolute_error_raw": item["mae_sum"] / count,
            "mean_signed_bias_raw": item["bias_sum"] / count,
            "mean_scenario_std_raw": item["spread_sum"] / count,
            **{
                f"coverage_{nominal}": item[f"coverage_{nominal}_sum"] / count
                for nominal in levels
            },
            **{
                f"mean_width_{nominal}_raw": item[f"width_{nominal}_sum"] / count
                for nominal in levels
            },
            "target_boundary_fraction": item["boundary_count"] / max(count, 1.0),
            "scenario_boundary_fraction": (
                item["scenario_boundary_count"]
                / max(item["scenario_boundary_total"], 1.0)
            ),
            **{
                f"boundary_coverage_{nominal}": (
                    item[f"boundary_coverage_{nominal}_sum"]
                    / max(item["boundary_count"], 1.0)
                )
                for nominal in levels
            },
            **{
                f"interior_coverage_{nominal}": (
                    item[f"interior_coverage_{nominal}_sum"]
                    / max(item["interior_count"], 1.0)
                )
                for nominal in levels
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
