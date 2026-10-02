"""Freeze train-only support-atom statistics for WorldMoE probabilistic heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset  # noqa: F401
import torch

from run_vpp_world_moe import _construct_world_corpus
from dcvpp.vpp_world_moe.config import load_world_moe_config
from dcvpp.vpp_world_moe.data import WEATHER_SLOT, WorldMoECorpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise RuntimeError("support-atom statistic has an empty denominator")
    return float(numerator / denominator)


def main() -> int:
    args = _parser().parse_args()
    config, repository_root, _ = load_world_moe_config(args.config)
    corpus = _construct_world_corpus(
        WorldMoECorpus,
        config,
        repository_root,
        smoke=False,
    )

    counts = {
        "workload_total": 0,
        "workload_zero": 0,
        "pv_total": 0,
        "pv_zero": 0,
        "pv_capacity": 0,
        "pv_daylight_total": 0,
        "pv_daylight_zero": 0,
        "pv_daylight_capacity": 0,
        "pv_night_total": 0,
        "pv_night_zero": 0,
        "pue_total": 0,
        "pue_minimum": 0,
        "pue_maximum": 0,
        "pue_nonzero_workload_total": 0,
        "pue_residual_minimum": 0,
        "pue_residual_maximum": 0,
        "pue_zero_workload_total": 0,
        "pue_zero_workload_maximum": 0,
    }
    dates: list[str] = []
    for batch in corpus.batches("train", 32, device="cpu"):
        target = batch["stochastic_target"].float()
        packet = batch["world_input"]
        dates.extend(str(value) for value in batch["operating_dates"])

        workload = target[..., 12:13]
        workload_zero = workload.abs().le(1.0e-8)
        counts["workload_total"] += workload.numel()
        counts["workload_zero"] += int(workload_zero.sum())

        pv = target[..., 13:17]
        pv_capacity = packet.constraints.pv_capacity[:, None, :].float()
        pv_zero = pv.abs().le(1.0e-8)
        pv_at_capacity = (pv - pv_capacity).abs().le(1.0e-6)
        weather_valid = packet.fast_mask[..., WEATHER_SLOT, None]
        daylight = (
            packet.fast_values[..., WEATHER_SLOT, 4:8].gt(1.0e-4)
            & weather_valid
        )
        night = (~daylight) & weather_valid
        counts["pv_total"] += pv.numel()
        counts["pv_zero"] += int(pv_zero.sum())
        counts["pv_capacity"] += int(pv_at_capacity.sum())
        counts["pv_daylight_total"] += int(daylight.sum())
        counts["pv_daylight_zero"] += int((daylight & pv_zero).sum())
        counts["pv_daylight_capacity"] += int((daylight & pv_at_capacity).sum())
        counts["pv_night_total"] += int(night.sum())
        counts["pv_night_zero"] += int((night & pv_zero).sum())

        pue = target[..., 17:21]
        pue_min = packet.constraints.pue_min[:, None, :].float()
        pue_max = packet.constraints.pue_max[:, None, :].float()
        pue_at_min = (pue - pue_min).abs().le(1.0e-6)
        pue_at_max = (pue - pue_max).abs().le(1.0e-6)
        zero_workload_nodes = workload_zero.expand_as(pue)
        nonzero_workload_nodes = ~zero_workload_nodes
        counts["pue_total"] += pue.numel()
        counts["pue_minimum"] += int(pue_at_min.sum())
        counts["pue_maximum"] += int(pue_at_max.sum())
        counts["pue_nonzero_workload_total"] += int(nonzero_workload_nodes.sum())
        counts["pue_residual_minimum"] += int(
            (nonzero_workload_nodes & pue_at_min).sum()
        )
        counts["pue_residual_maximum"] += int(
            (nonzero_workload_nodes & pue_at_max).sum()
        )
        counts["pue_zero_workload_total"] += int(zero_workload_nodes.sum())
        counts["pue_zero_workload_maximum"] += int(
            (zero_workload_nodes & pue_at_max).sum()
        )

    if len(dates) != 722 or any(not day.startswith(("2022-", "2023-")) for day in dates):
        raise RuntimeError("support-atom fit escaped the registered 2022-2023 train split")
    result = {
        "schema": "dcvpp.vpp-world-moe.support-atoms.train-fit.v1",
        "fit_split": "train_2022_2023_only",
        "fit_days": len(dates),
        "opened_final_test_2025": False,
        "night_shortwave_threshold_normalized": 1.0e-4,
        "prevalence": {
            "workload_zero": _ratio(counts["workload_zero"], counts["workload_total"]),
            "pv_zero_global": _ratio(counts["pv_zero"], counts["pv_total"]),
            "pv_capacity_global": _ratio(counts["pv_capacity"], counts["pv_total"]),
            "pv_daylight_zero": _ratio(
                counts["pv_daylight_zero"], counts["pv_daylight_total"]
            ),
            "pv_daylight_capacity": _ratio(
                counts["pv_daylight_capacity"], counts["pv_daylight_total"]
            ),
            "pv_night_zero": _ratio(counts["pv_night_zero"], counts["pv_night_total"]),
            "pue_minimum_global": _ratio(counts["pue_minimum"], counts["pue_total"]),
            "pue_maximum_global": _ratio(counts["pue_maximum"], counts["pue_total"]),
            "pue_residual_minimum_given_nonzero_workload": _ratio(
                counts["pue_residual_minimum"], counts["pue_nonzero_workload_total"]
            ),
            "pue_residual_maximum_given_nonzero_workload": _ratio(
                counts["pue_residual_maximum"], counts["pue_nonzero_workload_total"]
            ),
            "pue_maximum_given_zero_workload": _ratio(
                counts["pue_zero_workload_maximum"], counts["pue_zero_workload_total"]
            ),
        },
        "counts": counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
