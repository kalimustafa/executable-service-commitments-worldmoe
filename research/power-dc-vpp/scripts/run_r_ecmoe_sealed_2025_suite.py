#!/usr/bin/env python
"""Run the frozen five-model R-ECMoE final suite exactly once."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import pyarrow.dataset  # noqa: F401 - preload before CUDA/PyTorch on Windows
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.vpp_world_moe.config import WorldMoEConfig
from dcvpp.vpp_world_moe.r_ecmoe import RECMoEConfig
from dcvpp.vpp_world_moe.r_ecmoe_baseline_evaluation import (
    load_baseline_evaluation_runtime,
)
from dcvpp.vpp_world_moe.r_ecmoe_runtime import RECMoEInferenceRuntime
from dcvpp.vpp_world_moe.r_ecmoe_sealed_2025_suite import (
    FrozenSuiteRuntimes,
    RECMoESealed2025ModelSuiteEvaluator,
)
from dcvpp.vpp_world_moe.r_ecmoe_top1_evaluation import (
    load_retrained_top1_evaluation_runtime,
)


DEFAULT_MAIN_CHECKPOINT = Path(
    "artifacts/r_ecmoe_balanced_continued/"
    "r-ecmoe-balanced-continued-earlystop-from80-20260827-seed20240813/"
    "checkpoints/best_decision_value.pt"
)
DEFAULT_TOP1_CHECKPOINT = Path(
    "artifacts/r_ecmoe_top1_retrained/"
    "r-ecmoe-top1-retrained-gridclosed-formal-20260827-seed20240813/"
    "checkpoints/best_decision_value.pt"
)
DEFAULT_DENSE_CHECKPOINT = Path(
    "artifacts/r_ecmoe_dense_direct_baseline/"
    "r-ecmoe-dense-direct-gridclosed-formal-20260827-seed20240813/"
    "checkpoints/best_decision_value.pt"
)
DEFAULT_INDEPENDENT_CHECKPOINT = Path(
    "artifacts/r_ecmoe_independent_moe_baseline/"
    "r-ecmoe-independent-moe-gridclosed-formal-20260827-seed20240813/"
    "checkpoints/best_decision_value.pt"
)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _load_extended_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit(f"configuration must be a mapping: {resolved}")
    parent = payload.get("extends")
    local = {key: value for key, value in payload.items() if key != "extends"}
    if parent is None:
        return dict(local)
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge(_load_extended_yaml(parent_path), local)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise RuntimeError("corpus factory must use module:function syntax")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise RuntimeError("sealed corpus factory target is not callable")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-config",
        type=Path,
        default=Path("configs/r_ecmoe_balanced_continue_earlystop.yaml"),
    )
    parser.add_argument(
        "--marker",
        type=Path,
        default=Path(
            "artifacts/paper_finalization/r_ecmoe_2025_final_suite_marker.json"
        ),
    )
    parser.add_argument(
        "--main-checkpoint", type=Path, default=DEFAULT_MAIN_CHECKPOINT
    )
    parser.add_argument(
        "--top1-checkpoint", type=Path, default=DEFAULT_TOP1_CHECKPOINT
    )
    parser.add_argument(
        "--dense-checkpoint", type=Path, default=DEFAULT_DENSE_CHECKPOINT
    )
    parser.add_argument(
        "--independent-checkpoint",
        type=Path,
        default=DEFAULT_INDEPENDENT_CHECKPOINT,
    )
    parser.add_argument(
        "--corpus-factory",
        default=(
            "dcvpp.vpp_world_moe.r_ecmoe_sealed_corpus:"
            "build_sealed_final_corpus"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/r_ecmoe_final_2025_suite/one_shot_20260827"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-seed", type=int, default=20240813)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    main_config_path = args.main_config.expanduser().resolve()
    repository_root = main_config_path.parent.parent.resolve()
    config = _load_extended_yaml(main_config_path)
    if str(config.get("phase", "")).lower() != "r_ecmoe":
        raise SystemExit("main configuration phase must be r_ecmoe")
    source = config.get("source")
    model_mapping = config.get("model")
    if not isinstance(source, Mapping) or not isinstance(model_mapping, Mapping):
        raise SystemExit("main R-ECMoE source/model configuration is required")
    if int(model_mapping.get("active_top_k", 0)) != 2:
        raise SystemExit("suite main configuration must be the frozen Top-2 model")

    world_config_path = _resolve(repository_root, str(source["world_config"]))
    world_mapping = _load_extended_yaml(world_config_path)
    raw_world = world_mapping.get("model")
    if not isinstance(raw_world, Mapping):
        raise SystemExit("world model configuration is missing")
    world_config = WorldMoEConfig.from_mapping(raw_world)
    rec_config = RECMoEConfig.from_mapping(dict(model_mapping))
    w1_checkpoint = _resolve(repository_root, str(source["w1_checkpoint"]))
    main_checkpoint = _resolve(repository_root, args.main_checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    # Every loader is validation-only and rejects non-best or unsafe lineage.
    main_runtime = RECMoEInferenceRuntime.from_w1_checkpoint(
        world_config,
        rec_config,
        w1_checkpoint,
        device,
        r_checkpoint_path=main_checkpoint,
    )
    top1_runtime = load_retrained_top1_evaluation_runtime(
        _resolve(repository_root, args.top1_checkpoint), device=device
    )
    dense_runtime = load_baseline_evaluation_runtime(
        _resolve(repository_root, args.dense_checkpoint), device=device
    )
    independent_runtime = load_baseline_evaluation_runtime(
        _resolve(repository_root, args.independent_checkpoint), device=device
    )
    evaluator = RECMoESealed2025ModelSuiteEvaluator(
        FrozenSuiteRuntimes(
            main=main_runtime,
            top1=top1_runtime,
            dense=dense_runtime,
            independent=independent_runtime,
        ),
        # Import/construction of final data remains delayed until suite_started.
        lambda marker: _factory(str(args.corpus_factory))(
            config=world_mapping,
            repository_root=repository_root,
            frozen_evaluation_marker=dict(marker),
        ),
        _resolve(repository_root, args.marker),
        _resolve(repository_root, args.output_root),
        batch_size=args.batch_size,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        block_days=args.block_days,
    )
    result = evaluator.run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
