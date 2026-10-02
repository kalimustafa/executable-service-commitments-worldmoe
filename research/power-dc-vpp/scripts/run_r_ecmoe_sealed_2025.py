#!/usr/bin/env python
"""Run one frozen R-ECMoE final evaluation exactly once.

The sealed corpus is supplied as ``module:function``.  Import and construction
of that corpus are delayed until the evaluator has validated the frozen marker
and checkpoint and written its exclusive one-shot sentinel.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.vpp_world_moe.config import WorldMoEConfig
from dcvpp.vpp_world_moe.r_ecmoe import RECMoEConfig
from dcvpp.vpp_world_moe.r_ecmoe_runtime import RECMoEInferenceRuntime
from dcvpp.vpp_world_moe.r_ecmoe_sealed_2025 import (
    RECMoESealed2025Evaluator,
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
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise RuntimeError("sealed corpus factory target is not callable")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/r_ecmoe_balanced.yaml")
    )
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--r-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--corpus-factory",
        default=(
            "dcvpp.vpp_world_moe.r_ecmoe_sealed_corpus:"
            "build_sealed_final_corpus"
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    repository_root = config_path.parent.parent.resolve()
    config = _load_extended_yaml(config_path)
    if str(config.get("phase", "")).lower() != "r_ecmoe":
        raise SystemExit("configuration phase must be r_ecmoe")
    source = config.get("source")
    model_mapping = config.get("model")
    if not isinstance(source, Mapping) or not isinstance(model_mapping, Mapping):
        raise SystemExit("R-ECMoE source and model configuration are required")
    world_config_path = _resolve(repository_root, str(source["world_config"]))
    world_mapping = _load_extended_yaml(world_config_path)
    world_model_mapping = world_mapping.get("model")
    if not isinstance(world_model_mapping, Mapping):
        raise SystemExit("world model configuration is missing")
    world_config = WorldMoEConfig.from_mapping(world_model_mapping)
    rec_config = RECMoEConfig.from_mapping(dict(model_mapping))
    w1_checkpoint = _resolve(repository_root, str(source["w1_checkpoint"]))
    r_checkpoint = _resolve(repository_root, args.r_checkpoint)
    marker_path = _resolve(repository_root, args.marker)
    output_root = (
        _resolve(repository_root, args.output_root)
        if args.output_root is not None
        else r_checkpoint.parent.parent / "evaluation_2025_one_shot"
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    runtime = RECMoEInferenceRuntime.from_w1_checkpoint(
        world_config,
        rec_config,
        w1_checkpoint,
        device,
        r_checkpoint_path=r_checkpoint,
    )

    # Deliberately resolve/import the final-data implementation only after the
    # evaluator has reserved the authorized one-shot attempt.
    def sealed_corpus_factory(marker: Mapping[str, Any]) -> Any:
        create = _factory(str(args.corpus_factory))
        return create(
            config=world_mapping,
            repository_root=repository_root,
            frozen_evaluation_marker=dict(marker),
        )

    result = RECMoESealed2025Evaluator(
        runtime,
        sealed_corpus_factory,
        marker_path,
        output_root,
        batch_size=int(args.batch_size),
    ).run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
