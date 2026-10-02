"""Train one preregistered R-ECMoE architecture baseline on train/validation.

This entry point intentionally exposes no inference, evaluation, or locked-test
mode.  Dense Direct R and Independent R-MoE use the same released W1 source,
decision curriculum, optimizer budget, and selector as the balanced primary
model while writing independent lineages.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Type

import numpy as np
import pyarrow.dataset  # noqa: F401 - preload before CUDA/PyTorch on Windows
import torch
from torch import nn
import yaml

from dcvpp.vpp_world_moe.config import WorldMoEConfig
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.model import VPPWorldMoE
from dcvpp.vpp_world_moe.r_ecmoe import RECMoEConfig
from dcvpp.vpp_world_moe.r_ecmoe_runner import RECMoERunner

try:  # Direct ``python scripts/...`` execution.
    from run_r_ecmoe import (  # type: ignore[import-not-found]
        _atomic_text,
        _load_extended_yaml,
        _load_w1,
        _resolve,
        _runner_config,
    )
except ModuleNotFoundError as exc:  # Imported/runpy execution from repository root.
    if exc.name != "run_r_ecmoe":
        raise
    from scripts.run_r_ecmoe import (
        _atomic_text,
        _load_extended_yaml,
        _load_w1,
        _resolve,
        _runner_config,
    )


BASELINE_SCHEMA = "dcvpp.r_ecmoe.baseline.training.v1"
BASELINE_SPECS = {
    "dense_direct": "DenseDirectR",
    "independent_r_moe": "IndependentRMoE",
}


def _baseline_class(baseline_type: str) -> Type[nn.Module]:
    """Resolve the preregistered field-to-class mapping before construction."""

    value = str(baseline_type).strip().lower()
    expected_name = BASELINE_SPECS.get(value)
    if expected_name is None:
        raise SystemExit(
            "model.baseline_type must be dense_direct or independent_r_moe"
        )
    from dcvpp.vpp_world_moe.r_ecmoe_baselines import (
        DenseDirectR,
        IndependentRMoE,
    )

    classes: dict[str, Type[nn.Module]] = {
        "dense_direct": DenseDirectR,
        "independent_r_moe": IndependentRMoE,
    }
    model_class = classes[value]
    if model_class.__name__ != expected_name or not issubclass(model_class, nn.Module):
        raise SystemExit(
            f"baseline class contract mismatch: {value} must resolve to {expected_name}"
        )
    declared_type = getattr(model_class, "baseline_type", value)
    if str(declared_type).strip().lower() != value:
        raise SystemExit(
            f"baseline class declaration disagrees with model.baseline_type={value}"
        )
    return model_class


def _baseline_model_config(model_mapping: Mapping[str, Any]) -> RECMoEConfig:
    values = dict(model_mapping)
    values.pop("baseline_type", None)
    return RECMoEConfig.from_mapping(values)


def _build_baseline_model(
    world_model: VPPWorldMoE,
    model_mapping: Mapping[str, Any],
) -> nn.Module:
    baseline_type = str(model_mapping.get("baseline_type", "")).strip().lower()
    model_class = _baseline_class(baseline_type)
    # Field/class agreement is proven above, before either config or model is
    # instantiated.  Baselines deliberately share the primary RECMoEConfig.
    model_config = _baseline_model_config(model_mapping)
    return model_class(world_model, model_config)


def _write_baseline_run_contract(
    run_root: Path,
    *,
    run_id: str,
    baseline_type: str,
    model_class_name: str,
    resolved_config: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    config_path: Path,
    world_config_path: Path,
    resume: bool,
) -> dict[str, Any]:
    """Atomically freeze an independent, resume-exact baseline lineage."""

    config_output = run_root / "config_resolved.yaml"
    meta_output = run_root / "run_meta.json"
    config_value = dict(resolved_config)
    stable_meta = {
        "schema": "dcvpp.r_ecmoe.baseline.run-meta.v1",
        "phase": "r_ecmoe",
        "run_id": run_id,
        "profile": str(config_value.get("model", {}).get("profile", "")),
        "baseline_type": baseline_type,
        "model_class": model_class_name,
        "config_path": str(config_path.resolve()),
        "world_config_path": str(world_config_path.resolve()),
        "source_w1": dict(source_meta),
        "locked_test_opened": False,
        "locked_2025_opened": False,
        "weather_coverage_gate": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
        "outcome_basis": "realized_historical_replay",
    }
    if resume:
        if not config_output.is_file() or not meta_output.is_file():
            raise SystemExit(
                "baseline resume requires config_resolved.yaml and run_meta.json"
            )
        existing_config = yaml.safe_load(config_output.read_text(encoding="utf-8"))
        existing_meta = json.loads(meta_output.read_text(encoding="utf-8"))
        comparable_meta = dict(existing_meta)
        comparable_meta.pop("created_at_utc", None)
        frozen_cost_scale = comparable_meta.pop(
            "train_frozen_baseline_daily_cost", None
        )
        if frozen_cost_scale is not None:
            if isinstance(frozen_cost_scale, bool):
                raise SystemExit(
                    "resume train_frozen_baseline_daily_cost must be finite and positive"
                )
            try:
                numeric_cost_scale = float(frozen_cost_scale)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    "resume train_frozen_baseline_daily_cost must be finite and positive"
                ) from exc
            if not math.isfinite(numeric_cost_scale) or numeric_cost_scale <= 0.0:
                raise SystemExit(
                    "resume train_frozen_baseline_daily_cost must be finite and positive"
                )
        if existing_config != config_value or comparable_meta != stable_meta:
            raise SystemExit("baseline resume contract differs from current inputs")
        return existing_meta
    if config_output.exists() or meta_output.exists():
        raise SystemExit("baseline run contract already exists; use --mode resume")
    meta = {
        **stable_meta,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    _atomic_text(
        config_output,
        yaml.safe_dump(config_value, sort_keys=False, allow_unicode=True),
    )
    _atomic_text(
        meta_output,
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return meta


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "train", "resume"), default="smoke"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    repository_root = config_path.parent.parent.resolve()
    config = _load_extended_yaml(config_path)
    if str(config.get("schema", "")) != BASELINE_SCHEMA:
        raise SystemExit(f"baseline configuration schema must be {BASELINE_SCHEMA}")
    if str(config.get("phase", "")).lower() != "r_ecmoe":
        raise SystemExit("baseline configuration phase must be r_ecmoe")

    source = config.get("source")
    if not isinstance(source, Mapping) or source.get("locked_test_enabled") is not False:
        raise SystemExit("baseline development must keep the locked 2025 test disabled")
    model_mapping = config.get("model")
    if not isinstance(model_mapping, Mapping):
        raise SystemExit("baseline model configuration is required")
    baseline_type = str(model_mapping.get("baseline_type", "")).strip().lower()
    model_class = _baseline_class(baseline_type)

    world_config_path = _resolve(repository_root, str(source["world_config"]))
    world_mapping = _load_extended_yaml(world_config_path)
    world_model_config = WorldMoEConfig.from_mapping(world_mapping["model"])
    checkpoint = _resolve(repository_root, str(source["w1_checkpoint"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise SystemExit("baseline training configuration must be a mapping")
    seed = int(training.get("seed", 20240813))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    outputs = config.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise SystemExit("baseline outputs configuration must be a mapping")
    prefix = str(outputs.get("run_id_prefix", "")).strip()
    if not prefix or Path(prefix).name != prefix:
        raise SystemExit("outputs.run_id_prefix must be one nonempty path component")
    run_id = args.run_id or (
        prefix
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-seed{seed}"
    )
    if Path(run_id).name != run_id:
        raise SystemExit("run id must be one path component")
    output_root = _resolve(repository_root, str(outputs["root"]))

    world_model, source_meta = _load_w1(world_model_config, checkpoint)
    _write_baseline_run_contract(
        output_root / run_id,
        run_id=run_id,
        baseline_type=baseline_type,
        model_class_name=model_class.__name__,
        resolved_config=config,
        source_meta=source_meta,
        config_path=config_path,
        world_config_path=world_config_path,
        resume=args.mode == "resume",
    )
    model = _build_baseline_model(world_model, model_mapping)
    corpus = WorldMoECorpus(
        world_mapping,
        repository_root,
        weather_coverage_gate=False,
    )
    runner = RECMoERunner(
        model,  # type: ignore[arg-type] - structural RECMoE runner contract
        corpus,
        _runner_config(config),
        output_root,
        run_id,
        device,
        (checkpoint, config_path, world_config_path),
    )
    result = (
        runner.smoke()
        if args.mode == "smoke"
        else runner.fit(resume=args.mode == "resume")
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
