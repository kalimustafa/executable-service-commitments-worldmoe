"""2024 evaluation loader for a separately trained Top-1 R-ECMoE model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from .config import WorldMoEConfig
from .r_ecmoe import RECMoE, RECMoEConfig
from .r_ecmoe_baseline_evaluation import (
    Baseline2024Evaluator,
    BaselineEvaluationError,
    LoadedBaselineEvaluationRuntime,
    _atomic_json,
    _finite_metric,
    _load_extended_yaml,
    _load_released_w1,
)
from .r_ecmoe_replay import RealizedHistoricalReplay


TOP1_COMPARISON_ROLE = "separately_trained_top1_conditional_tail"
TOP1_MODEL_ROLE = "retrained_top1_conditional_tail"
PRIMARY_RUN_META_SCHEMA = "dcvpp.r_ecmoe.run-meta.v1"
PRIMARY_TRAINING_SCHEMA = "dcvpp.r_ecmoe.training.v1"


def _validate_top1_checkpoint(
    checkpoint_path: Path,
) -> tuple[Mapping[str, Any], Path, Mapping[str, Any], Mapping[str, Any]]:
    path = checkpoint_path.resolve()
    if path.name != "best_decision_value.pt" or not path.is_file():
        raise BaselineEvaluationError(
            "retrained Top-1 evaluation accepts only best_decision_value.pt"
        )
    run_root = path.parent.parent
    meta_path = run_root / "run_meta.json"
    config_path = run_root / "config_resolved.yaml"
    if not meta_path.is_file() or not config_path.is_file():
        raise BaselineEvaluationError(
            "retrained Top-1 best requires its independent run lineage"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(meta, Mapping) or not isinstance(resolved, Mapping):
        raise BaselineEvaluationError("retrained Top-1 run lineage is malformed")
    if meta.get("schema") != PRIMARY_RUN_META_SCHEMA:
        raise BaselineEvaluationError("Top-1 checkpoint is not a primary R training run")
    if resolved.get("schema") != PRIMARY_TRAINING_SCHEMA:
        raise BaselineEvaluationError("Top-1 resolved training schema is invalid")
    comparison = resolved.get("comparison")
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("role") != TOP1_COMPARISON_ROLE
        or comparison.get("separately_retrained") is not True
        or comparison.get("same_checkpoint_intervention") is not False
        or comparison.get("primary_comparator") != "main_top2"
    ):
        raise BaselineEvaluationError(
            "same-checkpoint Top-1 intervention cannot impersonate retrained Top-1"
        )
    model_mapping = resolved.get("model")
    if (
        not isinstance(model_mapping, Mapping)
        or str(model_mapping.get("profile", "")).lower() != "balanced"
        or int(model_mapping.get("active_top_k", 0)) != 1
        or int(model_mapping.get("adapter_bottleneck", 0)) != 32
        or int(model_mapping.get("temporal_rank", 0)) != 16
    ):
        raise BaselineEvaluationError(
            "retrained Top-1 must preserve balanced width/rank with active_top_k=1"
        )
    source = resolved.get("source")
    if not isinstance(source, Mapping) or source.get("locked_test_enabled") is not False:
        raise BaselineEvaluationError("retrained Top-1 enabled locked test data")
    outputs = resolved.get("outputs")
    if (
        not isinstance(outputs, Mapping)
        or "top1_retrained" not in str(outputs.get("root", ""))
    ):
        raise BaselineEvaluationError("retrained Top-1 lacks an independent output root")
    meta_required = {
        "phase": "r_ecmoe",
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
    }
    if any(meta.get(key) != expected for key, expected in meta_required.items()):
        raise BaselineEvaluationError("retrained Top-1 run metadata is unsafe")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise BaselineEvaluationError("retrained Top-1 checkpoint is malformed")
    payload_required = {
        "schema": "dcvpp.r_ecmoe.runner.v1",
        "phase": "r_ecmoe",
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
    }
    if any(payload.get(key) != expected for key, expected in payload_required.items()):
        raise BaselineEvaluationError("retrained Top-1 checkpoint violates lineage")
    metrics = payload.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("validation_scope") != "full"
        or metrics.get("selection_eligible") is not True
        or metrics.get("outcome_basis") != "realized_historical_replay"
    ):
        raise BaselineEvaluationError(
            "retrained Top-1 is not an eligible realized-replay full best"
        )
    _finite_metric(metrics, "empirical_cvar95")
    _finite_metric(metrics, "worst_origin_sla")
    update = payload.get("global_update")
    if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
        raise BaselineEvaluationError(
            "retrained Top-1 checkpoint does not prove independent training"
        )
    return payload, run_root, meta, resolved


def load_retrained_top1_evaluation_runtime(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> LoadedBaselineEvaluationRuntime:
    """Load only a separately trained Top-1+conditional-tail R checkpoint."""

    path = Path(checkpoint_path).resolve()
    payload, run_root, meta, resolved = _validate_top1_checkpoint(path)
    run_id = str(meta.get("run_id", ""))
    if not run_id or payload.get("run_id") != run_id:
        raise BaselineEvaluationError("retrained Top-1 checkpoint/run id disagrees")
    config_source = Path(str(meta.get("config_path", ""))).resolve()
    world_config_path = Path(str(meta.get("world_config_path", ""))).resolve()
    if not config_source.is_file() or not world_config_path.is_file():
        raise BaselineEvaluationError("retrained Top-1 source configs are unavailable")
    repository_root = config_source.parent.parent.resolve()
    world_mapping = _load_extended_yaml(world_config_path)
    raw_world = world_mapping.get("model")
    if not isinstance(raw_world, Mapping):
        raise BaselineEvaluationError("world config omitted model mapping")
    world_config = WorldMoEConfig.from_mapping(raw_world)
    source_w1 = meta.get("source_w1")
    if not isinstance(source_w1, Mapping):
        raise BaselineEvaluationError("retrained Top-1 metadata omitted source W1")
    w1_path = Path(str(source_w1.get("path", ""))).resolve()
    if not w1_path.is_file():
        raise BaselineEvaluationError("retrained Top-1 W1 source is unavailable")
    world_model = _load_released_w1(world_config, w1_path)
    model_mapping = resolved["model"]
    rec_config = RECMoEConfig.from_mapping(dict(model_mapping))
    model = RECMoE(world_model, rec_config)
    r_state = payload.get("r_state")
    if not isinstance(r_state, Mapping) or not r_state:
        raise BaselineEvaluationError("retrained Top-1 checkpoint omitted R state")
    if any(str(name).startswith("world_model.") for name in r_state):
        raise BaselineEvaluationError("retrained Top-1 R state contains frozen W1")
    if not any(str(name).startswith("continuation_router.") for name in r_state) or not any(
        str(name).startswith("tail_adapter.") for name in r_state
    ):
        raise BaselineEvaluationError(
            "retrained Top-1 checkpoint is not the full conditional-tail R model"
        )
    incompatible = model.load_state_dict(r_state, strict=False)
    non_world_missing = [
        name for name in incompatible.missing_keys if not name.startswith("world_model.")
    ]
    if non_world_missing or incompatible.unexpected_keys:
        raise BaselineEvaluationError(
            "retrained Top-1 R state does not load exactly"
        )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise BaselineEvaluationError("CUDA was requested but is unavailable")
    model.to(resolved_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # The generic separately-trained evaluator admits targets only through this
    # post-forward callable.  It does not alter the primary model implementation.
    model.realized_historical_replay = RealizedHistoricalReplay.from_model(model)  # type: ignore[attr-defined]
    scale = payload.get("train_frozen_cost_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise BaselineEvaluationError("retrained Top-1 lacks train-only cost scale")
    return LoadedBaselineEvaluationRuntime(
        model=model,  # type: ignore[arg-type]
        baseline_type=TOP1_MODEL_ROLE,
        model_class="RECMoE",
        checkpoint_path=path,
        checkpoint=payload,
        run_root=run_root,
        run_id=run_id,
        device=resolved_device,
        cost_scale=float(scale),
        world_config=world_config,
        world_mapping=world_mapping,
        repository_root=repository_root,
        w1_checkpoint_path=w1_path,
    )


class RetrainedTop12024Evaluator(Baseline2024Evaluator):
    """Formal six-decision 2024 evaluator for the independently trained Top-1."""

    def __init__(self, runtime: LoadedBaselineEvaluationRuntime, *args: Any, **kwargs: Any) -> None:
        if runtime.baseline_type != TOP1_MODEL_ROLE or runtime.model_class != "RECMoE":
            raise BaselineEvaluationError("runtime is not separately trained Top-1 R-ECMoE")
        super().__init__(runtime, *args, **kwargs)

    def run(self) -> Mapping[str, Any]:
        meta = dict(super().run())
        meta.update(
            {
                "schema": "dcvpp.r_ecmoe.retrained_top1_evaluation_2024.v1",
                "comparison_role": TOP1_COMPARISON_ROLE,
                "active_top_k": 1,
                "conditional_tail": True,
                "same_checkpoint_intervention": False,
            }
        )
        _atomic_json(self.output_root / "baseline_evaluation_2024_meta.json", meta)
        return meta


__all__ = [
    "PRIMARY_RUN_META_SCHEMA",
    "PRIMARY_TRAINING_SCHEMA",
    "RetrainedTop12024Evaluator",
    "TOP1_COMPARISON_ROLE",
    "TOP1_MODEL_ROLE",
    "load_retrained_top1_evaluation_runtime",
]
