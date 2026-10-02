"""Formal 2024 realized-replay evaluation for separately trained R baselines."""

from __future__ import annotations

from contextlib import nullcontext
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import yaml

from .config import WorldMoEConfig
from .model import VPPWorldMoE, load_plannerless_released_w1_state
from .r_ecmoe import RECMoEConfig, RiskConditionPacket
from .r_ecmoe_baselines import DenseDirectR, IndependentRMoE
from .r_ecmoe_evaluation import (
    EVALUATION_DECISIONS,
    EvaluationVariant,
    _atomic_csv,
    _atomic_json,
    _cvar95,
    _execution_rows,
    _percentile,
    moving_block_bootstrap_indices,
)


BASELINE_RUN_META_SCHEMA = "dcvpp.r_ecmoe.baseline.run-meta.v1"
BASELINE_TRAINING_SCHEMA = "dcvpp.r_ecmoe.baseline.training.v1"
CHECKPOINT_SCHEMA = "dcvpp.r_ecmoe.runner.v1"
COMPARISON_DESIGN = "separately_retrained_architecture"


class BaselineEvaluationError(RuntimeError):
    """A separately trained baseline cannot preserve formal 2024 lineage."""


@dataclass(frozen=True)
class LoadedBaselineEvaluationRuntime:
    model: DenseDirectR | IndependentRMoE
    baseline_type: str
    model_class: str
    checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    run_root: Path
    run_id: str
    device: torch.device
    cost_scale: float
    world_config: WorldMoEConfig
    world_mapping: Mapping[str, Any]
    repository_root: Path
    w1_checkpoint_path: Path

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _load_extended_yaml(path: Path) -> dict[str, Any]:
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BaselineEvaluationError(f"configuration must be a mapping: {source}")
    parent = raw.get("extends")
    if parent is None:
        return dict(raw)
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = source.parent / parent_path
    return _deep_merge(_load_extended_yaml(parent_path), raw)


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise BaselineEvaluationError(
            f"baseline best checkpoint lacks finite {name}"
        )
    return float(value)


def _validate_checkpoint(
    checkpoint_path: Path,
) -> tuple[Mapping[str, Any], Path, Mapping[str, Any], Mapping[str, Any]]:
    path = checkpoint_path.resolve()
    if path.name != "best_decision_value.pt" or not path.is_file():
        raise BaselineEvaluationError(
            "baseline evaluation accepts only best_decision_value.pt"
        )
    run_root = path.parent.parent
    meta_path = run_root / "run_meta.json"
    config_path = run_root / "config_resolved.yaml"
    if not meta_path.is_file() or not config_path.is_file():
        raise BaselineEvaluationError(
            "baseline best requires its separate run_meta and resolved config"
        )
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(meta, Mapping) or not isinstance(resolved, Mapping):
        raise BaselineEvaluationError("baseline run lineage is malformed")
    if meta.get("schema") != BASELINE_RUN_META_SCHEMA:
        raise BaselineEvaluationError(
            "primary/same-checkpoint artifacts are not separately trained baselines"
        )
    meta_required = {
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
        "outcome_basis": "realized_historical_replay",
    }
    if any(meta.get(key) != expected for key, expected in meta_required.items()):
        raise BaselineEvaluationError("baseline run metadata violates formal lineage")
    if resolved.get("schema") != BASELINE_TRAINING_SCHEMA:
        raise BaselineEvaluationError("baseline resolved training schema is invalid")
    resolved_source = resolved.get("source")
    if (
        not isinstance(resolved_source, Mapping)
        or resolved_source.get("locked_test_enabled") is not False
    ):
        raise BaselineEvaluationError("baseline resolved config enabled locked test data")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise BaselineEvaluationError("baseline checkpoint payload is malformed")
    required = {
        "schema": CHECKPOINT_SCHEMA,
        "phase": "r_ecmoe",
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
    }
    if any(payload.get(key) != expected for key, expected in required.items()):
        raise BaselineEvaluationError("baseline checkpoint violates formal lineage")
    metrics = payload.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("validation_scope") != "full"
        or metrics.get("selection_eligible") is not True
        or metrics.get("outcome_basis") != "realized_historical_replay"
    ):
        raise BaselineEvaluationError(
            "baseline checkpoint is not an eligible realized-replay full best"
        )
    _finite_metric(metrics, "empirical_cvar95")
    _finite_metric(metrics, "worst_origin_sla")
    update = payload.get("global_update")
    if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
        raise BaselineEvaluationError(
            "baseline checkpoint does not prove a separate completed training run"
        )
    return payload, run_root, meta, resolved


def _load_released_w1(
    config: WorldMoEConfig, checkpoint_path: Path
) -> VPPWorldMoE:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or str(payload.get("phase", "")).lower() != "w1":
        raise BaselineEvaluationError("baseline source is not a released W1 checkpoint")
    if payload.get("locked_2025_opened") is not False:
        raise BaselineEvaluationError("baseline W1 source opened locked 2025")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise BaselineEvaluationError("released W1 checkpoint omitted model_state")
    model = VPPWorldMoE(config, include_legacy_planner=False)
    try:
        load_plannerless_released_w1_state(model, state)
    except RuntimeError as exc:
        raise BaselineEvaluationError(
            "released W1 checkpoint does not load strictly"
        ) from exc
    return model


def load_baseline_evaluation_runtime(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> LoadedBaselineEvaluationRuntime:
    """Load one separately trained Dense/Independent best without opening data."""

    path = Path(checkpoint_path).resolve()
    payload, run_root, meta, resolved = _validate_checkpoint(path)
    model_mapping = resolved.get("model")
    source = resolved.get("source")
    if not isinstance(model_mapping, Mapping) or not isinstance(source, Mapping):
        raise BaselineEvaluationError("baseline resolved config omitted model/source")
    baseline_type = str(model_mapping.get("baseline_type", "")).strip().lower()
    classes = {
        "dense_direct": DenseDirectR,
        "independent_r_moe": IndependentRMoE,
    }
    model_class = classes.get(baseline_type)
    if model_class is None:
        raise BaselineEvaluationError("unsupported baseline_type in resolved config")
    if (
        meta.get("baseline_type") != baseline_type
        or meta.get("model_class") != model_class.__name__
    ):
        raise BaselineEvaluationError("baseline type/class lineage disagrees")
    run_id = str(meta.get("run_id", ""))
    if not run_id or payload.get("run_id") != run_id:
        raise BaselineEvaluationError("baseline checkpoint/run id disagrees")

    config_source = Path(str(meta.get("config_path", ""))).resolve()
    if not config_source.is_file():
        raise BaselineEvaluationError("baseline original config path is unavailable")
    repository_root = config_source.parent.parent.resolve()
    world_config_path = Path(str(meta.get("world_config_path", ""))).resolve()
    if not world_config_path.is_file():
        raise BaselineEvaluationError("baseline world config path is unavailable")
    world_mapping = _load_extended_yaml(world_config_path)
    raw_world_model = world_mapping.get("model")
    if not isinstance(raw_world_model, Mapping):
        raise BaselineEvaluationError("world config omitted model mapping")
    world_config = WorldMoEConfig.from_mapping(raw_world_model)
    source_w1 = meta.get("source_w1")
    if not isinstance(source_w1, Mapping):
        raise BaselineEvaluationError("baseline run metadata omitted source W1")
    w1_path = Path(str(source_w1.get("path", ""))).resolve()
    if not w1_path.is_file():
        raise BaselineEvaluationError("baseline released W1 checkpoint is unavailable")
    world_model = _load_released_w1(world_config, w1_path)
    rec_values = dict(model_mapping)
    rec_values.pop("baseline_type", None)
    rec_config = RECMoEConfig.from_mapping(rec_values)
    model = model_class(world_model, rec_config)

    r_state = payload.get("r_state")
    if not isinstance(r_state, Mapping) or not r_state:
        raise BaselineEvaluationError("baseline checkpoint omitted R state")
    if any(
        str(name).startswith("world_model.")
        or "continuation_router" in str(name)
        or "tail_adapter" in str(name)
        for name in r_state
    ):
        raise BaselineEvaluationError(
            "baseline checkpoint contains forbidden primary/unused parameters"
        )
    required_prefix = (
        "dense_decision_block."
        if baseline_type == "dense_direct"
        else "independent_router."
    )
    if not any(str(name).startswith(required_prefix) for name in r_state):
        raise BaselineEvaluationError("baseline checkpoint has the wrong architecture state")
    incompatible = model.load_state_dict(r_state, strict=False)
    non_world_missing = [
        name for name in incompatible.missing_keys if not name.startswith("world_model.")
    ]
    if non_world_missing or incompatible.unexpected_keys:
        raise BaselineEvaluationError(
            "baseline R state does not load exactly into its declared architecture"
        )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise BaselineEvaluationError("CUDA was requested but is unavailable")
    model.to(resolved_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scale = payload.get("train_frozen_cost_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise BaselineEvaluationError("baseline checkpoint lacks train-only cost scale")
    return LoadedBaselineEvaluationRuntime(
        model=model,
        baseline_type=baseline_type,
        model_class=model_class.__name__,
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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(runtime: LoadedBaselineEvaluationRuntime) -> Any:
    if runtime.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _aggregate_days(
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_type: str,
    run_id: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in decision_rows:
        grouped.setdefault(str(row["operating_date"]), []).append(row)
    metrics = (
        "active_horizon",
        "realized_operating_cost",
        "cost_normalization_denominator",
        "normalized_operating_cost",
        "deadline_expired_work",
        "unserved_energy",
        "action_total_variation",
        "battery_throughput",
        "compute_energy",
        "migration_energy",
        "pv_curtailed_energy",
        "route_entropy",
        "tail_activation_rate",
        "equivalent_latency_ms_per_sample",
        *(f"route_load_{index}" for index in range(7)),
    )
    descriptions = {
        "dense_direct": "separately trained parameter-matched dense residual policy",
        "independent_r_moe": (
            "separately trained parameter-matched independent Top-2 R-MoE"
        ),
        "retrained_top1_conditional_tail": (
            "separately trained R-ECMoE deployment Top-1 with conditional tail"
        ),
    }
    description = descriptions.get(baseline_type)
    if description is None:
        raise BaselineEvaluationError("unregistered separately trained model role")
    rows: list[dict[str, Any]] = []
    for operating_date, values in sorted(grouped.items()):
        if len(values) != len(EVALUATION_DECISIONS):
            raise BaselineEvaluationError(
                f"{operating_date}/{baseline_type} does not have six decisions"
            )
        row: dict[str, Any] = {
            "operating_date": operating_date,
            "variant": baseline_type,
            "comparison_design": COMPARISON_DESIGN,
            "separately_retrained": True,
            "intervention_description": description,
            "training_run_id": run_id,
            "decision_count": len(values),
            "outcome_basis": "realized_historical_replay",
            "hard_residual_max": max(
                float(value["hard_residual_max"]) for value in values
            ),
        }
        for metric in metrics:
            row[metric] = sum(float(value[metric]) for value in values) / len(values)
        rows.append(row)
    return rows


def _summary(
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_type: str,
    run_id: str,
    expected_days: int,
) -> dict[str, Any]:
    if len(daily_rows) != expected_days:
        raise BaselineEvaluationError("baseline omitted formal validation days")
    costs = [float(value["normalized_operating_cost"]) for value in daily_rows]
    row: dict[str, Any] = {
        "variant": baseline_type,
        "comparison_design": COMPARISON_DESIGN,
        "separately_retrained": True,
        "training_run_id": run_id,
        "days": len(daily_rows),
        "outcome_basis": "realized_historical_replay",
        "mean_realized_operating_cost": sum(
            float(value["realized_operating_cost"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_normalized_operating_cost": sum(costs) / len(costs),
        "median_normalized_operating_cost": _percentile(costs, 0.50),
        "cvar95_normalized_operating_cost": _cvar95(costs),
        "mean_deadline_expired_work": sum(
            float(value["deadline_expired_work"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_unserved_energy": sum(
            float(value["unserved_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "maximum_hard_residual": max(
            float(value["hard_residual_max"]) for value in daily_rows
        ),
        "mean_action_total_variation": sum(
            float(value["action_total_variation"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_battery_throughput": sum(
            float(value["battery_throughput"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_compute_energy": sum(
            float(value["compute_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_migration_energy": sum(
            float(value["migration_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_pv_curtailed_energy": sum(
            float(value["pv_curtailed_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_route_entropy": sum(
            float(value["route_entropy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_tail_activation_rate": sum(
            float(value["tail_activation_rate"]) for value in daily_rows
        )
        / len(daily_rows),
        "latency_mean_ms_per_sample": sum(
            float(value["equivalent_latency_ms_per_sample"]) for value in daily_rows
        )
        / len(daily_rows),
        "latency_p95_ms_per_sample": _percentile(
            [
                float(value["equivalent_latency_ms_per_sample"])
                for value in daily_rows
            ],
            0.95,
        ),
    }
    for expert in range(7):
        row[f"mean_route_load_{expert}"] = sum(
            float(value[f"route_load_{expert}"]) for value in daily_rows
        ) / len(daily_rows)
    return row


def _csv_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise BaselineEvaluationError(f"invalid boolean evidence value: {value!r}")


def compare_with_primary_top2(
    daily_rows: Sequence[Mapping[str, Any]],
    primary_per_day_path: str | Path,
    *,
    expected_days: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    block_days: int,
    hard_residual_tolerance: float,
) -> dict[str, Any]:
    """Pair a retrained baseline only with the registered primary Top-2 rows."""

    source = Path(primary_per_day_path).resolve()
    if source.name != "per_day_variant.csv" or not source.is_file():
        raise BaselineEvaluationError(
            "primary comparison requires its formal per_day_variant.csv"
        )
    with source.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    primary_rows = [row for row in all_rows if row.get("variant") == "top2"]
    if len(primary_rows) != expected_days:
        raise BaselineEvaluationError("primary comparator omitted formal Top-2 days")
    if any(
        row.get("comparison_design") != "same_checkpoint_intervention"
        or _csv_bool(row.get("separately_retrained"))
        or row.get("outcome_basis") != "realized_historical_replay"
        or row.get("intervention_description")
        != "registered main Top-2 conditional-tail configuration"
        for row in primary_rows
    ):
        raise BaselineEvaluationError(
            "primary comparator must be the registered same-checkpoint Top-2"
        )
    baseline = {str(row["operating_date"]): row for row in daily_rows}
    primary = {str(row["operating_date"]): row for row in primary_rows}
    if set(baseline) != set(primary) or len(baseline) != expected_days:
        raise BaselineEvaluationError("baseline and primary Top-2 dates are not paired")
    dates = sorted(baseline)
    if any(date.fromisoformat(value).year != 2024 for value in dates):
        raise BaselineEvaluationError("paired comparison attempted non-2024 data")
    for operating_date in dates:
        if not math.isclose(
            float(baseline[operating_date]["cost_normalization_denominator"]),
            float(primary[operating_date]["cost_normalization_denominator"]),
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        ):
            raise BaselineEvaluationError(
                "baseline and primary Top-2 use different cost scales"
            )
    deltas = [
        float(baseline[day]["normalized_operating_cost"])
        - float(primary[day]["normalized_operating_cost"])
        for day in dates
    ]
    blocks = moving_block_bootstrap_indices(
        len(dates),
        block_days=block_days,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    bootstrap = [
        sum(deltas[index] for index in sample) / len(sample) for sample in blocks
    ]
    return {
        "baseline_variant": str(daily_rows[0]["variant"]),
        "comparator": "primary_top2",
        "comparison_design": "separately_retrained_vs_primary_top2",
        "baseline_separately_retrained": True,
        "comparator_separately_retrained": False,
        "outcome_basis": "realized_historical_replay",
        "days": len(dates),
        "cost_delta_baseline_minus_primary_top2": sum(deltas) / len(deltas),
        "cost_delta_ci95_lower": _percentile(bootstrap, 0.025),
        "cost_delta_ci95_upper": _percentile(bootstrap, 0.975),
        "baseline_daily_cost_win_rate": sum(value < 0.0 for value in deltas)
        / len(deltas),
        "deadline_delta": sum(
            float(baseline[day]["deadline_expired_work"])
            - float(primary[day]["deadline_expired_work"])
            for day in dates
        )
        / len(dates),
        "unserved_delta": sum(
            float(baseline[day]["unserved_energy"])
            - float(primary[day]["unserved_energy"])
            for day in dates
        )
        / len(dates),
        "baseline_hard_violation_rate": sum(
            float(baseline[day]["hard_residual_max"]) > hard_residual_tolerance
            for day in dates
        )
        / len(dates),
        "primary_top2_hard_violation_rate": sum(
            float(primary[day]["hard_residual_max"]) > hard_residual_tolerance
            for day in dates
        )
        / len(dates),
        "bootstrap_block_days": block_days,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "primary_top2_source": str(source),
    }


class Baseline2024Evaluator:
    """Evaluate one separately trained baseline on the full 2024 validation set."""

    def __init__(
        self,
        runtime: LoadedBaselineEvaluationRuntime,
        corpus: Any,
        output_root: str | Path,
        *,
        batch_size: int = 32,
        expected_validation_days: int = 364,
        primary_top2_per_day: str | Path | None = None,
        bootstrap_seed: int = 20240813,
        bootstrap_resamples: int = 2000,
        block_days: int = 7,
    ) -> None:
        if not isinstance(runtime, LoadedBaselineEvaluationRuntime):
            raise BaselineEvaluationError("runtime must be a loaded baseline runtime")
        if not callable(getattr(corpus, "batches", None)):
            raise BaselineEvaluationError("corpus must expose validation batches")
        if batch_size < 1 or expected_validation_days < 2:
            raise BaselineEvaluationError("evaluation batch/day counts must be positive")
        self.runtime = runtime
        self.model = runtime.model
        self.corpus = corpus
        self.output_root = Path(output_root).resolve()
        self.batch_size = int(batch_size)
        self.expected_validation_days = int(expected_validation_days)
        self.primary_top2_per_day = primary_top2_per_day
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.block_days = int(block_days)
        self.variant = EvaluationVariant(
            name=runtime.baseline_type,
            active_top_k=(
                1
                if runtime.baseline_type
                in {"dense_direct", "retrained_top1_conditional_tail"}
                else 2
            ),
            tail_mode=(
                "conditional"
                if runtime.baseline_type == "retrained_top1_conditional_tail"
                else "off"
            ),
            route_prior_beta=0.0,
            description="separately trained matched architecture baseline",
        ).validate()

    def _batch_cost_scale(self, batch: Mapping[str, Any]) -> float:
        value = batch.get("train_frozen_baseline_daily_cost")
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise BaselineEvaluationError("validation batch omitted train-only cost scale")
        numeric = float(value.detach().float().cpu())
        if not math.isclose(
            numeric, self.runtime.cost_scale, rel_tol=1.0e-7, abs_tol=1.0e-7
        ):
            raise BaselineEvaluationError(
                "baseline evaluation cost scale differs from its training checkpoint"
            )
        return numeric

    def run(self) -> Mapping[str, Any]:
        self.model.eval()
        decision_rows: list[dict[str, Any]] = []
        reference_dates: tuple[str, ...] | None = None
        forward_calls = 0
        replay_calls = 0
        with torch.inference_mode():
            for stage_id, origin, horizon in EVALUATION_DECISIONS:
                dates_for_decision: list[str] = []
                for batch in self.corpus.batches(
                    "validation",
                    self.batch_size,
                    device=self.runtime.device,
                    shuffle_seed=None,
                    max_days=None,
                    origin=origin,
                    require_h96=False,
                    include_next_day=False,
                ):
                    dates = tuple(str(value) for value in batch.get("operating_dates", ()))
                    packet = batch.get("world_input")
                    if (
                        not dates
                        or packet is None
                        or len(dates) != int(packet.fast_values.shape[0])
                    ):
                        raise BaselineEvaluationError(
                            "validation batch omitted dates or world_input"
                        )
                    if any(date.fromisoformat(value).year != 2024 for value in dates):
                        raise BaselineEvaluationError(
                            "baseline evaluation attempted non-2024 data"
                        )
                    if not bool((packet.origin_index == origin).all()):
                        raise BaselineEvaluationError("validation packet origin mismatch")
                    dates_for_decision.extend(dates)
                    anchors = batch.get("forecast_anchors", {})
                    caps = batch.get("forecast_residual_caps", {})
                    da_anchor = anchors.get("da") if isinstance(anchors, Mapping) else None
                    da_cap = caps.get("da") if isinstance(caps, Mapping) else None
                    if not isinstance(da_anchor, Tensor) or not isinstance(da_cap, Tensor):
                        raise BaselineEvaluationError("validation batch omitted DA anchors")
                    risk = RiskConditionPacket.defaults(
                        len(dates),
                        device=self.runtime.device,
                        horizon=horizon,
                        stage=stage_id,
                    )
                    before = int(self.model.w1_backbone_calls)
                    _synchronize(self.runtime.device)
                    started = time.perf_counter()
                    with _autocast(self.runtime):
                        decision = self.model(
                            packet,
                            risk,
                            da_anchor=da_anchor,
                            da_residual_cap=da_cap,
                            return_diagnostics=True,
                        )
                    _synchronize(self.runtime.device)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    if int(self.model.w1_backbone_calls) - before != 1:
                        raise BaselineEvaluationError(
                            "each baseline decision must execute W1 exactly once"
                        )
                    forward_calls += 1
                    # Realized targets enter only after the complete separately
                    # trained baseline forward has fixed routes and raw actions.
                    realized = self.model.realized_historical_replay(
                        decision,
                        batch=batch,
                        risk=risk,
                        source_split="validation",
                    )
                    replay_calls += 1
                    decision_rows.extend(
                        _execution_rows(
                            decision,
                            realized,
                            variant=self.variant,
                            dates=dates,
                            stage_id=stage_id,
                            origin=origin,
                            horizon=horizon,
                            cost_scale=self._batch_cost_scale(batch),
                            equivalent_latency_ms_per_sample=latency_ms / len(dates),
                        )
                    )
                if (
                    len(dates_for_decision) != self.expected_validation_days
                    or len(set(dates_for_decision)) != self.expected_validation_days
                ):
                    raise BaselineEvaluationError(
                        "baseline evaluation did not consume the complete validation year"
                    )
                ordered = tuple(dates_for_decision)
                if reference_dates is None:
                    reference_dates = ordered
                elif ordered != reference_dates:
                    raise BaselineEvaluationError(
                        "six baseline decisions do not share identical 2024 dates"
                    )
        daily_rows = _aggregate_days(
            decision_rows,
            baseline_type=self.runtime.baseline_type,
            run_id=self.runtime.run_id,
        )
        summary = _summary(
            daily_rows,
            baseline_type=self.runtime.baseline_type,
            run_id=self.runtime.run_id,
            expected_days=self.expected_validation_days,
        )
        _atomic_csv(self.output_root / "per_day_variant.csv", daily_rows)
        _atomic_csv(self.output_root / "baseline_summary.csv", [summary])
        paired: dict[str, Any] | None = None
        if self.primary_top2_per_day is not None:
            tolerance = float(
                self.runtime.checkpoint.get("resolved_config", {}).get(
                    "hard_residual_tolerance", 1.0e-4
                )
            )
            paired = compare_with_primary_top2(
                daily_rows,
                self.primary_top2_per_day,
                expected_days=self.expected_validation_days,
                bootstrap_seed=self.bootstrap_seed,
                bootstrap_resamples=self.bootstrap_resamples,
                block_days=self.block_days,
                hard_residual_tolerance=tolerance,
            )
            _atomic_csv(self.output_root / "paired_primary_top2.csv", [paired])
        meta = {
            "schema": "dcvpp.r_ecmoe.baseline_evaluation_2024.v1",
            "recorded_at_utc": _utc_now(),
            "split": "validation",
            "year": 2024,
            "validation_days": self.expected_validation_days,
            "decisions_per_day": len(EVALUATION_DECISIONS),
            "decision_origins": [value[1] for value in EVALUATION_DECISIONS],
            "baseline_type": self.runtime.baseline_type,
            "model_class": self.runtime.model_class,
            "comparison_design": COMPARISON_DESIGN,
            "separately_retrained": True,
            "training_run_id": self.runtime.run_id,
            "best_checkpoint": str(self.runtime.checkpoint_path),
            "w1_checkpoint": str(self.runtime.w1_checkpoint_path),
            "outcome_basis": "realized_historical_replay",
            "train_frozen_baseline_daily_cost": self.runtime.cost_scale,
            "cost_normalization": (
                "train_frozen_baseline_daily_cost * active_horizon / 96"
            ),
            "model_forward_calls": forward_calls,
            "realized_replay_calls": replay_calls,
            "primary_top2_comparison": paired is not None,
            "bootstrap": (
                None
                if paired is None
                else {
                    "method": "circular_moving_block_over_daily_paired_differences",
                    "block_days": self.block_days,
                    "resamples": self.bootstrap_resamples,
                    "seed": self.bootstrap_seed,
                }
            ),
            "locked_2025_opened": False,
            "external_solver": False,
            "candidate_search": False,
            "hash_computed": False,
        }
        _atomic_json(self.output_root / "baseline_evaluation_2024_meta.json", meta)
        return meta


__all__ = [
    "BASELINE_RUN_META_SCHEMA",
    "BASELINE_TRAINING_SCHEMA",
    "Baseline2024Evaluator",
    "BaselineEvaluationError",
    "COMPARISON_DESIGN",
    "LoadedBaselineEvaluationRuntime",
    "compare_with_primary_top2",
    "load_baseline_evaluation_runtime",
]
