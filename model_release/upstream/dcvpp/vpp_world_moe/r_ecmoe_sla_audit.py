"""Read-only 2024 SLA attainability audit for one primary R latest checkpoint."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from dataclasses import dataclass
from datetime import date, datetime, timezone

import torch
from torch import Tensor
import yaml

from .config import WorldMoEConfig
from .r_ecmoe import RECMoE, RECMoEConfig, RECMoEOutput, RiskConditionPacket
from .r_ecmoe_baseline_evaluation import _load_extended_yaml, _load_released_w1
from .r_ecmoe_evaluation import (
    EVALUATION_DECISIONS,
    REGISTERED_VARIANTS,
    EvaluationVariant,
    _atomic_csv,
    _atomic_json,
    _execute_variant,
    _percentile,
)
from .r_ecmoe_replay import RealizedHistoricalReplay


PRIMARY_RUN_META_SCHEMA = "dcvpp.r_ecmoe.run-meta.v1"
PRIMARY_TRAINING_SCHEMA = "dcvpp.r_ecmoe.training.v1"


class SLAuditError(RuntimeError):
    """The read-only SLA audit cannot preserve its 2024 evidence boundary."""


@dataclass(frozen=True)
class LoadedSLAuditRuntime:
    model: RECMoE
    checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    run_config_path: Path
    resolved_config: Mapping[str, Any]
    run_root: Path
    run_id: str
    device: torch.device
    cost_scale: float
    sla_threshold: float
    world_config: WorldMoEConfig
    world_mapping: Mapping[str, Any]
    repository_root: Path
    w1_checkpoint_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_positive(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise SLAuditError(f"{label} must be finite and positive")
    return float(value)


def load_primary_latest_for_sla_audit(
    latest_checkpoint: str | Path,
    run_config: str | Path,
    *,
    device: torch.device | str,
) -> LoadedSLAuditRuntime:
    """Load primary ``latest.pt`` without applying checkpoint eligibility gates."""

    checkpoint_path = Path(latest_checkpoint).resolve()
    if checkpoint_path.name != "latest.pt" or not checkpoint_path.is_file():
        raise SLAuditError("SLA audit accepts only primary latest.pt")
    run_root = checkpoint_path.parent.parent
    config_path = Path(run_config).resolve()
    expected_config = (run_root / "config_resolved.yaml").resolve()
    if config_path != expected_config or not config_path.is_file():
        raise SLAuditError("SLA audit requires the same run config_resolved.yaml")
    meta_path = run_root / "run_meta.json"
    if not meta_path.is_file():
        raise SLAuditError("primary latest checkpoint omitted run metadata")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(meta, Mapping) or not isinstance(resolved, Mapping):
        raise SLAuditError("primary run metadata/config is malformed")
    if meta.get("schema") != PRIMARY_RUN_META_SCHEMA:
        raise SLAuditError("SLA audit accepts only a primary R run")
    if resolved.get("schema") != PRIMARY_TRAINING_SCHEMA:
        raise SLAuditError("primary R resolved config schema is invalid")
    required_meta = {
        "phase": "r_ecmoe",
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
    }
    if any(meta.get(key) != expected for key, expected in required_meta.items()):
        raise SLAuditError("primary R run metadata violates the audit boundary")
    source = resolved.get("source")
    model_mapping = resolved.get("model")
    selection = resolved.get("selection")
    if (
        not isinstance(source, Mapping)
        or source.get("locked_test_enabled") is not False
        or not isinstance(model_mapping, Mapping)
        or not isinstance(selection, Mapping)
    ):
        raise SLAuditError("primary run config omitted source/model/selection")
    if int(model_mapping.get("active_top_k", 0)) != 2:
        raise SLAuditError("SLA audit primary learned policy must be Top-2")
    threshold = _finite_positive(selection.get("sla_tolerance"), label="SLA threshold")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SLAuditError("primary latest checkpoint is malformed")
    required_payload = {
        "schema": "dcvpp.r_ecmoe.runner.v1",
        "phase": "r_ecmoe",
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
    }
    if any(payload.get(key) != expected for key, expected in required_payload.items()):
        raise SLAuditError("primary latest checkpoint violates audit lineage")
    run_id = str(meta.get("run_id", ""))
    if not run_id or payload.get("run_id") != run_id:
        raise SLAuditError("primary latest checkpoint/run id disagrees")
    if payload.get("tuned_dynamics_state") not in (None, {}):
        raise SLAuditError("SLA audit requires frozen released W1 dynamics")
    update = payload.get("global_update")
    if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
        raise SLAuditError("primary latest checkpoint contains no trained update")

    original_config = Path(str(meta.get("config_path", ""))).resolve()
    world_config_path = Path(str(meta.get("world_config_path", ""))).resolve()
    if not original_config.is_file() or not world_config_path.is_file():
        raise SLAuditError("primary run source configs are unavailable")
    repository_root = original_config.parent.parent.resolve()
    world_mapping = _load_extended_yaml(world_config_path)
    raw_world = world_mapping.get("model")
    if not isinstance(raw_world, Mapping):
        raise SLAuditError("world config omitted model mapping")
    world_config = WorldMoEConfig.from_mapping(raw_world)
    source_w1 = meta.get("source_w1")
    if not isinstance(source_w1, Mapping):
        raise SLAuditError("primary run metadata omitted source W1")
    w1_path = Path(str(source_w1.get("path", ""))).resolve()
    if not w1_path.is_file():
        raise SLAuditError("released W1 checkpoint is unavailable")
    world_model = _load_released_w1(world_config, w1_path)
    rec_config = RECMoEConfig.from_mapping(dict(model_mapping))
    model = RECMoE(world_model, rec_config)
    r_state = payload.get("r_state")
    if not isinstance(r_state, Mapping) or any(
        str(name).startswith("world_model.") for name in r_state
    ):
        raise SLAuditError("primary latest R state is malformed")
    incompatible = model.load_state_dict(r_state, strict=False)
    non_world_missing = [
        name for name in incompatible.missing_keys if not name.startswith("world_model.")
    ]
    if non_world_missing or incompatible.unexpected_keys:
        raise SLAuditError("primary latest R state does not load exactly")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise SLAuditError("CUDA was requested but is unavailable")
    model.to(resolved_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scale = _finite_positive(
        payload.get("train_frozen_cost_scale"), label="train-only cost scale"
    )
    return LoadedSLAuditRuntime(
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint=payload,
        run_config_path=config_path,
        resolved_config=resolved,
        run_root=run_root,
        run_id=run_id,
        device=resolved_device,
        cost_scale=scale,
        sla_threshold=threshold,
        world_config=world_config,
        world_mapping=world_mapping,
        repository_root=repository_root,
        w1_checkpoint_path=w1_path,
    )


def _normalized_batch_mean_abs(value: Tensor, scale: Tensor) -> Tensor:
    numerator = value.detach().float().abs().flatten(start_dim=1).mean(dim=1)
    denominator = scale.detach().float().abs()
    if denominator.ndim > 1:
        denominator = denominator.flatten(start_dim=1).mean(dim=1)
    return numerator / denominator.clamp_min(1.0e-6)


def sla_components(output: RECMoEOutput) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-sample SLA severity, deadline fraction and unserved severity."""

    active = output.decision_active_mask.detach().float()
    constraints = output.expert_packet.constraints
    reference = output.expert_packet.reference
    workload = constraints.initial_queue.detach().float() + (
        reference.arrivals.detach().float() * active[..., None]
    ).sum(dim=1)
    expired = output.deadline_expired_work.detach().float() * active
    deadline_fraction = expired.flatten(start_dim=1).sum(dim=1) / workload.sum(
        dim=1
    ).clamp_min(1.0e-6)
    power_scale = (
        constraints.grid_import_limit.detach().float()
        + constraints.grid_export_limit.detach().float()
    )
    unserved_severity = _normalized_batch_mean_abs(
        output.actions.unserved_power.detach().float() * active[..., None],
        power_scale,
    )
    severity = 0.5 * (deadline_fraction + unserved_severity)
    return severity, deadline_fraction, unserved_severity


def _empirical_cvar95(values: Sequence[float]) -> tuple[float, int]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise SLAuditError("CVaR95 requires finite realized daily costs")
    count = max(1, int(math.ceil(0.05 * len(values))))
    tail = sorted((float(value) for value in values), reverse=True)[:count]
    return sum(tail) / count, count


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sample_rows(
    output: RECMoEOutput,
    *,
    dates: Sequence[str],
    variant: str,
    stage_id: int,
    origin: int,
    horizon: int,
    cost_scale: float,
    latency_ms_per_sample: float,
) -> list[dict[str, Any]]:
    severity, deadline, unserved = sla_components(output)
    active_horizon = output.decision_active_mask.detach().float().sum(dim=1)
    if not bool(
        torch.equal(active_horizon, active_horizon.new_full(active_horizon.shape, horizon))
    ):
        raise SLAuditError("audit replay active horizon differs from registration")
    denominator = float(cost_scale) * active_horizon / 96.0
    raw_cost = output.differentiable_operating_cost.detach().float()
    normalized_cost = raw_cost / denominator
    rows: list[dict[str, Any]] = []
    for index, operating_date in enumerate(dates):
        rows.append(
            {
                "operating_date": operating_date,
                "variant": variant,
                "stage_id": stage_id,
                "decision_origin": origin,
                "horizon": horizon,
                "decision_label": (
                    f"{'da' if stage_id == 0 else 'rt'}_o{origin}_h{horizon}"
                ),
                "sla_severity": float(severity[index].cpu()),
                "deadline_fraction": float(deadline[index].cpu()),
                "unserved_normalized_severity": float(unserved[index].cpu()),
                "realized_operating_cost": float(raw_cost[index].cpu()),
                "cost_normalization_denominator": float(denominator[index].cpu()),
                "normalized_operating_cost": float(normalized_cost[index].cpu()),
                "latency_ms_per_sample": float(latency_ms_per_sample),
                "outcome_basis": "realized_historical_replay",
            }
        )
    return rows


def _summary_stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    return {
        f"mean_{prefix}": sum(values) / len(values),
        f"p95_{prefix}": _percentile(values, 0.95),
        f"max_{prefix}": max(values),
    }


def _origin_summaries(
    daily_rows: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = {}
    for row in daily_rows:
        key = (
            str(row["variant"]),
            int(row["stage_id"]),
            int(row["decision_origin"]),
            int(row["horizon"]),
        )
        grouped.setdefault(key, []).append(row)
    result = []
    for (variant, stage, origin, horizon), rows in sorted(grouped.items()):
        costs = [float(row["normalized_operating_cost"]) for row in rows]
        cvar, tail_count = _empirical_cvar95(costs)
        raw_costs = [float(row["realized_operating_cost"]) for row in rows]
        raw_cvar, _ = _empirical_cvar95(raw_costs)
        severity_values = [float(row["sla_severity"]) for row in rows]
        result.append(
            {
                "variant": variant,
                "stage_id": stage,
                "decision_origin": origin,
                "horizon": horizon,
                "decision_label": f"{'da' if stage == 0 else 'rt'}_o{origin}_h{horizon}",
                "days": len(rows),
                "registered_sla_threshold": threshold,
                **_summary_stats(severity_values, "sla_severity"),
                "mean_sla_severity_minus_threshold": (
                    sum(severity_values) / len(severity_values) - threshold
                ),
                **_summary_stats(
                    [float(row["deadline_fraction"]) for row in rows],
                    "deadline_fraction",
                ),
                **_summary_stats(
                    [float(row["unserved_normalized_severity"]) for row in rows],
                    "unserved_normalized_severity",
                ),
                "mean_normalized_operating_cost": sum(costs) / len(costs),
                "cvar95_normalized_operating_cost": cvar,
                "mean_realized_operating_cost": sum(raw_costs) / len(raw_costs),
                "cvar95_realized_operating_cost": raw_cvar,
                "cvar95_tail_days": tail_count,
                "outcome_basis": "realized_historical_replay",
            }
        )
    return result


def _year_summaries(
    daily_rows: Sequence[Mapping[str, Any]],
    origin_rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for row in daily_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    result = []
    for variant, rows in sorted(by_variant.items()):
        by_date: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_date.setdefault(str(row["operating_date"]), []).append(row)
        if any(len(values) != len(EVALUATION_DECISIONS) for values in by_date.values()):
            raise SLAuditError("each audit date must have all six decisions")
        daily_costs = [
            sum(float(row["normalized_operating_cost"]) for row in values)
            / len(values)
            for _, values in sorted(by_date.items())
        ]
        daily_raw_costs = [
            sum(float(row["realized_operating_cost"]) for row in values)
            / len(values)
            for _, values in sorted(by_date.items())
        ]
        cvar, tail_count = _empirical_cvar95(daily_costs)
        raw_cvar, _ = _empirical_cvar95(daily_raw_costs)
        variant_origins = [row for row in origin_rows if row["variant"] == variant]
        severity = [float(row["sla_severity"]) for row in rows]
        deadline = [float(row["deadline_fraction"]) for row in rows]
        unserved = [float(row["unserved_normalized_severity"]) for row in rows]
        result.append(
            {
                "variant": variant,
                "days": len(by_date),
                "decisions_per_day": len(EVALUATION_DECISIONS),
                "registered_sla_threshold": threshold,
                **_summary_stats(severity, "sla_severity"),
                **_summary_stats(deadline, "deadline_fraction"),
                **_summary_stats(unserved, "unserved_normalized_severity"),
                "worst_origin_mean_sla_severity": max(
                    float(row["mean_sla_severity"]) for row in variant_origins
                ),
                "mean_normalized_operating_cost": sum(daily_costs) / len(daily_costs),
                "cvar95_normalized_operating_cost": cvar,
                "mean_realized_operating_cost": sum(daily_raw_costs)
                / len(daily_raw_costs),
                "cvar95_realized_operating_cost": raw_cvar,
                "cvar95_tail_days": tail_count,
                "outcome_basis": "realized_historical_replay",
            }
        )
    return result


class PrimarySLA2024Audit:
    """Compare learned Top-2 and zero-raw safe tensor without selecting a model."""

    def __init__(
        self,
        runtime: LoadedSLAuditRuntime,
        corpus: Any,
        output_root: str | Path,
        *,
        batch_size: int = 32,
        expected_validation_days: int = 364,
    ) -> None:
        if not isinstance(runtime, LoadedSLAuditRuntime):
            raise SLAuditError("runtime must be a loaded primary SLA audit runtime")
        if not callable(getattr(corpus, "batches", None)):
            raise SLAuditError("corpus must expose validation batches")
        self.runtime = runtime
        self.model = runtime.model
        self.corpus = corpus
        self.output_root = Path(output_root).resolve()
        self.batch_size = int(batch_size)
        self.expected_validation_days = int(expected_validation_days)
        if self.batch_size < 1 or self.expected_validation_days < 2:
            raise SLAuditError("audit batch/day counts must be positive")
        selected = {
            variant.name: variant
            for variant in REGISTERED_VARIANTS
            if variant.name in {"top2", "safe_tensor"}
        }
        if set(selected) != {"top2", "safe_tensor"}:
            raise SLAuditError("registered Top-2/safe tensor audit variants are missing")
        self.variants: tuple[EvaluationVariant, ...] = (
            selected["top2"],
            selected["safe_tensor"],
        )
        self.replay = RealizedHistoricalReplay.from_model(self.model)

    def _cost_scale(self, batch: Mapping[str, Any]) -> float:
        value = batch.get("train_frozen_baseline_daily_cost")
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise SLAuditError("validation batch omitted train-only cost scale")
        numeric = float(value.detach().float().cpu())
        if not math.isclose(
            numeric, self.runtime.cost_scale, rel_tol=1.0e-7, abs_tol=1.0e-7
        ):
            raise SLAuditError("audit cost scale differs from training checkpoint")
        return numeric

    def run(self) -> Mapping[str, Any]:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        reference_dates: tuple[str, ...] | None = None
        w1_encode_calls = 0
        variant_decision_calls = 0
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
                    inputs = batch.get("world_input")
                    if (
                        not dates
                        or inputs is None
                        or len(dates) != int(inputs.fast_values.shape[0])
                    ):
                        raise SLAuditError("validation batch omitted dates/world_input")
                    if any(date.fromisoformat(value).year != 2024 for value in dates):
                        raise SLAuditError("SLA audit attempted non-2024 data")
                    if not bool((inputs.origin_index == origin).all()):
                        raise SLAuditError("SLA audit packet origin mismatch")
                    dates_for_decision.extend(dates)
                    anchors = batch.get("forecast_anchors", {})
                    caps = batch.get("forecast_residual_caps", {})
                    da_anchor = anchors.get("da") if isinstance(anchors, Mapping) else None
                    da_cap = caps.get("da") if isinstance(caps, Mapping) else None
                    if not isinstance(da_anchor, Tensor) or not isinstance(da_cap, Tensor):
                        raise SLAuditError("validation batch omitted causal DA anchors")
                    risk = RiskConditionPacket.defaults(
                        len(dates),
                        device=self.runtime.device,
                        horizon=horizon,
                        stage=stage_id,
                    )
                    before = int(self.model.w1_backbone_calls)
                    _synchronize(self.runtime.device)
                    started = time.perf_counter()
                    with _autocast(self.runtime.device):
                        packet = self.model.encode_expert_context(
                            inputs,
                            da_anchor=da_anchor,
                            da_residual_cap=da_cap,
                        )
                        decisions = [
                            (variant, _execute_variant(self.model, packet, risk, variant))
                            for variant in self.variants
                        ]
                    _synchronize(self.runtime.device)
                    latency = (time.perf_counter() - started) * 1000.0 / len(dates)
                    if int(self.model.w1_backbone_calls) - before != 1:
                        raise SLAuditError(
                            "learned and safe audit must share one W1 packet"
                        )
                    w1_encode_calls += 1
                    variant_decision_calls += len(decisions)
                    scale = self._cost_scale(batch)
                    # Both decisions are complete before any realized target is
                    # admitted to the shared historical replay boundary.
                    for variant, decision in decisions:
                        realized = self.replay(
                            decision,
                            batch=batch,
                            risk=risk,
                            source_split="validation",
                        )
                        replay_calls += 1
                        rows.extend(
                            _sample_rows(
                                realized,
                                dates=dates,
                                variant=(
                                    "learned_top2"
                                    if variant.name == "top2"
                                    else "safe_tensor_zero_raw"
                                ),
                                stage_id=stage_id,
                                origin=origin,
                                horizon=horizon,
                                cost_scale=scale,
                                latency_ms_per_sample=latency,
                            )
                        )
                if (
                    len(dates_for_decision) != self.expected_validation_days
                    or len(set(dates_for_decision)) != self.expected_validation_days
                ):
                    raise SLAuditError("audit did not consume complete 2024 validation")
                ordered = tuple(dates_for_decision)
                if reference_dates is None:
                    reference_dates = ordered
                elif ordered != reference_dates:
                    raise SLAuditError("audit decisions do not share identical dates")
        origin_summary = _origin_summaries(rows, self.runtime.sla_threshold)
        year_summary = _year_summaries(
            rows, origin_summary, self.runtime.sla_threshold
        )
        by_variant = {str(row["variant"]): row for row in year_summary}
        learned = by_variant["learned_top2"]
        safe = by_variant["safe_tensor_zero_raw"]
        threshold = self.runtime.sla_threshold
        audit_conclusion = {
            "registered_sla_threshold": threshold,
            "learned_worst_origin_mean_sla": learned[
                "worst_origin_mean_sla_severity"
            ],
            "safe_worst_origin_mean_sla": safe[
                "worst_origin_mean_sla_severity"
            ],
            "threshold_below_safe_baseline_level": threshold
            < float(safe["worst_origin_mean_sla_severity"]),
            "learned_minus_safe_worst_origin_sla": float(
                learned["worst_origin_mean_sla_severity"]
            )
            - float(safe["worst_origin_mean_sla_severity"]),
            "threshold_was_not_changed": True,
            "checkpoint_was_not_selected": True,
        }
        _atomic_csv(self.output_root / "sla_audit_daily.csv", rows)
        _atomic_csv(self.output_root / "sla_audit_by_origin.csv", origin_summary)
        _atomic_csv(self.output_root / "sla_audit_year.csv", year_summary)
        meta = {
            "schema": "dcvpp.r_ecmoe.sla_gate_audit_2024.v1",
            "recorded_at_utc": _utc_now(),
            "split": "validation",
            "year": 2024,
            "validation_days": self.expected_validation_days,
            "decisions_per_day": len(EVALUATION_DECISIONS),
            "decision_origins": [value[1] for value in EVALUATION_DECISIONS],
            "variants": ["learned_top2", "safe_tensor_zero_raw"],
            "same_w1_packet_per_batch": True,
            "outcome_basis": "realized_historical_replay",
            "source_checkpoint": str(self.runtime.checkpoint_path),
            "checkpoint_role": "latest_read_only_may_be_ineligible",
            "run_config": str(self.runtime.run_config_path),
            "run_id": self.runtime.run_id,
            "w1_checkpoint": str(self.runtime.w1_checkpoint_path),
            "train_frozen_baseline_daily_cost": self.runtime.cost_scale,
            "w1_encode_calls": w1_encode_calls,
            "variant_decision_calls": variant_decision_calls,
            "realized_replay_calls": replay_calls,
            "audit_conclusion": audit_conclusion,
            "selection_eligibility_ignored": True,
            "selector_invoked": False,
            "threshold_modified": False,
            "best_checkpoint_written": False,
            "locked_2025_opened": False,
            "external_solver": False,
            "candidate_search": False,
            "hash_computed": False,
        }
        _atomic_json(self.output_root / "sla_gate_audit_2024.json", meta)
        return meta


__all__ = [
    "LoadedSLAuditRuntime",
    "PRIMARY_RUN_META_SCHEMA",
    "PRIMARY_TRAINING_SCHEMA",
    "PrimarySLA2024Audit",
    "SLAuditError",
    "load_primary_latest_for_sla_audit",
    "sla_components",
]
