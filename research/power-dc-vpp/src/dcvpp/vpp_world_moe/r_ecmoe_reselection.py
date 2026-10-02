"""Read-only 2024 revalidation and selector-only R checkpoint reselection."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from .r_ecmoe_runner import (
    TRAIN_DECISION_SCHEDULE,
    RECMoERunner,
    RECMoERunnerConfig,
)
from .r_ecmoe_sla_audit import LoadedSLAuditRuntime


RESELECTION_META_SCHEMA = "dcvpp.r_ecmoe.reselection.run-meta.v1"
RESELECTION_SUMMARY_SCHEMA = "dcvpp.r_ecmoe.reselection.summary.v1"
OLD_RUN_INVALID_REASON = (
    "old_selector_used_sla_0.01_below_safe_worst_origin_0.04083646_"
    "and_penalized_latency_inside_budget"
)
AUDITED_SAFE_WORST_ORIGIN_SLA = 0.04083646
AUDITED_LEARNED_WORST_ORIGIN_SLA = 0.01646862
EXPECTED_VALIDATION_DAYS = 364
EXPECTED_VALIDATION_ORIGINS = (0, 4, 16, 48, 95)


class ReselectionError(RuntimeError):
    """Selector-only revalidation cannot preserve the independent lineage."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selection_without_sla(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("sla_tolerance", None)
    return result


def _runner_config_from_resolved(
    resolved: Mapping[str, Any],
) -> RECMoERunnerConfig:
    """Independently reconstruct the complete runner contract from YAML."""

    training = resolved.get("training", {})
    loss = resolved.get("loss", {})
    selection = resolved.get("selection", {})
    outputs = resolved.get("outputs", {})
    if not all(
        isinstance(value, Mapping) for value in (training, loss, selection, outputs)
    ):
        raise ReselectionError(
            "resolved training/loss/selection/outputs must be mappings"
        )

    def optional_int(name: str) -> int | None:
        value = training.get(name)
        return None if value is None else int(value)

    raw_schedule = training.get(
        "train_decision_schedule", TRAIN_DECISION_SCHEDULE
    )
    raw_origins = training.get(
        "validation_origins",
        (
            (training["validation_origin"],)
            if "validation_origin" in training
            else EXPECTED_VALIDATION_ORIGINS
        ),
    )
    try:
        schedule = tuple(
            tuple(int(item) for item in decision) for decision in raw_schedule
        )
        origins = tuple(int(item) for item in raw_origins)
    except (TypeError, ValueError) as exc:
        raise ReselectionError("resolved decision schedule is malformed") from exc
    raw = {
        "epochs": int(training.get("epochs", training.get("maximum_epochs", 80))),
        "ra_epochs": int(
            training.get("ra_epochs", training.get("stage_ra_epochs", 13))
        ),
        "batch_size": int(training.get("batch_size", 32)),
        "validation_batch_size": int(training.get("validation_batch_size", 16)),
        "learning_rate": float(
            training.get(
                "learning_rate", training.get("base_learning_rate", 2.0e-4)
            )
        ),
        "weight_decay": float(training.get("weight_decay", 1.0e-4)),
        "adamw_beta2": float(training.get("adamw_beta2", 0.95)),
        "gradient_clip_norm": float(training.get("gradient_clip_norm", 1.0)),
        "warmup_updates": int(training.get("warmup_updates", 0)),
        "cosine_decay_updates": int(training.get("cosine_decay_updates", 10_000)),
        "minimum_learning_rate_ratio": float(
            training.get("minimum_learning_rate_ratio", 0.10)
        ),
        "use_bf16": str(training.get("precision", "bf16")).lower() == "bf16",
        "allow_tf32": bool(training.get("allow_tf32", True)),
        "fused_adamw": bool(training.get("fused_adamw", True)),
        "seed": int(training.get("seed", 20240813)),
        "max_train_days": optional_int("max_train_days"),
        "max_validation_days": optional_int("max_validation_days"),
        "smoke_max_days": int(training.get("smoke_max_days", 4)),
        "train_decision_schedule": schedule,
        "validation_origins": origins,
        "checkpoint_interval_updates": int(
            training.get("checkpoint_interval_updates", 100)
        ),
        "train_log_interval_updates": int(
            training.get("train_log_interval_updates", 50)
        ),
        "validation_interval_epochs": int(
            training.get("validation_interval_epochs", 2)
        ),
        "sentinel_validation_days": int(
            training.get("sentinel_validation_days", 16)
        ),
        "full_validation_interval_epochs": int(
            training.get("full_validation_interval_epochs", 10)
        ),
        "early_stop_patience": int(training.get("early_stop_patience", 3)),
        "early_stop_min_relative_improvement": float(
            training.get("early_stop_min_relative_improvement", 0.001)
        ),
        "rolling_load_decay": float(training.get("rolling_load_decay", 0.90)),
        "safe_anchor_decay_power": float(
            training.get("safe_anchor_decay_power", 1.0)
        ),
        "decision_risk_weight": float(
            selection.get(
                "decision_risk_weight", selection.get("risk_weight", 0.25)
            )
        ),
        "decision_sla_weight": float(
            selection.get("decision_sla_weight", selection.get("sla_penalty", 1.0))
        ),
        "decision_constraint_weight": float(
            selection.get(
                "decision_constraint_weight",
                selection.get("constraint_penalty", 1.0),
            )
        ),
        "decision_latency_penalty": float(
            selection.get("decision_latency_penalty", 1.0)
        ),
        "hard_residual_tolerance": float(
            selection.get("hard_residual_tolerance", 1.0e-4)
        ),
        "sla_tolerance": float(
            selection.get(
                "sla_tolerance", selection.get("critical_queue_tolerance", 0.0)
            )
        ),
        "latency_limit_ms_per_sample": float(
            selection.get(
                "latency_limit_ms_per_sample",
                selection.get("latency_budget_ms", 1900.0),
            )
        ),
        "selection_latency_warmup": int(
            selection.get("selection_latency_warmup", 5)
        ),
        "selection_latency_repeats": int(
            selection.get("selection_latency_repeats", 30)
        ),
        "save_best_speed_feasible": bool(
            outputs.get(
                "save_best_speed_feasible",
                outputs.get("keep_best_speed_feasible", True),
            )
        ),
        "loss_weights": {
            "expected_operating_cost": float(loss.get("expected_cost", 1.0)),
            "upper_tail_cost": float(loss.get("tail_cost", 0.25)),
            "quantile_pinball": float(loss.get("quantile", 0.25)),
            "sla_deadline": float(loss.get("sla_deadline", 1.0)),
            "constraint_residual": float(loss.get("constraint", 1.0)),
            "route_continuation": float(loss.get("route_continuation", 0.02)),
            "load_balance": float(loss.get("load_balance", 0.01)),
            "tail_risk": float(loss.get("tail_expert", 0.05)),
            "action_smoothness": float(loss.get("smoothness", 0.02)),
            "safe_anchor": float(loss.get("behavior_anchor", 0.10)),
        },
    }
    return RECMoERunnerConfig.from_mapping(raw)


def validate_selector_revision(
    source_resolved: Mapping[str, Any],
    selector_resolved: Mapping[str, Any],
    runner_config: RECMoERunnerConfig,
) -> None:
    """Allow only the audited SLA threshold plus the code-level latency repair."""

    for section in ("source", "model", "training", "loss"):
        if source_resolved.get(section) != selector_resolved.get(section):
            raise ReselectionError(
                f"selector-only revalidation changed frozen {section} configuration"
            )
    source_selection = source_resolved.get("selection")
    revised_selection = selector_resolved.get("selection")
    if not isinstance(source_selection, Mapping) or not isinstance(
        revised_selection, Mapping
    ):
        raise ReselectionError("source/revised selector configuration is missing")
    if _selection_without_sla(source_selection) != _selection_without_sla(
        revised_selection
    ):
        raise ReselectionError("selector-only revalidation changed non-SLA selection")
    old_sla = float(source_selection.get("sla_tolerance", math.nan))
    new_sla = float(revised_selection.get("sla_tolerance", math.nan))
    if not math.isclose(old_sla, 0.01, rel_tol=0.0, abs_tol=1.0e-12):
        raise ReselectionError("source run does not carry the invalidated SLA=0.01")
    if not math.isclose(new_sla, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
        raise ReselectionError("revised selector must use audited SLA=0.02")
    if not math.isclose(
        float(runner_config.sla_tolerance), 0.02, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ReselectionError("resolved runner selector does not use SLA=0.02")
    for key in (
        "decision_risk_weight",
        "decision_sla_weight",
        "decision_constraint_weight",
        "decision_latency_penalty",
        "latency_limit_ms_per_sample",
        "hard_residual_tolerance",
        "selection_latency_warmup",
        "selection_latency_repeats",
    ):
        if key not in revised_selection:
            raise ReselectionError(f"revised selector omitted {key}")
        expected = float(revised_selection[key])
        actual = float(getattr(runner_config, key))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ReselectionError(
                f"resolved runner configuration disagrees with selector {key}"
            )
    expected_runner = _runner_config_from_resolved(selector_resolved)
    actual_payload = asdict(runner_config)
    expected_payload = asdict(expected_runner)
    if actual_payload != expected_payload:
        different = sorted(
            key
            for key in expected_payload
            if actual_payload.get(key) != expected_payload[key]
        )
        raise ReselectionError(
            "resolved runner configuration disagrees with selector contract: "
            + ", ".join(different)
        )
    if tuple(int(value) for value in runner_config.validation_origins) != (
        EXPECTED_VALIDATION_ORIGINS
    ):
        raise ReselectionError("reselection requires the registered six decisions")
    if runner_config.max_validation_days is not None:
        raise ReselectionError("reselection cannot truncate the 364-day validation")


def _validate_full_evaluation_metrics(
    metrics: Mapping[str, Any], *, selection_eligible: bool
) -> None:
    required = {
        "validation_scope": "full",
        "outcome_basis": "realized_historical_replay",
        "decision_points_evaluated": 6,
        "decision_origin": "0,0,4,16,48,95",
    }
    if any(metrics.get(key) != expected for key, expected in required.items()):
        raise ReselectionError(
            "reselection did not execute the registered full evaluation"
        )
    try:
        days = float(metrics.get("realized_validation_days", math.nan))
    except (TypeError, ValueError) as exc:
        raise ReselectionError("reselection validation-day count is malformed") from exc
    if not math.isclose(
        days, float(EXPECTED_VALIDATION_DAYS), rel_tol=0.0, abs_tol=0.0
    ):
        raise ReselectionError("reselection did not consume all 364 validation days")
    if bool(metrics.get("selection_eligible")) != bool(selection_eligible):
        raise ReselectionError(
            "reselection eligibility result is internally inconsistent"
        )


def validate_audit_artifact(
    audit_artifact: str | Path,
    *,
    source_checkpoint: Path,
    source_cost_scale: float,
) -> Mapping[str, Any]:
    path = Path(audit_artifact).resolve()
    if not path.is_file():
        raise ReselectionError("SLA audit artifact is unavailable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReselectionError("SLA audit artifact is malformed")
    required = {
        "schema": "dcvpp.r_ecmoe.sla_gate_audit_2024.v1",
        "split": "validation",
        "year": 2024,
        "validation_days": 364,
        "decisions_per_day": 6,
        "outcome_basis": "realized_historical_replay",
        "selector_invoked": False,
        "threshold_modified": False,
        "best_checkpoint_written": False,
        "locked_2025_opened": False,
    }
    if any(payload.get(key) != expected for key, expected in required.items()):
        raise ReselectionError("SLA audit artifact violates reselection evidence")
    if Path(str(payload.get("source_checkpoint", ""))).resolve() != source_checkpoint:
        raise ReselectionError("SLA audit and reselection source checkpoints differ")
    try:
        audit_scale = float(payload.get("train_frozen_baseline_daily_cost", math.nan))
        source_scale = float(source_cost_scale)
    except (TypeError, ValueError) as exc:
        raise ReselectionError("SLA audit cost scale is malformed") from exc
    if (
        not math.isfinite(source_scale)
        or source_scale <= 0.0
        or not math.isclose(
            audit_scale, source_scale, rel_tol=1.0e-7, abs_tol=1.0e-8
        )
    ):
        raise ReselectionError("SLA audit and source latest cost scales differ")
    conclusion = payload.get("audit_conclusion")
    if not isinstance(conclusion, Mapping):
        raise ReselectionError("SLA audit omitted its attainability conclusion")
    safe = float(conclusion.get("safe_worst_origin_mean_sla", math.nan))
    learned = float(conclusion.get("learned_worst_origin_mean_sla", math.nan))
    if not math.isclose(
        safe, AUDITED_SAFE_WORST_ORIGIN_SLA, rel_tol=1.0e-7, abs_tol=1.0e-8
    ) or not math.isclose(
        learned,
        AUDITED_LEARNED_WORST_ORIGIN_SLA,
        rel_tol=1.0e-7,
        abs_tol=1.0e-8,
    ):
        raise ReselectionError("SLA audit values differ from the registered repair basis")
    if conclusion.get("threshold_below_safe_baseline_level") is not True:
        raise ReselectionError("SLA audit did not invalidate the old threshold")
    return payload


def _checkpoint_payload(
    *,
    runner: RECMoERunner,
    metrics: Mapping[str, Any],
    source: LoadedSLAuditRuntime,
    audit_path: Path,
    selector_config_path: Path,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema": "dcvpp.r_ecmoe.runner.v1",
        "phase": "r_ecmoe",
        "w1_interface_mode": "single_final_risk_slot_packet_plus_parallel_response",
        "lineage_operation": "read_only_2024_revalidation_and_reselection",
        "training_not_repeated": True,
        "resume_for_training_allowed": False,
        "source_checkpoint": str(source.checkpoint_path),
        "source_run_id": source.run_id,
        "source_epoch": int(source.checkpoint.get("epoch", -1)),
        "source_global_update": int(source.checkpoint.get("global_update", -1)),
        "selector_config": str(selector_config_path),
        "audit_artifact": str(audit_path),
        "old_run_invalid_reason": OLD_RUN_INVALID_REASON,
        "freeze_state": {
            "w0": True,
            "w1_experts": True,
            "w1_router": True,
            "w1_dynamics": True,
            "r_trainable": False,
        },
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
        "outcome_basis": "realized_historical_replay",
        "run_id": run_id,
        "saved_at_utc": _utc_now(),
        "epoch": int(source.checkpoint.get("epoch", -1)),
        "global_update": int(source.checkpoint.get("global_update", -1)),
        "stage": "rb",
        "r_state": runner._r_state(),
        "tuned_dynamics_state": None,
        "resolved_config": asdict(runner.config),
        "metrics": dict(metrics),
        "source_paths": (
            str(source.checkpoint_path),
            str(source.run_config_path),
            str(selector_config_path),
            str(audit_path),
            str(source.w1_checkpoint_path),
        ),
        "train_frozen_cost_scale": runner._frozen_cost_scale,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "selection_formula_revision": {
            "latency_penalty": "max(0,p95/limit-1)",
            "sla_tolerance": 0.02,
        },
    }


def revalidate_and_reselect(
    source: LoadedSLAuditRuntime,
    corpus: Any,
    *,
    selector_resolved: Mapping[str, Any],
    selector_config_path: str | Path,
    runner_config: RECMoERunnerConfig,
    audit_artifact: str | Path,
    output_root: str | Path,
    run_id: str,
    source_paths: Sequence[str | Path] = (),
) -> Mapping[str, Any]:
    """Run one full validation and write a best only if the repaired selector passes."""

    if not isinstance(source, LoadedSLAuditRuntime):
        raise ReselectionError("source must be a loaded primary latest runtime")
    if source.checkpoint.get("epoch_completed") is not True:
        raise ReselectionError(
            "reselection requires a completed source latest checkpoint"
        )
    if not run_id or Path(run_id).name != run_id:
        raise ReselectionError("reselection run_id must be one path component")
    selector_path = Path(selector_config_path).resolve()
    if not selector_path.is_file():
        raise ReselectionError("selector config path is unavailable")
    audit_path = Path(audit_artifact).resolve()
    validate_selector_revision(
        source.resolved_config, selector_resolved, runner_config
    )
    validate_audit_artifact(
        audit_path,
        source_checkpoint=source.checkpoint_path,
        source_cost_scale=source.cost_scale,
    )
    root = Path(output_root).resolve()
    run_root = root / run_id
    if run_root.exists():
        raise ReselectionError("independent reselection lineage already exists")
    run_root.mkdir(parents=True)
    resolved_output = {
        **dict(selector_resolved),
        "reselection": {
            "training_not_repeated": True,
            "source_checkpoint": str(source.checkpoint_path),
            "source_run_id": source.run_id,
            "audit_artifact": str(audit_path),
            "old_run_invalid_reason": OLD_RUN_INVALID_REASON,
            "validation_scope": "full_2024_364_days_six_decisions",
            "selector_formula": "cost+risk*cvar+sla+constraint+latency*max(0,p95/limit-1)",
        },
    }
    meta = {
        "schema": RESELECTION_META_SCHEMA,
        "phase": "r_ecmoe_revalidation_reselection",
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "training_not_repeated": True,
        "source_checkpoint": str(source.checkpoint_path),
        "source_run_id": source.run_id,
        "source_global_update": int(source.checkpoint.get("global_update", -1)),
        "train_frozen_baseline_daily_cost": float(source.cost_scale),
        "audit_artifact": str(audit_path),
        "old_run_invalid_reason": OLD_RUN_INVALID_REASON,
        "selector_formula_revision": {
            "latency_penalty": "max(0,p95/limit-1)",
            "sla_tolerance": 0.02,
        },
        "locked_test_opened": False,
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
        "best_checkpoint_written": False,
    }
    _atomic_yaml(run_root / "config_resolved.yaml", resolved_output)
    _atomic_json(run_root / "run_meta.json", meta)

    # The loader freezes inference parameters.  Re-enable only R-owned tensors
    # long enough for the unchanged runner to construct its structural optimizer;
    # no backward or optimizer step is ever invoked, and all tensors are frozen
    # again before full validation starts.
    for name, parameter in source.model.named_parameters():
        parameter.requires_grad_(not name.startswith("world_model."))
    runner = RECMoERunner(
        source.model,
        corpus,
        runner_config,
        root,
        run_id,
        source.device,
        (
            source.checkpoint_path,
            source.run_config_path,
            selector_path,
            audit_path,
            source.w1_checkpoint_path,
            *source_paths,
        ),
    )
    runner.epoch = int(source.checkpoint.get("epoch", 0))
    runner.global_update = int(source.checkpoint.get("global_update", 0))
    # The selector-only lineage must reuse the exact train-frozen scale stored
    # by source latest and audited on 2024; never derive it again from today's
    # corpus materialization.
    runner._frozen_cost_scale = float(source.cost_scale)
    runner._frozen_cost_scale_tensor = torch.tensor(
        source.cost_scale, device=runner.device, dtype=torch.float32
    )
    runner.stage = "rb"
    runner.model.configure_stage("rb")
    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)
    validation = runner.validate(
        epoch=max(1, runner.epoch),
        smoke=False,
        scope="full",
    )
    metrics = dict(validation.metrics)
    eligible = bool(validation.selection_eligible)
    _validate_full_evaluation_metrics(metrics, selection_eligible=eligible)
    best_path = run_root / "checkpoints" / "best_decision_value.pt"
    if eligible:
        payload = _checkpoint_payload(
            runner=runner,
            metrics=metrics,
            source=source,
            audit_path=audit_path,
            selector_config_path=selector_path,
            run_id=run_id,
        )
        best_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = best_path.with_suffix(best_path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, best_path)
    meta.update(
        {
            "completed_at_utc": _utc_now(),
            "selection_eligible": eligible,
            "decision_value": float(validation.decision_value),
            "best_checkpoint_written": eligible,
            "best_checkpoint": str(best_path) if eligible else None,
            "train_frozen_baseline_daily_cost": runner._frozen_cost_scale,
        }
    )
    _atomic_json(run_root / "run_meta.json", meta)
    summary = {
        "schema": RESELECTION_SUMMARY_SCHEMA,
        "run_id": run_id,
        "training_not_repeated": True,
        "source_checkpoint": str(source.checkpoint_path),
        "audit_artifact": str(audit_path),
        "old_run_invalid_reason": OLD_RUN_INVALID_REASON,
        "validation_scope": "full",
        "validation_days": int(float(metrics.get("realized_validation_days", 0))),
        "decision_points_evaluated": int(metrics.get("decision_points_evaluated", 0)),
        "selection_eligible": eligible,
        "decision_value": float(validation.decision_value),
        "best_checkpoint": str(best_path) if eligible else None,
        "locked_2025_opened": False,
        "training_updates_executed": 0,
        "optimizer_steps_executed": 0,
    }
    _atomic_json(run_root / "reselection_summary.json", summary)
    if not eligible:
        raise ReselectionError(
            "repaired full selector remains ineligible; no best checkpoint was written"
        )
    return summary


__all__ = [
    "AUDITED_LEARNED_WORST_ORIGIN_SLA",
    "AUDITED_SAFE_WORST_ORIGIN_SLA",
    "OLD_RUN_INVALID_REASON",
    "RESELECTION_META_SCHEMA",
    "RESELECTION_SUMMARY_SCHEMA",
    "ReselectionError",
    "revalidate_and_reselect",
    "validate_audit_artifact",
    "validate_selector_revision",
]
