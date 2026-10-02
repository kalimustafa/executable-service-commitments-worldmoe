"""Strict aggregation of measured R-ECMoE paper results.

This module never runs a model or opens a dataset.  It only consolidates an
eligible selector-repaired primary checkpoint and already-materialized metric
artifacts.  Missing optional experiments remain explicit ``NOT_RUN`` rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


RESULTS_SCHEMA = "dcvpp.r_ecmoe.measured_results.v1"
PRIMARY_META_SCHEMA = "dcvpp.r_ecmoe.reselection.run-meta.v1"
PRIMARY_CHECKPOINT_SCHEMA = "dcvpp.r_ecmoe.runner.v1"
SELECTOR_CANDIDATE_AUDIT_SCHEMA = (
    "dcvpp.r_ecmoe.corrected_selector_candidate_audit.v1"
)
SELECTOR_CANDIDATE_AUDIT_STATUS = (
    "PROVEN_SELECTED_EPOCH_MINIMUM_AMONG_RECORDED_FULL_CANDIDATES"
)
OUTCOME_BASIS = "realized_historical_replay"
OPTIONAL_BASELINES = {
    "top1_retrained": "retrained_top1_conditional_tail",
    "dense_direct": "dense_direct",
    "independent_r_moe": "independent_r_moe",
}
SAME_CHECKPOINT_NAMES = {
    "safe_tensor": "safe_tensor",
    "top1": "top1_same_checkpoint_intervention",
    "top2": "primary_top2",
    "no_decision_tail": "no_decision_tail_same_checkpoint_intervention",
    "no_route_prior": "no_route_prior_same_checkpoint_intervention",
    "tail_always_on": "tail_always_on_same_checkpoint_intervention",
    "pooled_latent_instead_of_expert_states": (
        "pooled_latent_same_checkpoint_intervention"
    ),
    "no_sparse_coordinator": "no_sparse_coordinator_same_checkpoint_intervention",
    "no_risk_condition": "no_risk_condition_same_checkpoint_intervention",
}


class RECMoEResultAggregationError(RuntimeError):
    """Measured artifacts cannot support the requested result row."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise RECMoEResultAggregationError(f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RECMoEResultAggregationError(f"JSON must be a mapping: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RECMoEResultAggregationError(f"required CSV is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RECMoEResultAggregationError(f"CSV cannot be empty: {path}")
    return rows


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise RECMoEResultAggregationError(f"{label} must be finite numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RECMoEResultAggregationError(
            f"{label} must be finite numeric"
        ) from exc
    if not math.isfinite(numeric):
        raise RECMoEResultAggregationError(f"{label} must be finite numeric")
    return numeric


def _truth(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise RECMoEResultAggregationError(f"{label} is not a boolean evidence field")


def _artifact_file(value: str | Path, filename: str) -> Path:
    path = Path(value).expanduser().resolve()
    return path / filename if path.is_dir() else path


def _same_path(left: Any, right: Path) -> bool:
    if not isinstance(left, (str, Path)) or not str(left):
        return False
    return Path(left).expanduser().resolve() == right.resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _load_primary(best_checkpoint: str | Path) -> dict[str, Any]:
    best = Path(best_checkpoint).expanduser().resolve()
    if best.name != "best_decision_value.pt" or not best.is_file():
        raise RECMoEResultAggregationError(
            "primary must be an existing best_decision_value.pt"
        )
    run_root = best.parent.parent
    meta = _json(run_root / "run_meta.json")
    if (
        meta.get("schema") != PRIMARY_META_SCHEMA
        or meta.get("training_not_repeated") is not True
        or meta.get("best_checkpoint_written") is not True
        or meta.get("selection_eligible") is not True
        or not _same_path(meta.get("best_checkpoint"), best)
        or not str(meta.get("old_run_invalid_reason", ""))
    ):
        raise RECMoEResultAggregationError(
            "primary is not the independent selector-repaired reselection best"
        )
    payload = torch.load(best, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RECMoEResultAggregationError("primary checkpoint is malformed")
    metrics = payload.get("metrics")
    if (
        payload.get("schema") != PRIMARY_CHECKPOINT_SCHEMA
        or payload.get("lineage_operation")
        != "read_only_2024_revalidation_and_reselection"
        or payload.get("training_not_repeated") is not True
        or payload.get("resume_for_training_allowed") is not False
        or payload.get("locked_2025_opened") is not False
        or not isinstance(metrics, Mapping)
        or metrics.get("validation_scope") != "full"
        or metrics.get("selection_eligible") is not True
        or metrics.get("outcome_basis") != OUTCOME_BASIS
    ):
        raise RECMoEResultAggregationError(
            "old invalid or ineligible checkpoint cannot be primary"
        )
    _finite(metrics.get("empirical_cvar95"), label="primary empirical_cvar95")
    _finite(metrics.get("worst_origin_sla"), label="primary worst_origin_sla")
    audit = Path(str(payload.get("audit_artifact", ""))).expanduser().resolve()
    if not audit.is_file():
        raise RECMoEResultAggregationError("primary checkpoint audit artifact is missing")
    return {
        "best": best,
        "run_root": run_root,
        "meta": dict(meta),
        "checkpoint": dict(payload),
        "metrics": dict(metrics),
        "audit": audit,
    }


def _load_selector_candidate_audit(
    value: str | Path | None,
    *,
    expected_checkpoint: Path,
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a corrected-selector audit, validating its substantive claim.

    The input remains optional for backward-compatible aggregation. When it is
    supplied, the aggregator recomputes the eligible-candidate minimum from the
    materialized candidate rows instead of trusting the status string alone.
    """

    if value is None:
        return {
            "status": "NOT_PROVIDED",
            "strictly_validated": False,
            "evidence_path": None,
            "recorded_full_candidate_count": None,
            "eligible_full_candidate_count": None,
            "selected_epoch": None,
            "selected_epoch_is_minimum_among_recorded_full_candidates": None,
            "minimum_corrected_decision_value": None,
            "validation_split": None,
        }

    path = _artifact_file(value, "corrected_selector_candidate_audit.json")
    raw = _json(path)
    if raw.get("schema") != SELECTOR_CANDIDATE_AUDIT_SCHEMA:
        raise RECMoEResultAggregationError(
            "selector candidate audit has the wrong schema"
        )
    if raw.get("status") != SELECTOR_CANDIDATE_AUDIT_STATUS:
        raise RECMoEResultAggregationError(
            "selector candidate audit does not prove the recorded full-candidate minimum"
        )
    if (
        raw.get("locked_2025_opened") is not False
        or raw.get("training_performed") is not False
        or raw.get("checkpoint_updated") is not False
        or not _same_path(raw.get("selected_checkpoint"), expected_checkpoint)
    ):
        raise RECMoEResultAggregationError(
            "selector candidate audit violates frozen read-only lineage"
        )

    try:
        selected_epoch = int(raw.get("selected_epoch"))
        checkpoint_epoch = int(checkpoint_payload.get("epoch"))
        recorded_count = int(raw.get("recorded_full_candidate_count"))
        eligible_count = int(raw.get("eligible_full_candidate_count"))
    except (TypeError, ValueError) as exc:
        raise RECMoEResultAggregationError(
            "selector candidate audit has invalid epoch or candidate counts"
        ) from exc
    if (
        selected_epoch < 1
        or selected_epoch != checkpoint_epoch
        or recorded_count < 1
        or eligible_count < 1
        or eligible_count > recorded_count
    ):
        raise RECMoEResultAggregationError(
            "selector candidate audit disagrees with checkpoint epoch or counts"
        )

    candidates = raw.get("candidates")
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or len(candidates) != recorded_count
    ):
        raise RECMoEResultAggregationError(
            "selector candidate audit candidate rows do not match the recorded count"
        )
    eligible_rows: list[tuple[float, int]] = []
    seen_epochs: set[int] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise RECMoEResultAggregationError(
                "selector candidate audit contains a malformed candidate row"
            )
        try:
            epoch = int(candidate.get("epoch"))
        except (TypeError, ValueError) as exc:
            raise RECMoEResultAggregationError(
                f"selector candidate audit candidate {index} has an invalid epoch"
            ) from exc
        if epoch < 1 or epoch in seen_epochs:
            raise RECMoEResultAggregationError(
                "selector candidate audit candidate epochs must be positive and unique"
            )
        seen_epochs.add(epoch)
        eligible = _truth(
            candidate.get("eligible_under_corrected_selector"),
            label=f"selector candidate {epoch} eligibility",
        )
        score = _finite(
            candidate.get("corrected_decision_value"),
            label=f"selector candidate {epoch} corrected decision value",
        )
        if eligible:
            eligible_rows.append((score, epoch))
    if len(eligible_rows) != eligible_count:
        raise RECMoEResultAggregationError(
            "selector candidate audit eligible rows do not match the eligible count"
        )
    minimum_score, minimum_epoch = min(eligible_rows, key=lambda item: item)
    declared_minimum = _finite(
        raw.get("minimum_corrected_decision_value"),
        label="selector candidate audit minimum corrected decision value",
    )
    if minimum_epoch != selected_epoch or not math.isclose(
        minimum_score,
        declared_minimum,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RECMoEResultAggregationError(
            "selector candidate audit selected epoch is not the recomputed minimum"
        )

    return {
        "status": SELECTOR_CANDIDATE_AUDIT_STATUS,
        "strictly_validated": True,
        "evidence_path": str(path),
        "recorded_full_candidate_count": recorded_count,
        "eligible_full_candidate_count": eligible_count,
        "selected_epoch": selected_epoch,
        "selected_epoch_is_minimum_among_recorded_full_candidates": True,
        "minimum_corrected_decision_value": declared_minimum,
        "validation_split": str(raw.get("validation_split", "")),
    }


def _paper_row(
    summary: Mapping[str, Any],
    *,
    model: str,
    evidence_role: str,
    status: str,
    split: str,
    year: int | str,
    checkpoint_relation: str,
    separately_retrained: bool,
    same_checkpoint_intervention: bool,
    source: Path | str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "model": model,
        "evidence_role": evidence_role,
        "split": split,
        "year": year,
        "days": summary.get("days", ""),
        "checkpoint_relation": checkpoint_relation,
        "separately_retrained_evidence": separately_retrained,
        "same_checkpoint_intervention": same_checkpoint_intervention,
        "outcome_basis": summary.get("outcome_basis", ""),
        "source": str(source),
    }
    mapping = {
        "mean_realized_operating_cost": "mean_realized_operating_cost",
        "mean_normalized_operating_cost": "mean_normalized_operating_cost",
        "cvar95_normalized_operating_cost": "cvar95_normalized_operating_cost",
        "mean_deadline_expired_work": "mean_deadline_expired_work",
        "mean_unserved_energy": "mean_unserved_energy",
        "maximum_hard_residual": "maximum_hard_residual",
        "mean_action_total_variation": "mean_action_total_variation",
        "mean_battery_throughput": "mean_battery_throughput",
        "mean_compute_energy": "mean_compute_energy",
        "mean_migration_energy": "mean_migration_energy",
        "mean_pv_curtailed_energy": "mean_pv_curtailed_energy",
        "mean_grid_import_energy": "mean_grid_import_energy",
        "mean_grid_export_energy": "mean_grid_export_energy",
        "mean_grid_peak_import_ratio": "mean_grid_peak_import_ratio",
        "mean_grid_peak_export_ratio": "mean_grid_peak_export_ratio",
        "mean_grid_peak_ramp_ratio": "mean_grid_peak_ramp_ratio",
        "mean_grid_rebound_ratio": "mean_grid_rebound_ratio",
        "mean_compute_concentration_hhi": "mean_compute_concentration_hhi",
        "mean_terminal_unfinished_work": "mean_terminal_unfinished_work",
        "mean_terminal_soc_fraction": "mean_terminal_soc_fraction",
        "mean_route_entropy": "mean_route_entropy",
        "mean_route_gini": "mean_route_gini",
        "mean_tail_activation_rate": "mean_tail_activation_rate",
        "mean_tail_probability": "mean_tail_probability",
        "mean_tail_event_rate": "mean_tail_event_rate",
        "mean_tail_probability_brier": "mean_tail_probability_brier",
        "mean_tail_calibration_gap": "mean_tail_calibration_gap",
        "tail_calibration_target": "tail_calibration_target",
    }
    for output_name, source_name in mapping.items():
        row[output_name] = summary.get(source_name, "")
    return row


def _load_primary_evaluation(
    evaluation: str | Path, primary_best: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    meta_path = _artifact_file(evaluation, "evaluation_2024_meta.json")
    root = meta_path.parent
    meta = _json(meta_path)
    if (
        meta.get("schema") != "dcvpp.r_ecmoe.evaluation_2024.v1"
        or int(meta.get("year", -1)) != 2024
        or int(meta.get("validation_days", -1)) != 364
        or meta.get("outcome_basis") != OUTCOME_BASIS
        or meta.get("comparison_design") != "same_checkpoint_intervention_only"
        or meta.get("shared_checkpoint_for_all_variants") is not True
        or meta.get("separately_retrained_variant_claims") is not False
        or not _same_path(meta.get("best_checkpoint"), primary_best)
    ):
        raise RECMoEResultAggregationError("primary 2024 evaluation lineage is invalid")
    summaries = _csv(root / "variant_summary.csv")
    indexed = {row.get("variant", ""): row for row in summaries}
    if set(SAME_CHECKPOINT_NAMES) - set(indexed):
        raise RECMoEResultAggregationError("primary evaluation omitted registered variants")
    paper_rows: list[dict[str, Any]] = []
    for raw_name, output_name in SAME_CHECKPOINT_NAMES.items():
        summary = indexed[raw_name]
        if (
            summary.get("comparison_design") != "same_checkpoint_intervention"
            or _truth(
                summary.get("separately_retrained"),
                label=f"{raw_name} separately_retrained",
            )
            or summary.get("outcome_basis") != OUTCOME_BASIS
            or int(float(summary.get("days", 0))) != 364
        ):
            raise RECMoEResultAggregationError(
                f"primary variant {raw_name} is not same-checkpoint evidence"
            )
        is_primary = raw_name == "top2"
        paper_rows.append(
            _paper_row(
                summary,
                model=output_name,
                evidence_role=("primary_model" if is_primary else "ablation_or_safe"),
                status="COMPLETED",
                split="validation",
                year=2024,
                checkpoint_relation=(
                    "primary_reselected_checkpoint"
                    if is_primary
                    else "shared_primary_checkpoint_intervention"
                ),
                # The main row is the primary checkpoint, not a separately
                # retrained comparator.  Only independent baseline runs may
                # carry separately_retrained_evidence=True.
                separately_retrained=False,
                same_checkpoint_intervention=not is_primary,
                source=root / "variant_summary.csv",
            )
        )
    comparisons: list[dict[str, Any]] = []
    for row in _csv(root / "paired_comparisons.csv"):
        if (
            row.get("primary_variant") != "top2"
            or row.get("comparison_design") != "same_checkpoint_intervention"
            or _truth(
                row.get("separately_retrained"),
                label="same-checkpoint separately_retrained",
            )
            or row.get("outcome_basis") != OUTCOME_BASIS
        ):
            raise RECMoEResultAggregationError(
                "same-checkpoint comparison mislabeled its evidence design"
            )
        comparisons.append(
            {
                "status": "COMPLETED",
                "primary_model": "primary_top2",
                "comparator": SAME_CHECKPOINT_NAMES.get(
                    str(row.get("comparator")), str(row.get("comparator"))
                ),
                "comparison_design": "same_checkpoint_intervention",
                "checkpoint_relation": "shared_primary_checkpoint",
                "separately_retrained_evidence": False,
                "same_checkpoint_intervention": True,
                "split": "validation",
                "year": 2024,
                "days": row.get("days", ""),
                "outcome_basis": OUTCOME_BASIS,
                "delta_direction": "primary_top2_minus_comparator",
                "cost_delta": row.get("cost_delta_top2_minus_comparator", ""),
                "cost_delta_ci95_lower": row.get("cost_delta_ci95_lower", ""),
                "cost_delta_ci95_upper": row.get("cost_delta_ci95_upper", ""),
                "deadline_delta": row.get("deadline_delta", ""),
                "unserved_delta": row.get("unserved_delta", ""),
                "source": str(root / "paired_comparisons.csv"),
            }
        )
    return paper_rows, comparisons, meta


def _load_baseline_evaluation(
    label: str, value: str | Path, primary_best: Path
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    expected = OPTIONAL_BASELINES[label]
    meta_path = _artifact_file(value, "baseline_evaluation_2024_meta.json")
    root = meta_path.parent
    meta = _json(meta_path)
    if (
        meta.get("comparison_design") != "separately_retrained_architecture"
        or meta.get("separately_retrained") is not True
        or meta.get("baseline_type") != expected
        or meta.get("outcome_basis") != OUTCOME_BASIS
        or int(meta.get("year", -1)) != 2024
        or int(meta.get("validation_days", -1)) != 364
        or meta.get("primary_top2_comparison") is not True
    ):
        raise RECMoEResultAggregationError(
            f"{label} is not a completed separately retrained evaluation"
        )
    checkpoint = Path(str(meta.get("best_checkpoint", ""))).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint == primary_best:
        raise RECMoEResultAggregationError(
            f"{label} does not have an independent trained checkpoint"
        )
    summary_rows = _csv(root / "baseline_summary.csv")
    if len(summary_rows) != 1:
        raise RECMoEResultAggregationError(f"{label} baseline summary must have one row")
    summary = summary_rows[0]
    if (
        summary.get("variant") != expected
        or summary.get("comparison_design")
        != "separately_retrained_architecture"
        or not _truth(
            summary.get("separately_retrained"), label=f"{label} separately_retrained"
        )
        or summary.get("outcome_basis") != OUTCOME_BASIS
    ):
        raise RECMoEResultAggregationError(f"{label} summary lineage is invalid")
    paired_rows = _csv(root / "paired_primary_top2.csv")
    if len(paired_rows) != 1:
        raise RECMoEResultAggregationError(f"{label} paired result must have one row")
    paired = paired_rows[0]
    if (
        paired.get("baseline_variant") != expected
        or paired.get("comparator") != "primary_top2"
        or paired.get("comparison_design")
        != "separately_retrained_vs_primary_top2"
        or not _truth(
            paired.get("baseline_separately_retrained"),
            label=f"{label} baseline retrained",
        )
        or paired.get("outcome_basis") != OUTCOME_BASIS
    ):
        raise RECMoEResultAggregationError(f"{label} paired lineage is invalid")
    paper = _paper_row(
        summary,
        model=label,
        evidence_role="separately_retrained_comparator",
        status="COMPLETED",
        split="validation",
        year=2024,
        checkpoint_relation="independent_trained_checkpoint",
        separately_retrained=True,
        same_checkpoint_intervention=False,
        source=root / "baseline_summary.csv",
    )
    lower = _finite(paired.get("cost_delta_ci95_lower"), label=f"{label} CI lower")
    upper = _finite(paired.get("cost_delta_ci95_upper"), label=f"{label} CI upper")
    comparison = {
        "status": "COMPLETED",
        "primary_model": "primary_top2",
        "comparator": label,
        "comparison_design": "separately_retrained_architecture",
        "checkpoint_relation": "independent_trained_checkpoints",
        "separately_retrained_evidence": True,
        "same_checkpoint_intervention": False,
        "split": "validation",
        "year": 2024,
        "days": paired.get("days", ""),
        "outcome_basis": OUTCOME_BASIS,
        "delta_direction": "primary_top2_minus_comparator",
        "cost_delta": -_finite(
            paired.get("cost_delta_baseline_minus_primary_top2"),
            label=f"{label} cost delta",
        ),
        "cost_delta_ci95_lower": -upper,
        "cost_delta_ci95_upper": -lower,
        "deadline_delta": -_finite(
            paired.get("deadline_delta"), label=f"{label} deadline delta"
        ),
        "unserved_delta": -_finite(
            paired.get("unserved_delta"), label=f"{label} unserved delta"
        ),
        "source": str(root / "paired_primary_top2.csv"),
    }
    return paper, comparison, meta


def _not_run_paper(model: str, role: str, *, split: str, year: int) -> dict[str, Any]:
    return _paper_row(
        {},
        model=model,
        evidence_role=role,
        status="NOT_RUN",
        split=split,
        year=year,
        checkpoint_relation="NOT_RUN",
        separately_retrained=False,
        same_checkpoint_intervention=False,
        source="NOT_RUN",
    )


def _not_run_comparison(model: str) -> dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "primary_model": "primary_top2",
        "comparator": model,
        "comparison_design": "separately_retrained_architecture_planned",
        "checkpoint_relation": "NOT_RUN",
        "separately_retrained_evidence": False,
        "same_checkpoint_intervention": False,
        "split": "validation",
        "year": 2024,
        "days": "",
        "outcome_basis": "",
        "delta_direction": "primary_top2_minus_comparator",
        "cost_delta": "",
        "cost_delta_ci95_lower": "",
        "cost_delta_ci95_upper": "",
        "deadline_delta": "",
        "unserved_delta": "",
        "source": "NOT_RUN",
    }


def _load_performance(
    value: str | Path,
    *,
    model: str,
    expected_checkpoint: Path,
    expected_baseline_type: str | None = None,
    require_components: bool = False,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    path = _artifact_file(value, "inference_latency.json")
    result = _json(path)
    latency = result.get("latency_batch1", result)
    throughput = result.get("throughput_batch")
    if not isinstance(latency, Mapping) or not isinstance(throughput, Mapping):
        raise RECMoEResultAggregationError(
            "performance artifact must contain batch1 latency and batched throughput"
        )
    if (
        int(latency.get("batch_size", 0)) != 1
        or int(throughput.get("batch_size", 0)) != 16
        or int(latency.get("horizon", 0)) != 96
        or int(throughput.get("horizon", 0)) != 96
        or int(latency.get("warmup", -1)) != 5
        or int(throughput.get("warmup", -1)) != 5
        or int(latency.get("repeats", -1)) != 30
        or int(throughput.get("repeats", -1)) != 30
    ):
        raise RECMoEResultAggregationError("performance batch roles are invalid")
    latency_values = latency.get("full_inference_latency_ms")
    memory1 = latency.get("peak_memory_mib")
    memoryn = throughput.get("peak_memory_mib")
    parameters = latency.get("parameters")
    components = latency.get("component_latency_ms")
    if not all(
        isinstance(item, Mapping)
        for item in (latency_values, memory1, memoryn, parameters)
    ):
        raise RECMoEResultAggregationError("performance artifact is incomplete")
    for profile_name, profile in (("latency", latency), ("throughput", throughput)):
        calls = profile.get("w1_backbone_calls")
        if (
            not isinstance(calls, Mapping)
            or int(calls.get("per_full_inference", 0)) != 1
            or profile.get("candidate_search") is not False
            or profile.get("external_solver") is not False
        ):
            raise RECMoEResultAggregationError(
                f"{profile_name} performance violates one-W1 solver-free contract"
            )
    if expected_baseline_type is not None:
        if (
            result.get("schema")
            != "dcvpp.r_ecmoe.baseline.inference_benchmark.v1"
            or result.get("baseline_type") != expected_baseline_type
            or not _same_path(result.get("best_checkpoint"), expected_checkpoint)
            or result.get("training_performed") is not False
            or result.get("locked_2025_opened") is not False
            or result.get("candidate_search") is not False
            or result.get("external_solver") is not False
        ):
            raise RECMoEResultAggregationError(
                f"{model} performance does not match its baseline checkpoint lineage"
            )
    else:
        checkpoint_load = latency.get("checkpoint_load")
        if (
            not isinstance(checkpoint_load, Mapping)
            or not _same_path(
                checkpoint_load.get("r_checkpoint_path"), expected_checkpoint
            )
            or checkpoint_load.get("locked_2025_opened") is not False
        ):
            raise RECMoEResultAggregationError(
                f"{model} performance does not match its R checkpoint lineage"
            )
    component_names = {
        "w1_packet_mean_ms": "w1_packet",
        "r_route_adapter_heads_mean_ms": "r_route_adapter_coordinator_and_heads",
        "action_response_mean_ms": "action_response",
        "constraint_assembler_mean_ms": "constraint_assembler",
        "post_risk_value_cost_mean_ms": "post_decision_risk_value_and_cost",
    }
    row: dict[str, Any] = {
        "status": "COMPLETED",
        "model": model,
        "batch1_p50_ms": _finite(latency_values.get("p50"), label="batch1 p50"),
        "batch1_p95_ms": _finite(latency_values.get("p95"), label="batch1 p95"),
        "throughput_batch_size": int(throughput["batch_size"]),
        "throughput_windows_per_second": _finite(
            throughput.get("throughput_windows_per_second"), label="throughput"
        ),
        "batch1_peak_allocated_mib": _finite(
            memory1.get("peak_allocated"), label="batch1 peak memory"
        ),
        "throughput_peak_allocated_mib": _finite(
            memoryn.get("peak_allocated"), label="throughput peak memory"
        ),
        "total_parameters": int(parameters.get("total_parameters", 0)),
        "active_parameters_per_sample_mean": _finite(
            parameters.get("active_parameters_per_sample_mean"),
            label="active parameters per sample",
        ),
        "w1_total_parameters": int(parameters.get("w1_total_parameters", 0)),
        "r_total_parameters": int(parameters.get("r_total_parameters", 0)),
        "r_overhead_mean_ms": "",
        "r_overhead_vs_w1_percent": "",
        "source": str(path),
    }
    if min(row["total_parameters"], row["w1_total_parameters"], row["r_total_parameters"]) < 0:
        raise RECMoEResultAggregationError("parameter counts cannot be negative")
    if isinstance(components, Mapping) and components.get("complete") is True:
        overhead = components.get("r_overhead_relative_to_w1")
        if not isinstance(overhead, Mapping):
            raise RECMoEResultAggregationError("R overhead evidence is missing")
        row["r_overhead_mean_ms"] = _finite(
            overhead.get("mean"), label="R overhead mean"
        )
        row["r_overhead_vs_w1_percent"] = _finite(
            overhead.get("mean_percent"), label="R overhead percent"
        )
        for output, component in component_names.items():
            item = components.get(component)
            if not isinstance(item, Mapping):
                raise RECMoEResultAggregationError(
                    f"component timing missing: {component}"
                )
            row[output] = _finite(item.get("mean"), label=output)
    elif require_components:
        raise RECMoEResultAggregationError("component latency evidence is incomplete")
    else:
        for output in component_names:
            row[output] = ""
    return row, result


def _not_run_performance(model: str) -> dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "model": model,
        **{name: "" for name in PERFORMANCE_FIELDS if name not in {"status", "model"}},
        "source": "NOT_RUN",
    }


def _attach_comparison_performance(
    row: dict[str, Any],
    *,
    primary: Mapping[str, Any],
    comparator: Mapping[str, Any] | None,
) -> None:
    row.update(
        {
            "primary_batch1_p50_ms": primary["batch1_p50_ms"],
            "primary_batch1_p95_ms": primary["batch1_p95_ms"],
            "primary_batch16_windows_per_second": primary[
                "throughput_windows_per_second"
            ],
        }
    )
    if comparator is None or comparator.get("status") != "COMPLETED":
        row.update(
            {
                "performance_status": "NOT_RUN",
                "comparator_batch1_p50_ms": "",
                "comparator_batch1_p95_ms": "",
                "comparator_batch16_windows_per_second": "",
                "comparator_peak_memory_mib": "",
                "comparator_total_parameters": "",
                "comparator_active_parameters_per_sample_mean": "",
            }
        )
        return
    row.update(
        {
            "performance_status": "COMPLETED",
            "comparator_batch1_p50_ms": comparator["batch1_p50_ms"],
            "comparator_batch1_p95_ms": comparator["batch1_p95_ms"],
            "comparator_batch16_windows_per_second": comparator[
                "throughput_windows_per_second"
            ],
            "comparator_peak_memory_mib": comparator[
                "throughput_peak_allocated_mib"
            ],
            "comparator_total_parameters": comparator["total_parameters"],
            "comparator_active_parameters_per_sample_mean": comparator[
                "active_parameters_per_sample_mean"
            ],
        }
    )


def _load_sla_audit(value: str | Path, expected: Path) -> Mapping[str, Any]:
    path = _artifact_file(value, "sla_gate_audit_2024.json")
    if path.resolve() != expected.resolve():
        raise RECMoEResultAggregationError(
            "SLA audit must be the exact artifact recorded by the primary best"
        )
    meta = _json(path)
    conclusion = meta.get("audit_conclusion")
    if (
        meta.get("schema") != "dcvpp.r_ecmoe.sla_gate_audit_2024.v1"
        or int(meta.get("year", -1)) != 2024
        or int(meta.get("validation_days", -1)) != 364
        or meta.get("outcome_basis") != OUTCOME_BASIS
        or not isinstance(conclusion, Mapping)
    ):
        raise RECMoEResultAggregationError("SLA audit lineage is invalid")
    for name in (
        "registered_sla_threshold",
        "learned_worst_origin_mean_sla",
        "safe_worst_origin_mean_sla",
        "learned_minus_safe_worst_origin_sla",
    ):
        _finite(conclusion.get(name), label=f"SLA audit {name}")
    return meta


def _load_one_shot(
    value: str | Path, primary_best: Path
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    supplied = Path(value).expanduser().resolve()
    if supplied.is_dir():
        completed_path = supplied / "one_shot_meta.json"
        failed_path = supplied / "one_shot_failed.json"
        if completed_path.exists() and failed_path.exists():
            raise RECMoEResultAggregationError(
                "one-shot directory cannot be both completed and failed"
            )
        path = failed_path if failed_path.is_file() else completed_path
    else:
        path = supplied
    meta = _json(path)
    if path.name == "one_shot_failed.json":
        error = str(meta.get("error", "")).strip()
        error_type = str(meta.get("error_type", "")).strip()
        one_shot_id = str(meta.get("one_shot_id", "")).strip()
        if (
            meta.get("schema") != "dcvpp.r_ecmoe.sealed_2025_failed.v1"
            or meta.get("status")
            != "FAILED_ONE_SHOT_CONSUMED_NO_RERUN_ALLOWED"
            or not one_shot_id
            or not error_type
            or "daily final modalities" not in error.lower()
            or "partial" not in error.lower()
        ):
            raise RECMoEResultAggregationError(
                "one-shot failure is not strict data-incomplete evidence"
            )
        row = _not_run_paper(
            "primary_top2", "sealed_test", split="final_test", year=2025
        )
        row.update(
            {
                "status": "FAILED_ONE_SHOT_DATA_INCOMPLETE",
                "checkpoint_relation": "frozen_primary_attempt_consumed_before_inference",
                "source": str(path),
            }
        )
        return row, {
            "status": "FAILED_ONE_SHOT_DATA_INCOMPLETE",
            "attempted": True,
            "completed": False,
            "one_shot_id": one_shot_id,
            "error_type": error_type,
            "error": error,
            "source": str(path),
        }
    summary = meta.get("summary")
    if (
        meta.get("schema") != "dcvpp.r_ecmoe.sealed_2025_evaluation.v1"
        or meta.get("status") != "COMPLETED"
        or int(meta.get("one_shot_run_index", -1)) != 1
        or int(meta.get("year", -1)) != 2025
        or meta.get("training_performed") is not False
        or meta.get("checkpoint_selection_performed") is not False
        or meta.get("repeat_or_overwrite_allowed") is not False
        or meta.get("outcome_basis") != OUTCOME_BASIS
        or not _same_path(meta.get("best_checkpoint"), primary_best)
        or not isinstance(summary, Mapping)
    ):
        raise RECMoEResultAggregationError("sealed one-shot result is invalid")
    return (
        _paper_row(
            summary,
            model="primary_top2",
            evidence_role="sealed_test",
            status="COMPLETED",
            split="final_test",
            year=2025,
            checkpoint_relation="same_frozen_primary_checkpoint",
            separately_retrained=False,
            same_checkpoint_intervention=False,
            source=path,
        ),
        {
            "status": "COMPLETED",
            "attempted": True,
            "completed": True,
            "one_shot_id": str(meta.get("one_shot_id")),
            "error_type": None,
            "error": None,
            "source": str(path),
        },
    )


PAPER_FIELDS = (
    "status", "model", "evidence_role", "split", "year", "days",
    "checkpoint_relation", "separately_retrained_evidence",
    "same_checkpoint_intervention", "outcome_basis",
    "mean_realized_operating_cost", "mean_normalized_operating_cost",
    "cvar95_normalized_operating_cost", "mean_deadline_expired_work",
    "mean_unserved_energy", "maximum_hard_residual",
    "mean_action_total_variation", "mean_battery_throughput",
    "mean_compute_energy", "mean_migration_energy",
    "mean_pv_curtailed_energy", "mean_grid_import_energy",
    "mean_grid_export_energy", "mean_grid_peak_import_ratio",
    "mean_grid_peak_export_ratio", "mean_grid_peak_ramp_ratio",
    "mean_grid_rebound_ratio", "mean_compute_concentration_hhi",
    "mean_terminal_unfinished_work", "mean_terminal_soc_fraction",
    "mean_route_entropy", "mean_route_gini", "mean_tail_activation_rate",
    "mean_tail_probability", "mean_tail_event_rate",
    "mean_tail_probability_brier", "mean_tail_calibration_gap",
    "tail_calibration_target", "source",
)
COMPARISON_FIELDS = (
    "status", "primary_model", "comparator", "comparison_design",
    "checkpoint_relation", "separately_retrained_evidence",
    "same_checkpoint_intervention", "split", "year", "days", "outcome_basis",
    "delta_direction", "cost_delta", "cost_delta_ci95_lower",
    "cost_delta_ci95_upper", "deadline_delta", "unserved_delta",
    "performance_status", "primary_batch1_p50_ms", "primary_batch1_p95_ms",
    "primary_batch16_windows_per_second", "comparator_batch1_p50_ms",
    "comparator_batch1_p95_ms", "comparator_batch16_windows_per_second",
    "comparator_peak_memory_mib", "comparator_total_parameters",
    "comparator_active_parameters_per_sample_mean", "source",
)
PERFORMANCE_FIELDS = (
    "status", "model", "batch1_p50_ms", "batch1_p95_ms",
    "throughput_batch_size", "throughput_windows_per_second",
    "batch1_peak_allocated_mib", "throughput_peak_allocated_mib",
    "total_parameters", "active_parameters_per_sample_mean",
    "w1_total_parameters", "r_total_parameters", "w1_packet_mean_ms",
    "r_route_adapter_heads_mean_ms", "action_response_mean_ms",
    "constraint_assembler_mean_ms", "post_risk_value_cost_mean_ms",
    "r_overhead_mean_ms", "r_overhead_vs_w1_percent", "source",
)
GOAL_REGISTRY_FIELDS = (
    "category", "goal_item", "status", "requiredness", "evidence_design",
    "source", "reason",
)


def _goal_experiment_registry(
    *,
    experiments: Mapping[str, Mapping[str, Any]],
    evaluation_root: Path,
) -> list[dict[str, Any]]:
    """Register the complete Goal section-21 scope, including honest gaps."""

    same_checkpoint_source = str(evaluation_root / "variant_summary.csv")

    def row(
        category: str,
        goal_item: str,
        status: str,
        requiredness: str,
        evidence_design: str,
        source: str,
        reason: str = "",
    ) -> dict[str, Any]:
        return {
            "category": category,
            "goal_item": goal_item,
            "status": status,
            "requiredness": requiredness,
            "evidence_design": evidence_design,
            "source": source,
            "reason": reason,
        }

    registered: list[dict[str, Any]] = [
        row("main_comparison", "safe_deterministic_neural_rule", "COMPLETED", "registered", "same_checkpoint_tensor_baseline", same_checkpoint_source),
        row("main_comparison", "legacy_w0_w1_u0_p1_j1_r1", "NOT_RUN_REQUIRED", "registered", "independent_terminal_policy", "NOT_RUN", "no complete comparable P1/J1/R1 terminal checkpoint exists"),
        row("main_comparison", "dense_direct_r", str(experiments["dense_direct"]["status"]), "registered", "separately_retrained_architecture", str(experiments["dense_direct"].get("source", "NOT_RUN"))),
        row("main_comparison", "independent_r_moe", str(experiments["independent_r_moe"]["status"]), "registered", "separately_retrained_architecture", str(experiments["independent_r_moe"].get("source", "NOT_RUN"))),
        row("main_comparison", "r_ecmoe_top1_conditional_tail", str(experiments["top1_retrained"]["status"]), "registered", "separately_retrained_deployment_top1", str(experiments["top1_retrained"].get("source", "NOT_RUN"))),
        row("main_comparison", "r_ecmoe_top2_conditional_tail", "COMPLETED", "registered", "primary_reselected_checkpoint", same_checkpoint_source),
        row("main_comparison", "partial_w1_joint_tuning", "OPTIONAL_NOT_RUN", "explicitly_optional", "independent_joint_tuning", "NOT_RUN", "R-C remained disabled because the registered frozen-W1 main was used"),
        row("main_comparison", "one_step_corrector", "OPTIONAL_NOT_RUN", "explicitly_optional", "paper_ablation", "NOT_RUN", "optional corrector was not implemented or enabled"),
        row("main_comparison", "oracle_information_policy_gap", "OPTIONAL_NOT_RUN", "explicitly_optional", "nondeployable_diagnostic", "NOT_RUN", "no oracle-information diagnostic was run"),
        row("ablation", "no_w1_route_prior", "COMPLETED", "registered", "same_checkpoint_intervention", same_checkpoint_source),
        row("ablation", "pooled_latent_instead_of_expert_states", "COMPLETED", "registered", "same_checkpoint_intervention", same_checkpoint_source),
        row("ablation", "no_tail", "COMPLETED", "registered", "same_checkpoint_intervention", same_checkpoint_source),
        row("ablation", "tail_always_on", "COMPLETED", "registered", "same_checkpoint_intervention", same_checkpoint_source),
        row("ablation", "top1_vs_top2", "COMPLETED", "registered", "same_checkpoint_plus_separately_retrained", same_checkpoint_source),
        row("ablation", "adapter_widths", "NOT_RUN_REQUIRED", "registered", "separately_retrained_width_only", "NOT_RUN", "fast smoke changes Top-k and temporal rank and is not an isolated width ablation"),
        row("ablation", "no_sparse_coordinator", "COMPLETED", "registered", "same_checkpoint_intervention", same_checkpoint_source),
        row("ablation", "no_risk_condition", "COMPLETED", "registered", "same_checkpoint_intervention", same_checkpoint_source),
        row("ablation", "no_structured_action_parameterization", "NOT_IMPLEMENTED_REQUIRED", "registered", "independent_architecture", "NOT_RUN", "no unstructured-action comparator has been implemented"),
        row("ablation", "w1_freeze_strategies", "OPTIONAL_NOT_RUN", "conditional_on_r_c", "independent_freeze_strategy", "NOT_RUN", "R-C was not activated"),
        row("ablation", "no_route_loss", "NOT_RUN_REQUIRED", "registered", "separately_retrained_loss_ablation", "NOT_RUN"),
        row("ablation", "no_load_balance", "NOT_RUN_REQUIRED", "registered", "separately_retrained_loss_ablation", "NOT_RUN"),
        row("ablation", "no_w1_retention", "NOT_APPLICABLE_OPTIONAL", "conditional_on_r_c", "retention_ablation", "NOT_RUN", "the main freezes W1 and has no R-C retention term"),
        row("ablation", "no_cache", "NOT_IMPLEMENTED_REQUIRED", "registered", "runtime_ablation", "NOT_RUN", "no deployment cache comparison is implemented"),
        row("ablation", "one_step_corrector_value_latency", "OPTIONAL_NOT_RUN", "explicitly_optional", "paper_ablation", "NOT_RUN"),
    ]
    return registered


def aggregate_r_ecmoe_results(
    *,
    primary_best: str | Path,
    output_root: str | Path,
    inference_latency: str | Path | None = None,
    evaluation_2024: str | Path | None = None,
    sla_audit: str | Path | None = None,
    top1_evaluation: str | Path | None = None,
    dense_evaluation: str | Path | None = None,
    independent_evaluation: str | Path | None = None,
    top1_inference_latency: str | Path | None = None,
    dense_inference_latency: str | Path | None = None,
    independent_inference_latency: str | Path | None = None,
    selector_candidate_audit: str | Path | None = None,
    one_shot_2025: str | Path | None = None,
) -> Mapping[str, Any]:
    """Aggregate already-measured evidence without model/data execution."""

    primary = _load_primary(primary_best)
    run_root: Path = primary["run_root"]
    best: Path = primary["best"]
    evaluation_value = evaluation_2024 or (run_root / "evaluation_2024")
    performance_value = inference_latency or (run_root / "inference_latency.json")
    audit_value = sla_audit or primary["audit"]

    paper_rows, comparison_rows, _evaluation_meta = _load_primary_evaluation(
        evaluation_value, best
    )
    performance_row, _performance_raw = _load_performance(
        performance_value,
        model="primary_top2",
        expected_checkpoint=best,
        require_components=True,
    )
    audit_meta = _load_sla_audit(audit_value, primary["audit"])
    selector_audit = _load_selector_candidate_audit(
        selector_candidate_audit,
        expected_checkpoint=best,
        checkpoint_payload=primary["checkpoint"],
    )

    optional_values = {
        "top1_retrained": top1_evaluation,
        "dense_direct": dense_evaluation,
        "independent_r_moe": independent_evaluation,
    }
    optional_performance_values = {
        "top1_retrained": top1_inference_latency,
        "dense_direct": dense_inference_latency,
        "independent_r_moe": independent_inference_latency,
    }
    performance_rows = [performance_row]
    performance_by_model: dict[str, Mapping[str, Any]] = {
        "primary_top2": performance_row
    }
    experiments: dict[str, dict[str, Any]] = {
        "primary_2024": {"status": "COMPLETED", "design": "primary_reselected"}
    }
    for label, value in optional_values.items():
        if value is None:
            if optional_performance_values[label] is not None:
                raise RECMoEResultAggregationError(
                    f"{label} performance cannot be accepted without its 2024 evaluation"
                )
            paper_rows.append(
                _not_run_paper(
                    label, "separately_retrained_comparator", split="validation", year=2024
                )
            )
            comparison_rows.append(_not_run_comparison(label))
            experiments[label] = {
                "status": "NOT_RUN",
                "design": "separately_retrained_architecture",
                "performance_status": "NOT_RUN",
            }
            missing_performance = _not_run_performance(label)
            performance_rows.append(missing_performance)
            performance_by_model[label] = missing_performance
            continue
        paper, comparison, baseline_meta = _load_baseline_evaluation(
            label, value, best
        )
        paper_rows.append(paper)
        comparison_rows.append(comparison)
        raw_performance = optional_performance_values[label]
        if raw_performance is None:
            comparator_performance = _not_run_performance(label)
            performance_status = "NOT_RUN"
        else:
            baseline_checkpoint = Path(
                str(baseline_meta.get("best_checkpoint", ""))
            ).expanduser().resolve()
            comparator_performance, _raw = _load_performance(
                raw_performance,
                model=label,
                expected_checkpoint=baseline_checkpoint,
                expected_baseline_type=(
                    None if label == "top1_retrained" else OPTIONAL_BASELINES[label]
                ),
                require_components=(label == "top1_retrained"),
            )
            performance_status = "COMPLETED"
        performance_rows.append(comparator_performance)
        performance_by_model[label] = comparator_performance
        experiments[label] = {
            "status": "COMPLETED",
            "design": "separately_retrained_architecture",
            "source": str(Path(value).expanduser().resolve()),
            "performance_status": performance_status,
        }

    for comparison in comparison_rows:
        _attach_comparison_performance(
            comparison,
            primary=performance_row,
            comparator=performance_by_model.get(str(comparison["comparator"])),
        )

    if one_shot_2025 is None:
        paper_rows.append(
            _not_run_paper(
                "primary_top2", "sealed_test", split="final_test", year=2025
            )
        )
        experiments["sealed_2025"] = {
            "status": "NOT_RUN",
            "design": "one_shot",
            "attempted": False,
            "completed": False,
        }
        one_shot_evidence: Mapping[str, Any] | None = None
    else:
        one_shot_row, one_shot_evidence = _load_one_shot(one_shot_2025, best)
        paper_rows.append(one_shot_row)
        experiments["sealed_2025"] = {
            "design": "one_shot",
            **dict(one_shot_evidence),
        }

    evaluation_root = _artifact_file(
        evaluation_value, "evaluation_2024_meta.json"
    ).parent
    goal_registry = _goal_experiment_registry(
        experiments=experiments,
        evaluation_root=evaluation_root,
    )
    goal_required_missing = [
        str(row["goal_item"])
        for row in goal_registry
        if str(row["status"]) in {"NOT_RUN_REQUIRED", "NOT_IMPLEMENTED_REQUIRED"}
    ]
    goal_optional_not_run = [
        str(row["goal_item"])
        for row in goal_registry
        if str(row["status"]) in {
            "OPTIONAL_NOT_RUN",
            "NOT_APPLICABLE_OPTIONAL",
        }
    ]
    output = Path(output_root).expanduser().resolve()
    measured = {
        "schema": RESULTS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "primary": {
            "status": "COMPLETED",
            "best_checkpoint": str(best),
            "run_id": primary["checkpoint"].get("run_id"),
            "source_invalid_run": primary["checkpoint"].get("source_checkpoint"),
            "old_run_invalid_reason": primary["checkpoint"].get(
                "old_run_invalid_reason"
            ),
            "validation_scope": primary["metrics"].get("validation_scope"),
            "selection_eligible": True,
            "outcome_basis": OUTCOME_BASIS,
            "empirical_cvar95": primary["metrics"].get("empirical_cvar95"),
            "worst_origin_sla": primary["metrics"].get("worst_origin_sla"),
        },
        "experiments": experiments,
        "paper_results": paper_rows,
        "comparison_results": comparison_rows,
        "performance_results": performance_rows,
        "selector_candidate_audit": selector_audit,
        "goal_experiment_registry": goal_registry,
        "goal_required_missing_experiments": goal_required_missing,
        "goal_optional_not_run_experiments": goal_optional_not_run,
        "sla_audit": {
            "status": "COMPLETED",
            "source": str(primary["audit"]),
            "audit_conclusion": dict(audit_meta["audit_conclusion"]),
        },
        "source_artifacts": {
            "primary_evaluation": str(
                _artifact_file(evaluation_value, "evaluation_2024_meta.json")
            ),
            "performance": str(
                _artifact_file(performance_value, "inference_latency.json")
            ),
            "selector_candidate_audit": selector_audit["evidence_path"],
            "one_shot_2025": (
                None
                if one_shot_evidence is None
                else str(one_shot_evidence["source"])
            ),
        },
        "missing_experiments": [
            name for name, value in experiments.items() if value["status"] == "NOT_RUN"
        ]
        + [
            f"{name}_performance"
            for name, value in experiments.items()
            if value.get("status") == "COMPLETED"
            and value.get("performance_status") == "NOT_RUN"
        ],
        "failed_experiments": [
            name
            for name, value in experiments.items()
            if str(value.get("status", "")).startswith("FAILED_")
        ],
        "comparison_contract": {
            "same_checkpoint_interventions_are_not_retrained_comparators": True,
            "separately_retrained_rows_require_independent_checkpoint_and_paired_2024": True,
        },
    }

    _atomic_csv(output / "paper_results.csv", paper_rows, PAPER_FIELDS)
    _atomic_csv(
        output / "comparison_results.csv", comparison_rows, COMPARISON_FIELDS
    )
    _atomic_csv(
        output / "performance_results.csv", performance_rows, PERFORMANCE_FIELDS
    )
    _atomic_csv(
        output / "goal_experiment_registry.csv",
        goal_registry,
        GOAL_REGISTRY_FIELDS,
    )
    _atomic_json(output / "measured_results.json", measured)

    primary_row = next(row for row in paper_rows if row["model"] == "primary_top2" and row["year"] == 2024)
    missing = measured["missing_experiments"]
    sealed_status = str(experiments["sealed_2025"]["status"])
    lines = [
        "# R-ECMoE measured results",
        "",
        "Primary evidence: selector-repaired eligible checkpoint with full 2024 realized historical replay.",
        "",
        f"- 2024 normalized operating cost: {primary_row['mean_normalized_operating_cost']}",
        f"- 2024 empirical CVaR95: {primary_row['cvar95_normalized_operating_cost']}",
        f"- Worst-origin SLA: {primary['metrics']['worst_origin_sla']}",
        f"- Batch-1 latency p50/p95: {performance_row['batch1_p50_ms']} / {performance_row['batch1_p95_ms']} ms",
        f"- Batched throughput: {performance_row['throughput_windows_per_second']} windows/s at batch {performance_row['throughput_batch_size']}",
        f"- Corrected selector candidate audit: {selector_audit['status']}",
        f"- Recorded full-eval candidates: {selector_audit['recorded_full_candidate_count']}",
        f"- Eligible full-eval candidates: {selector_audit['eligible_full_candidate_count']}",
        f"- Selected epoch: {selector_audit['selected_epoch']}",
        "- Selected epoch is minimum among recorded full-eval candidates: "
        f"{selector_audit['selected_epoch_is_minimum_among_recorded_full_candidates']}",
        f"- Selector candidate audit evidence: {selector_audit['evidence_path']}",
        "",
        "Same-checkpoint interventions and separately retrained architecture comparisons are reported in distinct rows.",
        "",
        "Core aggregate inputs missing (NOT_RUN): "
        + (", ".join(missing) if missing else "none"),
        "Registered non-optional Goal experiments not completed: "
        + (", ".join(goal_required_missing) if goal_required_missing else "none"),
        "Explicitly optional or conditional experiments not run: "
        + (", ".join(goal_optional_not_run) if goal_optional_not_run else "none"),
        f"2025 one-shot status: {sealed_status}",
        *(
            [
                f"2025 failure type: {experiments['sealed_2025']['error_type']}",
                f"2025 failure: {experiments['sealed_2025']['error']}",
            ]
            if sealed_status == "FAILED_ONE_SHOT_DATA_INCOMPLETE"
            else []
        ),
        "",
    ]
    _atomic_text(output / "summary.md", "\n".join(lines))
    return measured


__all__ = [
    "RECMoEResultAggregationError",
    "RESULTS_SCHEMA",
    "aggregate_r_ecmoe_results",
]
