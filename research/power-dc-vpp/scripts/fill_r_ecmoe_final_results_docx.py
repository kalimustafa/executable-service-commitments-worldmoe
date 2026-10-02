r"""Fill the R-ECMoE manuscript from already-aggregated measured evidence.

This command is deliberately read-only with respect to model/data artifacts.  It
opens the source DOCX and the four files created by
``aggregate_r_ecmoe_results``.  When, and only when, the aggregate declares a
completed sealed final evaluation, it additionally opens the recorded
``one_shot_meta.json`` to verify the no-training/no-selection-feedback contract.
It never opens raw 2025 data and cannot run evaluation or training.

Example (do not overwrite the source manuscript)::

    python scripts/fill_r_ecmoe_final_results_docx.py \
      --input-docx ..\Data_Center_VPP_English_Manuscript_and_Experimental_Protocol_R_ECMoE_RealizedReplay_20260827.docx \
      --results-root artifacts\paper_finalization\aggregate_r_ecmoe_results

The default output name replaces ``RealizedReplay`` with ``Final_Results``.
Existing outputs are refused.  The edit keeps the source styles and table cell
formatting by changing text in existing runs rather than rebuilding the DOCX.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from docx import Document
from docx.document import Document as DocumentType
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph


RESULTS_SCHEMA = "dcvpp.r_ecmoe.measured_results.v1"
OUTCOME_BASIS = "realized_historical_replay"
REGISTERED_VALIDATION_YEAR = 2024
REGISTERED_VALIDATION_DAYS = 364
AUDITED_OLD_SLA_TOLERANCE = 0.01
REVISED_SLA_TOLERANCE = 0.02
REGISTERED_LATENCY_BUDGET_MS = 1900.0
REGISTERED_MEMORY_BUDGET_MIB = 10.0 * 1024.0
REGISTERED_BOOTSTRAP_BLOCK_DAYS = 7

REQUIRED_RESULT_FILES = (
    "measured_results.json",
    "paper_results.csv",
    "comparison_results.csv",
    "performance_results.csv",
)

TABLE_SAME_CHECKPOINT = 29
TABLE_FIVE_MODELS = 30
TABLE_HYPOTHESES = 31
TABLE_ARTIFACT_PATHS = 35


class FinalResultsFillError(RuntimeError):
    """The source topology or measured evidence is not safe to publish."""


@dataclass(frozen=True)
class ResultBundle:
    root: Path
    measured_path: Path
    paper_path: Path
    comparison_path: Path
    performance_path: Path
    measured: Mapping[str, Any]
    paper_rows: tuple[Mapping[str, str], ...]
    comparison_rows: tuple[Mapping[str, str], ...]
    performance_rows: tuple[Mapping[str, str], ...]
    sealed_2025_status: str
    sealed_2025_meta: Mapping[str, Any] | None
    sealed_2025_failure: Mapping[str, Any] | None


@dataclass(frozen=True)
class EvidenceView:
    primary: Mapping[str, Any]
    primary_row: Mapping[str, str]
    safe_row: Mapping[str, str]
    no_tail_row: Mapping[str, str]
    no_route_row: Mapping[str, str]
    performance: Mapping[str, str]
    performances: Mapping[str, Mapping[str, str]]
    comparisons: Mapping[str, Mapping[str, str]]
    audit: Mapping[str, Any]
    sealed_2025_status: str
    final_2025_row: Mapping[str, str] | None
    final_2025_meta: Mapping[str, Any] | None
    final_2025_failure: Mapping[str, Any] | None


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FinalResultsFillError(f"required measured result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FinalResultsFillError(f"JSON must contain an object: {path}")
    return value


def _read_csv(path: Path) -> tuple[Mapping[str, str], ...]:
    if not path.is_file():
        raise FinalResultsFillError(f"required measured result is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise FinalResultsFillError(f"CSV cannot be empty: {path}")
    return rows


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise FinalResultsFillError(f"{label} must be finite numeric evidence")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalResultsFillError(
            f"{label} must be finite numeric evidence"
        ) from exc
    if not math.isfinite(result):
        raise FinalResultsFillError(f"{label} must be finite numeric evidence")
    return result


def _integer(value: Any, *, label: str) -> int:
    result = _finite(value, label=label)
    if not result.is_integer():
        raise FinalResultsFillError(f"{label} must be an integer")
    return int(result)


def _truth(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise FinalResultsFillError(f"{label} must be true or false")


def _completed(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "")).strip().upper() == "COMPLETED"


def _csv_matches_embedded(
    csv_rows: Sequence[Mapping[str, str]],
    embedded: Any,
    *,
    identity_fields: Sequence[str],
    label: str,
) -> None:
    if not isinstance(embedded, Sequence) or isinstance(embedded, (str, bytes)):
        raise FinalResultsFillError(f"measured_results.{label} must be a list")
    if not all(isinstance(row, Mapping) for row in embedded):
        raise FinalResultsFillError(f"measured_results.{label} has a non-object row")

    def identity(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(field, "")) for field in identity_fields)

    embedded_by_identity = {identity(row): row for row in embedded}
    csv_by_identity = {identity(row): row for row in csv_rows}
    if (
        len(embedded_by_identity) != len(embedded)
        or len(csv_by_identity) != len(csv_rows)
        or set(csv_by_identity) != set(embedded_by_identity)
    ):
        raise FinalResultsFillError(
            f"{label}.csv does not match the measured_results registry"
        )
    for row_identity, csv_row in csv_by_identity.items():
        embedded_row = embedded_by_identity[row_identity]
        for field, csv_value in csv_row.items():
            embedded_value = embedded_row.get(field, "")
            normalized_embedded = "" if embedded_value is None else str(embedded_value)
            normalized_csv = "" if csv_value is None else str(csv_value)
            if normalized_csv != normalized_embedded:
                raise FinalResultsFillError(
                    f"{label}.csv value drift for {row_identity!r}, field {field!r}"
                )


def _same_recorded_path(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    return os.path.normcase(os.path.normpath(left_text)) == os.path.normcase(
        os.path.normpath(right_text)
    )


def _validate_completed_one_shot_meta(
    *,
    path_value: Any,
    primary_best: Any,
) -> Mapping[str, Any]:
    """Read only the final metadata artifact, never the recorded 2025 data."""

    path_text = str(path_value or "").strip()
    if not path_text:
        raise FinalResultsFillError("completed sealed 2025 result lacks one_shot_meta")
    path = Path(path_text).expanduser().resolve()
    if path.name != "one_shot_meta.json":
        raise FinalResultsFillError("sealed 2025 source must be one_shot_meta.json")
    meta = _read_json(path)
    required_false = (
        "optimizer_created",
        "training_performed",
        "checkpoint_selection_performed",
        "checkpoint_updated",
        "thresholds_updated",
        "data_statistics_updated",
        "final_results_may_modify_selection",
        "repeat_or_overwrite_allowed",
        "external_solver",
        "candidate_search",
    )
    invalid_false = [name for name in required_false if meta.get(name) is not False]
    if (
        meta.get("schema") != "dcvpp.r_ecmoe.sealed_2025_evaluation.v1"
        or meta.get("status") != "COMPLETED"
        or _integer(meta.get("one_shot_run_index"), label="2025 one-shot index") != 1
        or _integer(meta.get("year"), label="2025 final year") != 2025
        or meta.get("split") != "final_test"
        or meta.get("inference_only") is not True
        or meta.get("outcome_basis") != OUTCOME_BASIS
        or not _same_recorded_path(meta.get("best_checkpoint"), primary_best)
        or invalid_false
    ):
        details = f"; invalid false flags={invalid_false}" if invalid_false else ""
        raise FinalResultsFillError(
            "sealed 2025 evidence is not a unique frozen no-feedback one-shot" + details
        )
    one_shot_id = meta.get("one_shot_id")
    if not isinstance(one_shot_id, str) or not one_shot_id.strip():
        raise FinalResultsFillError("sealed 2025 evidence lacks a one-shot id")
    expected_days = _integer(
        meta.get("expected_evaluable_days"), label="2025 expected evaluable days"
    )
    observed_days = _integer(
        meta.get("observed_evaluable_days"), label="2025 observed evaluable days"
    )
    if not (1 <= expected_days <= 366) or observed_days != expected_days:
        raise FinalResultsFillError("sealed 2025 final evaluation has incomplete dates")
    if _integer(meta.get("decisions_per_day"), label="2025 decisions per day") != 6:
        raise FinalResultsFillError("sealed 2025 final evaluation omitted decisions")
    summary = meta.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema") != "dcvpp.r_ecmoe.sealed_2025_summary.v1"
        or summary.get("variant") != "top2"
        or summary.get("outcome_basis") != OUTCOME_BASIS
        or _integer(summary.get("days"), label="2025 summary days") != observed_days
    ):
        raise FinalResultsFillError("sealed 2025 summary is invalid")
    return meta


def _validate_failed_one_shot_evidence(
    *,
    experiment: Mapping[str, Any],
    path_value: Any,
) -> Mapping[str, Any]:
    path_text = str(path_value or "").strip()
    if not path_text:
        raise FinalResultsFillError("failed sealed 2025 attempt lacks failure evidence")
    path = Path(path_text).expanduser().resolve()
    if path.name != "one_shot_failed.json":
        raise FinalResultsFillError("failed sealed 2025 source must be one_shot_failed.json")
    failure = _read_json(path)
    error = str(failure.get("error", "")).strip()
    error_type = str(failure.get("error_type", "")).strip()
    one_shot_id = str(failure.get("one_shot_id", "")).strip()
    if (
        failure.get("schema") != "dcvpp.r_ecmoe.sealed_2025_failed.v1"
        or failure.get("status")
        != "FAILED_ONE_SHOT_CONSUMED_NO_RERUN_ALLOWED"
        or not error_type
        or not one_shot_id
        or "daily final modalities" not in error.lower()
        or "partial" not in error.lower()
        or experiment.get("attempted") is not True
        or experiment.get("completed") is not False
        or experiment.get("one_shot_id") != one_shot_id
        or experiment.get("error_type") != error_type
        or experiment.get("error") != error
        or not _same_recorded_path(experiment.get("source"), path)
    ):
        raise FinalResultsFillError(
            "sealed 2025 failure is not consumed data-incomplete one-shot evidence"
        )
    return failure


def load_result_bundle(results_root: str | Path) -> ResultBundle:
    """Load only the aggregate outputs; recorded source paths stay unopened."""

    root = Path(results_root).expanduser().resolve()
    paths = {name: root / name for name in REQUIRED_RESULT_FILES}
    measured = _read_json(paths["measured_results.json"])
    if measured.get("schema") != RESULTS_SCHEMA:
        raise FinalResultsFillError("unsupported measured_results schema")

    paper_rows = _read_csv(paths["paper_results.csv"])
    comparison_rows = _read_csv(paths["comparison_results.csv"])
    performance_rows = _read_csv(paths["performance_results.csv"])
    _csv_matches_embedded(
        paper_rows,
        measured.get("paper_results"),
        identity_fields=("model", "split", "year", "evidence_role"),
        label="paper_results",
    )
    _csv_matches_embedded(
        comparison_rows,
        measured.get("comparison_results"),
        identity_fields=("primary_model", "comparator", "comparison_design"),
        label="comparison_results",
    )
    _csv_matches_embedded(
        performance_rows,
        measured.get("performance_results"),
        identity_fields=("model", "status"),
        label="performance_results",
    )

    primary = measured.get("primary")
    if not isinstance(primary, Mapping):
        raise FinalResultsFillError("measured_results.primary must be an object")
    if (
        primary.get("status") != "COMPLETED"
        or primary.get("selection_eligible") is not True
        or primary.get("validation_scope") != "full"
        or primary.get("outcome_basis") != OUTCOME_BASIS
    ):
        raise FinalResultsFillError(
            "only an eligible, full-2024, realized-replay primary may be filled"
        )

    experiments = measured.get("experiments")
    if not isinstance(experiments, Mapping):
        raise FinalResultsFillError("measured_results.experiments must be an object")
    primary_2024 = experiments.get("primary_2024")
    sealed_2025 = experiments.get("sealed_2025")
    if not isinstance(primary_2024, Mapping) or primary_2024.get("status") != "COMPLETED":
        raise FinalResultsFillError("the registered 2024 primary evaluation is incomplete")
    if (
        not isinstance(sealed_2025, Mapping)
        or sealed_2025.get("design") != "one_shot"
        or sealed_2025.get("status")
        not in {
            "NOT_RUN",
            "COMPLETED",
            "FAILED_ONE_SHOT_DATA_INCOMPLETE",
        }
    ):
        raise FinalResultsFillError(
            "sealed 2025 experiment must be NOT_RUN, completed once, or consumed with data incomplete"
        )
    source_artifacts = measured.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise FinalResultsFillError("measured_results.source_artifacts must be an object")
    sealed_path = source_artifacts.get("one_shot_2025")
    sealed_status = str(sealed_2025.get("status"))
    sealed_meta: Mapping[str, Any] | None = None
    sealed_failure: Mapping[str, Any] | None = None
    if sealed_status == "COMPLETED":
        sealed_meta = _validate_completed_one_shot_meta(
            path_value=sealed_path,
            primary_best=primary.get("best_checkpoint"),
        )
    elif sealed_status == "FAILED_ONE_SHOT_DATA_INCOMPLETE":
        sealed_failure = _validate_failed_one_shot_evidence(
            experiment=sealed_2025,
            path_value=sealed_path,
        )
    else:
        if sealed_path not in {None, "", "NOT RUN"}:
            raise FinalResultsFillError(
                "NOT_RUN sealed 2025 experiment cannot record a completed source"
            )

    return ResultBundle(
        root=root,
        measured_path=paths["measured_results.json"],
        paper_path=paths["paper_results.csv"],
        comparison_path=paths["comparison_results.csv"],
        performance_path=paths["performance_results.csv"],
        measured=measured,
        paper_rows=paper_rows,
        comparison_rows=comparison_rows,
        performance_rows=performance_rows,
        sealed_2025_status=sealed_status,
        sealed_2025_meta=sealed_meta,
        sealed_2025_failure=sealed_failure,
    )


def _paper_row(
    rows: Sequence[Mapping[str, str]], model: str, *, allow_not_run: bool = False
) -> Mapping[str, str]:
    found = [
        row
        for row in rows
        if row.get("model") == model
        and str(row.get("year")) == str(REGISTERED_VALIDATION_YEAR)
        and row.get("split") == "validation"
    ]
    if len(found) != 1:
        raise FinalResultsFillError(f"expected one 2024 validation row for {model}")
    row = found[0]
    status = str(row.get("status", "")).upper()
    if status not in {"COMPLETED", "NOT_RUN"}:
        raise FinalResultsFillError(f"{model} has unsupported status {status!r}")
    if status == "NOT_RUN" and not allow_not_run:
        raise FinalResultsFillError(f"required result {model} is NOT_RUN")
    if status == "COMPLETED":
        if (
            row.get("outcome_basis") != OUTCOME_BASIS
            or _integer(row.get("days"), label=f"{model} days")
            != REGISTERED_VALIDATION_DAYS
        ):
            raise FinalResultsFillError(
                f"{model} is not complete 364-date 2024 realized replay"
            )
    return row


def _sealed_final_row(bundle: ResultBundle) -> Mapping[str, str] | None:
    found = [
        row
        for row in bundle.paper_rows
        if row.get("model") == "primary_top2"
        and row.get("split") == "final_test"
        and str(row.get("year")) == "2025"
        and row.get("evidence_role") == "sealed_test"
    ]
    if len(found) != 1:
        raise FinalResultsFillError("expected one registered sealed 2025 result row")
    row = found[0]
    if bundle.sealed_2025_status == "FAILED_ONE_SHOT_DATA_INCOMPLETE":
        if (
            str(row.get("status", "")).upper()
            != "FAILED_ONE_SHOT_DATA_INCOMPLETE"
            or row.get("checkpoint_relation")
            != "frozen_primary_attempt_consumed_before_inference"
            or not _same_recorded_path(
                row.get("source"),
                bundle.measured["source_artifacts"].get("one_shot_2025"),
            )
        ):
            raise FinalResultsFillError("2025 failure row conflicts with consumed attempt")
        numeric_fields = (
            "mean_realized_operating_cost",
            "mean_normalized_operating_cost",
            "cvar95_normalized_operating_cost",
            "mean_deadline_expired_work",
            "mean_unserved_energy",
            "maximum_hard_residual",
        )
        if any(str(row.get(field, "")).strip() for field in numeric_fields):
            raise FinalResultsFillError("failed 2025 attempt cannot contain scores")
        return None
    if bundle.sealed_2025_status == "NOT_RUN":
        if str(row.get("status", "")).upper() != "NOT_RUN":
            raise FinalResultsFillError("2025 paper row conflicts with NOT_RUN lineage")
        return None
    if bundle.sealed_2025_meta is None:
        raise FinalResultsFillError("completed 2025 lineage lacks completion metadata")
    if (
        str(row.get("status", "")).upper() != "COMPLETED"
        or row.get("checkpoint_relation") != "same_frozen_primary_checkpoint"
        or _truth(
            row.get("separately_retrained_evidence"),
            label="2025 separately-retrained marker",
        )
        or _truth(
            row.get("same_checkpoint_intervention"),
            label="2025 intervention marker",
        )
        or row.get("outcome_basis") != OUTCOME_BASIS
        or _integer(row.get("days"), label="2025 paper days")
        != _integer(
            bundle.sealed_2025_meta.get("observed_evaluable_days"),
            label="2025 observed days",
        )
        or not _same_recorded_path(
            row.get("source"), bundle.measured["source_artifacts"].get("one_shot_2025")
        )
    ):
        raise FinalResultsFillError("completed 2025 paper row has invalid lineage")
    summary = bundle.sealed_2025_meta["summary"]
    metrics = (
        "mean_realized_operating_cost",
        "mean_normalized_operating_cost",
        "cvar95_normalized_operating_cost",
        "mean_deadline_expired_work",
        "mean_unserved_energy",
        "maximum_hard_residual",
        "mean_action_total_variation",
        "mean_battery_throughput",
        "mean_compute_energy",
        "mean_migration_energy",
        "mean_pv_curtailed_energy",
        "mean_route_entropy",
        "mean_tail_activation_rate",
    )
    for metric in metrics:
        paper_value = _finite(row.get(metric), label=f"2025 paper {metric}")
        summary_value = _finite(summary.get(metric), label=f"2025 summary {metric}")
        if not math.isclose(
            paper_value, summary_value, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise FinalResultsFillError(f"2025 paper metric drift: {metric}")
    return row


def _validate_row_lineage(row: Mapping[str, str], *, model: str) -> None:
    if not _completed(row):
        return
    separate = _truth(
        row.get("separately_retrained_evidence"),
        label=f"{model} separately_retrained_evidence",
    )
    intervention = _truth(
        row.get("same_checkpoint_intervention"),
        label=f"{model} same_checkpoint_intervention",
    )
    relation = row.get("checkpoint_relation")
    if model == "primary_top2":
        valid = not separate and not intervention and relation == "primary_reselected_checkpoint"
    elif model == "safe_tensor":
        valid = not separate and intervention and relation == "shared_primary_checkpoint_intervention"
    else:
        valid = separate and not intervention and relation == "independent_trained_checkpoint"
    if not valid:
        raise FinalResultsFillError(
            f"{model} checkpoint relation conflicts with its evidence role"
        )


def make_evidence_view(bundle: ResultBundle) -> EvidenceView:
    primary = bundle.measured["primary"]
    primary_row = _paper_row(bundle.paper_rows, "primary_top2")
    safe_row = _paper_row(bundle.paper_rows, "safe_tensor")
    no_tail_row = _paper_row(
        bundle.paper_rows, "no_decision_tail_same_checkpoint_intervention"
    )
    no_route_row = _paper_row(
        bundle.paper_rows, "no_route_prior_same_checkpoint_intervention"
    )
    for model, row in (
        ("primary_top2", primary_row),
        ("safe_tensor", safe_row),
    ):
        _validate_row_lineage(row, model=model)
    if not _truth(
        no_tail_row.get("same_checkpoint_intervention"),
        label="no-tail same-checkpoint marker",
    ) or not _truth(
        no_route_row.get("same_checkpoint_intervention"),
        label="no-route same-checkpoint marker",
    ):
        raise FinalResultsFillError("registered ablations are not same-checkpoint evidence")

    allowed_performance_models = {
        "primary_top2",
        "top1_retrained",
        "dense_direct",
        "independent_r_moe",
    }
    performances: dict[str, Mapping[str, str]] = {}
    for row in bundle.performance_rows:
        model = str(row.get("model", ""))
        status = str(row.get("status", "")).upper()
        if model not in allowed_performance_models or status not in {
            "COMPLETED",
            "NOT_RUN",
        }:
            raise FinalResultsFillError(
                f"unsupported performance row {model!r}/{status!r}"
            )
        if model in performances:
            raise FinalResultsFillError(f"duplicate performance row {model}")
        performances[model] = row
    performance = performances.get("primary_top2")
    if performance is None or not _completed(performance):
        raise FinalResultsFillError("one completed primary Top-2 performance row is required")
    for model, row in performances.items():
        if not _completed(row):
            continue
        for field in (
            "batch1_p50_ms",
            "batch1_p95_ms",
            "throughput_batch_size",
            "throughput_windows_per_second",
            "batch1_peak_allocated_mib",
            "throughput_peak_allocated_mib",
            "total_parameters",
            "active_parameters_per_sample_mean",
        ):
            _finite(row.get(field), label=f"{model} performance {field}")
        if model != "primary_top2":
            model_row = _paper_row(bundle.paper_rows, model, allow_not_run=True)
            if not _completed(model_row):
                raise FinalResultsFillError(
                    f"{model} performance exists without a completed trained model"
                )

    comparisons: dict[str, Mapping[str, str]] = {}
    for row in bundle.comparison_rows:
        name = str(row.get("comparator", ""))
        if not name or not _completed(row):
            continue
        if row.get("primary_model") != "primary_top2":
            raise FinalResultsFillError("comparison does not use primary_top2")
        if (
            row.get("outcome_basis") != OUTCOME_BASIS
            or _integer(row.get("days"), label=f"comparison {name} days")
            != REGISTERED_VALIDATION_DAYS
            or row.get("delta_direction") != "primary_top2_minus_comparator"
        ):
            raise FinalResultsFillError(f"comparison {name} has invalid scope")
        if name in comparisons:
            raise FinalResultsFillError(f"duplicate comparison {name}")
        comparisons[name] = row

    sla_audit = bundle.measured.get("sla_audit")
    if not isinstance(sla_audit, Mapping) or sla_audit.get("status") != "COMPLETED":
        raise FinalResultsFillError("completed validation-only SLA audit is required")
    audit = sla_audit.get("audit_conclusion")
    if not isinstance(audit, Mapping):
        raise FinalResultsFillError("SLA audit conclusion is missing")
    old = _finite(
        audit.get("registered_sla_threshold"), label="audited old SLA threshold"
    )
    if not math.isclose(old, AUDITED_OLD_SLA_TOLERANCE, rel_tol=0.0, abs_tol=1.0e-12):
        raise FinalResultsFillError("SLA audit is not the registered 0.01 audit")
    learned = _finite(
        audit.get("learned_worst_origin_mean_sla"), label="learned audit SLA"
    )
    safe = _finite(
        audit.get("safe_worst_origin_mean_sla"), label="safe audit SLA"
    )
    primary_sla = _finite(primary.get("worst_origin_sla"), label="primary worst SLA")
    # The selector and annual audit aggregate the same six-origin replay through
    # different serialization paths, so allow only sub-1e-6 rounding drift.
    if not math.isclose(learned, primary_sla, rel_tol=1.0e-4, abs_tol=1.0e-6):
        raise FinalResultsFillError("primary worst SLA conflicts with the audit")
    if not (learned <= REVISED_SLA_TOLERANCE < safe):
        raise FinalResultsFillError(
            "audit does not support the revised 0.02 validation tolerance"
        )

    return EvidenceView(
        primary=primary,
        primary_row=primary_row,
        safe_row=safe_row,
        no_tail_row=no_tail_row,
        no_route_row=no_route_row,
        performance=performance,
        performances=performances,
        comparisons=comparisons,
        audit=audit,
        sealed_2025_status=bundle.sealed_2025_status,
        final_2025_row=_sealed_final_row(bundle),
        final_2025_meta=bundle.sealed_2025_meta,
        final_2025_failure=bundle.sealed_2025_failure,
    )


def _format_number(value: Any, *, digits: int = 6) -> str:
    return f"{_finite(value, label='reported number'):.{digits}f}"


def _format_delta(value: Any, *, digits: int = 6) -> str:
    return f"{_finite(value, label='reported delta'):+.{digits}f}"


def _format_residual(value: Any) -> str:
    return f"{_finite(value, label='reported residual'):.3e}"


def _format_int(value: Any) -> str:
    return f"{_integer(value, label='reported integer'):,}"


def _comparison_text(row: Mapping[str, str] | None) -> str:
    if row is None:
        return "NOT RUN"
    delta = _format_delta(row.get("cost_delta"))
    lower = _format_delta(row.get("cost_delta_ci95_lower"))
    upper = _format_delta(row.get("cost_delta_ci95_upper"))
    return (
        f"{delta} (95% {REGISTERED_BOOTSTRAP_BLOCK_DAYS}-day fixed-seed "
        f"moving-block CI [{lower}, {upper}])"
    )


def _cvar_delta(primary: Mapping[str, str], comparator: Mapping[str, str] | None) -> str:
    if comparator is None or not _completed(comparator):
        return "NOT RUN"
    value = _finite(
        primary.get("cvar95_normalized_operating_cost"), label="primary CVaR95"
    ) - _finite(
        comparator.get("cvar95_normalized_operating_cost"),
        label="comparator CVaR95",
    )
    return _format_delta(value)


def _require_paragraph(document: DocumentType, starts_with: str) -> Paragraph:
    matches = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.strip().startswith(starts_with)
    ]
    if len(matches) != 1:
        raise FinalResultsFillError(
            f"paragraph anchor must occur once: {starts_with!r}; found {len(matches)}"
        )
    return matches[0]


def _set_paragraph(paragraph: Paragraph, text: str) -> None:
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def _set_cell(cell: _Cell, text: str) -> None:
    if not cell.paragraphs:
        paragraph = cell.add_paragraph()
    else:
        paragraph = cell.paragraphs[0]
    _set_paragraph(paragraph, text)
    for extra in cell.paragraphs[1:]:
        _set_paragraph(extra, "")


def _require_table(
    document: DocumentType,
    index: int,
    *,
    rows: int,
    columns: int,
    first_cell: str,
) -> Table:
    if len(document.tables) <= index:
        raise FinalResultsFillError(f"table {index} is missing")
    table = document.tables[index]
    if len(table.rows) != rows or len(table.columns) != columns:
        raise FinalResultsFillError(
            f"table {index} topology changed: "
            f"{len(table.rows)}x{len(table.columns)} != {rows}x{columns}"
        )
    if table.cell(0, 0).text.strip() != first_cell:
        raise FinalResultsFillError(
            f"table {index} header changed: {table.cell(0, 0).text!r}"
        )
    return table


def _require_row_labels(table: Table, labels: Sequence[str], *, table_index: int) -> None:
    actual = [row.cells[0].text.strip() for row in table.rows[1:]]
    if actual != list(labels):
        raise FinalResultsFillError(
            f"table {table_index} row labels changed: {actual!r}"
        )


def _remove_row(table: Table, row_index: int) -> None:
    row = table.rows[row_index]
    table._tbl.remove(row._tr)


def _find_optional_model(bundle: ResultBundle, model: str) -> Mapping[str, str]:
    return _paper_row(bundle.paper_rows, model, allow_not_run=True)


def _model_value(row: Mapping[str, str], field: str, *, formatter) -> str:
    if not _completed(row):
        return "NOT RUN"
    value = row.get(field, "")
    if value == "":
        return "NOT RUN"
    return formatter(value)


def _hypothesis_decisions(view: EvidenceView) -> dict[str, str]:
    safe_comparison = view.comparisons.get("safe_tensor")
    h3_supported = (
        safe_comparison is not None
        and _finite(
            safe_comparison.get("cost_delta_ci95_upper"), label="H3 CI upper"
        )
        < 0.0
    )

    primary_cvar = _finite(
        view.primary_row.get("cvar95_normalized_operating_cost"),
        label="primary CVaR95",
    )
    no_tail_cvar = _finite(
        view.no_tail_row.get("cvar95_normalized_operating_cost"),
        label="no-tail CVaR95",
    )
    no_prior = view.comparisons.get("no_route_prior_same_checkpoint_intervention")
    pooled = view.comparisons.get("pooled_latent_instead_of_expert_states")
    h5_supported = (
        no_prior is not None
        and pooled is not None
        and _finite(no_prior.get("cost_delta_ci95_upper"), label="H5 no-prior CI")
        < 0.0
        and _finite(pooled.get("cost_delta_ci95_upper"), label="H5 pooled CI")
        < 0.0
    )

    p95 = _finite(view.performance.get("batch1_p95_ms"), label="batch-1 p95")
    peak = max(
        _finite(
            view.performance.get("batch1_peak_allocated_mib"),
            label="batch-1 peak memory",
        ),
        _finite(
            view.performance.get("throughput_peak_allocated_mib"),
            label="throughput peak memory",
        ),
    )
    h6_supported = (
        p95 <= REGISTERED_LATENCY_BUDGET_MS
        and peak <= REGISTERED_MEMORY_BUDGET_MIB
    )

    return {
        "H0": "NOT RUN in the aggregate result package; no new physical-core gate row is supplied.",
        "H1": "NOT RUN in the aggregate result package; the released W1 capability gate remains separate evidence.",
        "H2": "NOT RUN: the aggregate contains no registered paired R-A versus R-B trajectory with common 2024 decisions.",
        "H3": (
            "SUPPORTED: the paired main-minus-safe normalized composite-cost "
            f"difference is {_comparison_text(safe_comparison)}."
            if h3_supported
            else (
                "NOT RUN: the aggregate contains no paired main-versus-safe result."
                if safe_comparison is None
                else "NOT SUPPORTED: the paired 95% interval does not favor Top-2 over safe tensor."
            )
        ),
        "H4": (
            "NOT SUPPORTED: conditional-tail Top-2 did not lower empirical "
            f"CVaR95 ({_format_number(primary_cvar)} versus no-tail "
            f"{_format_number(no_tail_cvar)})."
            if primary_cvar >= no_tail_cvar
            else "NOT RUN: Top-2 lowered empirical CVaR95, but the aggregate lacks "
            "the no-tail worst-origin composite-SLA severity required by H4."
        ),
        "H5": (
            "SUPPORTED: both no-prior and pooled-state paired intervals favor the main model."
            if h5_supported
            else (
                "NOT RUN: no-prior or pooled-state paired evidence is missing."
                if no_prior is None or pooled is None
                else "NOT SUPPORTED: the registered no-prior and pooled-state paired evidence does not jointly favor the main model."
            )
        ),
        "H6": (
            "SUPPORTED: batch-1 p95 is "
            f"{p95:.1f} ms and measured peak allocation is {peak:.1f} MiB. "
            "Top-k is token-wise; window-level active parameters are the measured count across all routed tokens."
            if h6_supported
            else "NOT SUPPORTED: measured p95 latency or peak allocation exceeds the registered budget."
        ),
    }


def _edit_paragraphs(
    document: DocumentType, bundle: ResultBundle, view: EvidenceView
) -> None:
    primary_cost = _format_number(
        view.primary_row.get("mean_normalized_operating_cost")
    )
    primary_cvar = _format_number(
        view.primary_row.get("cvar95_normalized_operating_cost")
    )
    safe_cost = _format_number(view.safe_row.get("mean_normalized_operating_cost"))
    safe_cvar = _format_number(
        view.safe_row.get("cvar95_normalized_operating_cost")
    )
    primary_sla = _format_number(view.primary.get("worst_origin_sla"))
    safe_sla = _format_number(view.audit.get("safe_worst_origin_mean_sla"))
    hard = _format_residual(view.primary_row.get("maximum_hard_residual"))
    p95 = _finite(view.performance.get("batch1_p95_ms"), label="batch-1 p95")
    comparisons = view.comparisons
    safe_comparison = comparisons.get("safe_tensor")
    hypotheses = _hypothesis_decisions(view)
    verdicts = {
        hypothesis: text.split(":", 1)[0]
        for hypothesis, text in hypotheses.items()
    }
    missing = bundle.measured.get("missing_experiments", [])
    missing_text = ", ".join(str(item) for item in missing) if missing else "none"
    performance_labels = {
        "top1_retrained": "separately trained Top-1",
        "dense_direct": "separately trained dense-direct R",
        "independent_r_moe": "separately trained independent R-MoE",
    }
    optional_performance_parts = [
        f"{performance_labels[model]} p95 "
        f"{_finite(row.get('batch1_p95_ms'), label=f'{model} p95'):.1f} ms"
        for model, row in view.performances.items()
        if model in performance_labels and _completed(row)
    ]
    optional_performance_text = (
        "; ".join(optional_performance_parts)
        if optional_performance_parts
        else "independently trained comparator latency NOT RUN"
    )
    if view.sealed_2025_status == "FAILED_ONE_SHOT_DATA_INCOMPLETE":
        final_evidence_note = (
            "The unique permitted 2025 final evaluation was attempted but failed "
            "before inference because the required final daily bundle was not "
            "materialized. The one-shot attempt is consumed, no 2025 score exists, "
            "selection was unchanged, and rerun is prohibited."
        )
        final_abstract = final_evidence_note
        final_result = final_evidence_note
        final_conclusion = final_evidence_note
    elif view.final_2025_row is None:
        final_evidence_note = "The sealed 2025 final evaluation remains NOT RUN and unopened."
        final_abstract = "The sealed 2025 final evaluation remains NOT RUN."
        final_result = "The sealed 2025 result is NOT RUN; no final-year metric is inferred."
        final_conclusion = "The sealed 2025 final evaluation remains NOT RUN."
    else:
        final_days = _integer(view.final_2025_row.get("days"), label="2025 days")
        final_cost = _format_number(
            view.final_2025_row.get("mean_normalized_operating_cost")
        )
        final_cvar = _format_number(
            view.final_2025_row.get("cvar95_normalized_operating_cost")
        )
        final_deadline = _format_number(
            view.final_2025_row.get("mean_deadline_expired_work")
        )
        final_unserved = _format_number(
            view.final_2025_row.get("mean_unserved_energy")
        )
        final_hard = _format_residual(
            view.final_2025_row.get("maximum_hard_residual")
        )
        final_evidence_note = (
            "The frozen checkpoint was opened for exactly one 2025 final-test "
            "evaluation; training, checkpoint selection, threshold/statistic updates, "
            "reruns, and feedback to model selection were disabled."
        )
        final_abstract = (
            f"The unique frozen 2025 final evaluation covered {final_days} registered "
            f"dates and reported dimensionless normalized composite cost {final_cost}, "
            f"empirical CVaR95 {final_cvar}, mean newly expired work {final_deadline}, "
            f"mean unserved energy {final_unserved}, and maximum registered hard "
            f"residual {final_hard}."
        )
        final_result = final_abstract + " It did not modify model selection."
        final_conclusion = (
            final_abstract
            + " This was the unique frozen final evaluation and supplied no selection feedback."
        )

    replacements = {
        "No numerical performance claim is invented.": (
            "Numerical statements in this final-results version are populated only "
            "from measured_results.json and its registered CSVs. Missing experiments "
            "remain NOT RUN; same-checkpoint interventions are never presented as "
            f"separately trained models. {final_evidence_note}"
        ),
        "Artificial-intelligence training, cloud services, and online inference are increasing data-center electricity demand": (
            "Artificial-intelligence workloads create coupled electricity-cost, service, and physical-feasibility decisions for data-center virtual power plants. "
            "We develop WorldMoE-VPP R-ECMoE, a solver-free risk-conditioned continuation of a released causal sparse world model, and evaluate structured day-ahead and real-time actions through post-decision realized historical replay. "
            f"On {REGISTERED_VALIDATION_DAYS} registered 2024 dates, rather than all 366 calendar dates, the eligible selector-repaired Top-2 checkpoint achieved a dimensionless normalized composite operating cost of {primary_cost} and empirical CVaR95 of {primary_cvar}; the same-input safe tensor baseline achieved {safe_cost} and {safe_cvar}. "
            f"The paired main-minus-safe cost difference was {_comparison_text(safe_comparison)}. Worst-origin composite SLA severity was {primary_sla} versus {safe_sla} for safe tensor, and the maximum registered hard residual was {hard}. "
            f"Batch-1 p95 inference latency was {p95:.1f} ms. A validation-only feasibility audit revised the selector SLA tolerance from 0.01 to 0.02, and latency affects selection only through budget excess. Hypothesis decisions were H2 {verdicts['H2']}, H3 {verdicts['H3']}, H4 {verdicts['H4']}, and H5 {verdicts['H5']}. "
            f"{final_abstract} Missing experiments are {missing_text}. Costs are dimensionless composite objectives, not US-dollar savings, and all findings are trace-driven counterfactual co-simulations."
        ),
        "Validation checkpoints must be finite": (
            "Validation checkpoints must be finite, come from R-B, satisfy the registered hard-residual gate, meet the audit-revised worst-origin composite SLA tolerance of 0.02, and remain within the registered batch-1 h96 p95 latency budget. The previous 0.01 SLA tolerance was invalidated by the validation-only attainability audit."
        ),
        "Eligible full-validation checkpoints minimize": (
            "Eligible full-validation checkpoints minimize the registered dimensionless composite decision value over the six decisions on each of 364 registered 2024 dates. The latency term is max(0, p95/budget - 1): measurements within budget receive no latency preference or penalty, and only budget excess is charged."
        ),
        "The development fit is 2022-01-01 through 2023-12-31": (
            "The development fit is 2022-01-01 through 2023-12-31 (722 valid days); validation and formal checkpoint selection use 364 registered dates in 2024, not all 366 calendar dates. The primary model is selected only on 2024. "
            + final_evidence_note
        ),
        "Cells marked TO BE GENERATED remain empty": (
            f"The eligible independently reselected checkpoint is evaluated on {REGISTERED_VALIDATION_DAYS} registered 2024 dates and six common decisions per date using realized historical replay. The main normalized composite cost is {primary_cost}, empirical CVaR95 is {primary_cvar}, worst-origin composite SLA severity is {primary_sla}, and the maximum registered hard residual is {hard}. {final_result} Values absent from the aggregate are labeled NOT RUN rather than inferred."
        ),
        "Report cost-quantile scores and calibration": (
            "The result package reports realized-replay normalized composite cost, empirical CVaR95, composite SLA severity, registered hard residuals, routing behavior, and measured runtime. "
            f"The primary batch-1 p95 is {p95:.1f} ms, total parameters are {_format_int(view.performance.get('total_parameters'))}, and measured active parameters per sample are {_format_int(view.performance.get('active_parameters_per_sample_mean'))}; {optional_performance_text}. Sparse Top-k routing is token-wise, while window-level active parameters are reported from the measured execution across all routed tokens."
        ),
        "The headline trained-model table compares": (
            "Table 30 contains exactly five headline rows: safe tensor, separately trained dense-direct R, separately trained independent R-MoE, separately trained Top-1 continuation, and the eligible balanced Top-2 model. Same-checkpoint Top-1/tail/route/expert/coordinator interventions appear only in Table 29. The Top-2 row reports both 2024 selection evidence and the frozen 2025 one-shot result when available; comparator inference p95 is populated only from its own benchmark."
        ),
        "A hypothesis is supported only when": (
            f"Measured decisions are: H2 {verdicts['H2']}; H3 {verdicts['H3']}; H4 {verdicts['H4']}; H5 {verdicts['H5']}; H6 {verdicts['H6']}. Table 31 records the evidence and explicitly leaves missing foundation or curriculum tests as NOT RUN."
        ),
        "The preregistered study asks whether W1 expert continuation improves": (
            f"Across {REGISTERED_VALIDATION_DAYS} registered 2024 dates, the eligible balanced Top-2 R-ECMoE checkpoint achieved a paired dimensionless normalized composite-cost difference of {_comparison_text(safe_comparison)} relative to the safe tensor baseline. The measured decisions were H2 {verdicts['H2']}, H3 {verdicts['H3']}, H4 {verdicts['H4']}, and H5 {verdicts['H5']}. The selector uses the audit-revised 0.02 composite-SLA tolerance and charges only latency-budget excess. {final_conclusion} Separately trained rows remain NOT RUN when their independent artifacts are absent. These findings are counterfactual realized-replay results, not dollar savings or observed NLR market participation."
        ),
    }
    for anchor, text in replacements.items():
        _set_paragraph(_require_paragraph(document, anchor), text)


def _edit_same_checkpoint_table(
    document: DocumentType, view: EvidenceView
) -> None:
    table = _require_table(
        document,
        TABLE_SAME_CHECKPOINT,
        rows=5,
        columns=5,
        first_cell="Same-checkpoint intervention",
    )
    _require_row_labels(
        table,
        (
            "No decision tail",
            "No W1 route prior",
            "Tail always on",
            "Pooled state / no sparse coordinator",
        ),
        table_index=TABLE_SAME_CHECKPOINT,
    )
    headers = (
        "Same-checkpoint intervention",
        "Mean normalized cost delta (main - intervention; 95% 7-day fixed-seed moving-block CI)",
        "Empirical CVaR95 delta (main - intervention)",
        "Worst-origin composite SLA severity delta",
        "Latency increment",
    )
    for cell, value in zip(table.rows[0].cells, headers, strict=True):
        _set_cell(cell, value)

    no_tail = view.comparisons.get("no_decision_tail_same_checkpoint_intervention")
    no_route = view.comparisons.get("no_route_prior_same_checkpoint_intervention")
    tail_always = view.comparisons.get("tail_always_on")
    pooled = view.comparisons.get("pooled_latent_instead_of_expert_states")
    no_coordinator = view.comparisons.get("no_sparse_coordinator")
    rows = (
        (
            "No decision tail",
            _comparison_text(no_tail),
            _cvar_delta(view.primary_row, view.no_tail_row),
            "NOT RUN",
            "NOT RUN",
        ),
        (
            "No W1 route prior",
            _comparison_text(no_route),
            _cvar_delta(view.primary_row, view.no_route_row),
            "NOT RUN",
            "NOT RUN",
        ),
        (
            "Tail always on",
            _comparison_text(tail_always),
            "NOT RUN",
            "NOT RUN",
            "NOT RUN",
        ),
        (
            "Pooled state / no sparse coordinator",
            "Pooled: "
            + _comparison_text(pooled)
            + "\nNo coordinator: "
            + _comparison_text(no_coordinator),
            "Pooled: NOT RUN\nNo coordinator: NOT RUN",
            "NOT RUN",
            "NOT RUN",
        ),
    )
    for table_row, values in zip(table.rows[1:], rows, strict=True):
        for cell, value in zip(table_row.cells, values, strict=True):
            _set_cell(cell, value)


def _edit_five_model_table(
    document: DocumentType, bundle: ResultBundle, view: EvidenceView
) -> None:
    table = _require_table(
        document,
        TABLE_FIVE_MODELS,
        rows=8,
        columns=7,
        first_cell="Trained policy",
    )
    _require_row_labels(
        table,
        (
            "Safe tensor baseline",
            "Dense-direct R",
            "Independent R-MoE",
            "R-ECMoE Top-1 trained",
            "R-ECMoE Top-2 balanced",
            "Top-2 no-tail intervention",
            "Top-2 no-route-prior intervention",
        ),
        table_index=TABLE_FIVE_MODELS,
    )
    headers = (
        "Headline model (training relation)",
        "Mean normalized composite cost (dimensionless)",
        "Empirical CVaR95 (dimensionless)",
        "Worst-origin composite SLA severity",
        "Maximum registered hard residual",
        "Batch-1 p95 (ms)",
        "Status",
    )
    for cell, value in zip(table.rows[0].cells, headers, strict=True):
        _set_cell(cell, value)

    dense = _find_optional_model(bundle, "dense_direct")
    independent = _find_optional_model(bundle, "independent_r_moe")
    top1 = _find_optional_model(bundle, "top1_retrained")
    for model, row in (
        ("dense_direct", dense),
        ("independent_r_moe", independent),
        ("top1_retrained", top1),
    ):
        _validate_row_lineage(row, model=model)

    learned_sla = _format_number(view.primary.get("worst_origin_sla"))
    safe_sla = _format_number(view.audit.get("safe_worst_origin_mean_sla"))
    if view.sealed_2025_status == "FAILED_ONE_SHOT_DATA_INCOMPLETE":
        primary_cost = (
            "2024 validation: "
            + _model_value(
                view.primary_row,
                "mean_normalized_operating_cost",
                formatter=_format_number,
            )
            + "\n2025 one-shot final: FAILED - NO SCORE"
        )
        primary_cvar = (
            "2024 validation: "
            + _model_value(
                view.primary_row,
                "cvar95_normalized_operating_cost",
                formatter=_format_number,
            )
            + "\n2025 one-shot final: FAILED - NO SCORE"
        )
        primary_service = (
            f"2024 worst-origin composite: {learned_sla}\n"
            "2025 one-shot final: FAILED - NO SCORE"
        )
        primary_residual = (
            "2024 validation: "
            + _model_value(
                view.primary_row,
                "maximum_hard_residual",
                formatter=_format_residual,
            )
            + "\n2025 one-shot final: FAILED - NO SCORE"
        )
        primary_status = (
            "2024 COMPLETED; 2025 FAILED_DATA_INCOMPLETE - attempt consumed, no rerun"
        )
    elif view.final_2025_row is None:
        primary_cost = _model_value(
            view.primary_row,
            "mean_normalized_operating_cost",
            formatter=_format_number,
        )
        primary_cvar = _model_value(
            view.primary_row,
            "cvar95_normalized_operating_cost",
            formatter=_format_number,
        )
        primary_service = learned_sla + "\n2025 one-shot final: NOT RUN"
        primary_residual = _model_value(
            view.primary_row, "maximum_hard_residual", formatter=_format_residual
        )
        primary_status = "COMPLETED - eligible full-2024 best; 2025 NOT RUN"
    else:
        primary_cost = (
            "2024 validation: "
            + _model_value(
                view.primary_row,
                "mean_normalized_operating_cost",
                formatter=_format_number,
            )
            + "\n2025 one-shot final: "
            + _model_value(
                view.final_2025_row,
                "mean_normalized_operating_cost",
                formatter=_format_number,
            )
        )
        primary_cvar = (
            "2024 validation: "
            + _model_value(
                view.primary_row,
                "cvar95_normalized_operating_cost",
                formatter=_format_number,
            )
            + "\n2025 one-shot final: "
            + _model_value(
                view.final_2025_row,
                "cvar95_normalized_operating_cost",
                formatter=_format_number,
            )
        )
        primary_service = (
            f"2024 worst-origin composite: {learned_sla}\n"
            "2025 composite: NOT RUN; mean expired work: "
            + _format_number(view.final_2025_row.get("mean_deadline_expired_work"))
            + "; mean unserved energy: "
            + _format_number(view.final_2025_row.get("mean_unserved_energy"))
        )
        primary_residual = (
            "2024 validation: "
            + _model_value(
                view.primary_row,
                "maximum_hard_residual",
                formatter=_format_residual,
            )
            + "\n2025 one-shot final: "
            + _model_value(
                view.final_2025_row,
                "maximum_hard_residual",
                formatter=_format_residual,
            )
        )
        primary_status = "COMPLETED - eligible 2024 best; unique frozen 2025 final"
    rows: tuple[tuple[str, ...], ...] = (
        (
            "Safe tensor baseline (same causal packet; not a trained comparator)",
            _model_value(
                view.safe_row, "mean_normalized_operating_cost", formatter=_format_number
            ),
            _model_value(
                view.safe_row,
                "cvar95_normalized_operating_cost",
                formatter=_format_number,
            ),
            safe_sla,
            _model_value(
                view.safe_row, "maximum_hard_residual", formatter=_format_residual
            ),
            "NOT RUN",
            "COMPLETED - tensor baseline",
        ),
        _headline_model_row(
            "Dense-direct R", dense, view.performances.get("dense_direct")
        ),
        _headline_model_row(
            "Independent R-MoE",
            independent,
            view.performances.get("independent_r_moe"),
        ),
        _headline_model_row(
            "R-ECMoE Top-1 + conditional tail",
            top1,
            view.performances.get("top1_retrained"),
        ),
        (
            "R-ECMoE Top-2 balanced (eligible reselected primary)",
            primary_cost,
            primary_cvar,
            primary_service,
            primary_residual,
            f"{_finite(view.performance.get('batch1_p95_ms'), label='batch-1 p95'):.1f}",
            primary_status,
        ),
    )
    for table_row, values in zip(table.rows[1:6], rows, strict=True):
        for cell, value in zip(table_row.cells, values, strict=True):
            _set_cell(cell, value)

    # The last two legacy rows were same-checkpoint interventions duplicated
    # from Table 29.  Removing them makes Table 30 the registered five-model
    # headline table without changing the remaining table style.
    _remove_row(table, 7)
    _remove_row(table, 6)


def _headline_model_row(
    label: str,
    row: Mapping[str, str],
    performance: Mapping[str, str] | None,
) -> tuple[str, ...]:
    if not _completed(row):
        return (label, "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN")
    return (
        f"{label} (separately trained)",
        _model_value(row, "mean_normalized_operating_cost", formatter=_format_number),
        _model_value(
            row, "cvar95_normalized_operating_cost", formatter=_format_number
        ),
        "NOT RUN",
        _model_value(row, "maximum_hard_residual", formatter=_format_residual),
        (
            f"{_finite(performance.get('batch1_p95_ms'), label=f'{label} p95'):.1f}"
            if performance is not None and _completed(performance)
            else "NOT RUN"
        ),
        "COMPLETED - separately trained",
    )


def _edit_hypothesis_table(document: DocumentType, view: EvidenceView) -> None:
    table = _require_table(
        document,
        TABLE_HYPOTHESES,
        rows=8,
        columns=2,
        first_cell="Hypothesis",
    )
    _require_row_labels(
        table,
        ("H0", "H1", "H2", "H3", "H4", "H5", "H6"),
        table_index=TABLE_HYPOTHESES,
    )
    decisions = _hypothesis_decisions(view)
    _set_cell(table.cell(0, 1), "Measured decision and evidence")
    for row in table.rows[1:]:
        hypothesis = row.cells[0].text.strip()
        _set_cell(row.cells[1], decisions[hypothesis])


def _as_path_text(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "NOT RUN"


def _sibling_path(path_text: Any, filename: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return "NOT RUN"
    return str(Path(text).with_name(filename))


def _optional_baseline_paths(bundle: ResultBundle) -> str:
    entries: list[str] = []
    for label in ("dense_direct", "independent_r_moe", "top1_retrained"):
        row = _find_optional_model(bundle, label)
        if _completed(row):
            entries.append(f"{label}: {_as_path_text(row.get('source'))}")
        else:
            entries.append(f"{label}: NOT RUN")
    return "\n".join(entries)


def _optional_performance_paths(bundle: ResultBundle) -> str:
    indexed = {str(row.get("model", "")): row for row in bundle.performance_rows}
    entries: list[str] = []
    for label in ("dense_direct", "independent_r_moe", "top1_retrained"):
        row = indexed.get(label)
        if row is not None and _completed(row):
            entries.append(f"{label} benchmark: {_as_path_text(row.get('source'))}")
        else:
            entries.append(f"{label} benchmark: NOT RUN")
    return "\n".join(entries)


def _edit_artifact_table(
    document: DocumentType, bundle: ResultBundle
) -> None:
    table = _require_table(
        document,
        TABLE_ARTIFACT_PATHS,
        rows=16,
        columns=2,
        first_cell="Artifact",
    )
    primary = bundle.measured["primary"]
    source_artifacts = bundle.measured.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise FinalResultsFillError("measured_results.source_artifacts is missing")
    best_text = _as_path_text(primary.get("best_checkpoint"))
    best_run_root = Path(best_text).parent.parent if best_text != "NOT RUN" else None
    evaluation_meta = _as_path_text(source_artifacts.get("primary_evaluation"))
    performance_source = _as_path_text(source_artifacts.get("performance"))
    sealed_path = source_artifacts.get("one_shot_2025")
    if bundle.sealed_2025_status == "FAILED_ONE_SHOT_DATA_INCOMPLETE":
        sealed_path_text = _as_path_text(sealed_path)
        sealed_destination = (
            "2025 one-shot failure evidence: final daily bundle not materialized; "
            "attempt consumed, no score, no rerun, no selection feedback"
        )
    elif bundle.sealed_2025_meta is None:
        sealed_path_text = "NOT RUN"
        sealed_destination = "Sealed 2025 one-shot final evaluation: NOT RUN"
    else:
        sealed_path_text = _as_path_text(sealed_path)
        sealed_destination = (
            "Unique frozen 2025 final metadata; no training, reselection, rerun, "
            "threshold/statistic update, or selection feedback"
        )

    # Retain the released foundation references already registered in the
    # manuscript, then replace superseded R-run paths with the aggregate's
    # selector-repaired lineage.  These retained strings are not opened.
    released_w1_path = table.cell(1, 0).text.strip()
    w1_capability_path = table.cell(2, 0).text.strip()
    balanced_config_path = table.cell(4, 0).text.strip()
    rows = (
        (released_w1_path, "Released W1 source, epoch 56/update 3360"),
        (w1_capability_path, "Released W1 packet and freeze capability evidence"),
        (balanced_config_path, "Balanced architecture, curriculum, loss, and selector"),
        (
            _as_path_text(primary.get("source_invalid_run")),
            "Superseded source checkpoint; old selector invalid, training not repeated",
        ),
        (best_text, "Eligible independently reselected primary checkpoint"),
        (
            str(best_run_root / "run_meta.json") if best_run_root else "NOT RUN",
            "Reselection lineage, source checkpoint, and invalid-selector reason",
        ),
        (
            str(best_run_root / "config_resolved.yaml") if best_run_root else "NOT RUN",
            "Resolved selector: composite SLA tolerance 0.02 and budget-excess latency",
        ),
        (
            _as_path_text(bundle.measured.get("sla_audit", {}).get("source")),
            "Validation-only SLA attainability audit: 0.01 revised to 0.02",
        ),
        (evaluation_meta, "Complete 364-date 2024 realized-replay evaluation metadata"),
        (
            _sibling_path(evaluation_meta, "variant_summary.csv")
            + "\n"
            + _sibling_path(evaluation_meta, "per_day_variant.csv"),
            "Same-checkpoint summary plus registered 2024 date-by-variant outcomes",
        ),
        (
            str(bundle.comparison_path),
            "Paired comparisons with same-checkpoint and separately trained roles separated",
        ),
        (
            performance_source + "\n" + str(bundle.performance_path),
            "Primary benchmark source plus aggregated performance row",
        ),
        (
            str(bundle.measured_path) + "\n" + str(bundle.paper_path),
            "Authoritative measured registry plus five-model result table",
        ),
        (
            _optional_baseline_paths(bundle)
            + "\n"
            + _optional_performance_paths(bundle),
            "Independent dense, independent-MoE, and retrained Top-1 evaluation/benchmark sources",
        ),
        (sealed_path_text, sealed_destination),
    )
    for table_row, values in zip(table.rows[1:], rows, strict=True):
        for cell, value in zip(table_row.cells, values, strict=True):
            _set_cell(cell, value)


def _mark_remaining_missing_not_run(document: DocumentType) -> None:
    """Convert legacy result placeholders without manufacturing measurements."""

    for paragraph in document.paragraphs:
        if "TO BE GENERATED" in paragraph.text:
            _set_paragraph(
                paragraph, paragraph.text.replace("TO BE GENERATED", "NOT RUN")
            )
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    if "TO BE GENERATED" in paragraph.text:
                        _set_paragraph(
                            paragraph,
                            paragraph.text.replace("TO BE GENERATED", "NOT RUN"),
                        )


def _validate_final_document(document: DocumentType, view: EvidenceView) -> None:
    if len(document.tables) != 36:
        raise FinalResultsFillError("final document must retain all 36 tables")
    if len(document.tables[TABLE_FIVE_MODELS].rows) != 6:
        raise FinalResultsFillError("Table 30 must contain one header plus five models")
    joined = "\n".join(paragraph.text for paragraph in document.paragraphs)
    joined += "\n" + "\n".join(
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    )
    forbidden = (
        "No numerical performance claim is invented",
        "Cells marked TO BE GENERATED",
        "TO BE GENERATED",
        "TRAINING",
        "PENDING BEST",
        "$",
        "only 2/7",
        "366-day 2024",
    )
    remaining = [token for token in forbidden if token in joined]
    if remaining:
        raise FinalResultsFillError(
            "superseded or unsafe result wording remains: " + ", ".join(remaining)
        )
    required = (
        "364 registered 2024 dates",
        "dimensionless normalized composite",
        "composite SLA severity",
        "registered hard residual",
        "H3",
        "H4",
        "H5",
        "NOT RUN",
        "budget excess",
        "token-wise",
    )
    missing = [token for token in required if token not in joined]
    if missing:
        raise FinalResultsFillError(
            "final document omitted required result wording: " + ", ".join(missing)
        )
    if view.sealed_2025_status == "FAILED_ONE_SHOT_DATA_INCOMPLETE":
        failure_terms = (
            "FAILED - NO SCORE",
            "daily bundle was not materialized",
            "attempt is consumed",
            "rerun is prohibited",
        )
        absent = [term for term in failure_terms if term not in joined]
        if absent:
            raise FinalResultsFillError(
                "failed 2025 one-shot wording is incomplete: " + ", ".join(absent)
            )


def default_output_path(input_docx: str | Path) -> Path:
    source = Path(input_docx)
    stem = source.stem
    if "_RealizedReplay_" in stem:
        stem = stem.replace("_RealizedReplay_", "_Final_Results_", 1)
    else:
        stem = stem + "_Final_Results"
    return source.with_name(stem + source.suffix)


def fill_final_results_docx(
    *,
    input_docx: str | Path,
    results_root: str | Path,
    output_docx: str | Path | None = None,
) -> Path:
    """Create a styled final-results copy without opening any recorded sources."""

    source = Path(input_docx).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".docx":
        raise FinalResultsFillError(f"source DOCX is missing: {source}")
    output = (
        default_output_path(source)
        if output_docx is None
        else Path(output_docx).expanduser().resolve()
    )
    output = output.resolve()
    if output == source:
        raise FinalResultsFillError("source manuscript must not be overwritten")
    if output.exists():
        raise FinalResultsFillError(f"output already exists: {output}")

    bundle = load_result_bundle(results_root)
    view = make_evidence_view(bundle)
    document = Document(source)
    if len(document.tables) != 36:
        raise FinalResultsFillError(
            f"source manuscript topology changed: expected 36 tables, found {len(document.tables)}"
        )

    _edit_paragraphs(document, bundle, view)
    _edit_same_checkpoint_table(document, view)
    _edit_five_model_table(document, bundle, view)
    _edit_hypothesis_table(document, view)
    _edit_artifact_table(document, bundle)
    _mark_remaining_missing_not_run(document)
    _validate_final_document(document, view)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FinalResultsFillError(f"temporary output already exists: {temporary}")
    try:
        document.save(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-docx", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument(
        "--output-docx",
        type=Path,
        default=None,
        help="Default: source name with RealizedReplay replaced by Final_Results",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = fill_final_results_docx(
        input_docx=args.input_docx,
        results_root=args.results_root,
        output_docx=args.output_docx,
    )
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
