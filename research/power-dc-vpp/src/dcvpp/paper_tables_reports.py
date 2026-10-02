from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TABLE_SOURCE_SCHEMA_VERSION = "paper_table_source_v1"
REPORT_SOURCE_SCHEMA_VERSION = "paper_report_source_v1"
TABLE_WRITER_VERSION = "paper_tables_reports_v1"

REFERENCE_BOUNDARY = (
    "Referenced public-data evidence only; observed values are source records, "
    "not measurements made by this study."
)
NLR_BOUNDARY = (
    "Derived from referenced NLR/SRRL records; observed denotes source-level "
    "records, not investigator measurement."
)
NYISO_BOUNDARY = (
    "Forecast evaluation on referenced NYISO records; no facility operation or "
    "realized market return is claimed."
)
WORKLOAD_BOUNDARY = (
    "NLR workload-trace evidence only; replayed dates are not additional "
    "independent observations."
)
TRANSFER_BOUNDARY = (
    "NLR-calibrated parameter transfer to counterfactual NYISO twins; no NLR "
    "facility was operated in NYISO."
)
XCEL_BOUNDARY = (
    "Tariff-rule counterfactual only; zero observed event calls and no observed "
    "demand-response savings."
)
COUNTERFACTUAL_BOUNDARY = (
    "Counterfactual co-simulation only; not realized profit, observed grid "
    "benefit, or an actual NLR-in-NYISO operation."
)
DART_COUNTERFACTUAL_BOUNDARY = (
    "DART decision value is constrained data-center counterfactual replay on "
    "referenced NYISO prices; it is not virtual-trader P&L, realized commercial "
    "profit, or observed grid benefit."
)


@dataclass(frozen=True)
class TableDefinition:
    name: str
    writer: str
    source_patterns: tuple[str, ...]
    required_groups: tuple[tuple[str, ...], ...]
    evidence_role: str = "derived"
    source_relationship: str = "derived_from_third_party_referenced_public_data"
    claim_boundary: str = REFERENCE_BOUNDARY

    @property
    def relative_path(self) -> str:
        return f"artifacts/tables/{self.name}"


def _table(
    name: str,
    writer: str,
    patterns: Sequence[str],
    required_groups: Sequence[Sequence[str]],
    *,
    evidence_role: str = "derived",
    relationship: str = "derived_from_third_party_referenced_public_data",
    boundary: str = REFERENCE_BOUNDARY,
) -> TableDefinition:
    return TableDefinition(
        name=name,
        writer=writer,
        source_patterns=tuple(patterns),
        required_groups=tuple(tuple(group) for group in required_groups),
        evidence_role=evidence_role,
        source_relationship=relationship,
        claim_boundary=boundary,
    )


_FORECAST_METRICS = (
    # Confirmatory H1/H2 evidence.  Keep this namespace first and label it
    # separately from the frozen v1 auxiliary lineage below.
    "artifacts/real_forecasting_extensions/run_id=*/metrics/**/*.json",
    "artifacts/real_forecasting_extensions/run_id=*/manifests/*.json",
    "artifacts/real_forecasting_extensions/run_id=*/extension_run_manifest.json",
    # Frozen v1 is auxiliary lineage / non-confirmatory baseline evidence.
    "artifacts/real_forecasting/*/metrics/metrics.json",
    "artifacts/real_forecasting/*/metrics/*leaderboard*.json",
    "artifacts/real_forecasting/*/metrics/**/*.json",
)
_FORECAST_PREDICTIONS = (
    "artifacts/real_forecasting_extensions/run_id=*/predictions/**/*.npz",
    "artifacts/real_forecasting/*/predictions/*.parquet",
    "artifacts/real_forecasting/*/predictions/**/*.npz",
)
_OPT_RESULTS = (
    # This is the production runner namespace.
    "artifacts/optimization_branch/*/optimization_results.csv",
    "artifacts/optimization_branch/*/results/**/*.json",
    "artifacts/optimization_branch/*/rolling_checkpoints/*.json",
    # Legacy compatibility only.
    "artifacts/research/optimization/*/optimization_results.csv",
    "artifacts/research/optimization/*/results/**/*.json",
    "artifacts/research/optimization/*/rolling_checkpoints/*.json",
)
_OPT_SENSITIVITY = (
    "artifacts/optimization_branch/*/optimization_sensitivity.csv",
    "artifacts/optimization_branch/*/sensitivity_results.json",
    "artifacts/optimization_branch/*/timezone_sensitivity*.json",
    "artifacts/optimization_branch/*/timezone_sensitivity*.csv",
    "artifacts/research/optimization/*/optimization_sensitivity.csv",
    "artifacts/research/optimization/*/sensitivity_results.json",
    "artifacts/research/optimization/*/timezone_sensitivity*.json",
    "artifacts/research/optimization/*/timezone_sensitivity*.csv",
)
_DART_PROBABILITY = (
    "artifacts/dart_value_select/*/tables/dart_probability_quality.csv",
    "artifacts/dart_value_select/*/probability_quality.csv",
    "artifacts/dart_value_select/*/probability_quality.json",
    "artifacts/dart_forecast_oos/*/probability_quality.json",
)
_DART_REPLAY = (
    "artifacts/dart_value_select/*/tables/dart_decision_value.csv",
    "artifacts/dart_value_select/*/tables/dart_regime_value.csv",
    "artifacts/dart_value_select/*/manifests/dart_statistical_selection_manifest.json",
    "artifacts/dart_replay/*/replay_manifest.json",
    "artifacts/dart_replay/*/panel_manifest.json",
    "artifacts/dart_replay/*/results/**/*.json",
    "artifacts/dart_replay/*/a3_metric_contradiction.json",
    "artifacts/dart_value_select/*/replay/**/*.json",
)
_DART_SELECTION = (
    "artifacts/dart_value_select/*/tables/dart_selection_methods.csv",
    "artifacts/dart_value_select/*/manifests/dart_statistical_selection_manifest.json",
    "artifacts/dart_value_select/*/selection_methods.csv",
    "artifacts/dart_value_select/*/selection_result.json",
    "artifacts/dart_value_select/*/statistical_selection.json",
)


TABLE_INPUT_REGISTRY: tuple[TableDefinition, ...] = (
    _table(
        "nlr_twin_validation.csv",
        "direct",
        ("artifacts/research/physical/run_id=*/nlr_physical_validation_metrics.parquet",),
        (("model",), ("split",), ("facility_rmse_kw", "facility_mae_kw")),
        boundary=NLR_BOUNDARY,
    ),
    _table(
        "nlr_pue_validation.csv",
        "direct",
        ("artifacts/research/physical/run_id=*/pue_validation.parquet",),
        (("model",), ("split",), ("pue_mae",)),
        boundary=NLR_BOUNDARY,
    ),
    _table(
        "nlr_parameter_uncertainty.csv",
        "direct",
        ("artifacts/research/physical/run_id=*/uncertainty_calibration.parquet",),
        (("method",), ("nominal_coverage",), ("locked_test_interval_coverage",)),
        boundary=NLR_BOUNDARY,
    ),
    _table(
        "energy_coverage_by_partition.csv",
        "direct",
        ("artifacts/research/physical/run_id=*/nlr295_energy_coverage.parquet",),
        (("split",), ("dimension",), ("job_rows",)),
        boundary=NLR_BOUNDARY,
    ),
    _table(
        "pue_component_closure.csv",
        "direct",
        ("artifacts/research/physical/run_id=*/component_closure.parquet",),
        (("row_count",), ("closure_mae_kw",)),
        boundary=NLR_BOUNDARY,
    ),
    _table(
        "nyiso_da_forecast_main.csv",
        "forecast_main_da",
        _FORECAST_METRICS,
        (("task",), ("model",), ("split",), ("mae", "mean_pinball", "rmse")),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "nyiso_rt_forecast_main.csv",
        "forecast_main_rt",
        _FORECAST_METRICS,
        (("task",), ("model",), ("split",), ("mae", "mean_pinball", "rmse")),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "nyiso_da_rt_consistency.csv",
        "da_rt_consistency",
        _FORECAST_PREDICTIONS,
        (("zone", "aggregation"), ("paired_interval_count", "n"), ("da_rt_mae", "mae")),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "energy_forecast_by_zone.csv",
        "energy_by_zone",
        _FORECAST_PREDICTIONS,
        (("task",), ("zone",), ("model",), ("mae",)),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "energy_forecast_by_horizon.csv",
        "energy_by_horizon",
        _FORECAST_PREDICTIONS,
        (("task",), ("forecast_horizon_minutes", "horizon"), ("model",), ("mae",)),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "energy_forecast_spikes.csv",
        "energy_spikes",
        _FORECAST_METRICS + _FORECAST_PREDICTIONS,
        (("task",), ("model",), ("spike_f1", "spike_mae", "tail_mae", "tail_f1")),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "energy_forecast_ablation.csv",
        "energy_ablation",
        _FORECAST_METRICS,
        (("task",), ("model", "variant"), ("split",), ("mae", "mean_pinball", "rmse")),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "workload_forecast_main.csv",
        "forecast_main_workload",
        _FORECAST_METRICS,
        (("task",), ("model",), ("split",), ("mae", "joint_log_rmse", "rmse")),
        boundary=WORKLOAD_BOUNDARY,
    ),
    _table(
        "workload_forecast_by_cohort.csv",
        "workload_by_cohort",
        _FORECAST_PREDICTIONS,
        (("cohort", "cohort_index"), ("model",), ("mae", "rmse")),
        boundary=WORKLOAD_BOUNDARY,
    ),
    _table(
        "workload_forecast_ablation.csv",
        "workload_ablation",
        _FORECAST_METRICS,
        (("task",), ("model", "variant"), ("split",), ("mae", "joint_log_rmse", "rmse")),
        boundary=WORKLOAD_BOUNDARY,
    ),
    _table(
        "calibration_summary.csv",
        "calibration",
        _FORECAST_METRICS + _FORECAST_PREDICTIONS,
        (("task",), ("model",), ("split",), ("coverage_90", "empirical_coverage")),
        boundary=REFERENCE_BOUNDARY,
    ),
    _table(
        "digital_twin_transfer.csv",
        "direct",
        ("artifacts/research/physical/run_id=*/digital_twin_transfer_parameters.parquet",),
        (("site_id",), ("twin_id",), ("counterfactual",)),
        evidence_role="transferred_nyiso_twin",
        boundary=TRANSFER_BOUNDARY,
    ),
    _table(
        "xcel_local_counterfactual.csv",
        "xcel",
        (
            "artifacts/tables/xcel_local_tariff_replay.csv",
            "artifacts/research/**/xcel_local_tariff_replay.csv",
            "artifacts/manifests/xcel_counterfactual_manifest.json",
        ),
        (("month",), ("schedule",), ("source_coverage_fraction",), ("event_calls",)),
        evidence_role="xcel_counterfactual",
        relationship="tariff_defined_counterfactual",
        boundary=XCEL_BOUNDARY,
    ),
    _table(
        "evidence_layer_audit.csv",
        "inventory",
        (),
        (("artifact_id",), ("status",), ("evidence_role",), ("claim_boundary",)),
    ),
    _table(
        "scenario_diagnostics.csv",
        "scenario_diagnostics",
        (
            "artifacts/optimization_branch/*/scenario_diagnostics.json",
            "artifacts/research/optimization/*/scenario_diagnostics.json",
        ),
        (("operating_date",), ("regime",), ("reduced_count",), ("energy_score", "reduction_mse")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "scenario_dependence.csv",
        "scenario_dependence",
        (
            "artifacts/optimization_branch/*/scenario_diagnostics.json",
            "artifacts/research/optimization/*/scenario_diagnostics.json",
        ),
        (("regime",), ("mean_pairwise_rank_correlation",), ("all_four_upper_quintile_probability",)),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "scenario_reduction.csv",
        "scenario_reduction",
        (
            "artifacts/optimization_branch/*/scenario_diagnostics.json",
            "artifacts/optimization_branch/*/scenario_decision_value_subset.json",
            "artifacts/research/optimization/*/scenario_diagnostics.json",
            "artifacts/research/optimization/*/scenario_decision_value_subset.json",
        ),
        (("initial_count",), ("reduced_count",), ("reduction_mse", "decision_value_gap")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "extreme_cooccurrence.csv",
        "extreme_cooccurrence",
        (
            "artifacts/optimization_branch/*/scenario_diagnostics.json",
            "artifacts/research/optimization/*/scenario_diagnostics.json",
        ),
        (("regime",), ("full_cooccurrence_rate",), ("reduced_cooccurrence_rate",)),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "counterfactual_optimization_main.csv",
        "optimization_main",
        _OPT_RESULTS,
        (("policy_id", "policy", "variant"), ("expected_total_cost_usd", "expected_cost"), ("counterfactual_cost_difference_vs_baseline",), ("counterfactual",)),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "optimization_cost_breakdown.csv",
        "optimization_cost",
        _OPT_RESULTS,
        (("policy_id", "policy", "variant"), ("cost_component", "expected_total_cost_usd", "expected_cost")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "optimization_service.csv",
        "optimization_service",
        _OPT_RESULTS,
        (("policy_id", "policy", "variant"), ("expected_sla_shortfall_work_units", "sla_shortfall")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "optimization_grid_facing_proxies.csv",
        "optimization_grid",
        _OPT_RESULTS,
        (("policy_id", "policy", "variant"), ("grid_energy_mwh", "expected_grid_energy_mwh", "peak_import_mw", "rebound_peak_mw")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "risk_frontier.csv",
        "risk_frontier",
        _OPT_RESULTS + _OPT_SENSITIVITY,
        (("risk_weight", "risk_aversion", "cvar_weight"), ("expected_total_cost_usd", "expected_cost"), ("cvar_total_cost_usd", "cvar")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "dependence_stress.csv",
        "dependence_stress",
        _OPT_RESULTS,
        (("regime", "dependence_regime"), ("expected_total_cost_usd", "expected_cost"), ("cvar_total_cost_usd", "cvar")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "optimization_sensitivity.csv",
        "optimization_sensitivity",
        _OPT_SENSITIVITY,
        (("sensitivity_dimension",), ("variant_value", "setting_value", "clock_interpretation"), ("expected_total_cost_usd", "expected_cost")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "solver_performance.csv",
        "solver",
        _OPT_RESULTS,
        (("termination_condition",), ("solve_time_seconds", "solver_time_seconds"), ("number_of_variables", "variable_count")),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "failures.csv",
        "failures",
        (
            "artifacts/optimization_branch/*/failures.csv",
            "artifacts/optimization_branch/*/cell_status.json",
            "artifacts/optimization_branch/*/optimization_results.csv",
            "artifacts/optimization_branch/*/summary.json",
            "artifacts/research/optimization/*/failures.csv",
            "artifacts/research/optimization/*/cell_status.json",
            "artifacts/research/optimization/*/optimization_results.csv",
            "artifacts/research/optimization/*/summary.json",
        ),
        (("event_type",), ("event_count",), ("status",)),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "hypothesis_tests.csv",
        "hypotheses",
        (
            "artifacts/optimization_branch/*/h0_h6_aggregation.json",
            "artifacts/optimization_branch/*/statistical_inference.json",
            "artifacts/optimization_branch/*/paired_policy_inference.csv",
            "artifacts/optimization_branch/*/sensitivity_effects.csv",
            "artifacts/optimization_branch/*/forecast_dm_tests.csv",
            "artifacts/optimization_branch/*/forecast_decision_linkage.csv",
            "artifacts/research/optimization/*/h0_h6_aggregation.json",
            "artifacts/research/optimization/*/statistical_inference.json",
            "artifacts/research/optimization/*/paired_policy_inference.csv",
            "artifacts/research/optimization/*/sensitivity_effects.csv",
            "artifacts/research/optimization/*/forecast_dm_tests.csv",
            "artifacts/research/optimization/*/forecast_decision_linkage.csv",
        ),
        (("hypothesis_id",), ("status",), ("statistical_inference_consumed", "evidence_complete")),
        evidence_role="derived",
        relationship="derived_from_counterfactual_optimization_and_frozen_upstream_metrics",
        boundary=COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "dart_probability_quality.csv",
        "dart_probability_quality",
        _DART_PROBABILITY,
        (
            ("candidate_id", "model"),
            ("evaluation_window", "split"),
            ("crps",),
            ("tail_crps",),
            ("awcrps",),
            ("action_brier",),
            ("forl",),
            ("sample_count", "n"),
        ),
        boundary=NYISO_BOUNDARY,
    ),
    _table(
        "dart_decision_value.csv",
        "dart_decision_value",
        _DART_REPLAY,
        (
            ("candidate_id", "model"),
            ("experiment_id", "regime"),
            ("incremental_operational_value_usd",),
            ("prescriptive_regret_usd", "regret_usd"),
            ("profit_capture_ratio",),
            ("pcr_numerator_operational_value_usd",),
            ("pcr_denominator_perfect_information_value_usd",),
            ("counterfactual",),
        ),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=DART_COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "dart_selection_methods.csv",
        "dart_selection_methods",
        _DART_SELECTION,
        (
            ("selection_method", "method"),
            ("candidate_id", "selected_model"),
            ("selection_oos_years",),
            ("locked_test_year",),
            ("test_used_for_selection",),
            ("lcb_incremental_operational_value_usd", "lcb_7_day"),
            ("mcs_member",),
        ),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=DART_COUNTERFACTUAL_BOUNDARY,
    ),
    _table(
        "dart_regime_value.csv",
        "dart_regime_value",
        _DART_REPLAY,
        (
            ("experiment_id", "regime"),
            ("candidate_id", "model"),
            (
                "mean_incremental_operational_value_usd",
                "incremental_operational_value_usd",
            ),
            ("source_cell_count", "sample_count", "n"),
            ("counterfactual",),
        ),
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        boundary=DART_COUNTERFACTUAL_BOUNDARY,
    ),
)


TABLE_DEFINITION_BY_NAME = {definition.name: definition for definition in TABLE_INPUT_REGISTRY}


class PaperArtifactInputError(ValueError):
    pass


@dataclass
class LoadedSource:
    path: Path
    sha256: str
    bytes: int
    row_count: int
    frame: pd.DataFrame | None
    payload: Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_jsonable(item) for item in value]
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _jsonable(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise PaperArtifactInputError(f"cannot write empty table {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "synthetic"}


def _frame_is_synthetic(frame: pd.DataFrame) -> bool:
    for column in ("synthetic_fixture", "demo", "prototype_only"):
        if column in frame and frame[column].map(_truthy).any():
            return True
    if "synthetic" in frame and frame["synthetic"].map(_truthy).any():
        # A transfer table may truthfully contain a false-valued synthetic field.
        return True
    for column in ("status", "evidence_role", "source_record_type"):
        if column in frame:
            values = frame[column].astype(str).str.lower()
            if values.str.contains("demo|prototype_only|synthetic_fallback|pending_real_experiment", regex=True).any():
                return True
    return False


def _payload_is_incomplete(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("synthetic_fallback") is True or payload.get("synthetic_fallback_used") is True:
        return True
    status = str(payload.get("status", "")).strip().lower()
    return status in {"failed", "failure", "partial", "pending", "in_progress", "blocked"}


def _payload_is_demo_or_synthetic(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        if any(
            payload.get(key) is True
            for key in (
                "demo",
                "synthetic",
                "synthetic_fixture",
                "synthetic_fallback",
                "synthetic_fallback_used",
                "prototype_only",
            )
        ):
            return True
        if str(payload.get("profile", "")).strip().lower() in {"demo", "synthetic"}:
            return True
        return any(_payload_is_demo_or_synthetic(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_payload_is_demo_or_synthetic(value) for value in payload)
    return False


def _verify_dart_payload_hashes(payload: Any, path: Path) -> None:
    if not isinstance(payload, Mapping):
        return
    for hash_key, value_key in (
        ("output_sha256", "result"),
        ("results_sha256", "results"),
        ("rows_sha256", "rows"),
    ):
        declared = payload.get(hash_key)
        if declared is None:
            continue
        if value_key not in payload or str(declared).lower() != _json_sha256(payload[value_key]):
            raise PaperArtifactInputError(
                f"DART embedded {hash_key} does not match {value_key} in {path}"
            )


def _dart_final_manifest_for(path: Path) -> Path | None:
    for parent in path.resolve().parents:
        candidate = parent / "manifests" / "dart_value_select_manifest.json"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_dart_manifest_output(manifest: Path, value: Any) -> Path:
    output = Path(str(value))
    if output.is_absolute():
        return output.resolve()
    run_root = manifest.parent.parent
    from_run = (run_root / output).resolve()
    if from_run.is_file():
        return from_run
    return (manifest.parents[3] / output).resolve()


def _verify_formal_dart_output(path: Path) -> dict[str, Any]:
    manifest_path = _dart_final_manifest_for(path)
    if manifest_path is None:
        raise PaperArtifactInputError(
            f"formal DART source lacks a sibling dart_value_select_manifest.json: {path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PaperArtifactInputError(
            f"unreadable DART final manifest: {type(exc).__name__}"
        ) from exc
    chronology = manifest.get("chronology", {}) if isinstance(manifest, Mapping) else {}
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("status") != "COMPLETE"
        or not isinstance(chronology, Mapping)
        or chronology.get("selection_years") != [2022, 2023, 2024]
        or chronology.get("locked_test_year") != 2025
        or chronology.get("selection_frozen_before_locked_test") is not True
        or chronology.get("selection_updated_from_2025") is not False
        or manifest.get("realized_commercial_profit_claim") is not False
    ):
        raise PaperArtifactInputError("DART final manifest fails chronology/claim gates")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise PaperArtifactInputError("DART final manifest has no hashed outputs")
    target = path.resolve()
    matched = False
    required_names = {
        "dart_probability_quality.csv",
        "dart_decision_value.csv",
        "dart_selection_methods.csv",
        "dart_regime_value.csv",
        "dart_replay_manifest.json",
        "dart_statistical_selection_manifest.json",
    }
    seen: set[str] = set()
    for record in outputs:
        if not isinstance(record, Mapping):
            raise PaperArtifactInputError("DART final manifest has malformed output provenance")
        output = _resolve_dart_manifest_output(manifest_path, record.get("path"))
        expected = str(record.get("sha256", "")).lower()
        if len(expected) != 64 or not output.is_file() or _sha256(output) != expected:
            raise PaperArtifactInputError(
                f"DART final-manifest output hash changed: {record.get('path')}"
            )
        seen.add(output.name)
        matched = matched or output == target
    if not required_names.issubset(seen):
        raise PaperArtifactInputError("DART final manifest omits required outputs")
    if not matched:
        raise PaperArtifactInputError("DART source is not hash-pinned by its final manifest")
    return {
        "path": manifest_path.as_posix(),
        "sha256": _sha256(manifest_path),
        "experiment_fingerprint": manifest.get("experiment_fingerprint"),
        "config_sha256": manifest.get("config_sha256"),
    }


def _load_source(
    root: Path,
    path: Path,
    *,
    expected_sha256: str | None = None,
    allow_test_fixtures: bool = False,
) -> LoadedSource:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise PaperArtifactInputError(f"source is missing or empty: {path}")
    digest = _sha256(path)
    if expected_sha256 and digest.lower() != str(expected_sha256).strip().lower():
        raise PaperArtifactInputError(f"declared SHA-256 mismatch for {path}")
    hash_sidecar = Path(str(path) + ".sha256")
    if hash_sidecar.is_file():
        declared = hash_sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
        if declared != digest:
            raise PaperArtifactInputError(f"hash sidecar does not match {path}")
    lowered_path = path.as_posix().lower()
    if not allow_test_fixtures and any(
        marker in lowered_path
        for marker in ("/artifacts/demo/", "/demo/", "/fixture/", "/fixtures/")
    ):
        raise PaperArtifactInputError(f"demo/fixture source is not formal evidence: {path}")

    frame: pd.DataFrame | None = None
    payload: Any = None
    row_count = 0
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        row_count = int(len(frame))
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
        row_count = int(len(frame))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            row_count = len(payload)
        elif isinstance(payload, Mapping):
            row_count = 1
        else:
            raise PaperArtifactInputError(f"unsupported JSON root in {path}")
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            payload = {key: np.asarray(bundle[key]) for key in bundle.files}
        row_count = max((int(value.shape[0]) for value in payload.values() if value.ndim), default=0)
    else:
        raise PaperArtifactInputError(f"unsupported table source type: {path.suffix}")
    if frame is not None:
        if frame.empty:
            raise PaperArtifactInputError(f"source table has no rows: {path}")
        if _frame_is_synthetic(frame) and not allow_test_fixtures:
            raise PaperArtifactInputError(f"synthetic/demo rows cannot satisfy paper evidence: {path}")
    if _payload_is_incomplete(payload):
        raise PaperArtifactInputError(f"source has incomplete machine status: {path}")
    if "dart" in lowered_path:
        if _payload_is_demo_or_synthetic(payload) and not allow_test_fixtures:
            raise PaperArtifactInputError(
                f"demo/synthetic DART output cannot satisfy paper evidence: {path}"
            )
        _verify_dart_payload_hashes(payload, path)
        if not allow_test_fixtures:
            _verify_formal_dart_output(path)
    return LoadedSource(path, digest, path.stat().st_size, row_count, frame, payload)


def _expand(root: Path, patterns: Iterable[str]) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        candidate = Path(str(pattern))
        if candidate.is_absolute() and not any(char in str(pattern) for char in "*?["):
            candidates = [candidate] if candidate.is_file() else []
        else:
            candidates = sorted(root.glob(str(pattern).replace("\\", "/")))
        for path in candidates:
            if not path.is_file():
                continue
            key = str(path.resolve()).lower()
            if key not in seen:
                output.append(path)
                seen.add(key)
    return output


def _mapping_for(branch_results: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for mapping_name in ("paper_table_inputs", "table_inputs", "artifact_mappings"):
        mappings = branch_results.get(mapping_name, {})
        if not isinstance(mappings, Mapping):
            continue
        for key in (name, Path(name).stem, f"artifacts/tables/{name}", f"table:{Path(name).stem}"):
            if key not in mappings:
                continue
            value = mappings[key]
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, (str, Path)):
                return {"source": str(value)}
            if isinstance(value, (list, tuple)):
                return {"sources": [str(item) for item in value]}
    return None


def _declared_paths(root: Path, value: Any) -> list[Path]:
    output: dict[str, Path] = {}
    seen: set[int] = set()

    def visit(nested: Any, key_hint: str = "") -> None:
        if isinstance(nested, Mapping):
            identity = id(nested)
            if identity in seen:
                return
            seen.add(identity)
            for key, child in nested.items():
                visit(child, str(key).lower())
        elif isinstance(nested, (list, tuple)):
            for child in nested:
                visit(child, key_hint)
        elif isinstance(nested, (str, Path)):
            candidate = Path(str(nested))
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_file() and (
                key_hint.endswith(("path", "file", "table", "manifest"))
                or candidate.suffix.lower() in {".json", ".csv", ".parquet", ".npz"}
            ):
                output[str(candidate.resolve()).lower()] = candidate

    visit(value)
    return list(output.values())


def _declared_path_relevant(definition: TableDefinition, path: Path) -> bool:
    normalized = path.as_posix().lower()
    writer = definition.writer
    if writer.startswith("forecast_") or writer in {
        "da_rt_consistency",
        "energy_by_zone",
        "energy_by_horizon",
        "energy_spikes",
        "energy_ablation",
        "workload_by_cohort",
        "workload_ablation",
        "calibration",
    }:
        return "real_forecasting" in normalized
    if writer.startswith("scenario_") or writer == "extreme_cooccurrence":
        return "scenario" in path.name.lower() and "optimization" in normalized
    if writer.startswith("optimization_") or writer in {
        "risk_frontier",
        "dependence_stress",
        "solver",
        "failures",
        "hypotheses",
    }:
        return "optimization" in normalized or path.name in {
            "h0_h6_aggregation.json",
            "statistical_inference.json",
            "paired_policy_inference.csv",
            "sensitivity_effects.csv",
            "forecast_dm_tests.csv",
            "forecast_decision_linkage.csv",
        }
    if writer == "xcel":
        return "xcel" in normalized
    if writer.startswith("dart_"):
        return "dart" in normalized
    if writer == "direct":
        return "physical" in normalized
    return False


def _resolve_sources(
    root: Path,
    definition: TableDefinition,
    branch_results: Mapping[str, Any],
) -> tuple[list[Path], Mapping[str, Any] | None]:
    explicit = _mapping_for(branch_results, definition.name)
    patterns: list[str] = []
    if explicit:
        value = explicit.get("sources", explicit.get("source", explicit.get("path")))
        if isinstance(value, (str, Path)):
            patterns.append(str(value))
        elif isinstance(value, (list, tuple)):
            patterns.extend(str(item) for item in value)
    else:
        patterns.extend(definition.source_patterns)
    target = (root / definition.relative_path).resolve()
    resolved = [path for path in _expand(root, patterns) if path.resolve() != target]
    if explicit is None:
        for path in _declared_paths(root, branch_results):
            if path.resolve() != target and _declared_path_relevant(definition, path):
                resolved.append(path)
    unique: dict[str, Path] = {}
    for path in resolved:
        unique[str(path.resolve()).lower()] = path
    return list(unique.values()), explicit


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, nested in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(nested, Mapping):
            output.update(_flatten(nested, name))
        elif isinstance(nested, (str, int, float, bool)) or nested is None:
            output[name] = nested
    return output


def _metric_aliases(values: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(values)
    for canonical, suffixes in {
        "mae": ("_mae",),
        "rmse": ("_rmse",),
        "mean_pinball": ("_mean_pinball", "_mean_weighted_pinball"),
        "coverage_90": ("_coverage_90",),
        "mean_width_90": ("_mean_width_90",),
    }.items():
        if canonical in output:
            continue
        match = next(
            (value for key, value in output.items() if any(str(key).endswith(suffix) for suffix in suffixes)),
            None,
        )
        if match is not None:
            output[canonical] = match
    return output


def _latest_sources(sources: Sequence[LoadedSource], suffix: str | None = None) -> list[LoadedSource]:
    eligible = [source for source in sources if suffix is None or source.path.suffix.lower() == suffix]
    if not eligible:
        return []
    # Explicit mappings are resolved first; otherwise the newest completed run
    # is the least surprising source when stale formal runs coexist locally.
    newest = max(eligible, key=lambda source: source.path.stat().st_mtime_ns)
    return [newest]


def _metric_rows_from_payload(payload: Any, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        return rows
    task_payload = payload.get("tasks", {}).get(task) if isinstance(payload.get("tasks"), Mapping) else None
    if isinstance(task_payload, Mapping):
        for split in ("validation", "locked_test"):
            values = task_payload.get(split)
            if isinstance(values, list):
                for row in values:
                    if isinstance(row, Mapping):
                        rows.append({"task": task, "split": split, **_metric_aliases(_flatten(row))})
        for model, model_payload in task_payload.items():
            if model in {"validation", "locked_test"} or not isinstance(model_payload, Mapping):
                continue
            for split in ("validation", "locked_test"):
                metrics = model_payload.get(split)
                if isinstance(metrics, Mapping):
                    rows.append({"task": task, "model": model, "split": split, **_metric_aliases(_flatten(metrics))})

    for key in ("deployable_baselines", "trials", "seed_results", "variants", "leaderboard"):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            declared_task = str(value.get("task", payload.get("task", ""))).lower()
            if declared_task and task not in declared_task and not (
                task == "nyiso_da" and "da_lbmp" in declared_task
            ) and not (task == "nyiso_rt" and "rt_lbmp" in declared_task):
                continue
            model = value.get("model", value.get("variant", value.get("model_class", key)))
            for split_key, split in (("validation_metrics", "validation"), ("locked_test_metrics", "locked_test")):
                metrics = value.get(split_key)
                if isinstance(metrics, Mapping):
                    rows.append(
                        {
                            "task": task,
                            "model": model,
                            "variant": value.get("variant"),
                            "seed": value.get("seed", value.get("selected_seed")),
                            "deployable": value.get("deployable"),
                            "split": split,
                            **_metric_aliases(_flatten(metrics)),
                        }
                    )
    # Per-seed extension metrics use top-level task/variant and metrics.{split}.
    declared = str(payload.get("task", "")).lower()
    matches_task = task in declared or (task == "nyiso_da" and "da_lbmp" in declared) or (
        task == "nyiso_rt" and "rt_lbmp" in declared
    ) or (task == "nlr_workload" and "workload" in declared)
    if matches_task and isinstance(payload.get("metrics"), Mapping):
        for split in ("validation", "locked_test"):
            metrics = payload["metrics"].get(split)
            if isinstance(metrics, Mapping):
                rows.append(
                    {
                        "task": task,
                        "model": payload.get("model", payload.get("variant", "advanced")),
                        "variant": payload.get("variant"),
                        "seed": payload.get("seed", payload.get("selected_seed")),
                        "deployable": payload.get("deployable"),
                        "split": split,
                        **_metric_aliases(_flatten(metrics)),
                    }
                )
    return rows


def _forecast_scope(source: LoadedSource) -> str:
    return (
        "confirmatory_h1_h2_extension"
        if "real_forecasting_extensions" in source.path.as_posix().lower()
        else "auxiliary_v1_not_for_h1_h2"
    )


def _forecast_main(
    sources: Sequence[LoadedSource],
    task: str,
    *,
    allow_auxiliary_only: bool = False,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        extracted = _metric_rows_from_payload(source.payload, task)
        if extracted:
            for row in extracted:
                row["evidence_scope"] = _forecast_scope(source)
            rows.extend(extracted)
            used.append(source)
    if not rows:
        raise PaperArtifactInputError(f"no locked forecast metrics found for {task}")
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["split"].astype(str).isin(["validation", "locked_test"])].copy()
    if frame.empty:
        raise PaperArtifactInputError(f"forecast metrics for {task} have no validation/test rows")
    confirmatory = frame["evidence_scope"].eq("confirmatory_h1_h2_extension")
    if not confirmatory.any() and not allow_auxiliary_only:
        raise PaperArtifactInputError(
            f"{task} main table requires completed real_forecasting_extensions evidence; v1 is auxiliary only"
        )
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame, used, {
        "task": task,
        "locked_test_not_used_for_selection": True,
        "confirmatory_row_count": int(confirmatory.sum()),
        "v1_role": "auxiliary_lineage_not_for_H1_H2",
    }


def _prediction_frame(source: LoadedSource, task: str | None = None) -> pd.DataFrame | None:
    if source.frame is None:
        return None
    frame = source.frame.copy()
    if task and "task" in frame:
        frame = frame.loc[frame["task"].astype(str).eq(task)]
    if frame.empty:
        return None
    return frame


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "observed_target",
        "task",
        "split",
        "zone",
        "timestamp_utc",
        "timestamp_local",
        "operating_date",
        "prediction_origin_utc",
        "feature_data_cutoff_utc",
        "latest_target_feature_timestamp_utc",
        "weather_feature_timestamp_utc",
        "forecast_horizon_minutes",
        "selected_advanced_seed",
    }
    candidates = []
    for column in frame.columns:
        if column in excluded or column.startswith("raw_q"):
            continue
        if column in {"q50", "baseline_persistence", "baseline_seasonal", "ridge_autoregression", "advanced_lightgbm", "predicted_target", "prediction"}:
            candidates.append(column)
    return candidates


def _error_metrics(target: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    paired = pd.DataFrame({"target": pd.to_numeric(target, errors="coerce"), "prediction": pd.to_numeric(prediction, errors="coerce")}).dropna()
    if paired.empty:
        return {"n": 0, "mae": None, "rmse": None, "mean_error": None}
    error = paired["prediction"] - paired["target"]
    return {
        "n": int(len(paired)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error": float(error.mean()),
    }


def _extension_energy_arrays(source: LoadedSource) -> tuple[str, np.ndarray, np.ndarray] | None:
    if not isinstance(source.payload, Mapping) or source.path.suffix.lower() != ".npz":
        return None
    normalized = source.path.as_posix().lower()
    if "real_forecasting_extensions" not in normalized:
        return None
    if "/dgpt_e_primary/" not in normalized:
        return None
    task = "nyiso_da" if "/da_lbmp/" in normalized else "nyiso_rt" if "/rt_lbmp/" in normalized else ""
    if not task or "locked_test_target" not in source.payload or "locked_test_prediction" not in source.payload:
        return None
    target = np.asarray(source.payload["locked_test_target"])
    prediction = np.asarray(source.payload["locked_test_prediction"])
    if prediction.shape[:-1] != target.shape or prediction.shape[-1] < 3 or target.ndim != 3:
        return None
    return task, target, prediction[..., prediction.shape[-1] // 2]


def _energy_grouped(
    sources: Sequence[LoadedSource],
    group: str,
    *,
    allow_auxiliary_only: bool = False,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        extension = _extension_energy_arrays(source)
        if extension is not None:
            task, target_array, prediction_array = extension
            if group == "zone":
                for zone_index in range(target_array.shape[1]):
                    metrics = _error_metrics(
                        pd.Series(target_array[:, zone_index, :].reshape(-1)),
                        pd.Series(prediction_array[:, zone_index, :].reshape(-1)),
                    )
                    rows.append(
                        {
                            "task": task,
                            "zone": f"zone_index_{zone_index}",
                            "zone_index": zone_index,
                            "model": "dgpt_e_primary",
                            "split": "locked_test",
                            "evidence_scope": "confirmatory_h1_h2_extension",
                            **metrics,
                        }
                    )
            else:
                for horizon_index in range(target_array.shape[2]):
                    metrics = _error_metrics(
                        pd.Series(target_array[:, :, horizon_index].reshape(-1)),
                        pd.Series(prediction_array[:, :, horizon_index].reshape(-1)),
                    )
                    rows.append(
                        {
                            "task": task,
                            "forecast_horizon_minutes": 15 * (horizon_index + 1),
                            "horizon_index": horizon_index,
                            "model": "dgpt_e_primary",
                            "split": "locked_test",
                            "evidence_scope": "confirmatory_h1_h2_extension",
                            **metrics,
                        }
                    )
            used.append(source)
            continue
        frame = _prediction_frame(source)
        if frame is None or "observed_target" not in frame or "task" not in frame:
            continue
        frame = frame.loc[frame["task"].astype(str).isin(["nyiso_da", "nyiso_rt"])]
        if "split" in frame:
            frame = frame.loc[frame["split"].astype(str).eq("locked_test")]
        grouping_column = "zone" if group == "zone" else "forecast_horizon_minutes"
        if grouping_column not in frame or frame.empty:
            continue
        models = _prediction_columns(frame)
        for keys, subset in frame.groupby(["task", grouping_column], dropna=False):
            task, group_value = keys
            for model in models:
                metrics = _error_metrics(subset["observed_target"], subset[model])
                if metrics["n"]:
                    rows.append({"task": task, grouping_column: group_value, "model": model, "split": "locked_test", "evidence_scope": _forecast_scope(source), **metrics})
        if rows:
            used.append(source)
    if not rows:
        raise PaperArtifactInputError(f"no locked-test energy prediction rows support grouping by {group}")
    frame = pd.DataFrame(rows)
    confirmatory = frame["evidence_scope"].eq("confirmatory_h1_h2_extension")
    if not confirmatory.any() and not allow_auxiliary_only:
        raise PaperArtifactInputError(
            f"energy-by-{group} table requires extension primary predictions; v1 is auxiliary only"
        )
    return frame, used, {
        "grouping": group,
        "negative_prices_preserved": True,
        "confirmatory_row_count": int(confirmatory.sum()),
        "v1_role": "auxiliary_lineage_not_for_H1_H2",
    }


def _da_rt_consistency(
    sources: Sequence[LoadedSource],
    *,
    allow_auxiliary_only: bool = False,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    extension_arrays: dict[str, list[tuple[np.ndarray, np.ndarray, LoadedSource]]] = {
        "nyiso_da": [],
        "nyiso_rt": [],
    }
    for source in sources:
        extension = _extension_energy_arrays(source)
        if extension is not None:
            task, target, prediction = extension
            extension_arrays[task].append((target, prediction, source))
    if all(extension_arrays.values()):
        rows: list[dict[str, Any]] = []
        used: list[LoadedSource] = []
        da_target = np.mean(np.stack([item[0] for item in extension_arrays["nyiso_da"]]), axis=0)
        da_prediction = np.mean(np.stack([item[1] for item in extension_arrays["nyiso_da"]]), axis=0)
        rt_target = np.mean(np.stack([item[0] for item in extension_arrays["nyiso_rt"]]), axis=0)
        rt_prediction = np.mean(np.stack([item[1] for item in extension_arrays["nyiso_rt"]]), axis=0)
        days = min(da_target.shape[0], rt_target.shape[0])
        zones = min(da_target.shape[1], rt_target.shape[1])
        horizons = min(da_target.shape[2], rt_target.shape[2])
        for zone_index in range(zones):
            da_pred = da_prediction[:days, zone_index, :horizons].reshape(-1)
            rt_pred = rt_prediction[:days, zone_index, :horizons].reshape(-1)
            da_obs = da_target[:days, zone_index, :horizons].reshape(-1)
            rt_obs = rt_target[:days, zone_index, :horizons].reshape(-1)
            rows.append(
                {
                    "zone": f"zone_index_{zone_index}",
                    "zone_index": zone_index,
                    "aggregation": "paired_extension_locked_test_intervals",
                    "paired_interval_count": int(len(da_pred)),
                    "da_rt_mae": float(np.mean(np.abs(da_pred - rt_pred))),
                    "da_rt_mean_difference": float(np.mean(da_pred - rt_pred)),
                    "observed_da_rt_mae": float(np.mean(np.abs(da_obs - rt_obs))),
                    "prediction_correlation": float(np.corrcoef(da_pred, rt_pred)[0, 1]),
                    "evidence_scope": "confirmatory_h1_h2_extension",
                }
            )
        used.extend(item[2] for values in extension_arrays.values() for item in values)
        return pd.DataFrame(rows), used, {
            "negative_prices_preserved": True,
            "paired_shape": [days, zones, horizons],
            "seed_aggregation": "mean_across_preregistered_extension_seeds",
        }
    frames: dict[str, tuple[pd.DataFrame, LoadedSource]] = {}
    for source in sources:
        frame = _prediction_frame(source)
        if frame is None or "task" not in frame or "observed_target" not in frame:
            continue
        tasks = set(frame["task"].astype(str))
        for task in ("nyiso_da", "nyiso_rt"):
            if task not in tasks:
                continue
            task_frame = frame.loc[frame["task"].astype(str).eq(task)].copy()
            if "split" in task_frame:
                task_frame = task_frame.loc[task_frame["split"].astype(str).eq("locked_test")]
            prediction_columns = _prediction_columns(task_frame)
            if task_frame.empty or not prediction_columns:
                continue
            preferred = "q50" if "q50" in prediction_columns else prediction_columns[-1]
            task_frame["prediction_value"] = pd.to_numeric(task_frame[preferred], errors="coerce")
            frames[task] = (task_frame, source)
    if set(frames) != {"nyiso_da", "nyiso_rt"}:
        raise PaperArtifactInputError("DA/RT consistency requires both locked-test prediction sources")
    da, da_source = frames["nyiso_da"]
    rt, rt_source = frames["nyiso_rt"]
    keys = [column for column in ("timestamp_utc", "zone") if column in da and column in rt]
    if not keys:
        raise PaperArtifactInputError("DA/RT prediction sources have no common timestamp/zone keys")
    paired = da[keys + ["prediction_value", "observed_target"]].rename(
        columns={"prediction_value": "da_prediction", "observed_target": "da_observed"}
    ).merge(
        rt[keys + ["prediction_value", "observed_target"]].rename(
            columns={"prediction_value": "rt_prediction", "observed_target": "rt_observed"}
        ),
        on=keys,
        how="inner",
    )
    if paired.empty:
        raise PaperArtifactInputError("DA/RT locked-test predictions have zero paired intervals")
    group_columns = ["zone"] if "zone" in paired else []
    groups = paired.groupby(group_columns, dropna=False) if group_columns else [("all", paired)]
    rows = []
    for key, subset in groups:
        diff = subset["da_prediction"] - subset["rt_prediction"]
        observed_diff = subset["da_observed"] - subset["rt_observed"]
        rows.append(
            {
                "zone": key if not isinstance(key, tuple) else key[0],
                "aggregation": "paired_locked_test_intervals",
                "paired_interval_count": int(len(subset)),
                "da_rt_mae": float(diff.abs().mean()),
                "da_rt_mean_difference": float(diff.mean()),
                "observed_da_rt_mae": float(observed_diff.abs().mean()),
                "prediction_correlation": float(subset[["da_prediction", "rt_prediction"]].corr().iloc[0, 1]),
            }
        )
    if not allow_auxiliary_only:
        raise PaperArtifactInputError(
            "DA/RT consistency requires extension primary predictions; v1 is auxiliary only"
        )
    for row in rows:
        row["evidence_scope"] = "auxiliary_v1_not_for_h1_h2"
    return pd.DataFrame(rows), [da_source, rt_source], {
        "negative_prices_preserved": True,
        "join_keys": keys,
        "v1_role": "auxiliary_lineage_not_for_H1_H2",
    }


def _energy_spikes(
    sources: Sequence[LoadedSource],
    *,
    allow_auxiliary_only: bool = False,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        source_used = False
        for task in ("nyiso_da", "nyiso_rt"):
            for row in _metric_rows_from_payload(source.payload, task):
                spike = {key: value for key, value in row.items() if "spike" in key or "tail" in key}
                if spike and row.get("split") == "locked_test":
                    rows.append({key: value for key, value in row.items() if key in {"task", "model", "variant", "seed", "split"}} | {"evidence_scope": _forecast_scope(source)} | spike)
                    source_used = True
        frame = _prediction_frame(source)
        if frame is None or not {"task", "observed_target"}.issubset(frame):
            if source_used:
                used.append(source)
            continue
        if "split" in frame:
            frame = frame.loc[frame["split"].astype(str).eq("locked_test")]
        for task, subset in frame.groupby("task"):
            if task not in {"nyiso_da", "nyiso_rt"} or subset.empty:
                continue
            target = pd.to_numeric(subset["observed_target"], errors="coerce")
            threshold = float(target.abs().quantile(0.95))
            tail = subset.loc[target.abs().ge(threshold)]
            for model in _prediction_columns(subset):
                metrics = _error_metrics(tail["observed_target"], tail[model])
                if metrics["n"]:
                    rows.append({"task": task, "model": model, "split": "locked_test", "evidence_scope": _forecast_scope(source), "spike_definition": "absolute_target_training_or_locked_p95", "spike_threshold": threshold, "spike_mae": metrics["mae"], "spike_rmse": metrics["rmse"], "n": metrics["n"]})
                    source_used = True
        if source_used:
            used.append(source)
    if not rows:
        raise PaperArtifactInputError("no spike/tail forecast evidence found")
    frame = pd.DataFrame(rows).drop_duplicates()
    confirmatory = frame["evidence_scope"].eq("confirmatory_h1_h2_extension")
    if not confirmatory.any() and not allow_auxiliary_only:
        raise PaperArtifactInputError("energy spike table requires extension evidence; v1 is auxiliary only")
    return frame, used, {"negative_prices_preserved": True, "confirmatory_row_count": int(confirmatory.sum())}


def _forecast_ablation(
    sources: Sequence[LoadedSource],
    task_family: str,
    *,
    allow_auxiliary_only: bool = False,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    tasks = ("nyiso_da", "nyiso_rt") if task_family == "energy" else ("nlr_workload",)
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        extracted: list[dict[str, Any]] = []
        for task in tasks:
            extracted.extend(_metric_rows_from_payload(source.payload, task))
        extracted = [row for row in extracted if row.get("split") in {"validation", "locked_test"}]
        if extracted:
            for row in extracted:
                row["evidence_scope"] = _forecast_scope(source)
            rows.extend(extracted)
            used.append(source)
    if not rows:
        raise PaperArtifactInputError(f"no {task_family} baseline/ablation metrics found")
    frame = pd.DataFrame(rows).drop_duplicates()
    confirmatory = frame["evidence_scope"].eq("confirmatory_h1_h2_extension")
    if not confirmatory.any() and not allow_auxiliary_only:
        raise PaperArtifactInputError(
            f"{task_family} ablation table requires real_forecasting_extensions evidence; v1 is auxiliary only"
        )
    return frame, used, {
        "same_locked_test_required": True,
        "family": task_family,
        "confirmatory_row_count": int(confirmatory.sum()),
        "v1_role": "auxiliary_lineage_not_for_H1_H2",
    }


def _workload_by_cohort(
    sources: Sequence[LoadedSource],
    *,
    allow_auxiliary_only: bool = False,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        frame = _prediction_frame(source, "nlr_workload")
        if frame is not None:
            cohort_column = next((column for column in ("cohort", "cohort_id", "cohort_index", "partition") if column in frame), None)
            if cohort_column and "observed_target" in frame:
                if "split" in frame:
                    frame = frame.loc[frame["split"].astype(str).eq("locked_test")]
                for cohort, subset in frame.groupby(cohort_column, dropna=False):
                    for model in _prediction_columns(frame):
                        metrics = _error_metrics(subset["observed_target"], subset[model])
                        if metrics["n"]:
                            rows.append({"cohort": cohort, "model": model, "split": "locked_test", "evidence_scope": _forecast_scope(source), **metrics})
                if rows:
                    used.append(source)
        if not isinstance(source.payload, Mapping) or source.path.suffix.lower() != ".npz":
            continue
        target_key = next((key for key in source.payload if "locked_test" in key and "target" in key and "count" in key), None)
        prediction_keys = [key for key in source.payload if "locked_test" in key and "prediction" in key and "count" in key]
        if not target_key or not prediction_keys:
            continue
        target = np.asarray(source.payload[target_key])
        if target.ndim < 2:
            continue
        cohort_count = target.shape[-1]
        for prediction_key in prediction_keys:
            prediction = np.asarray(source.payload[prediction_key])
            if prediction.shape != target.shape:
                continue
            for cohort in range(cohort_count):
                metrics = _error_metrics(pd.Series(target[..., cohort].reshape(-1)), pd.Series(prediction[..., cohort].reshape(-1)))
                rows.append({"cohort_index": cohort, "model": prediction_key, "split": "locked_test", "evidence_scope": _forecast_scope(source), **metrics})
        if rows:
            used.append(source)
    if not rows:
        raise PaperArtifactInputError("workload cohort table requires real cohort-level predictions")
    frame = pd.DataFrame(rows)
    confirmatory = frame["evidence_scope"].eq("confirmatory_h1_h2_extension")
    if not confirmatory.any() and not allow_auxiliary_only:
        raise PaperArtifactInputError("workload cohort table requires extension predictions; v1 is auxiliary only")
    return frame, used, {
        "cohort_labels": "source labels or zero-based cohort_index",
        "confirmatory_row_count": int(confirmatory.sum()),
    }


def _calibration(
    sources: Sequence[LoadedSource],
    *,
    allow_auxiliary_only: bool = False,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        extracted = []
        for task in ("nyiso_da", "nyiso_rt", "nlr_workload"):
            for row in _metric_rows_from_payload(source.payload, task):
                calibration = {key: value for key, value in row.items() if any(token in key for token in ("coverage", "width", "calibr", "quantile_crossing"))}
                if calibration:
                    extracted.append({key: value for key, value in row.items() if key in {"task", "model", "variant", "seed", "split"}} | {"evidence_scope": _forecast_scope(source)} | calibration)
        if extracted:
            rows.extend(extracted)
            used.append(source)
        frame = _prediction_frame(source)
        if frame is None or not {"task", "observed_target", "q05", "q95"}.issubset(frame):
            continue
        if "split" in frame:
            frame = frame.loc[frame["split"].astype(str).isin(["validation", "locked_test"])]
        for (task, split), subset in frame.groupby(["task", "split"], dropna=False):
            observed = pd.to_numeric(subset["observed_target"], errors="coerce")
            q05 = pd.to_numeric(subset["q05"], errors="coerce")
            q95 = pd.to_numeric(subset["q95"], errors="coerce")
            valid = observed.notna() & q05.notna() & q95.notna()
            if valid.any():
                rows.append({"task": task, "model": "quantile_model", "split": split, "evidence_scope": _forecast_scope(source), "nominal_coverage": 0.9, "coverage_90": float(((observed[valid] >= q05[valid]) & (observed[valid] <= q95[valid])).mean()), "mean_width_90": float((q95[valid] - q05[valid]).mean()), "n": int(valid.sum())})
                used.append(source)
    if not rows:
        raise PaperArtifactInputError("no calibration metrics or quantile predictions found")
    frame = pd.DataFrame(rows).drop_duplicates()
    confirmatory = frame["evidence_scope"].eq("confirmatory_h1_h2_extension")
    if not confirmatory.any() and not allow_auxiliary_only:
        raise PaperArtifactInputError("calibration summary requires extension evidence; v1 is auxiliary only")
    unique_used = list({source.path: source for source in used}.values())
    return frame, unique_used, {
        "locked_test_not_used_for_calibration": True,
        "confirmatory_row_count": int(confirmatory.sum()),
        "v1_role": "auxiliary_lineage_not_for_H1_H2",
    }


def _scenario_rows(sources: Sequence[LoadedSource]) -> tuple[list[dict[str, Any]], list[LoadedSource]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        payload = source.payload
        candidates = payload if isinstance(payload, list) else payload.get("rows", payload.get("diagnostics", [])) if isinstance(payload, Mapping) else []
        if isinstance(candidates, list):
            extracted = [_flatten(row) for row in candidates if isinstance(row, Mapping)]
            if extracted:
                rows.extend(extracted)
                used.append(source)
    if not rows:
        raise PaperArtifactInputError("scenario diagnostics JSON has no rows")
    return rows, used


def _scenario_table(sources: Sequence[LoadedSource], kind: str) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows, used = _scenario_rows(sources)
    if kind == "diagnostics":
        return pd.DataFrame(rows), used, {"view": "complete_flattened_diagnostics"}
    selected: list[dict[str, Any]] = []
    if kind == "dependence":
        tokens = ("regime", "kappa", "rank_correlation", "upper_quintile", "autocorrelation", "dependence")
        for row in rows:
            selected.append({key: value for key, value in row.items() if any(token in key for token in tokens) or key in {"operating_date", "seed"}})
    elif kind == "reduction":
        tokens = ("initial_count", "reduced_count", "reduction", "distance", "decision_value", "mapping", "regime", "operating_date", "seed")
        for row in rows:
            selected.append({key: value for key, value in row.items() if any(token in key for token in tokens)})
        for source in sources:
            if isinstance(source.payload, Mapping) and "decision" in source.path.name.lower():
                selected.append(_flatten(source.payload))
                if source not in used:
                    used.append(source)
    else:
        for row in rows:
            selected.append({key.replace("extreme_event_", ""): value for key, value in row.items() if key.startswith("extreme_event_") or key in {"regime", "operating_date", "seed"}})
    frame = pd.DataFrame(selected).dropna(axis=1, how="all")
    if frame.empty:
        raise PaperArtifactInputError(f"scenario diagnostics lack {kind} fields")
    return frame, used, {"view": kind}


def _optimization_frames(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource], list[Mapping[str, Any]]]:
    frames: list[pd.DataFrame] = []
    payload_rows: list[Mapping[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        if source.frame is not None:
            frames.append(source.frame.copy())
            used.append(source)
        elif isinstance(source.payload, Mapping):
            payload = source.payload
            if isinstance(payload.get("metrics"), Mapping):
                payload_rows.append({**_flatten(payload.get("metrics", {})), **{key: value for key, value in payload.items() if isinstance(value, (str, int, float, bool))}})
            elif any(key in payload for key in ("expected_total_cost_usd", "expected_cost", "objective", "termination_condition")):
                payload_rows.append(_flatten(payload))
            used.append(source)
    if payload_rows:
        frames.append(pd.DataFrame(payload_rows))
    frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if frame.empty:
        raise PaperArtifactInputError("optimization results contain no scalar rows")
    return frame, used, payload_rows


def _col(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _optimization_main(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    frame, used, _ = _optimization_frames(sources)
    policy = _col(frame, ("policy_id", "policy", "variant", "policy_name"))
    cost = _col(frame, ("expected_total_cost_usd", "expected_cost", "expected_cost_usd"))
    if not policy or not cost:
        raise PaperArtifactInputError("optimization main requires policy and expected cost fields")
    frame["policy_id"] = frame[policy].astype(str)
    frame["expected_total_cost_usd"] = pd.to_numeric(frame[cost], errors="coerce")
    group_keys = [column for column in ("operating_date", "seed", "regime", "phase") if column in frame]
    baseline_mask = frame["policy_id"].isin(["B0", "baseline", "Baseline"])
    if group_keys and baseline_mask.any():
        baselines = frame.loc[baseline_mask, group_keys + ["expected_total_cost_usd"]].rename(columns={"expected_total_cost_usd": "baseline_expected_cost_usd"})
        frame = frame.merge(baselines.drop_duplicates(group_keys), on=group_keys, how="left")
    else:
        baseline_value = frame.loc[baseline_mask, "expected_total_cost_usd"].mean() if baseline_mask.any() else frame["expected_total_cost_usd"].max()
        frame["baseline_expected_cost_usd"] = baseline_value
    frame["counterfactual_cost_difference_vs_baseline"] = frame["expected_total_cost_usd"] - frame["baseline_expected_cost_usd"]
    frame["counterfactual"] = True
    return frame, used, {"baseline_policy": "B0 when present; otherwise recorded highest-cost reference", "realized_profit_claimed": False}


def _optimization_cost(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    frame, used, _ = _optimization_frames(sources)
    id_columns = [column for column in ("operating_date", "seed", "phase", "policy_id", "policy", "variant", "regime") if column in frame]
    cost_columns = [column for column in frame.columns if ("cost" in column.lower() or "revenue" in column.lower()) and pd.api.types.is_numeric_dtype(pd.to_numeric(frame[column], errors="coerce"))]
    if not cost_columns:
        raise PaperArtifactInputError("optimization results expose no scalar cost components")
    melted = frame.melt(id_vars=id_columns, value_vars=cost_columns, var_name="cost_component", value_name="component_value_usd")
    melted = melted.dropna(subset=["component_value_usd"])
    if melted.empty:
        raise PaperArtifactInputError("optimization cost components are all missing")
    return melted, used, {"components": cost_columns, "scenario_weighting": "upstream solver probabilities or recorded expectations"}


def _optimization_view(sources: Sequence[LoadedSource], kind: str) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    frame, used, _ = _optimization_frames(sources)
    identifiers = [column for column in ("operating_date", "seed", "phase", "policy_id", "policy", "variant", "regime", "sensitivity_dimension", "variant_value") if column in frame]
    tokens = {
        "service": ("sla", "migrat", "checkpoint", "throughput", "curtail", "completed_work", "unfinished_work"),
        "grid": ("grid", "import", "rebound", "ramp", "concentration", "peak"),
        "solver": ("termination", "solver", "solve_time", "mip_gap", "variable", "constraint", "time_limit", "upper_bound"),
    }[kind]
    columns = identifiers + [column for column in frame.columns if any(token in column.lower() for token in tokens) and column not in identifiers]
    columns = list(dict.fromkeys(columns))
    if len(columns) == len(identifiers):
        raise PaperArtifactInputError(f"optimization results expose no {kind} metrics")
    output = frame.loc[:, columns].dropna(how="all", subset=[column for column in columns if column not in identifiers])
    if output.empty:
        raise PaperArtifactInputError(f"optimization {kind} metrics are empty")
    return output, used, {"view": kind}


def _filter_view(sources: Sequence[LoadedSource], kind: str) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    frame, used, _ = _optimization_frames(sources)
    if kind == "risk":
        risk = _col(frame, ("risk_weight", "risk_aversion", "cvar_weight", "risk_lambda"))
        expected = _col(frame, ("expected_total_cost_usd", "expected_cost"))
        cvar = _col(frame, ("cvar_total_cost_usd", "cvar", "cvar_cost_usd"))
        if not all((risk, expected, cvar)):
            raise PaperArtifactInputError("risk frontier requires risk weight, expected cost, and CVaR")
        columns = [column for column in ("operating_date", "seed", "policy_id", "policy", "variant", risk, expected, cvar, "alpha") if column and column in frame]
    else:
        regime = _col(frame, ("regime", "dependence_regime", "scenario_regime"))
        expected = _col(frame, ("expected_total_cost_usd", "expected_cost"))
        cvar = _col(frame, ("cvar_total_cost_usd", "cvar", "cvar_cost_usd"))
        if not all((regime, expected, cvar)):
            raise PaperArtifactInputError("dependence stress requires regime, expected cost, and CVaR")
        columns = [column for column in ("operating_date", "seed", "policy_id", "policy", "variant", regime, expected, cvar, "mixture_stress_mass", "kappa") if column and column in frame]
    return frame.loc[:, list(dict.fromkeys(columns))], used, {"view": kind}


def _sensitivity(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    used: list[LoadedSource] = []
    for source in sources:
        if source.frame is not None:
            frame = source.frame.copy()
            if "sensitivity_dimension" not in frame:
                frame["sensitivity_dimension"] = "ofat"
            frames.append(frame)
            used.append(source)
        elif isinstance(source.payload, Mapping) and "timezone" in source.path.name.lower():
            payload = _flatten(source.payload)
            row = {
                "sensitivity_dimension": "timezone_clock_interpretation",
                "variant_value": payload.get("clock_interpretation", payload.get("sensitivity_clock_interpretation", "paired_primary_vs_fixed_mst")),
                **payload,
            }
            frames.append(pd.DataFrame([row]))
            used.append(source)
    if not frames:
        raise PaperArtifactInputError("no completed OFAT/timezone sensitivity evidence found")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "expected_total_cost_usd" not in frame:
        alias = _col(frame, ("expected_cost", "primary_expected_cost_usd", "fixed_mst_expected_cost_usd"))
        if alias:
            frame["expected_total_cost_usd"] = frame[alias]
    if "variant_value" not in frame:
        frame["variant_value"] = frame.get("setting_value", frame.get("clock_interpretation"))
    return frame, used, {"timezone_paired_economic_included": bool(any("timezone" in source.path.name.lower() for source in used))}


def _failures(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluated = 0
    used: list[LoadedSource] = []
    for source in sources:
        if source.frame is not None:
            frame = source.frame.copy()
            evaluated += len(frame)
            if source.path.name == "failures.csv":
                for row in _records(frame):
                    rows.append({"event_type": "recorded_failure", "event_count": 1, "status": row.get("status", "FAILED"), **row})
            else:
                for row in _records(frame):
                    termination = str(row.get("termination_condition", "")).lower()
                    gap = pd.to_numeric(pd.Series([row.get("mip_gap")]), errors="coerce").iloc[0]
                    if termination and termination not in {"optimal", "feasible", "success"}:
                        rows.append({"event_type": "nonoptimal_termination", "event_count": 1, "status": termination, **row})
                    if pd.notna(gap) and float(gap) > 0:
                        rows.append({"event_type": "nonzero_mip_gap", "event_count": 1, "status": "RECORDED", "mip_gap": float(gap), **{key: row.get(key) for key in ("operating_date", "policy_id", "seed")}})
            used.append(source)
        elif isinstance(source.payload, Mapping):
            payload = source.payload
            records = payload.get("cells", payload.get("records", payload.get("cell_status", [])))
            if isinstance(records, list):
                evaluated += len(records)
                for record in records:
                    if not isinstance(record, Mapping):
                        continue
                    status = str(record.get("current_status", record.get("status", ""))).upper()
                    if status in {"FAILED", "TIMEOUT"}:
                        rows.append({"event_type": "cell_failure" if status == "FAILED" else "timeout", "event_count": 1, "status": status, **_flatten(record)})
            used.append(source)
    if not used:
        raise PaperArtifactInputError("failure ledger requires cell/results/summary evidence")
    if not rows:
        rows.append({"event_type": "run_summary", "event_count": 0, "status": "NO_FAILURES_RECORDED", "evaluated_cell_or_result_count": evaluated})
    return pd.DataFrame(rows), used, {"evaluated_cell_or_result_count": evaluated, "zero_event_summary_is_not_a_placeholder": True}


def _hypotheses(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    aggregation: LoadedSource | None = None
    statistical: LoadedSource | None = None
    four_stats: dict[str, LoadedSource] = {}
    for source in sources:
        if source.path.name == "h0_h6_aggregation.json" and isinstance(source.payload, Mapping):
            aggregation = source
        elif source.path.name == "statistical_inference.json" and isinstance(source.payload, Mapping):
            statistical = source
        elif source.path.name in {"paired_policy_inference.csv", "sensitivity_effects.csv", "forecast_dm_tests.csv", "forecast_decision_linkage.csv"}:
            four_stats[source.path.name] = source
    if aggregation is None:
        raise PaperArtifactInputError("hypothesis_tests requires h0_h6_aggregation.json")
    hypotheses = aggregation.payload.get("hypotheses", {})
    rows: list[dict[str, Any]] = []
    if isinstance(hypotheses, Mapping):
        iterable = hypotheses.items()
    elif isinstance(hypotheses, list):
        iterable = ((str(row.get("hypothesis_id", "")), row) for row in hypotheses if isinstance(row, Mapping))
    else:
        iterable = ()
    for identifier, record in iterable:
        if not isinstance(record, Mapping):
            continue
        row = {"hypothesis_id": str(record.get("hypothesis_id", identifier)), **_flatten(record)}
        row["statistical_inference_consumed"] = bool(record.get("statistical_inference_consumed", statistical is not None))
        row["evidence_complete"] = bool(statistical is not None and len(four_stats) == 4)
        rows.append(row)
    expected = {f"H{index}" for index in range(7)}
    if {str(row.get("hypothesis_id")) for row in rows} != expected:
        raise PaperArtifactInputError("hypothesis aggregation must contain exactly H0-H6")
    if statistical is None or statistical.payload.get("completed") is not True or len(four_stats) != 4:
        raise PaperArtifactInputError("H0-H6 table requires COMPLETE inference manifest and all four hashed statistics CSVs")
    outputs = statistical.payload.get("outputs", {})
    for key, expected_name in (
        ("paired_policy_inference", "paired_policy_inference.csv"),
        ("sensitivity_effects", "sensitivity_effects.csv"),
        ("forecast_dm_tests", "forecast_dm_tests.csv"),
        ("forecast_decision_linkage", "forecast_decision_linkage.csv"),
    ):
        record = outputs.get(key, {}) if isinstance(outputs, Mapping) else {}
        declared_hash = str(record.get("sha256", "")).lower() if isinstance(record, Mapping) else ""
        if declared_hash and declared_hash != four_stats[expected_name].sha256:
            raise PaperArtifactInputError(f"statistical inference hash mismatch for {expected_name}")
    used = [aggregation, statistical, *four_stats.values()]
    return pd.DataFrame(rows), used, {"hypothesis_ids": sorted(expected), "statistics_outputs": {name: source.sha256 for name, source in four_stats.items()}, "negative_results_retained": True}


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    flattened = _flatten(record)
    for name in names:
        if name in record:
            return record[name]
        if name in flattened:
            return flattened[name]
    return None


def _finite_float(value: Any, *, field: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperArtifactInputError(f"DART field {field} is absent or nonnumeric") from exc
    if not math.isfinite(output):
        raise PaperArtifactInputError(f"DART field {field} is not finite")
    return output


def _false_value(value: Any) -> bool:
    return value is False or str(value).strip().lower() in {"false", "0", "no"}


def _dart_records(
    sources: Sequence[LoadedSource],
) -> list[tuple[dict[str, Any], LoadedSource]]:
    output: list[tuple[dict[str, Any], LoadedSource]] = []
    for source in sources:
        if source.frame is not None:
            output.extend((dict(row), source) for row in source.frame.to_dict(orient="records"))
            continue
        payload = source.payload
        if not isinstance(payload, Mapping):
            continue
        emitted = False
        result = payload.get("result")
        if isinstance(result, Mapping):
            output.append((dict(result), source))
            emitted = True
        for key in ("results", "rows", "selection_methods", "methods", "model_rows"):
            values = payload.get(key)
            if isinstance(values, list):
                context = {
                    str(name): value
                    for name, value in payload.items()
                    if name not in {key, "results", "rows", "selection_methods", "methods", "model_rows"}
                    and (
                        isinstance(value, (str, int, float, bool))
                        or value is None
                    )
                }
                output.extend(
                    ({**context, **dict(row)}, source)
                    for row in values
                    if isinstance(row, Mapping)
                )
                emitted = True
        summaries = payload.get("model_summaries")
        if isinstance(summaries, Mapping):
            output.extend(
                ({"candidate_id": str(model), **dict(summary)}, source)
                for model, summary in summaries.items()
                if isinstance(summary, Mapping)
            )
            emitted = True
        if not emitted:
            output.append((dict(payload), source))
    return output


def _dart_probability_quality(
    sources: Sequence[LoadedSource],
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: dict[str, LoadedSource] = {}
    for raw, source in _dart_records(sources):
        metrics = raw.get("probability_quality", raw.get("metrics", raw))
        if not isinstance(metrics, Mapping):
            continue
        combined = {**raw, **dict(metrics)}
        candidate = _first_value(combined, ("candidate_id", "model", "candidate"))
        evaluation = _first_value(
            combined, ("evaluation_window", "phase", "split", "evaluation_period")
        )
        names = {
            "crps": ("crps",),
            "tail_crps": ("tail_crps", "tail_weighted_crps"),
            "awcrps": ("awcrps", "action_weighted_crps"),
            "action_brier": (
                "action_brier",
                "action_brier_score",
                "economic_action_brier",
            ),
            "forl": ("forl", "forecast_operational_regret_loss"),
            "sample_count": (
                "sample_count",
                "source_record_count",
                "n",
                "count",
            ),
        }
        values = {
            name: _first_value(combined, aliases) for name, aliases in names.items()
        }
        future_price_flag = _first_value(
            combined,
            (
                "feasibility_weight_uses_future_price",
                "future_price_used_in_f",
                "awcrps_future_price_used",
            ),
        )
        test_used = _first_value(
            combined, ("test_used_for_selection", "locked_test_used_for_selection")
        )
        sign = _first_value(combined, ("dart_sign_convention", "dart_formula"))
        if candidate is None or evaluation is None or any(
            values[name] is None for name in ("crps", "tail_crps", "sample_count")
        ):
            continue
        for nullable_name, status_aliases in {
            "awcrps": ("awcrps_status",),
            "action_brier": ("economic_action_brier_status", "action_brier_status"),
            "forl": ("forl_status",),
        }.items():
            if values[nullable_name] is None and _first_value(
                combined, status_aliases
            ) is None:
                raise PaperArtifactInputError(
                    f"null {nullable_name} requires an explicit status"
                )
        formal_bundle = _dart_final_manifest_for(source.path) is not None
        if future_price_flag is None and formal_bundle:
            future_price_flag = False
        if test_used is None and formal_bundle:
            test_used = False
        if sign is None and formal_bundle:
            sign = "DA_MINUS_RT"
        if not _false_value(future_price_flag):
            raise PaperArtifactInputError(
                "AWCRPS feasibility weights must explicitly exclude future prices"
            )
        if not _false_value(test_used):
            raise PaperArtifactInputError(
                "DART probability source must state that locked test was not used for selection"
            )
        if str(sign).replace(" ", "").upper() not in {
            "DA_MINUS_RT",
            "DA-RT",
            "DA_LBMP-RT_LBMP",
        }:
            raise PaperArtifactInputError("DART probability source lacks canonical DA-minus-RT sign")
        row = {
            "candidate_id": str(candidate),
            "evaluation_window": str(evaluation),
            "dart_sign_convention": "DA_MINUS_RT",
            **{
                name: (
                    int(value)
                    if name == "sample_count"
                    else None
                    if value is None
                    else _finite_float(value, field=name)
                )
                for name, value in values.items()
            },
            "forl_definition": _first_value(
                combined, ("forl_definition", "forecast_operational_regret_definition")
            ),
            "source_denominator": _first_value(combined, ("denominator",)),
            "feasibility_weight_uses_future_price": False,
            "test_used_for_selection": False,
        }
        for field_name, aliases in {
            "mae": ("mae",),
            "rmse": ("rmse",),
            "pinball_loss": ("pinball_loss", "mean_pinball"),
            "emphasized_tail_threshold_weighted_crps": (
                "emphasized_tail_threshold_weighted_crps",
                "threshold_weighted_crps",
            ),
            "q05_q95_threshold_weighted_crps": (
                "q05_q95_threshold_weighted_crps",
            ),
            "q01_q99_extreme_threshold_weighted_crps": (
                "q01_q99_extreme_threshold_weighted_crps",
            ),
            "tail_calibration_error": ("tail_calibration_error",),
            "awcrps_status": ("awcrps_status",),
            "feasibility_conditioned_tail_crps": (
                "feasibility_conditioned_tail_crps",
            ),
            "feasibility_conditioned_tail_crps_status": (
                "feasibility_conditioned_tail_crps_status",
            ),
            "economic_action_brier_status": (
                "economic_action_brier_status",
                "action_brier_status",
            ),
            "action_precision": ("action_precision",),
            "action_recall": ("action_recall",),
            "action_f1": ("action_f1",),
            "wrong_action_rate": ("wrong_action_rate",),
            "missed_opportunity_rate": ("missed_opportunity_rate",),
            "wrong_action_economic_loss_usd": (
                "wrong_action_economic_loss_usd",
            ),
            "missed_opportunity_economic_loss_usd": (
                "missed_opportunity_economic_loss_usd",
            ),
            "q99_event_brier": ("q99_event_brier",),
            "q99_pr_auc": ("q99_pr_auc", "spike_pr_auc"),
            "q99_precision": ("q99_precision", "spike_precision"),
            "q99_recall": ("q99_recall", "spike_recall"),
            "q99_f1": ("q99_f1", "spike_f1"),
            "q99_miss_rate": ("q99_miss_rate",),
            "interval_coverage_80": ("interval_coverage_80",),
            "interval_width_80": ("interval_width_80",),
            "interval_calibration_error_80": (
                "interval_calibration_error_80",
            ),
            "interval_coverage_90": ("interval_coverage_90",),
            "interval_width_90": ("interval_width_90",),
            "interval_calibration_error_90": (
                "interval_calibration_error_90",
            ),
            "energy_score": ("energy_score",),
            "variogram_score": ("variogram_score",),
            "forl_status": ("forl_status",),
            "tail_threshold_contract_sha256": (
                "tail_threshold_contract_sha256",
            ),
            "normalized_feasibility_weight_sha256": (
                "normalized_feasibility_weight_sha256",
            ),
            "normalized_feasibility_weight_sum": (
                "normalized_feasibility_weight_sum",
            ),
            "normalized_feasibility_weight_min": (
                "normalized_feasibility_weight_min",
            ),
            "normalized_feasibility_weight_max": (
                "normalized_feasibility_weight_max",
            ),
            "physical_flexible_mwh_sha256": ("physical_flexible_mwh_sha256",),
            "physical_flexible_mwh_total": ("physical_flexible_mwh_total",),
            "source_records_sha256": ("source_records_sha256",),
        }.items():
            optional = _first_value(combined, aliases)
            if optional is not None:
                row[field_name] = optional
        scalar_passthrough_prefixes = (
            "q99_",
            "q05_q95_",
            "q01_q99_",
            "threshold_weighted_",
            "emphasized_tail_",
            "feasibility_conditioned_",
            "wrong_action_",
            "missed_opportunity_",
            "interval_",
            "tail_threshold_contract_",
            "normalized_feasibility_weight_",
            "physical_flexible_mwh_",
            "metric_contract_",
        )
        for field_name, value in combined.items():
            if field_name.startswith(scalar_passthrough_prefixes) and not isinstance(
                value, (Mapping, list, tuple, set)
            ):
                row.setdefault(field_name, value)
        rows.append(row)
        used[str(source.path)] = source
    if not rows:
        raise PaperArtifactInputError(
            "no complete real DART probability rows contain CRPS, Tail-CRPS, AWCRPS, action Brier, and FORL"
        )
    return pd.DataFrame(rows).drop_duplicates(), list(used.values()), {
        "dart_sign_convention": "DA_MINUS_RT",
        "awcrps_denominator": "sum_candidate_independent_normalized_gate_time_F",
        "physical_flexible_energy_role": "separate_MWh_scale_not_the_AWCRPS_weight",
        "awcrps_future_realized_price_used": False,
        "locked_test_used_for_selection": False,
        "denominator_contract": {
            "awcrps": "sum(F_issue_time_normalized_to_0_1)",
            "action_brier": "number_of_scored_action_events",
            "forl": "source_defined_and_preserved_verbatim",
        },
    }


def _normalize_decision_row(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    record = raw.get("result", raw)
    if not isinstance(record, Mapping):
        return None
    decision = record.get("decision_value", record)
    if not isinstance(decision, Mapping):
        return None
    candidate = _first_value(record, ("candidate_id", "model", "candidate"))
    experiment = _first_value(record, ("experiment_id", "regime"))
    if candidate is None or experiment is None:
        return None
    candidate_cost = _first_value(
        decision,
        ("candidate_simulated_total_cost_usd", "candidate_cost_usd"),
    )
    baseline_cost = _first_value(
        decision,
        ("baseline_simulated_total_cost_usd", "baseline_cost_usd"),
    )
    pi_cost = _first_value(
        decision,
        ("perfect_information_cost_lower_bound_usd", "pi_cost_usd"),
    )
    incremental = _first_value(
        decision,
        (
            "incremental_operational_value_usd",
            "pcr_numerator_operational_value_usd",
            "cumulative_incremental_operational_value",
            "mean_daily_incremental_operational_value",
        ),
    )
    denominator = _first_value(
        decision, ("pcr_denominator_perfect_information_value_usd",)
    )
    exported_summary = (
        candidate_cost is None
        and baseline_cost is None
        and pi_cost is None
        and _first_value(decision, ("cumulative_incremental_operational_value",))
        is not None
    )
    if candidate_cost is not None and baseline_cost is not None and pi_cost is not None:
        candidate_cost_value = _finite_float(candidate_cost, field="candidate cost")
        baseline_cost_value = _finite_float(baseline_cost, field="baseline cost")
        pi_cost_value = _finite_float(pi_cost, field="perfect-information cost")
        expected_numerator = baseline_cost_value - candidate_cost_value
        expected_denominator = baseline_cost_value - pi_cost_value
    elif incremental is not None and denominator is not None:
        candidate_cost_value = baseline_cost_value = pi_cost_value = None
        expected_numerator = _finite_float(incremental, field="PCR numerator")
        expected_denominator = _finite_float(denominator, field="PCR denominator")
    elif exported_summary:
        candidate_cost_value = baseline_cost_value = pi_cost_value = None
        expected_numerator = _finite_float(
            _first_value(decision, ("cumulative_incremental_operational_value",)),
            field="exported cumulative PCR numerator",
        )
        exported_pcr = _finite_float(
            _first_value(decision, ("profit_capture_ratio", "pcr")),
            field="exported PCR",
        )
        if math.isclose(exported_pcr, 0.0, abs_tol=1e-15):
            return None
        expected_denominator = expected_numerator / exported_pcr
    else:
        return None
    if expected_denominator <= 0.0:
        raise PaperArtifactInputError(
            "PCR denominator must be positive perfect-information value over baseline"
        )
    declared_incremental = _finite_float(
        incremental if incremental is not None else expected_numerator,
        field="incremental operational value",
    )
    if not math.isclose(declared_incremental, expected_numerator, rel_tol=1e-9, abs_tol=1e-8):
        raise PaperArtifactInputError("DART PCR numerator disagrees with model-minus-baseline value")
    declared_pcr = _finite_float(
        _first_value(decision, ("profit_capture_ratio", "pcr")), field="PCR"
    )
    expected_pcr = expected_numerator / expected_denominator
    if not math.isclose(declared_pcr, expected_pcr, rel_tol=1e-9, abs_tol=1e-8):
        raise PaperArtifactInputError(
            "DART PCR must equal (model value - baseline value) / (PI value - baseline value)"
        )
    counterfactual = _first_value(decision, ("counterfactual",))
    if not _truthy(counterfactual):
        raise PaperArtifactInputError("DART decision value must be labeled counterfactual")
    profit_claim = _first_value(decision, ("realized_commercial_profit_claim",))
    if profit_claim is not None and not _false_value(profit_claim):
        raise PaperArtifactInputError("commercial-profit claims cannot source DART paper tables")
    regret = _first_value(
        decision,
        ("prescriptive_regret_usd", "regret_usd", "mean_prescriptive_regret"),
    )
    if regret is None:
        return None
    row = {
        "operating_date": _first_value(record, ("operating_date", "date")),
        "candidate_id": str(candidate),
        "experiment_id": str(experiment),
        "seed": _first_value(record, ("seed",)),
        "candidate_simulated_total_cost_usd": candidate_cost_value,
        "baseline_simulated_total_cost_usd": baseline_cost_value,
        "perfect_information_cost_lower_bound_usd": pi_cost_value,
        "counterfactual_realized_operational_value_usd": (
            -candidate_cost_value if candidate_cost_value is not None else None
        ),
        "incremental_operational_value_usd": _finite_float(
            _first_value(
                decision,
                (
                    "mean_daily_incremental_operational_value",
                    "incremental_operational_value_usd",
                    "cumulative_incremental_operational_value",
                ),
            ),
            field="incremental operational value",
        ),
        "prescriptive_regret_usd": _finite_float(regret, field="prescriptive regret"),
        "profit_capture_ratio": expected_pcr,
        "pcr_numerator_operational_value_usd": expected_numerator,
        "pcr_denominator_perfect_information_value_usd": expected_denominator,
        "pcr_formula_verified": True,
        "counterfactual": True,
        "realized_commercial_profit_claim": False,
        "perfect_information_deployable": False,
        "source_denominator": _first_value(decision, ("denominator",)),
        "pcr_denominator_recovered_from_exported_ratio": bool(exported_summary),
    }
    candidate_result = record.get("candidate_result")
    if isinstance(candidate_result, Mapping):
        flattened = _flatten(candidate_result)
        for name in (
            "cvar95_total_cost_usd",
            "cvar_total_cost_usd",
            "downside_deviation_usd",
            "probability_below_baseline",
            "worst_5pct_mean_daily_cost_usd",
            "sla_violation_rate",
            "inference_time_seconds",
        ):
            if name in flattened:
                row[name] = flattened[name]
    for field_name, aliases in {
        "cumulative_incremental_operational_value_usd": (
            "cumulative_incremental_operational_value",
        ),
        "equivalent_operating_cost_reduction_usd": (
            "equivalent_operating_cost_reduction",
        ),
        "operating_cost_reduction_rate": ("operating_cost_reduction_rate",),
        "cvar95_loss_usd": ("cvar95_loss", "cvar_loss"),
        "maximum_drawdown_usd": ("maximum_drawdown",),
        "probability_below_baseline": ("probability_below_baseline",),
        "weighted_sla_satisfaction": ("weighted_sla_satisfaction",),
        "deadline_violations": ("deadline_violations",),
        "infeasible_days": ("infeasible_days",),
        "invalid_days": ("invalid_days",),
        "forecast_plus_solve_latency_seconds": (
            "forecast_plus_solve_latency_seconds",
        ),
        "source_records_sha256": ("source_records_sha256",),
    }.items():
        optional = _first_value(decision, aliases)
        if optional is not None:
            row[field_name] = optional
    return row


def _dart_decision_value(
    sources: Sequence[LoadedSource],
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: dict[str, LoadedSource] = {}
    for raw, source in _dart_records(sources):
        row = _normalize_decision_row(raw)
        if row is None:
            continue
        rows.append(row)
        used[str(source.path)] = source
    if not rows:
        raise PaperArtifactInputError("no complete DART replay cells support decision-value derivation")
    return pd.DataFrame(rows).drop_duplicates(), list(used.values()), {
        "value_semantics": "counterfactual_realized_price_replay_not_commercial_profit",
        "denominator_contract": {
            "pcr_numerator": "model_realized_operational_value_minus_baseline_realized_operational_value",
            "pcr_denominator": "perfect_information_value_minus_baseline_realized_operational_value",
        },
        "pcr_formula_verified_for_every_row": True,
        "perfect_information_deployable": False,
    }


def _selection_years(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        try:
            years = sorted(int(item) for item in value)
        except (TypeError, ValueError):
            return None
        return "-".join(str(year) for year in years)
    if value is None:
        return None
    return str(value)


def _dart_selection_methods(
    sources: Sequence[LoadedSource],
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: dict[str, LoadedSource] = {}
    for source in sources:
        if source.frame is not None:
            candidates = source.frame.to_dict(orient="records")
            payload: Mapping[str, Any] = {}
        elif isinstance(source.payload, Mapping):
            payload = source.payload
            locked_rows = payload.get(
                "locked_test_selector_rows",
                payload.get("locked_selector_rows"),
            )
            has_locked_rows = isinstance(locked_rows, list) and bool(locked_rows)
            if has_locked_rows:
                candidates = [
                    dict(row) for row in locked_rows if isinstance(row, Mapping)
                ]
            else:
                nested = payload.get("selection_results_by_experiment", {})
                primary: Mapping[str, Any] = {}
                if isinstance(nested, Mapping):
                    for key in (
                        "A2_final_with_A1_noninferiority",
                        "A2",
                        "A2_pre_A1_gate",
                    ):
                        value = nested.get(key)
                        if isinstance(value, Mapping):
                            primary = value
                            break
            if not has_locked_rows and isinstance(primary, Mapping) and isinstance(
                primary.get("lcb_by_model"), Mapping
            ):
                lcb = primary["lcb_by_model"]
                robustness = primary.get("robustness_lcb_by_block", {})
                feasible = set(map(str, primary.get("feasible_models", [])))
                mcs = set(map(str, primary.get("mcs_set", [])))
                selected = primary.get("selected_model")
                candidates = []
                for model in sorted(map(str, lcb)):
                    candidates.append(
                        {
                            "selection_method": "DART-ValueSelect",
                            "experiment_id": "A2",
                            "candidate_id": model,
                            "selected_model": selected,
                            "selected": model == str(selected),
                            "selection_status": primary.get("status"),
                            "lcb_incremental_operational_value_usd": lcb.get(model),
                            "lcb_7_day": (
                                robustness.get("7", {}).get(model)
                                if isinstance(robustness, Mapping)
                                and isinstance(robustness.get("7"), Mapping)
                                else lcb.get(model)
                            ),
                            "lcb_14_day": (
                                robustness.get("14", {}).get(model)
                                if isinstance(robustness, Mapping)
                                and isinstance(robustness.get("14"), Mapping)
                                else None
                            ),
                            "lcb_28_day": (
                                robustness.get("28", {}).get(model)
                                if isinstance(robustness, Mapping)
                                and isinstance(robustness.get("28"), Mapping)
                                else None
                            ),
                            "feasible": model in feasible,
                            "mcs_member": model in mcs,
                            "selection_oos_years": payload.get("selection_years"),
                            "locked_test_year": payload.get("locked_test_year"),
                            "test_used_for_selection": payload.get(
                                "selection_updated_from_2025"
                            ),
                        }
                    )
            elif not has_locked_rows:
                explicit_rows = payload.get("selection_methods", payload.get("methods"))
                if isinstance(explicit_rows, list):
                    candidates = [
                        dict(row) for row in explicit_rows if isinstance(row, Mapping)
                    ]
                else:
                    lcb = payload.get("lcb_by_model", {})
                    feasible = set(map(str, payload.get("feasible_models", [])))
                    mcs = set(
                        map(
                            str,
                            payload.get(
                                "mcs_members", payload.get("mcs_set", [])
                            ),
                        )
                    )
                    robustness = payload.get("robustness_lcb_by_block", {})
                    models = (
                        sorted(set(map(str, lcb)) | feasible | mcs)
                        if isinstance(lcb, Mapping)
                        else []
                    )
                    candidates = [
                        {
                            "selection_method": "DART-ValueSelect",
                            "experiment_id": "A2",
                            "candidate_id": model,
                            "selected_model": payload.get("selected_model"),
                            "lcb_incremental_operational_value_usd": lcb.get(model),
                            "lcb_7_day": lcb.get(model),
                            "lcb_14_day": robustness.get("14", {}).get(model)
                            if isinstance(robustness, Mapping)
                            and isinstance(robustness.get("14"), Mapping)
                            else None,
                            "lcb_28_day": robustness.get("28", {}).get(model)
                            if isinstance(robustness, Mapping)
                            and isinstance(robustness.get("28"), Mapping)
                            else None,
                            "feasible": model in feasible,
                            "mcs_member": model in mcs,
                        }
                        for model in models
                    ]
        else:
            continue
        expanded_candidates: list[dict[str, Any]] = []
        for raw in candidates:
            raw_row = dict(raw)
            candidate_value = _first_value(
                raw_row, ("candidate_id", "model", "selected_model")
            )
            winner_value = raw_row.get("winner_set")
            if candidate_value is None and isinstance(winner_value, (list, tuple)):
                for member in winner_value:
                    expanded_candidates.append(
                        {**raw_row, "candidate_id": str(member), "winner_set_member": True}
                    )
            else:
                expanded_candidates.append(raw_row)
        for raw in expanded_candidates:
            combined = {**dict(payload), **dict(raw)}
            formal_source = _dart_final_manifest_for(source.path) is not None
            method = _first_value(
                combined, ("selection_method", "method", "selector")
            )
            candidate = _first_value(combined, ("candidate_id", "model", "selected_model"))
            years = _selection_years(
                _first_value(combined, ("selection_oos_years", "selection_years"))
            )
            locked_year = _first_value(combined, ("locked_test_year",))
            test_used = _first_value(
                combined, ("test_used_for_selection", "locked_test_used_for_selection")
            )
            lcb = _first_value(
                combined,
                ("lcb_incremental_operational_value_usd", "lcb_7_day"),
            )
            mcs_member = _first_value(combined, ("mcs_member",))
            if mcs_member is None and method != "DART-ValueSelect":
                mcs_member = False
            selector_summary = source.frame is not None and formal_source
            if years is None and selector_summary:
                years = "2022-2024"
            if locked_year is None and selector_summary:
                locked_year = 2025
            if test_used is None and selector_summary:
                test_used = False
            if mcs_member is None and selector_summary:
                mcs_member = False
            if candidate is not None and pd.isna(candidate):
                candidate = None
            if None in (method, candidate, years, locked_year, mcs_member):
                continue
            if lcb is None and not selector_summary:
                continue
            if years.replace(",", "-").replace(" ", "") not in {
                "2022-2023-2024",
                "2022-2024",
            }:
                raise PaperArtifactInputError("DART selection must use rolling OOS 2022-2024")
            if int(locked_year) != 2025 or not _false_value(test_used):
                raise PaperArtifactInputError("locked 2025 may be evaluated once but never used for selection")
            row = dict(raw)
            canonical_method = str(method)
            main_method = canonical_method in {
                "MAE-Select",
                "minimum_mae",
                "mae",
                "minimum_mae_with_crps_tie_break",
                "DART-ValueSelect",
            }
            row.update(
                {
                    "selection_method": canonical_method,
                    "candidate_id": str(candidate),
                    "primary_paper_comparison": main_method,
                    "supplementary_selector": not main_method,
                    "primary_replay_experiment": str(
                        _first_value(combined, ("experiment_id", "replay_experiment"))
                        or "A2"
                    ),
                    "selection_oos_years": "2022-2024",
                    "locked_test_year": 2025,
                    "test_used_for_selection": False,
                    "lcb_incremental_operational_value_usd": (
                        _finite_float(lcb, field="selection LCB")
                        if lcb is not None
                        else None
                    ),
                    "mcs_member": _truthy(mcs_member),
                    "counterfactual": True,
                    "derived_without_new_solves": _truthy(
                        _first_value(
                            combined,
                            ("derived_without_new_solves", "no_new_solve"),
                        )
                        if _first_value(
                            combined,
                            ("derived_without_new_solves", "no_new_solve"),
                        )
                        is not None
                        else selector_summary
                    ),
                    "row_type": (
                        "frozen_selector_locked_test_comparison"
                        if selector_summary
                        else "selection_candidate_lcb"
                    ),
                }
            )
            rows.append(row)
            used[str(source.path)] = source
    if not rows:
        raise PaperArtifactInputError(
            "no complete DART selection rows prove rolling 2022-2024 OOS selection and locked 2025 isolation"
        )
    output = pd.DataFrame(rows).drop_duplicates()
    locked_rows = output.loc[
        output["row_type"].astype(str) == "frozen_selector_locked_test_comparison"
    ]
    if not locked_rows.empty:
        normalized_methods = {
            str(value).strip().lower().replace("_", "-")
            for value in locked_rows["selection_method"]
        }
        has_mae = any(value in normalized_methods for value in {"mae-select", "minimum-mae", "mae", "minimum-mae-with-crps-tie-break"})
        has_value = "dart-valueselect" in normalized_methods
        if not (has_mae and has_value):
            raise PaperArtifactInputError(
                "locked DART main comparison requires both MAE-Select and DART-ValueSelect"
            )
        if not bool(locked_rows["derived_without_new_solves"].all()):
            raise PaperArtifactInputError(
                "locked selector comparisons must be derived from the same replay table without new solves"
            )
        if set(locked_rows["primary_replay_experiment"].astype(str)) != {"A2"}:
            raise PaperArtifactInputError("locked selector comparisons must use A2")
    return output, list(used.values()), {
        "selection_oos_years": [2022, 2023, 2024],
        "locked_test_year": 2025,
        "locked_test_used_for_selection": False,
        "selection_metric": "lower_confidence_bound_of_incremental_operational_value",
        "primary_paper_methods": ["MAE-Select", "DART-ValueSelect"],
        "primary_replay_experiment": "A2",
        "A1_role": "noninferiority_gate_only",
        "same_locked_replay_rows_without_new_solves": True,
        "bootstrap_block_lengths_days": [7, 14, 28],
        "denominator_contract": {
            "lcb": "paired_daily_incremental_operational_value_vs_frozen_baseline",
            "mcs": "same_feasible_candidate_loss_panel",
        },
    }


def _dart_regime_value(
    sources: Sequence[LoadedSource],
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    a3_rows: list[dict[str, Any]] = []
    used: dict[str, LoadedSource] = {}
    for raw, source in _dart_records(sources):
        cell = _normalize_decision_row(raw)
        if cell is not None:
            cell_rows.append(cell)
            used[str(source.path)] = source
            continue
        experiment = _first_value(raw, ("experiment_id", "regime"))
        candidate = _first_value(raw, ("candidate_id", "model"))
        value = _first_value(
            raw,
            (
                "mean_incremental_operational_value_usd",
                "mean_incremental_operational_value",
                "incremental_operational_value_usd",
            ),
        )
        if (
            str(experiment).upper() in {"A1", "A2"}
            and candidate is not None
            and value is not None
        ):
            regime_rows.append(
                {
                    "phase": _first_value(raw, ("phase", "evaluation_window")),
                    "experiment_id": str(experiment).upper(),
                    "candidate_id": str(candidate),
                    "regime": _first_value(raw, ("regime",)),
                    "mean_incremental_operational_value_usd": _finite_float(
                        value, field="regime mean value"
                    ),
                    "source_cell_count": int(
                        _first_value(raw, ("day_count", "source_cell_count", "n"))
                        or 0
                    ),
                    "status": _first_value(raw, ("status",)),
                    "source_denominator": _first_value(raw, ("denominator",)),
                    "counterfactual": True,
                }
            )
            used[str(source.path)] = source
            continue
        if str(experiment).upper() == "A3" and candidate is not None and value is not None:
            accuracy_metric = _first_value(raw, ("accuracy_metric",))
            accuracy_value = _first_value(
                raw,
                (
                    "accuracy_metric_value",
                    "mean_mae",
                    "mean_crps",
                    "mae",
                    "crps",
                ),
            )
            a3_rows.append(
                {
                    "experiment_id": "A3",
                    "candidate_id": str(candidate),
                    "mean_incremental_operational_value_usd": _finite_float(
                        value, field="A3 mean value"
                    ),
                    "rank_contradiction": bool(raw.get("rank_contradiction", False)),
                    "accuracy_metric": accuracy_metric,
                    "accuracy_metric_value": (
                        _finite_float(accuracy_value, field="A3 accuracy")
                        if accuracy_value is not None
                        else None
                    ),
                    "accuracy_rank": _first_value(raw, ("accuracy_rank",)),
                    "decision_value_rank": _first_value(
                        raw, ("decision_value_rank",)
                    ),
                    "source_cell_count": int(raw.get("source_cell_count", 1)),
                    "counterfactual": True,
                }
            )
            used[str(source.path)] = source
    for source in sources:
        payload = source.payload
        if not isinstance(payload, Mapping):
            continue
        for phase_key, phase in (
            ("A3_selection", "selection_2022_2024"),
            ("A3_locked_test", "locked_test_2025"),
        ):
            wrapper = payload.get(phase_key)
            if not isinstance(wrapper, Mapping):
                continue
            for metric in ("mae", "crps", "q99_tail_f1"):
                contradiction = wrapper.get(metric)
                if not isinstance(contradiction, Mapping):
                    continue
                if contradiction.get("derived_without_solves") is not True:
                    raise PaperArtifactInputError(
                        "A3 rows must be derived from completed replay without new solves"
                    )
                nested_rows = contradiction.get("rows")
                if not isinstance(nested_rows, list):
                    continue
                declared_rows_hash = contradiction.get("rows_sha256")
                if (
                    declared_rows_hash is not None
                    and str(declared_rows_hash).lower() != _json_sha256(nested_rows)
                ):
                    raise PaperArtifactInputError("A3 contradiction row hash changed")
                for raw in nested_rows:
                    if not isinstance(raw, Mapping):
                        continue
                    a3_rows.append(
                        {
                            "phase": phase,
                            "experiment_id": "A3",
                            "candidate_id": str(raw["candidate_id"]),
                            "regime": f"accuracy_{metric}",
                            "mean_incremental_operational_value_usd": _finite_float(
                                raw.get("mean_incremental_operational_value_usd"),
                                field="A3 mean value",
                            ),
                            "accuracy_metric": metric,
                            "accuracy_metric_value": _finite_float(
                                raw.get(f"mean_{metric}"), field=f"A3 {metric}"
                            ),
                            "accuracy_rank": int(raw["accuracy_rank"]),
                            "decision_value_rank": int(raw["decision_value_rank"]),
                            "rank_contradiction": bool(raw["rank_contradiction"]),
                            "source_cell_count": int(
                                wrapper.get("source_replay_result_count", 1)
                            ),
                            "source_denominator": (
                                "completed_replay_cells_grouped_by_candidate"
                            ),
                            "counterfactual": True,
                        }
                    )
                    used[str(source.path)] = source
    grouped_rows: list[dict[str, Any]] = []
    if cell_rows:
        frame = pd.DataFrame(cell_rows)
        for (experiment, candidate), subset in frame.groupby(
            ["experiment_id", "candidate_id"], dropna=False
        ):
            grouped_rows.append(
                {
                    "experiment_id": str(experiment),
                    "candidate_id": str(candidate),
                    "mean_incremental_operational_value_usd": float(
                        subset["incremental_operational_value_usd"].mean()
                    ),
                    "mean_prescriptive_regret_usd": float(
                        subset["prescriptive_regret_usd"].mean()
                    ),
                    "mean_profit_capture_ratio": float(
                        subset["profit_capture_ratio"].mean()
                    ),
                    "probability_negative_incremental_value": float(
                        subset["incremental_operational_value_usd"].lt(0).mean()
                    ),
                    "source_cell_count": int(len(subset)),
                    "counterfactual": True,
                }
            )
    output = pd.DataFrame([*grouped_rows, *regime_rows, *a3_rows])
    regimes = set(output.get("experiment_id", pd.Series(dtype=str)).astype(str).str.upper())
    if not {"A1", "A2", "A3"}.issubset(regimes):
        raise PaperArtifactInputError(
            "DART regime table requires complete A1 prediction-only, A2 full-deployment, and A3 contradiction evidence"
        )
    return output.drop_duplicates(), list(used.values()), {
        "required_regimes": ["A1", "A2", "A3"],
        "a3_is_posthoc_derived_without_new_solves": True,
        "denominator_contract": {
            "incremental_value": "baseline_cost_minus_candidate_cost",
            "pcr": "perfect_information_value_minus_baseline_value",
        },
        "commercial_profit_claimed": False,
    }


def _canonical_rows(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    frames = [source.frame for source in sources if source.frame is not None]
    if not frames:
        raise PaperArtifactInputError("canonical-row input requires a CSV or Parquet source")
    return pd.concat(frames, ignore_index=True, sort=False), [source for source in sources if source.frame is not None], {"explicit_canonical_rows": True}


def _derive_table(
    definition: TableDefinition,
    sources: Sequence[LoadedSource],
    explicit: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, list[LoadedSource], dict[str, Any]]:
    if explicit and explicit.get("canonical_rows") is True:
        frame, used, selection = _canonical_rows(sources)
        if isinstance(explicit.get("denominator_contract"), Mapping):
            selection["denominator_contract"] = dict(
                explicit["denominator_contract"]
            )
        return frame, used, selection
    writer = definition.writer
    allow_auxiliary_only = explicit is not None
    if writer == "direct":
        selected = _latest_sources([source for source in sources if source.frame is not None])
        return _canonical_rows(selected)
    if writer == "forecast_main_da":
        return _forecast_main(sources, "nyiso_da", allow_auxiliary_only=allow_auxiliary_only)
    if writer == "forecast_main_rt":
        return _forecast_main(sources, "nyiso_rt", allow_auxiliary_only=allow_auxiliary_only)
    if writer == "forecast_main_workload":
        return _forecast_main(sources, "nlr_workload", allow_auxiliary_only=allow_auxiliary_only)
    if writer == "da_rt_consistency":
        return _da_rt_consistency(sources, allow_auxiliary_only=allow_auxiliary_only)
    if writer == "energy_by_zone":
        return _energy_grouped(sources, "zone", allow_auxiliary_only=allow_auxiliary_only)
    if writer == "energy_by_horizon":
        return _energy_grouped(sources, "horizon", allow_auxiliary_only=allow_auxiliary_only)
    if writer == "energy_spikes":
        return _energy_spikes(sources, allow_auxiliary_only=allow_auxiliary_only)
    if writer == "energy_ablation":
        return _forecast_ablation(sources, "energy", allow_auxiliary_only=allow_auxiliary_only)
    if writer == "workload_by_cohort":
        return _workload_by_cohort(sources, allow_auxiliary_only=allow_auxiliary_only)
    if writer == "workload_ablation":
        return _forecast_ablation(sources, "workload", allow_auxiliary_only=allow_auxiliary_only)
    if writer == "calibration":
        return _calibration(sources, allow_auxiliary_only=allow_auxiliary_only)
    if writer == "xcel":
        frames = [source.frame for source in sources if source.frame is not None and "source_coverage_fraction" in source.frame]
        used = [source for source in sources if source.frame is not None and "source_coverage_fraction" in source.frame]
        if not frames:
            raise PaperArtifactInputError("Xcel table requires monthly tariff replay with coverage")
        frame = pd.concat(frames, ignore_index=True, sort=False)
        if "event_calls" not in frame:
            frame["event_calls"] = 0
        if pd.to_numeric(frame["event_calls"], errors="coerce").fillna(0).ne(0).any():
            raise PaperArtifactInputError("Xcel local case has observed event calls; claim boundary must be revisited")
        return frame, used, {"event_calls": 0, "realized_savings_claimed": False, "coverage_rows_retained": True}
    if writer == "scenario_diagnostics":
        return _scenario_table(sources, "diagnostics")
    if writer == "scenario_dependence":
        return _scenario_table(sources, "dependence")
    if writer == "scenario_reduction":
        return _scenario_table(sources, "reduction")
    if writer == "extreme_cooccurrence":
        return _scenario_table(sources, "extreme")
    if writer == "optimization_main":
        return _optimization_main(sources)
    if writer == "optimization_cost":
        return _optimization_cost(sources)
    if writer == "optimization_service":
        return _optimization_view(sources, "service")
    if writer == "optimization_grid":
        return _optimization_view(sources, "grid")
    if writer == "risk_frontier":
        return _filter_view(sources, "risk")
    if writer == "dependence_stress":
        return _filter_view(sources, "dependence")
    if writer == "optimization_sensitivity":
        return _sensitivity(sources)
    if writer == "solver":
        return _optimization_view(sources, "solver")
    if writer == "failures":
        return _failures(sources)
    if writer == "hypotheses":
        return _hypotheses(sources)
    if writer == "dart_probability_quality":
        return _dart_probability_quality(sources)
    if writer == "dart_decision_value":
        return _dart_decision_value(sources)
    if writer == "dart_selection_methods":
        return _dart_selection_methods(sources)
    if writer == "dart_regime_value":
        return _dart_regime_value(sources)
    raise PaperArtifactInputError(f"table writer {writer!r} requires deferred inventory evidence")


def _validate_required_columns(definition: TableDefinition, frame: pd.DataFrame) -> None:
    missing = [group for group in definition.required_groups if not any(column in frame.columns for column in group)]
    if missing:
        raise PaperArtifactInputError(
            f"{definition.name} lacks required semantic column groups: {missing}"
        )


def _source_records(root: Path, sources: Sequence[LoadedSource]) -> list[dict[str, Any]]:
    return [
        {
            "path": _relative(root, source.path),
            "sha256": source.sha256,
            "bytes": source.bytes,
            "row_count": source.row_count,
        }
        for source in sources
    ]


def _candidate_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    return _json_sha256(
        [
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "row_count": record.get("row_count"),
            }
            for record in records
        ]
    )


def _augment_metadata(
    frame: pd.DataFrame,
    definition: TableDefinition,
    source_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    output = frame.copy()
    output["evidence_role"] = definition.evidence_role
    output["source_relationship"] = definition.source_relationship
    output["measurement_relationship"] = "not_measured_by_this_study"
    output["claim_boundary"] = definition.claim_boundary
    output["source_artifacts_json"] = json.dumps([record["path"] for record in source_records], sort_keys=True)
    output["source_artifact_hashes_json"] = json.dumps({record["path"]: record["sha256"] for record in source_records}, sort_keys=True)
    if len(source_records) == 1:
        output["source_artifact"] = source_records[0]["path"]
        output["source_artifact_sha256"] = source_records[0]["sha256"]
    return output


def write_table_bundle(
    root: Path | str,
    definition: TableDefinition,
    frame: pd.DataFrame,
    sources: Sequence[LoadedSource],
    *,
    selection: Mapping[str, Any] | None = None,
    materialization: str = "derived_from_hashed_formal_sources",
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    if frame.empty or not sources:
        raise PaperArtifactInputError("table bundle requires nonempty rows and at least one hashed source")
    source_records = _source_records(project_root, sources)
    output = _augment_metadata(frame, definition, source_records)
    _validate_required_columns(definition, output)
    target = project_root / definition.relative_path
    _atomic_csv(target, output)
    # Hash the serialized CSV interpretation, not the pre-serialization float
    # objects, so validation is stable across pandas' decimal round trip.
    normalized_rows = _records(pd.read_csv(target))
    rows_sha = _json_sha256(normalized_rows)
    sidecar = Path(str(target) + ".source.json")
    selection_payload = _jsonable(selection or {})
    dart_run_manifests: list[dict[str, Any]] = []
    if definition.name.startswith("dart_"):
        seen_manifests: set[str] = set()
        for source in sources:
            manifest_path = _dart_final_manifest_for(source.path)
            if manifest_path is None:
                continue
            key = str(manifest_path).lower()
            if key not in seen_manifests:
                dart_run_manifests.append(
                    {
                        "path": _relative(project_root, manifest_path),
                        "sha256": _sha256(manifest_path),
                    }
                )
                seen_manifests.add(key)
    payload = {
        "schema_version": TABLE_SOURCE_SCHEMA_VERSION,
        "writer_version": TABLE_WRITER_VERSION,
        "status": "COMPLETE",
        "completed": True,
        "generated_at_utc": _utc_now(),
        "table": definition.name,
        "materialization": materialization,
        "candidate_fingerprint": _candidate_fingerprint(source_records),
        "source_files": source_records,
        "output_row_count": int(len(output)),
        "output_columns": [str(column) for column in output.columns],
        "output_rows_sha256": rows_sha,
        "selection": selection_payload,
        "denominator_contract": (
            selection_payload.get("denominator_contract", {})
            if isinstance(selection_payload, Mapping)
            else {}
        ),
        "provenance_contract": {
            "source_hashes_verified_at_write": True,
            "output_rows_sha256": rows_sha,
            "candidate_fingerprint": _candidate_fingerprint(source_records),
            "demo_synthetic_partial_sources_accepted": False,
            "formal_dart_run_manifests": dart_run_manifests,
        },
        "evidence_role": definition.evidence_role,
        "source_relationship": definition.source_relationship,
        "claim_boundary": definition.claim_boundary,
        "synthetic_fallback_used": False,
        "output": {
            "path": definition.relative_path,
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
        },
    }
    _atomic_json(sidecar, payload)
    return {
        "status": "MATERIALIZED",
        "path": definition.relative_path,
        "materialization": materialization,
        "sources": source_records,
        "source_rows_sha256": rows_sha,
        "sample_count": int(len(output)),
        "companion_outputs": {
            "source_sidecar": {
                "path": _relative(project_root, sidecar),
                "sha256": _sha256(sidecar),
                "bytes": sidecar.stat().st_size,
            }
        },
    }


def validate_table_bundle(path: Path | str) -> tuple[bool, str, dict[str, Any]]:
    target = Path(path).resolve()
    sidecar = Path(str(target) + ".source.json")
    if not target.is_file() or target.stat().st_size <= 0:
        return False, "table is missing or empty", {}
    if not sidecar.is_file():
        return False, "table source sidecar is missing", {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        frame = pd.read_csv(target)
    except Exception as exc:
        return False, f"unreadable table bundle: {type(exc).__name__}", {}
    if payload.get("schema_version") != TABLE_SOURCE_SCHEMA_VERSION or payload.get("completed") is not True:
        return False, "table sidecar schema/status is invalid", payload
    if frame.empty or int(payload.get("output_row_count", -1)) != len(frame):
        return False, "table row count is empty or disagrees with sidecar", payload
    if _frame_is_synthetic(frame) or payload.get("synthetic_fallback_used") is True:
        return False, "synthetic/demo evidence cannot satisfy a canonical table", payload
    if _sha256(target) != str(payload.get("output", {}).get("sha256", "")):
        return False, "table output SHA-256 disagrees with sidecar", payload
    rows_sha = _json_sha256(_records(frame))
    if rows_sha != str(payload.get("output_rows_sha256", "")):
        return False, "table row-content SHA-256 disagrees with sidecar", payload
    sources = payload.get("source_files")
    if not isinstance(sources, list) or not sources:
        return False, "table sidecar has no hashed source files", payload
    root = target
    while root.parent != root and root.name != "artifacts":
        root = root.parent
    project_root = root.parent if root.name == "artifacts" else target.parent
    for source in sources:
        if not isinstance(source, Mapping):
            return False, "table sidecar has malformed source entry", payload
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_absolute():
            source_path = project_root / source_path
        if not source_path.is_file() or _sha256(source_path) != str(source.get("sha256", "")):
            return False, f"table source hash is invalid for {source.get('path')}", payload
    if _candidate_fingerprint(sources) != str(payload.get("candidate_fingerprint", "")):
        return False, "table candidate fingerprint disagrees with source records", payload
    required_metadata = {"evidence_role", "source_relationship", "claim_boundary"}
    if not required_metadata.issubset(frame.columns):
        return False, "table lacks evidence/claim-boundary columns", payload
    if target.name.startswith("dart_"):
        if not isinstance(payload.get("denominator_contract"), Mapping) or not payload[
            "denominator_contract"
        ]:
            return False, "DART table sidecar lacks denominator semantics", payload
        provenance = payload.get("provenance_contract", {})
        if not isinstance(provenance, Mapping) or provenance.get(
            "demo_synthetic_partial_sources_accepted"
        ) is not False:
            return False, "DART table sidecar lacks strict provenance gate", payload
        for record in provenance.get("formal_dart_run_manifests", []):
            if not isinstance(record, Mapping):
                return False, "DART table has malformed run-manifest provenance", payload
            manifest_path = Path(str(record.get("path", "")))
            if not manifest_path.is_absolute():
                manifest_path = project_root / manifest_path
            if (
                not manifest_path.is_file()
                or str(record.get("sha256", "")) != _sha256(manifest_path)
            ):
                return False, "DART run-manifest provenance hash changed", payload
    return True, f"validated {len(frame)} source-backed rows", payload


def generate_paper_tables(
    root: Path | str,
    branch_results: Mapping[str, Any] | None = None,
    *,
    allow_test_fixtures: bool = False,
) -> dict[str, dict[str, Any]]:
    project_root = Path(root).resolve()
    results = dict(branch_results or {})
    output: dict[str, dict[str, Any]] = {}
    for definition in TABLE_INPUT_REGISTRY:
        if definition.writer == "inventory":
            output[definition.relative_path] = {
                "status": "MISSING_MANDATORY",
                "reason": "deferred until the artifact inventory has been assessed",
                "sources": [],
            }
            continue
        target = project_root / definition.relative_path
        valid, reason, sidecar = validate_table_bundle(target)
        if valid:
            output[definition.relative_path] = {
                "status": "MATERIALIZED",
                "path": definition.relative_path,
                "materialization": "existing_validated_table_bundle",
                "sources": list(sidecar.get("source_files", [])),
                "source_rows_sha256": sidecar.get("output_rows_sha256"),
                "sample_count": sidecar.get("output_row_count"),
                "companion_outputs": {
                    "source_sidecar": {
                        "path": _relative(project_root, Path(str(target) + ".source.json")),
                        "sha256": _sha256(Path(str(target) + ".source.json")),
                        "bytes": Path(str(target) + ".source.json").stat().st_size,
                    }
                },
            }
            continue
        paths, explicit = _resolve_sources(project_root, definition, results)
        loaded: list[LoadedSource] = []
        errors: list[str] = []
        expected_by_path: dict[str, str] = {}
        if explicit:
            declared = explicit.get("sha256")
            if declared and len(paths) == 1:
                expected_by_path[str(paths[0].resolve()).lower()] = str(declared)
            hashes = explicit.get("source_hashes", {})
            if isinstance(hashes, Mapping):
                for key, value in hashes.items():
                    candidate = Path(str(key))
                    if not candidate.is_absolute():
                        candidate = project_root / candidate
                    expected_by_path[str(candidate.resolve()).lower()] = str(value)
        for path in paths:
            try:
                loaded.append(
                    _load_source(
                        project_root,
                        path,
                        expected_sha256=expected_by_path.get(str(path.resolve()).lower()),
                        allow_test_fixtures=allow_test_fixtures,
                    )
                )
            except Exception as exc:
                errors.append(f"{_relative(project_root, path)}: {exc}")
        try:
            frame, used, selection = _derive_table(definition, loaded, explicit)
            output[definition.relative_path] = write_table_bundle(
                project_root,
                definition,
                frame,
                used,
                selection=selection,
            )
        except Exception as exc:
            output[definition.relative_path] = {
                "status": "MISSING_MANDATORY",
                "reason": str(exc) if loaded else ("; ".join(errors) or reason or "no formal source inputs found"),
                "sources": _source_records(project_root, loaded),
                "input_errors": errors,
            }
    return output


def write_inventory_audit_table(
    root: Path | str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    definition = TABLE_DEFINITION_BY_NAME["evidence_layer_audit.csv"]
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        raise PaperArtifactInputError("evidence-layer audit cannot be empty")
    source_paths: list[Path] = []
    for row in rows:
        value = row.get("canonical_path", row.get("relative_path"))
        if value:
            path = Path(str(value))
            if not path.is_absolute():
                path = project_root / path
            if path.is_file() and path.name != definition.name:
                source_paths.append(path)
    unique = list({str(path.resolve()).lower(): path for path in source_paths}.values())
    loaded = [
        LoadedSource(path.resolve(), _sha256(path), path.stat().st_size, 1, None, None)
        for path in unique
    ]
    if not loaded:
        raise PaperArtifactInputError("evidence-layer audit has no materialized source artifacts")
    return write_table_bundle(
        project_root,
        definition,
        frame,
        loaded,
        selection={"inventory_row_count": len(frame)},
        materialization="inventory_to_evidence_layer_audit",
    )


REPORT_NAMES = (
    "run_summary.html",
    "run_summary.md",
    "integrity_audit.json",
    "paper_mapping.md",
    "data_architecture_amendment.md",
    "evidence_layer_dictionary.md",
    "source_to_claim_matrix.md",
    "transfer_and_extrapolation_report.md",
    "failed_or_skipped_experiments.md",
)


def _report_sidecar(path: Path) -> Path:
    return Path(str(path) + ".source.json")


def _record_path(root: Path, record: Mapping[str, Any]) -> Path | None:
    value = record.get("relative_path", record.get("path"))
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _report_sources(root: Path, records: Sequence[Mapping[str, Any]]) -> list[LoadedSource]:
    paths: dict[str, Path] = {}
    for record in records:
        if str(record.get("status")) not in {"MATERIALIZED", "SKIPPED_OPTIONAL_SOURCE"}:
            continue
        path = _record_path(root, record)
        if path is None or path.parent.name == "report" or path.name == "final_manifest.json":
            continue
        paths[str(path.resolve()).lower()] = path
    return [LoadedSource(path.resolve(), _sha256(path), path.stat().st_size, 1, None, None) for path in paths.values()]


def _write_report_bundle(
    root: Path,
    name: str,
    content: str | Mapping[str, Any],
    sources: Sequence[LoadedSource],
    *,
    evidence_role: str,
    source_relationship: str,
    claim_boundary: str,
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    if not sources:
        raise PaperArtifactInputError(f"{name} has no hashed report inputs")
    target = root / "artifacts/report" / name
    if isinstance(content, Mapping):
        _atomic_json(target, content)
        facts_payload: Any = content
    else:
        _atomic_text(target, content)
        facts_payload = facts
    source_records = _source_records(root, sources)
    sidecar = _report_sidecar(target)
    payload = {
        "schema_version": REPORT_SOURCE_SCHEMA_VERSION,
        "writer_version": TABLE_WRITER_VERSION,
        "status": "COMPLETE",
        "completed": True,
        "generated_at_utc": _utc_now(),
        "report": name,
        "candidate_fingerprint": _candidate_fingerprint(source_records),
        "source_files": source_records,
        "source_fact_count": len(facts),
        "source_facts_sha256": _json_sha256(facts_payload),
        "source_facts": _jsonable(facts),
        "evidence_role": evidence_role,
        "source_relationship": source_relationship,
        "claim_boundary": claim_boundary,
        "synthetic_fallback_used": False,
        "output": {
            "path": _relative(root, target),
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
        },
    }
    _atomic_json(sidecar, payload)
    return {
        "status": "MATERIALIZED",
        "path": _relative(root, target),
        "materialization": "generated_from_hashed_artifact_records",
        "sources": source_records,
        "source_rows_sha256": payload["source_facts_sha256"],
        "sample_count": len(facts),
        "companion_outputs": {
            "source_sidecar": {
                "path": _relative(root, sidecar),
                "sha256": _sha256(sidecar),
                "bytes": sidecar.stat().st_size,
            }
        },
    }


def validate_report_bundle(path: Path | str) -> tuple[bool, str, dict[str, Any]]:
    target = Path(path).resolve()
    sidecar = _report_sidecar(target)
    if not target.is_file() or target.stat().st_size <= 40:
        return False, "report is missing or too short", {}
    if not sidecar.is_file():
        return False, "report source sidecar is missing", {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid report sidecar: {type(exc).__name__}", {}
    if payload.get("schema_version") != REPORT_SOURCE_SCHEMA_VERSION or payload.get("completed") is not True:
        return False, "report sidecar schema/status is invalid", payload
    if _sha256(target) != str(payload.get("output", {}).get("sha256", "")):
        return False, "report output SHA-256 disagrees with sidecar", payload
    if payload.get("synthetic_fallback_used") is True:
        return False, "synthetic report evidence is forbidden", payload
    sources = payload.get("source_files")
    if not isinstance(sources, list) or not sources:
        return False, "report has no hashed sources", payload
    root = target
    while root.parent != root and root.name != "artifacts":
        root = root.parent
    project_root = root.parent if root.name == "artifacts" else target.parent
    for source in sources:
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_absolute():
            source_path = project_root / source_path
        if not source_path.is_file() or _sha256(source_path) != str(source.get("sha256", "")):
            return False, f"report source hash is invalid for {source.get('path')}", payload
    if _candidate_fingerprint(sources) != str(payload.get("candidate_fingerprint", "")):
        return False, "report candidate fingerprint disagrees with sources", payload
    return True, "validated source-backed report", payload


def _markdown_artifact_table(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Artifact | Status | Evidence role | Claim boundary |",
        "|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| `{}` | `{}` | `{}` | {} |".format(
                record.get("relative_path", record.get("path")),
                record.get("status"),
                record.get("evidence_role", "derived"),
                str(record.get("claim_boundary", REFERENCE_BOUNDARY)).replace("|", "\\|"),
            )
        )
    return "\n".join(lines)


def generate_paper_reports(
    root: Path | str,
    artifact_records: Sequence[Mapping[str, Any]],
    branch_results: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    project_root = Path(root).resolve()
    records = [dict(record) for record in artifact_records]
    output: dict[str, dict[str, Any]] = {}
    sources = _report_sources(project_root, records)
    if not records or not sources:
        return {
            f"artifacts/report/{name}": {
                "status": "MISSING_MANDATORY",
                "reason": "report generation requires assessed, hashed artifact records",
                "sources": [],
            }
            for name in REPORT_NAMES
        }

    stable_records = [
        record
        for record in records
        if str(record.get("category")) != "report"
        and Path(str(record.get("relative_path", record.get("path", "")))).name
        not in {"final_manifest.json", "evidence_layer_audit.csv"}
    ]
    missing = [record for record in stable_records if record.get("requirement") == "mandatory" and record.get("status") != "MATERIALIZED"]
    skipped = [record for record in records if record.get("status") == "SKIPPED_OPTIONAL_SOURCE"]
    status = "COMPLETE" if not missing else "BLOCKED_MISSING_MANDATORY"
    by_name = {Path(str(record.get("relative_path", record.get("path", "")))).name: record for record in records}
    facts = {
        "status": status,
        "mandatory_artifact_count": sum(record.get("requirement") == "mandatory" for record in stable_records),
        "materialized_mandatory_count": sum(record.get("requirement") == "mandatory" and record.get("status") == "MATERIALIZED" for record in stable_records),
        "missing_mandatory_count": len(missing),
        "optional_source_skip_count": len(skipped),
        "full_year_extension_status": next((record.get("status") for record in records if "optimization" in str(record.get("artifact_id", ""))), None),
        "timezone_paired_economic_table": "artifacts/tables/optimization_sensitivity.csv",
        "xcel_coverage_table": "artifacts/tables/xcel_local_counterfactual.csv",
        "hypothesis_table": "artifacts/tables/hypothesis_tests.csv",
    }
    summary_md = (
        "# Paper Artifact Run Summary\n\n"
        f"- Inventory status: `{status}`\n"
        f"- Materialized mandatory artifacts: `{facts['materialized_mandatory_count']}` / `{facts['mandatory_artifact_count']}`\n"
        f"- Missing mandatory artifacts: `{len(missing)}`\n"
        f"- Optional robustness sources skipped with reasons: `{len(skipped)}`\n"
        "- Statistical inference: H0-H6 are mapped through `hypothesis_tests.csv`; the sidecar pins all four statistics outputs.\n"
        "- Timezone sensitivity: paired economic outcomes are mapped through `optimization_sensitivity.csv`.\n"
        "- Xcel: monthly source coverage and zero observed event calls are mapped through `xcel_local_counterfactual.csv`.\n"
        "- Resume/reproduction: hashed manifests and stage status are retained in the canonical manifest set.\n"
        "- Claim boundary: all market/VPP outcomes are counterfactual; no realized profit, observed DR saving, or demonstrated grid benefit is claimed.\n"
    )
    summary_html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Paper artifact summary</title></head><body>"
        "<h1>Paper Artifact Run Summary</h1>"
        f"<p>Status: <code>{html.escape(status)}</code></p>"
        f"<p>Materialized mandatory artifacts: {facts['materialized_mandatory_count']} / {facts['mandatory_artifact_count']}</p>"
        f"<p>Missing mandatory artifacts: {len(missing)}</p>"
        f"<p>Optional robustness sources skipped: {len(skipped)}</p>"
        "<p>H0-H6 statistics, timezone-paired economics, Xcel coverage, resume, and reproduction evidence are mapped to hashed canonical artifacts.</p>"
        "<p>All market and VPP outcomes are counterfactual, not realized returns or observed grid benefits.</p>"
        "</body></html>"
    )

    integrity = {
        "schema_version": "paper_artifact_inventory_v1",
        "status": status,
        "completed": not missing,
        "generated_at_utc": _utc_now(),
        **facts,
        "missing_mandatory": [
            {"artifact_id": record.get("artifact_id"), "path": record.get("relative_path"), "reason": record.get("reason")}
            for record in missing
        ],
        "optional_robustness": [
            {"source": record.get("artifact_id"), "status": record.get("status"), "reason": record.get("reason")}
            for record in skipped
        ],
        "forbidden_claims_present": False,
        "claim_boundary": COUNTERFACTUAL_BOUNDARY,
    }

    mapping = (
        "# Paper Artifact Mapping\n\n"
        "This mapping is generated from the assessed canonical registry and its hashed artifacts.\n\n"
        + _markdown_artifact_table(stable_records)
        + "\n\n## Paper-ready destinations\n\n"
        "| Destination | Canonical evidence |\n|---|---|\n"
        "| Physical validation | `nlr_twin_validation.csv`, `nlr_pue_validation.csv`, `nlr_parameter_uncertainty.csv` |\n"
        "| H1-H2 forecast evidence | `nyiso_da_forecast_main.csv`, `nyiso_rt_forecast_main.csv`, `workload_forecast_main.csv` |\n"
        "| Scenario quality and reduction | four canonical scenario tables |\n"
        "| B0-B8, risk, service, grid, sensitivities | ten canonical optimization/statistical tables |\n"
        "| H0-H6 decisions | `hypothesis_tests.csv` with four-statistics sidecar |\n"
        "| Reproduction/resume | canonical manifests and final manifest |\n"
    )

    architecture = (
        "# Data Architecture Amendment\n\n"
        "- NLR, SRRL, Xcel, NYISO, NASA POWER, and optional robustness traces remain independent source domains.\n"
        "- Local timestamps retain their declared source timezone and are converted through explicit timezone manifests; dates are not row-wise joined into a false synchronized observation.\n"
        "- NLR physical calibration is transferred to NYISO-zone twins only for counterfactual co-simulation.\n"
        "- NYISO DA/RT records are referenced market data, not measurements made by this study. Negative prices remain on their original scale.\n"
        "- Xcel outputs are local tariff-defined counterfactuals; monthly coverage is explicit and event calls remain zero.\n"
        "- Reconstructed weather, replayed workload, and optional robustness sources retain separate evidence roles.\n"
    )
    dictionary = (
        "# Evidence Layer Dictionary\n\n"
        "- `observed_nlr`: referenced NLR source record, not investigator measurement.\n"
        "- `observed_nyiso_market`: referenced NYISO market record.\n"
        "- `observed_srrl`: referenced SRRL record.\n"
        "- `derived`: deterministic transformation of hashed source records.\n"
        "- `reconstructed`: reconstructed exogenous series, including NASA POWER weather.\n"
        "- `replayed`: source workload trace placed in a non-joint counterfactual timeline.\n"
        "- `transferred_nyiso_twin`: NLR-calibrated parameters transferred to a NYISO-zone twin.\n"
        "- `xcel_counterfactual`: published tariff-rule calculation with zero observed event calls.\n"
        "- `optional_robustness`: independent source, never concatenated into primary NLR training.\n"
        "- `oracle`: perfect-information lower bound, excluded from deployable claims.\n"
        "- `counterfactual`: simulated VPP/market result, never a realized return or observed grid benefit.\n"
    )
    matrix_lines = [
        "# Source-to-Claim Matrix",
        "",
        "| Artifact | Evidence role | Source relationship | Claim boundary | Source hashes |",
        "|---|---|---|---|---|",
    ]
    for record in stable_records:
        matrix_lines.append(
            "| `{}` | `{}` | `{}` | {} | `{}` |".format(
                record.get("relative_path"),
                record.get("evidence_role", "derived"),
                record.get("source_relationship", "derived_from_mapped_sources"),
                str(record.get("claim_boundary", REFERENCE_BOUNDARY)).replace("|", "\\|"),
                json.dumps(record.get("source_hashes", {}), sort_keys=True).replace("|", "\\|"),
            )
        )
    source_matrix = "\n".join(matrix_lines) + "\n"

    transfer = by_name.get("digital_twin_transfer.csv")
    if not transfer or transfer.get("status") != "MATERIALIZED":
        transfer_error = "complete digital-twin transfer evidence is unavailable"
    else:
        transfer_error = None
    transfer_report = (
        "# Transfer and Extrapolation Report\n\n"
        "The NLR-calibrated physical parameter ensemble is transferred only to counterfactual NYISO-zone twins. "
        "The canonical transfer table retains site, climate-support distance, out-of-support fraction, physical multiplier intervals, source hashes, and the no-deployment claim boundary.\n\n"
        "This report does not assert that an NLR facility operated in NYISO and does not convert counterfactual savings into realized revenue.\n"
    )

    failure_lines = ["# Failed or Skipped Experiments", ""]
    failure_table = by_name.get("failures.csv")
    failure_lines.append(
        "The canonical `failures.csv` records failures, timeouts, nonoptimal terminations, nonzero gaps, or a source-backed zero-event audit row."
    )
    for record in missing:
        failure_lines.append(f"- `MISSING_MANDATORY` - `{record.get('relative_path')}`: {record.get('reason')}")
    for record in skipped:
        failure_lines.append(f"- `SKIPPED_OPTIONAL_SOURCE` - `{record.get('relative_path')}`: {record.get('reason')}")
    if failure_table:
        failure_lines.append(f"- Failure ledger status: `{failure_table.get('status')}`; SHA-256 `{failure_table.get('sha256')}`.")
    failures_report = "\n".join(failure_lines) + "\n"

    report_payloads: dict[str, tuple[str | Mapping[str, Any], str, str, str]] = {
        "run_summary.md": (summary_md, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
        "run_summary.html": (summary_html, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
        "integrity_audit.json": (integrity, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
        "paper_mapping.md": (mapping, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
        "data_architecture_amendment.md": (architecture, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
        "evidence_layer_dictionary.md": (dictionary, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
        "source_to_claim_matrix.md": (source_matrix, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
        "transfer_and_extrapolation_report.md": (transfer_report, "transferred_nyiso_twin", "derived_from_third_party_referenced_public_data", TRANSFER_BOUNDARY),
        "failed_or_skipped_experiments.md": (failures_report, "derived", "derived_from_mapped_sources", REFERENCE_BOUNDARY),
    }
    for name, (content, role, relationship, boundary) in report_payloads.items():
        relative = f"artifacts/report/{name}"
        target = project_root / relative
        valid, _, sidecar = validate_report_bundle(target)
        if valid:
            output[relative] = {
                "status": "MATERIALIZED",
                "path": relative,
                "materialization": "existing_validated_report_bundle",
                "sources": list(sidecar.get("source_files", [])),
                "source_rows_sha256": sidecar.get("source_facts_sha256"),
                "sample_count": sidecar.get("source_fact_count"),
                "companion_outputs": {
                    "source_sidecar": {
                        "path": _relative(project_root, _report_sidecar(target)),
                        "sha256": _sha256(_report_sidecar(target)),
                        "bytes": _report_sidecar(target).stat().st_size,
                    }
                },
            }
            continue
        if name == "transfer_and_extrapolation_report.md" and transfer_error:
            output[relative] = {"status": "MISSING_MANDATORY", "reason": transfer_error, "sources": []}
            continue
        try:
            output[relative] = _write_report_bundle(
                project_root,
                name,
                content,
                sources,
                evidence_role=role,
                source_relationship=relationship,
                claim_boundary=boundary,
                facts=facts,
            )
        except Exception as exc:
            output[relative] = {"status": "MISSING_MANDATORY", "reason": str(exc), "sources": []}
    return output


__all__ = [
    "PaperArtifactInputError",
    "REPORT_NAMES",
    "REPORT_SOURCE_SCHEMA_VERSION",
    "TABLE_DEFINITION_BY_NAME",
    "TABLE_INPUT_REGISTRY",
    "TABLE_SOURCE_SCHEMA_VERSION",
    "TableDefinition",
    "generate_paper_reports",
    "generate_paper_tables",
    "validate_report_bundle",
    "validate_table_bundle",
    "write_inventory_audit_table",
    "write_table_bundle",
]
