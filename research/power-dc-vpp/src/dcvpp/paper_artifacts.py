from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from dcvpp.core import load_yaml
from dcvpp.experiments.dart_partial_candidate import (
    PARTIAL_CANDIDATE_STATUS,
    PARTIAL_CANDIDATE_TERMINAL_CAP,
    validate_partial_candidate_pool_contract,
)


INVENTORY_SCHEMA_VERSION = "paper_artifact_inventory_v1"
FINAL_MANIFEST_SCHEMA_VERSION = "paper_final_manifest_v1"

STATUS_MATERIALIZED = "MATERIALIZED"
STATUS_MISSING_MANDATORY = "MISSING_MANDATORY"
STATUS_SKIPPED_OPTIONAL = "SKIPPED_OPTIONAL_SOURCE"
STATUS_INVALID = "INVALID_EVIDENCE"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    category: str
    relative_path: str
    requirement: str
    evidence_role: str
    source_relationship: str
    claim_boundary: str
    specification_lines: str


_MANIFEST_NAMES = (
    "source_manifest.json",
    "referenced_data_progress_manifest.json",
    "nlr295_inventory_manifest.json",
    "download_manifest.json",
    "data_quality_manifest.json",
    "timezone_manifest.json",
    "source_overlap_manifest.json",
    "split_manifest.json",
    "cohort_mapping_manifest.json",
    "feature_manifest.json",
    "model_manifest.json",
    "physical_calibration_manifest.json",
    "nyiso_da_calibration_manifest.json",
    "nyiso_rt_calibration_manifest.json",
    "workload_calibration_manifest.json",
    "digital_twin_transfer_manifest.json",
    "xcel_counterfactual_manifest.json",
    "resume_manifest.json",
    "robustness_manifest.json",
    "scenario_manifest.json",
    "solver_manifest.json",
    "dart_value_select_manifest.json",
    "dart_replay_manifest.json",
    "dart_statistical_selection_manifest.json",
    "final_manifest.json",
)

_FORECAST_TABLE_NAMES = (
    "nlr_twin_validation.csv",
    "nlr_pue_validation.csv",
    "nlr_parameter_uncertainty.csv",
    "energy_coverage_by_partition.csv",
    "pue_component_closure.csv",
    "nyiso_da_forecast_main.csv",
    "nyiso_rt_forecast_main.csv",
    "nyiso_da_rt_consistency.csv",
    "energy_forecast_by_zone.csv",
    "energy_forecast_by_horizon.csv",
    "energy_forecast_spikes.csv",
    "energy_forecast_ablation.csv",
    "workload_forecast_main.csv",
    "workload_forecast_by_cohort.csv",
    "workload_forecast_ablation.csv",
    "calibration_summary.csv",
    "digital_twin_transfer.csv",
    "xcel_local_counterfactual.csv",
    "evidence_layer_audit.csv",
)

_SCENARIO_TABLE_NAMES = (
    "scenario_diagnostics.csv",
    "scenario_dependence.csv",
    "scenario_reduction.csv",
    "extreme_cooccurrence.csv",
)

_OPTIMIZATION_TABLE_NAMES = (
    "counterfactual_optimization_main.csv",
    "optimization_cost_breakdown.csv",
    "optimization_service.csv",
    "optimization_grid_facing_proxies.csv",
    "risk_frontier.csv",
    "dependence_stress.csv",
    "optimization_sensitivity.csv",
    "solver_performance.csv",
    "failures.csv",
    "hypothesis_tests.csv",
)

_DART_TABLE_NAMES = (
    "dart_probability_quality.csv",
    "dart_decision_value.csv",
    "dart_selection_methods.csv",
    "dart_regime_value.csv",
)

_FIGURE_NAMES = (
    "data_timeline.png",
    "data_missingness.png",
    "nlr_physical_representative_week.png",
    "nyiso_counterfactual_representative_week.png",
    "nlr_physical_calibration.png",
    "pue_component_closure.png",
    "twin_transfer_support.png",
    "nyiso_da_calibration.png",
    "nyiso_rt_calibration.png",
    "workload_calibration.png",
    "price_spike_case.png",
    "workload_burst_case.png",
    "scenario_dependence.png",
    "dependence_stress.png",
    "risk_frontier.png",
    "policy_comparison.png",
    "typical_day_dispatch.png",
    "grid_rebound.png",
    "sensitivity_flexibility.png",
    "sensitivity_bandwidth.png",
    "sensitivity_storage.png",
    "solver_scaling.png",
    "dart_accuracy_value_contradiction.png",
    "dart_value_selection_lcb.png",
)

_REPORT_NAMES = (
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


def _claim_metadata(category: str, name: str) -> tuple[str, str, str]:
    lowered = name.lower()
    if "dart_probability_quality" in lowered:
        return (
            "derived",
            "derived_from_third_party_referenced_public_data",
            "Probabilistic DART evaluation on referenced NYISO records; no facility operation or realized market return is claimed.",
        )
    if "dart" in lowered:
        return (
            "counterfactual",
            "counterfactual_using_third_party_referenced_public_data",
            "DART decision value is a constrained data-center counterfactual on referenced NYISO prices; it is not virtual-trader P&L, realized profit, or observed grid benefit.",
        )
    if "xcel" in lowered:
        return (
            "xcel_counterfactual",
            "tariff_defined_counterfactual",
            "Tariff-rule counterfactual only; zero observed event calls and no observed demand-response savings.",
        )
    if "transfer" in lowered or "twin" in lowered:
        return (
            "transferred_nyiso_twin" if "transfer" in lowered else "derived",
            "derived_from_third_party_referenced_public_data",
            "NLR-calibrated parameter transfer; no NLR facility was operated in NYISO.",
        )
    if any(token in lowered for token in ("optimization", "risk", "policy", "dispatch", "rebound", "dependence_stress", "scenario")):
        return (
            "counterfactual",
            "counterfactual_using_third_party_referenced_public_data",
            "Counterfactual co-simulation only; not realized profit, observed grid benefit, or an actual NLR-in-NYISO operation.",
        )
    if lowered.startswith("nlr_") or "pue" in lowered or "energy_coverage" in lowered:
        return (
            "derived",
            "derived_from_third_party_referenced_public_data",
            "Derived from referenced NLR/SRRL records; observed denotes source-level records, not investigator measurement.",
        )
    if "nyiso" in lowered or "energy_forecast" in lowered or "price_spike" in lowered:
        return (
            "derived",
            "derived_from_third_party_referenced_public_data",
            "Forecast evaluation on referenced NYISO records; no facility operation or realized market return is claimed.",
        )
    if "workload" in lowered or "cohort" in lowered:
        return (
            "derived",
            "derived_from_third_party_referenced_public_data",
            "NLR workload-trace evidence only; replayed dates are not additional independent observations.",
        )
    return (
        "derived",
        "derived_from_mapped_sources",
        "Evidence mapping only; claims cannot exceed the hashed source artifacts and their declared evidence roles.",
    )


def _make_spec(
    category: str, name: str, lines: str, *, requirement: str = "mandatory"
) -> ArtifactSpec:
    evidence_role, source_relationship, boundary = _claim_metadata(category, name)
    directory = {
        "manifest": "artifacts/manifests",
        "table": "artifacts/tables",
        "figure": "artifacts/figures",
        "report": "artifacts/report",
    }[category]
    stem = Path(name).stem
    return ArtifactSpec(
        artifact_id=f"{category}:{stem}",
        category=category,
        relative_path=f"{directory}/{name}",
        requirement=requirement,
        evidence_role=evidence_role,
        source_relationship=source_relationship,
        claim_boundary=boundary,
        specification_lines=lines,
    )


CANONICAL_ARTIFACT_REGISTRY: tuple[ArtifactSpec, ...] = tuple(
    [_make_spec("manifest", name, "1939-1964") for name in _MANIFEST_NAMES]
    + [_make_spec("table", name, "1970-1992") for name in _FORECAST_TABLE_NAMES]
    + [_make_spec("table", name, "1994-2001") for name in _SCENARIO_TABLE_NAMES]
    + [_make_spec("table", name, "2003-2018") for name in _OPTIMIZATION_TABLE_NAMES]
    + [_make_spec("table", name, "DART-ValueSelect extension") for name in _DART_TABLE_NAMES]
    + [_make_spec("figure", name, "2020-2047") for name in _FIGURE_NAMES]
    + [_make_spec("report", name, "2049-2060") for name in _REPORT_NAMES]
)


DEFAULT_OPTIONAL_SOURCE_SPECS: tuple[ArtifactSpec, ...] = tuple(
    ArtifactSpec(
        artifact_id=f"optional_source:{source}",
        category="optional_source",
        relative_path=f"artifacts/optional/{source}/source_status.json",
        requirement="optional",
        evidence_role="optional_robustness",
        source_relationship="third_party_referenced_public_data",
        claim_boundary=(
            "Independent optional robustness source; never concatenated into the primary NLR training set."
        ),
        specification_lines="59,128-136,949-951",
    )
    for source in ("nlr301", "alibaba", "pm100", "acmetrace")
)


DEFAULT_SOURCE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "artifacts/manifests/source_manifest.json": (
        "artifacts/manifests/source_manifest.json",
    ),
    "artifacts/manifests/referenced_data_progress_manifest.json": (
        "artifacts/manifests/referenced_data_progress_manifest.json",
    ),
    "artifacts/manifests/nlr295_inventory_manifest.json": (
        "artifacts/manifests/nlr295_inventory_manifest.json",
    ),
    "artifacts/manifests/download_manifest.json": (
        "artifacts/state/nyiso_download_state.json",
        "artifacts/research/physical/run_id=*/srrl_download_progress.json",
        "artifacts/research/physical/run_id=*/nlr295_member_progress.json",
    ),
    "artifacts/manifests/data_quality_manifest.json": (
        "artifacts/real_forecasting/*/manifests/input_audits.json",
        "artifacts/research/physical/run_id=*/source_overlap_manifest.json",
        "artifacts/manifests/nlr295_inventory_manifest.json",
    ),
    "artifacts/manifests/timezone_manifest.json": (
        "artifacts/manifests/timezone_manifest.json",
    ),
    "artifacts/manifests/source_overlap_manifest.json": (
        "artifacts/research/physical/run_id=*/source_overlap_manifest.json",
    ),
    "artifacts/manifests/split_manifest.json": (
        "artifacts/real_forecasting/*/manifests/split_manifest.json",
        "artifacts/research/physical/run_id=*/physical_split_manifest.json",
    ),
    "artifacts/manifests/cohort_mapping_manifest.json": (),
    "artifacts/manifests/feature_manifest.json": (
        "artifacts/real_forecasting/*/manifests/input_audits.json",
        "artifacts/real_forecasting/*/manifests/model_manifest.json",
    ),
    "artifacts/manifests/model_manifest.json": (
        "artifacts/real_forecasting/*/manifests/model_manifest.json",
    ),
    "artifacts/manifests/physical_calibration_manifest.json": (
        "artifacts/research/physical/run_id=*/physical_calibration_manifest.json",
        "artifacts/research/physical/run_id=*/physical_branch_manifest.json",
    ),
    "artifacts/manifests/nyiso_da_calibration_manifest.json": (
        "artifacts/real_forecasting_extensions/run_id=*/extension_run_manifest.json",
        "artifacts/real_forecasting_extensions/run_id=*/metrics/da_lbmp/**/*.json",
        "artifacts/real_forecasting/*/run_manifest.json",
        "artifacts/real_forecasting/*/manifests/model_manifest.json",
        "artifacts/real_forecasting/*/metrics/metrics.json",
        "artifacts/real_forecasting/*/predictions/nyiso_da_predictions.parquet",
    ),
    "artifacts/manifests/nyiso_rt_calibration_manifest.json": (
        "artifacts/real_forecasting_extensions/run_id=*/extension_run_manifest.json",
        "artifacts/real_forecasting_extensions/run_id=*/metrics/rt_lbmp/**/*.json",
        "artifacts/real_forecasting/*/run_manifest.json",
        "artifacts/real_forecasting/*/manifests/model_manifest.json",
        "artifacts/real_forecasting/*/metrics/metrics.json",
        "artifacts/real_forecasting/*/predictions/nyiso_rt_predictions.parquet",
    ),
    "artifacts/manifests/workload_calibration_manifest.json": (
        "artifacts/real_forecasting_extensions/run_id=*/extension_run_manifest.json",
        "artifacts/real_forecasting_extensions/run_id=*/metrics/nlr_workload/**/*.json",
        "artifacts/real_forecasting/*/run_manifest.json",
        "artifacts/real_forecasting/*/manifests/model_manifest.json",
        "artifacts/real_forecasting/*/metrics/metrics.json",
        "artifacts/real_forecasting/*/predictions/nlr_workload_predictions.parquet",
    ),
    "artifacts/manifests/digital_twin_transfer_manifest.json": (
        "artifacts/research/physical/run_id=*/digital_twin_transfer_manifest.json",
    ),
    "artifacts/manifests/xcel_counterfactual_manifest.json": (
        "artifacts/manifests/xcel_counterfactual_manifest.json",
    ),
    "artifacts/manifests/resume_manifest.json": (
        "artifacts/state/pipeline_state.json",
        "artifacts/real_forecasting_extensions/run_id=*/extension_run_manifest.json",
        "artifacts/optimization_branch/*/summary.json",
        "artifacts/real_forecasting/*/run_manifest.json",
        "artifacts/research/optimization/*/summary.json",
    ),
    "artifacts/manifests/robustness_manifest.json": (
        "artifacts/research/physical/run_id=*/physical_transfer_ablation_manifest.json",
        "artifacts/optimization_branch/*/sensitivity_plan.json",
        "artifacts/optimization_branch/*/optimization_sensitivity.csv",
        "artifacts/research/optimization/*/sensitivity_plan.json",
        "artifacts/research/optimization/*/optimization_sensitivity.csv",
    ),
    "artifacts/manifests/scenario_manifest.json": (
        "artifacts/optimization_branch/*/scenario_*.json",
        "artifacts/optimization_branch/*/scenario_*.csv",
        "artifacts/research/optimization/*/scenario_*.json",
        "artifacts/research/optimization/*/scenario_*.csv",
    ),
    "artifacts/manifests/solver_manifest.json": (
        "artifacts/optimization_branch/*/summary.json",
        "artifacts/optimization_branch/*/optimization_results.csv",
        "artifacts/research/optimization/*/summary.json",
        "artifacts/research/optimization/*/optimization_results.csv",
    ),
    "artifacts/manifests/dart_value_select_manifest.json": (
        "artifacts/dart_value_select/*/manifests/dart_pipeline_manifest.json",
        "artifacts/dart_value_select/*/manifests/dart_value_select_manifest.json",
        "artifacts/dart_value_select/*/dart_value_select_manifest.json",
    ),
    "artifacts/manifests/dart_replay_manifest.json": (
        "artifacts/dart_value_select/*/manifests/dart_replay_manifest.json",
        "artifacts/dart_replay/*/replay_manifest.json",
    ),
    "artifacts/manifests/dart_statistical_selection_manifest.json": (
        "artifacts/dart_value_select/*/manifests/dart_statistical_selection_manifest.json",
        "artifacts/dart_value_select/*/statistical_selection.json",
    ),
    "artifacts/tables/nlr_twin_validation.csv": (
        "artifacts/research/physical/run_id=*/nlr_physical_validation_metrics.parquet",
    ),
    "artifacts/tables/nlr_pue_validation.csv": (
        "artifacts/research/physical/run_id=*/pue_validation.parquet",
    ),
    "artifacts/tables/nlr_parameter_uncertainty.csv": (
        "artifacts/research/physical/run_id=*/uncertainty_calibration.parquet",
    ),
    "artifacts/tables/energy_coverage_by_partition.csv": (
        "artifacts/research/physical/run_id=*/nlr295_energy_coverage.parquet",
    ),
    "artifacts/tables/pue_component_closure.csv": (
        "artifacts/research/physical/run_id=*/component_closure.parquet",
    ),
    "artifacts/tables/nyiso_da_forecast_main.csv": (
        "artifacts/real_forecasting/*/metrics/metrics.json",
    ),
    "artifacts/tables/nyiso_rt_forecast_main.csv": (
        "artifacts/real_forecasting/*/metrics/metrics.json",
    ),
    "artifacts/tables/workload_forecast_main.csv": (
        "artifacts/real_forecasting/*/metrics/metrics.json",
    ),
    "artifacts/tables/calibration_summary.csv": (
        "artifacts/real_forecasting/*/metrics/metrics.json",
    ),
    "artifacts/tables/digital_twin_transfer.csv": (
        "artifacts/research/physical/run_id=*/digital_twin_transfer_parameters.parquet",
    ),
    "artifacts/tables/xcel_local_counterfactual.csv": (
        "artifacts/tables/xcel_local_counterfactual.csv",
    ),
    "artifacts/tables/counterfactual_optimization_main.csv": (
        "artifacts/optimization_branch/*/optimization_results.csv",
        "artifacts/research/optimization/*/optimization_results.csv",
    ),
    "artifacts/tables/optimization_sensitivity.csv": (
        "artifacts/optimization_branch/*/optimization_sensitivity.csv",
        "artifacts/research/optimization/*/optimization_sensitivity.csv",
    ),
    "artifacts/tables/failures.csv": (
        "artifacts/optimization_branch/*/failures.csv",
        "artifacts/research/optimization/*/failures.csv",
    ),
    "artifacts/tables/hypothesis_tests.csv": (
        "artifacts/optimization_branch/*/paired_policy_inference.csv",
        "artifacts/optimization_branch/*/hypothesis_tests.csv",
        "artifacts/research/optimization/*/paired_policy_inference.csv",
        "artifacts/research/optimization/*/hypothesis_tests.csv",
    ),
    "artifacts/tables/dart_probability_quality.csv": (
        "artifacts/dart_value_select/*/tables/dart_probability_quality.csv",
        "artifacts/dart_value_select/*/probability_quality.json",
        "artifacts/dart_value_select/*/probability_quality.csv",
    ),
    "artifacts/tables/dart_decision_value.csv": (
        "artifacts/dart_value_select/*/tables/dart_decision_value.csv",
        "artifacts/dart_replay/*/results/**/*.json",
        "artifacts/dart_value_select/*/replay/**/*.json",
    ),
    "artifacts/tables/dart_selection_methods.csv": (
        "artifacts/dart_value_select/*/tables/dart_selection_methods.csv",
        "artifacts/dart_value_select/*/manifests/dart_statistical_selection_manifest.json",
        "artifacts/dart_value_select/*/selection_methods.csv",
        "artifacts/dart_value_select/*/selection_result.json",
    ),
    "artifacts/tables/dart_regime_value.csv": (
        "artifacts/dart_value_select/*/tables/dart_regime_value.csv",
        "artifacts/dart_replay/*/regime_value.csv",
        "artifacts/dart_replay/*/results/**/*.json",
        "artifacts/dart_value_select/*/replay/**/*.json",
    ),
}


SOURCE_BASENAME_ALIASES: dict[str, tuple[str, ...]] = {
    "hypothesis_tests.csv": (
        "hypothesis_tests.csv",
        "paired_policy_inference.csv",
    ),
    "counterfactual_optimization_main.csv": ("optimization_results.csv",),
    "optimization_sensitivity.csv": ("optimization_sensitivity.csv",),
    "scenario_diagnostics.csv": ("scenario_diagnostics.csv",),
    "scenario_dependence.csv": ("scenario_dependence.csv",),
    "scenario_reduction.csv": ("scenario_reduction.csv",),
    "extreme_cooccurrence.csv": ("extreme_cooccurrence.csv",),
    "failures.csv": ("failures.csv",),
    "dart_probability_quality.csv": (
        "dart_probability_quality.csv",
        "probability_quality.csv",
    ),
    "dart_decision_value.csv": ("dart_decision_value.csv", "decision_value.csv"),
    "dart_selection_methods.csv": (
        "dart_selection_methods.csv",
        "selection_methods.csv",
        "dart_statistical_selection_manifest.json",
    ),
    "dart_regime_value.csv": ("dart_regime_value.csv", "regime_value.csv"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _atomic_copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    return target


def _expand_patterns(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        candidates = (
            sorted(root.glob(normalized))
            if any(character in normalized for character in "*?[")
            else [root / normalized]
        )
        for candidate in candidates:
            if candidate.is_file():
                key = str(candidate.resolve()).lower()
                if key not in seen:
                    paths.append(candidate)
                    seen.add(key)
    return paths


def _collect_declared_paths(value: Any, root: Path) -> list[Path]:
    output: list[Path] = []
    seen: set[int] = set()

    def visit(nested: Any) -> None:
        if isinstance(nested, Mapping):
            identity = id(nested)
            if identity in seen:
                return
            seen.add(identity)
            for key, child in nested.items():
                if isinstance(child, (str, Path)) and (
                    str(key).lower().endswith(("path", "file", "table", "manifest"))
                    or Path(str(child)).suffix.lower()
                    in {".json", ".csv", ".parquet", ".png", ".md", ".html"}
                ):
                    candidate = Path(str(child))
                    if not candidate.is_absolute():
                        candidate = root / candidate
                    if candidate.is_file():
                        output.append(candidate)
                else:
                    visit(child)
        elif isinstance(nested, (list, tuple)):
            for child in nested:
                visit(child)

    visit(value)
    unique: dict[str, Path] = {}
    for path in output:
        unique[str(path.resolve()).lower()] = path
    return list(unique.values())


def _explicit_mapping(
    branch_results: Mapping[str, Any], spec: ArtifactSpec
) -> Mapping[str, Any] | None:
    mappings = branch_results.get("artifact_mappings", {})
    if not isinstance(mappings, Mapping):
        return None
    keys = (
        spec.relative_path,
        Path(spec.relative_path).name,
        Path(spec.relative_path).stem,
        spec.artifact_id,
    )
    for key in keys:
        if key not in mappings:
            continue
        value = mappings[key]
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, (str, Path)):
            return {"source": str(value)}
        if isinstance(value, (list, tuple)):
            return {"sources": [str(item) for item in value]}
        raise TypeError(f"artifact mapping for {key} has an unsupported type")
    return None


_BAD_STATUS_TOKENS = {
    "blocked",
    "failed",
    "failure",
    "partial",
    "pending",
    "in_progress",
    "not_fitted_preflight_ready",
    "not_instantiated_calibration_pending",
    "paper_preflight_ready",
    "missing_mandatory",
    "invalid",
}
_GOOD_STATUS_TOKENS = {
    "complete",
    "completed",
    "success",
    "succeeded",
    "passed",
    "computed",
    "frozen",
}
_PLACEHOLDER_MARKERS = (
    "pending_real_experiment",
    "paper results available: `false`",
    '"paper_results_available": false',
    "synthetic fallback: `true`",
    "synthetic fallback: true",
    '"synthetic_fallback": true',
    '"synthetic_fallback_used": true',
    "demo-only",
    "prototype_only",
)

_DART_MANIFEST_NAMES = {
    "dart_value_select_manifest.json",
    "dart_replay_manifest.json",
    "dart_statistical_selection_manifest.json",
}


def _dart_partial_contract(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate the sole admissible reduced DART pool, if one is declared.

    This helper intentionally does not infer readiness from a generic PARTIAL
    status.  Presence of either partial field requires the exact self-hashed
    iTransformer three-failure contract and its permanent paper-status cap.
    Callers separately constrain which manifest kinds may remain internally
    COMPLETE while carrying this downstream contract.
    """

    value = payload.get("candidate_pool_contract")
    cap = payload.get("terminal_status_cap")
    if value is None and cap in (None, ""):
        return None
    if value is None:
        raise ValueError("candidate_pool_contract is absent")
    if cap != PARTIAL_CANDIDATE_TERMINAL_CAP:
        raise ValueError("terminal_status_cap is not PARTIAL_PAPER")
    return validate_partial_candidate_pool_contract(value)


def _dart_manifest_partial_mode(
    payload: Mapping[str, Any],
    *,
    complete_with_partial_contract_allowed: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a strict partial contract and a validation error, if any."""

    status = _status_token(payload.get("status"))
    try:
        contract = _dart_partial_contract(payload)
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    if status == _status_token(PARTIAL_CANDIDATE_STATUS):
        if contract is None:
            return None, "partial DART status lacks the exact candidate-pool contract"
        return contract, None
    if contract is not None and not (
        complete_with_partial_contract_allowed and status == "complete"
    ):
        return None, "partial candidate contract is incompatible with manifest status"
    return contract, None


def _contains_prohibited_dart_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in {
                "demo",
                "synthetic",
                "synthetic_fixture",
                "synthetic_fallback",
                "synthetic_fallback_used",
                "prototype_only",
            } and nested is True:
                return True
            if _contains_prohibited_dart_marker(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_prohibited_dart_marker(item) for item in value)
    return False


def _project_root_for(path: Path) -> Path:
    current = path.resolve()
    for parent in (current, *current.parents):
        if parent.name.lower() == "artifacts":
            return parent.parent
    return path.resolve().parent


def _dart_run_final_manifest(path: Path) -> Path | None:
    for parent in path.resolve().parents:
        candidate = parent / "manifests" / "dart_value_select_manifest.json"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_dart_output_path(manifest_path: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    run_root = manifest_path.parent.parent
    from_run = (run_root / candidate).resolve()
    if from_run.is_file():
        return from_run
    return (_project_root_for(manifest_path) / candidate).resolve()


def _verify_dart_final_outputs(
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> tuple[bool, str]:
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False, "DART final manifest has no hashed output list"
    seen: set[str] = set()
    for record in outputs:
        if not isinstance(record, Mapping):
            return False, "DART final manifest contains a malformed output record"
        output = _resolve_dart_output_path(manifest_path, record.get("path"))
        expected = str(record.get("sha256", "")).lower()
        if len(expected) != 64 or not output.is_file() or _sha256(output) != expected:
            return False, f"DART output hash is invalid for {record.get('path')}"
        seen.add(output.name)
    required = {
        "dart_probability_quality.csv",
        "dart_decision_value.csv",
        "dart_selection_methods.csv",
        "dart_regime_value.csv",
        "dart_replay_manifest.json",
        "dart_statistical_selection_manifest.json",
    }
    if not required.issubset(seen):
        return False, "DART final manifest does not pin all required tables/manifests"
    return True, "all DART final-manifest outputs are hash-clean"


def _verify_dart_source_listed_by_final(path: Path) -> tuple[bool, str]:
    final_path = _dart_run_final_manifest(path)
    if final_path is None or final_path == path.resolve():
        return False, "DART source has no sibling final manifest provenance"
    try:
        final = json.loads(final_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid DART final manifest: {type(exc).__name__}"
    if not isinstance(final, Mapping):
        return False, "DART sibling final manifest is not an object"
    final_contract, contract_error = _dart_manifest_partial_mode(final)
    final_status = _status_token(final.get("status"))
    if contract_error is not None or final_status not in {
        "complete",
        _status_token(PARTIAL_CANDIDATE_STATUS),
    }:
        return False, "DART sibling final manifest is neither COMPLETE nor the exact admitted partial run"
    if final_status == "complete" and final_contract is not None:
        return False, "COMPLETE DART sibling final manifest carries a partial pool"
    valid, reason = _verify_dart_final_outputs(final_path, final)
    if not valid:
        return valid, reason
    try:
        source_payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid DART source manifest: {type(exc).__name__}"
    if not isinstance(source_payload, Mapping):
        return False, "DART source manifest is not an object"
    source_contract, source_error = _dart_manifest_partial_mode(
        source_payload,
        complete_with_partial_contract_allowed=True,
    )
    if source_error is not None:
        return False, f"DART source partial contract is invalid: {source_error}"
    final_pool_sha = (
        final_contract.get("contract_sha256") if final_contract is not None else None
    )
    source_pool_sha = (
        source_contract.get("contract_sha256") if source_contract is not None else None
    )
    if source_pool_sha != final_pool_sha:
        return False, "DART source and sibling final candidate pools differ"
    target = path.resolve()
    for record in final.get("outputs", []):
        if (
            isinstance(record, Mapping)
            and _resolve_dart_output_path(final_path, record.get("path")) == target
            and str(record.get("sha256", "")).lower() == _sha256(target)
        ):
            qualifier = (
                "exact PARTIAL_PAPER-capped"
                if final_contract is not None
                else "COMPLETE"
            )
            return True, f"source is hash-pinned by the {qualifier} DART final manifest"
    return False, "DART source is not hash-pinned by its final manifest"


def _pipeline_record_path(manifest_path: Path, record: Mapping[str, Any]) -> Path:
    value = record.get("path")
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    return (_project_root_for(manifest_path) / candidate).resolve()


def _self_hashed_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    core = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != _json_sha256(core):
        return None
    return value


def _verify_anchor_pipeline_outputs(
    root: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    resolved: Mapping[str, Path],
    readiness: Mapping[str, Any],
    candidate_pool_contract: Mapping[str, Any] | None = None,
) -> str | None:
    readiness_config = config.get("pre_dart_v1_readiness", {})
    amendment = (
        readiness_config.get("anchor_only_checkpoint_amendment")
        if isinstance(readiness_config, Mapping)
        else None
    )
    if (
        not isinstance(amendment, Mapping)
        or amendment.get("enabled") is not True
        or not isinstance(readiness.get("anchor_import_evidence"), Mapping)
    ):
        return None
    required = {
        "v3_anchor_snapshot_manifest",
        "v4_anchor_import_ledger",
        "v4_epoch_archive_inventory",
        "selection_training_manifest",
    }
    if not required.issubset(resolved):
        return "DART v4 pipeline omits anchor/import/archive provenance outputs"
    run_ids = payload.get("run_ids")
    destination = str(amendment.get("destination_run_id", ""))
    if (
        not isinstance(run_ids, Mapping)
        or run_ids.get("dart_forecast") != destination
        or payload.get("anchor_training_lineage_required") is not True
    ):
        return "DART pipeline does not pin the frozen forecast-v4 lineage"
    selection = _self_hashed_json(resolved["selection_training_manifest"])
    # Training-stage manifests use manifest_fingerprint rather than manifest_sha256.
    if selection is None:
        try:
            selection = json.loads(
                resolved["selection_training_manifest"].read_text(encoding="utf-8")
            )
        except Exception:
            selection = None
    if not isinstance(selection, Mapping):
        return "DART forecast-v4 selection manifest is invalid"
    selection_core = {
        key: value for key, value in selection.items() if key != "manifest_fingerprint"
    }
    selection_status = _status_token(selection.get("status"))
    if candidate_pool_contract is None:
        selection_admitted = (
            selection_status == "complete"
            and selection.get("downstream_ready") is True
            and selection.get("candidate_pool_contract") is None
        )
    else:
        try:
            selection_contract = validate_partial_candidate_pool_contract(
                selection.get("candidate_pool_contract"),
                failed_units=selection.get("failed_units"),
                require_failed_units=True,
            )
        except (TypeError, ValueError):
            selection_contract = None
        selection_admitted = (
            selection_status == _status_token(PARTIAL_CANDIDATE_STATUS)
            and selection.get("downstream_ready") is False
            and selection.get("partial_candidate_downstream_ready") is True
            and selection.get("terminal_status_cap")
            == PARTIAL_CANDIDATE_TERMINAL_CAP
            and selection_contract is not None
            and selection_contract.get("contract_sha256")
            == candidate_pool_contract.get("contract_sha256")
            and int(selection.get("expected_unit_count", -1)) == 126
            and int(selection.get("completed_unit_count", -1)) == 123
            and int(selection.get("eligible_expected_unit_count", -1)) == 108
            and int(selection.get("eligible_completed_unit_count", -1)) == 108
            and int(selection.get("failed_unit_count", -1)) == 3
            and int(selection.get("date_drop_count", -1)) == 0
            and selection.get("common_date_panel_preserved") is True
        )
    if (
        not selection_admitted
        or selection.get("run_id") != destination
        or selection.get("manifest_fingerprint") != _json_sha256(selection_core)
    ):
        return "DART forecast-v4 selection stage is not hash-clean under the admitted candidate pool"
    overlay = selection.get("anchor_overlay_recovery")
    compatibility = selection.get("anchor_only_compatibility")
    embedded_inventory = selection.get("v4_full_epoch_archive_inventory")
    if (
        not isinstance(overlay, Mapping)
        or overlay.get("status") != "V4_OVERLAY_PREPARED_NO_TRAINING_LAUNCHED"
        or overlay.get("training_launched_by_prepare") is not False
        or overlay.get("v3_files_moved_modified_or_deleted") != 0
        or not isinstance(compatibility, Mapping)
        or compatibility.get(
            "provisional_candidate_grid_changed_to_exact_five_epoch_anchors"
        )
        is not True
        or compatibility.get("arbitrary_non_anchor_replay_forbidden") is not True
        or not isinstance(embedded_inventory, Mapping)
    ):
        return "DART forecast-v4 stage omits its anchor/archive compatibility evidence"

    snapshot = _self_hashed_json(resolved["v3_anchor_snapshot_manifest"])
    if (
        snapshot is None
        or snapshot.get("status") != "COMPLETE_HASH_PINNED_READ_ONLY_SOURCE"
        or snapshot.get("source_run_id") != amendment.get("source_run_id")
        or snapshot.get("destination_run_id") != destination
        or snapshot.get("records_sha256")
        != amendment.get("source_anchor_records_sha256")
        or int(snapshot.get("record_count", -1))
        != int(amendment.get("source_anchor_record_count", -2))
        or int(snapshot.get("total_bytes", -1))
        != int(amendment.get("source_anchor_total_bytes", -2))
        or snapshot.get("completed_oos_unit_count") != 0
        or snapshot.get("copy_hardlink_or_source_rewrite_performed") is not False
        or amendment.get("source_snapshot_manifest_file_sha256")
        not in (None, _sha256(resolved["v3_anchor_snapshot_manifest"]))
        or amendment.get("source_snapshot_manifest_byte_count")
        not in (None, resolved["v3_anchor_snapshot_manifest"].stat().st_size)
        or amendment.get("source_snapshot_manifest_sha256")
        not in (None, snapshot.get("manifest_sha256"))
    ):
        return "DART v3 immutable anchor snapshot contract/hash changed"
    anchor_evidence = readiness.get("anchor_import_evidence")
    if (
        not isinstance(anchor_evidence, Mapping)
        or anchor_evidence.get("snapshot_manifest_file_sha256")
        != _sha256(resolved["v3_anchor_snapshot_manifest"])
        or anchor_evidence.get("snapshot_manifest_sha256")
        != snapshot.get("manifest_sha256")
    ):
        return "v8 readiness does not pin the DART v3 anchor snapshot"

    imported = _self_hashed_json(resolved["v4_anchor_import_ledger"])
    if (
        imported is None
        or imported.get("status") != "V4_OVERLAY_IMPORTED_NO_TRAINING_LAUNCHED"
        or imported.get("v3_snapshot_manifest_sha256")
        != _sha256(resolved["v3_anchor_snapshot_manifest"])
        or imported.get("v3_files_moved_modified_or_deleted") != 0
        or imported.get(
            "training_not_launched_during_import",
            imported.get("training_launched") is False,
        )
        is not True
    ):
        return "DART v4 anchor import ledger contract/hash changed"
    inventory = _self_hashed_json(resolved["v4_epoch_archive_inventory"])
    if (
        inventory is None
        or inventory.get("status") != "COMPLETE"
        or inventory.get("every_completed_epoch_full_state_persisted") is not True
        or inventory.get("formal_selection_candidates_divisible_by_five_only")
        is not True
        or int(inventory.get("track_count", -1))
        != len(inventory.get("track_manifests", ()))
        or inventory.get("track_manifests_sha256")
        != _json_sha256(inventory.get("track_manifests", ()))
    ):
        return "DART v4 full-epoch archive inventory contract/hash changed"
    if (
        embedded_inventory.get("sha256")
        != _sha256(resolved["v4_epoch_archive_inventory"])
        or embedded_inventory.get("manifest_sha256")
        != inventory.get("manifest_sha256")
        or embedded_inventory.get("track_manifests_sha256")
        != inventory.get("track_manifests_sha256")
    ):
        return "DART forecast-v4 stage does not pin its archive inventory"
    for record in inventory.get("track_manifests", ()):
        if not isinstance(record, Mapping):
            return "DART v4 full-epoch archive inventory contains a malformed record"
        track_path = Path(str(record.get("path", "")))
        if not track_path.is_absolute():
            track_path = root / track_path
        track = _self_hashed_json(track_path)
        if (
            track is None
            or record.get("sha256") != _sha256(track_path)
            or track.get("status") != "COMPLETE_THROUGH_CURRENT_EPOCH"
            or track.get("every_completed_epoch_full_state_persisted") is not True
            or track.get("formal_selection_candidates_divisible_by_five_only")
            is not True
        ):
            return "a DART v4 full-epoch archive track contract/hash changed"
    return None


def _verify_dart_pipeline_manifest(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[bool, str | None, str]:
    status = _status_token(payload.get("status"))
    candidate_pool_contract, contract_error = _dart_manifest_partial_mode(payload)
    partial_status = _status_token(PARTIAL_CANDIDATE_STATUS)
    status_admitted = (
        status == "complete" and candidate_pool_contract is None
    ) or (
        status == partial_status and candidate_pool_contract is not None
    )
    if (
        contract_error is not None
        or not status_admitted
        or payload.get("completed") is not True
        or payload.get("downstream_ready") is not True
        or payload.get("paper_results_available") is not True
        or payload.get("synthetic_fallback_used") is not False
        or payload.get("selection_operating_date_count") != 1096
        or payload.get("locked_operating_date_count") != 365
    ):
        return False, status, (
            "DART pipeline manifest is neither paper-ready COMPLETE nor the exact "
            f"PARTIAL_PAPER-capped candidate run: {contract_error or 'status/fields invalid'}"
        )
    if candidate_pool_contract is not None and (
        int(payload.get("attempted_candidate_count", -1)) != 9
        or int(payload.get("candidate_count", -1)) != 8
    ):
        return False, status, "DART partial pipeline does not preserve attempted9/eligible8"
    recorded_manifest_hash = payload.get("manifest_sha256")
    core = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded_manifest_hash != _json_sha256(core):
        return False, status, "DART pipeline manifest content hash is invalid"
    root = _project_root_for(path)
    pipeline_config = Path(str(payload.get("config_path", "")))
    if not pipeline_config.is_absolute():
        pipeline_config = root / pipeline_config
    pipeline_config_sha = payload.get("config_sha256")
    if (
        not pipeline_config.is_file()
        or not isinstance(pipeline_config_sha, str)
        or len(pipeline_config_sha) != 64
        or _sha256(pipeline_config) != pipeline_config_sha
    ):
        return False, status, "DART pipeline config bytes changed"
    outputs = payload.get("outputs")
    required = {
        "v1_readiness_manifest",
        "dart_value_select_manifest",
        "dart_replay_manifest",
        "dart_statistical_selection_manifest",
    }
    if not isinstance(outputs, Mapping) or not required.issubset(outputs):
        return False, status, "DART pipeline manifest omits required provenance outputs"
    resolved: dict[str, Path] = {}
    for name, record in outputs.items():
        if not isinstance(record, Mapping):
            return False, status, f"DART pipeline output {name} is malformed"
        output = _pipeline_record_path(path, record)
        expected = str(record.get("sha256", "")).lower()
        if len(expected) != 64 or not output.is_file() or _sha256(output) != expected:
            return False, status, f"DART pipeline output hash changed: {name}"
        resolved[str(name)] = output
    try:
        readiness = json.loads(
            resolved["v1_readiness_manifest"].read_text(encoding="utf-8")
        )
    except Exception as exc:
        return False, status, f"invalid v1 DART readiness manifest: {type(exc).__name__}"
    if not isinstance(readiness, Mapping):
        return False, status, "v1 DART readiness manifest is not an object"
    try:
        pipeline_config_payload = load_yaml(pipeline_config)
    except (OSError, ValueError, TypeError):
        return False, status, "DART pipeline config is not valid YAML"
    anchor_reason = _verify_anchor_pipeline_outputs(
        root,
        payload,
        pipeline_config_payload,
        resolved,
        readiness,
        candidate_pool_contract,
    )
    if anchor_reason is not None:
        return False, status, anchor_reason
    gates = readiness.get("completion_gates")
    policy_sha = readiness.get("frozen_policy_sha256")
    v1_sha = readiness.get("v1_forecast_manifest_sha256")
    if (
        readiness.get("status") != "COMPLETE"
        or readiness.get("formal_dart_training_allowed") is not True
        or readiness.get("native_rt_cross_head_label")
        != "AUXILIARY_NOT_DA_GATE_ELIGIBLE"
        or readiness.get("native_rt_selection_eligible") is not False
        or not isinstance(policy_sha, str)
        or len(policy_sha) != 64
        or not isinstance(v1_sha, str)
        or len(v1_sha) != 64
        or not isinstance(gates, Mapping)
    ):
        return False, status, "v1 readiness does not prove the frozen auxiliary-only gate"
    readiness_fingerprint = readiness.get("manifest_fingerprint")
    readiness_core = {
        key: value for key, value in readiness.items() if key != "manifest_fingerprint"
    }
    if readiness_fingerprint != _json_sha256(readiness_core):
        return False, status, "v1 readiness manifest fingerprint is invalid"
    readiness_outputs = readiness.get("outputs")
    if not isinstance(readiness_outputs, list) or not readiness_outputs:
        return False, status, "v1 readiness manifest has no hashed outputs"
    for record in readiness_outputs:
        if not isinstance(record, Mapping):
            return False, status, "v1 readiness output provenance is malformed"
        output = Path(str(record.get("path", "")))
        if not output.is_absolute():
            output = root / output
        if (
            not output.is_file()
            or str(record.get("sha256", "")).lower() != _sha256(output)
        ):
            return False, status, "v1 readiness output hash changed"
    v1_manifest = Path(str(readiness.get("v1_forecast_manifest_path", "")))
    if not v1_manifest.is_absolute():
        v1_manifest = root / v1_manifest
    policy_freeze = Path(str(readiness.get("policy_freeze_path", "")))
    if not policy_freeze.is_absolute():
        policy_freeze = root / policy_freeze
    if (
        not v1_manifest.is_file()
        or _sha256(v1_manifest) != v1_sha
        or not policy_freeze.is_file()
        or _sha256(policy_freeze)
        != str(readiness.get("policy_freeze_file_sha256", "")).lower()
    ):
        return False, status, "v1 readiness does not hash-pin v1/policy-freeze bytes"
    try:
        frozen_policy = json.loads(policy_freeze.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, status, f"invalid v1 readiness policy freeze: {type(exc).__name__}"
    if not isinstance(frozen_policy, Mapping):
        return False, status, "v1 readiness policy freeze is not an object"
    policy_contract = frozen_policy.get("contract")
    if (
        not isinstance(policy_contract, Mapping)
        or frozen_policy.get("frozen_policy_sha256") != policy_sha
        or _json_sha256(policy_contract) != policy_sha
        or (
            frozen_policy.get("frozen_before_v1_complete_manifest") is not True
            and not (
                frozen_policy.get(
                    "post_result_structural_missingness_and_three_seed_amendment"
                )
                is True
                and frozen_policy.get(
                    "validation_informed_selection_method_amendment"
                )
                is True
            )
        )
    ):
        return False, status, "v1 readiness frozen policy contract is invalid"
    frozen_config = Path(str(policy_contract.get("config_path", "")))
    if not frozen_config.is_absolute():
        frozen_config = root / frozen_config
    evaluator_path = Path(__file__).resolve().parent / "experiments/v1_dart_readiness.py"
    if (
        not frozen_config.is_file()
        or policy_contract.get("config_sha256") != pipeline_config_sha
        or _sha256(frozen_config) != pipeline_config_sha
        or not evaluator_path.is_file()
        or policy_contract.get("evaluator_code_sha256") != _sha256(evaluator_path)
    ):
        return False, status, "v1 frozen readiness config/code bytes changed"
    pre_result_amendment = frozen_policy.get("pre_result_amendment")
    if not isinstance(pre_result_amendment, bool):
        return False, status, "v1 readiness policy amendment flag is absent"
    post_result_amendment = (
        frozen_policy.get(
            "post_result_structural_missingness_and_three_seed_amendment"
        )
        is True
    )
    method_amendment = (
        frozen_policy.get("validation_informed_selection_method_amendment") is True
    )
    anchor_amendment = frozen_policy.get("anchor_only_checkpoint_amendment") is True
    policy_sections = policy_contract.get("policy_sections")
    readiness_policy = (
        policy_sections.get("pre_dart_v1_readiness")
        if isinstance(policy_sections, Mapping)
        else None
    )
    if (
        pre_result_amendment
        or post_result_amendment
        or method_amendment
        or anchor_amendment
    ):
        superseded = frozen_policy.get("superseded_freeze")
        if not isinstance(superseded, Mapping):
            return False, status, "v1 superseded policy-freeze record is absent"
        superseded_path = Path(str(superseded.get("path", "")))
        if not superseded_path.is_absolute():
            superseded_path = root / superseded_path
        if (
            not superseded_path.is_file()
            or superseded.get("file_sha256") != _sha256(superseded_path)
        ):
            return False, status, "v1 superseded policy-freeze bytes changed"
        ledger_path = Path(str(readiness.get("amendment_ledger_path", "")))
        if not ledger_path.is_absolute():
            ledger_path = root / ledger_path
        try:
            amendment_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, status, f"invalid readiness amendment ledger: {type(exc).__name__}"
        if not isinstance(amendment_ledger, Mapping):
            return False, status, "readiness amendment ledger is not an object"
        declared_ledger_sha = readiness.get("amendment_ledger_sha256")
        ledger_fingerprint = amendment_ledger.get("ledger_fingerprint")
        ledger_core = {
            key: value
            for key, value in amendment_ledger.items()
            if key != "ledger_fingerprint"
        }
        replacement = amendment_ledger.get("replacement")
        amendment_status = str(amendment_ledger.get("status", "")).upper()
        chronology_valid = (
            pre_result_amendment
            and amendment_status == "AMENDED_PRE_RESULT"
            and amendment_ledger.get("amended_before_v1_complete_manifest") is True
        ) or (
            post_result_amendment
            and amendment_status
            == "AMENDED_POST_RESULT_STRUCTURAL_MISSINGNESS_AND_THREE_SEED_PANEL"
            and amendment_ledger.get("amended_before_v1_complete_manifest") is False
            and amendment_ledger.get("performance_independent") is True
        ) or (
            anchor_amendment
            and amendment_status
            == str(
                readiness_policy.get("anchor_only_checkpoint_amendment", {}).get(
                    "amendment_status",
                    "AMENDED_PRE_OOS_ANCHOR_ONLY_CHECKPOINT_SELECTION_AFTER_FAIL_CLOSED_REPLAY_DEFECT",
                )
            ).upper()
            and amendment_ledger.get("performance_independent") is True
            and int(
                amendment_ledger.get("formal_dart_completed_oos_units", -1)
            )
            == 0
            and amendment_ledger.get("formal_dart_replay_started") is False
            and amendment_ledger.get("locked_2025_dart_metrics_consumed") is False
            and amendment_ledger.get(
                "market_replay_milp_action_or_value_results_consumed"
            )
            is False
        )
        method_boundary_valid = (
            not method_amendment
            or (
                amendment_ledger.get(
                    "selection_method_amendment_validation_informed"
                )
                is True
                and amendment_ledger.get("locked_2025_dart_metrics_consumed") is False
                and amendment_ledger.get("formal_dart_training_or_replay_existed")
                is False
            )
        )
        if (
            declared_ledger_sha != _sha256(ledger_path)
            or ledger_fingerprint != _json_sha256(ledger_core)
            or not chronology_valid
            or not method_boundary_valid
            or amendment_ledger.get("superseded") != superseded
            or not isinstance(replacement, Mapping)
            or replacement.get("frozen_policy_sha256") != policy_sha
            or replacement.get("file_sha256")
            != readiness.get("policy_freeze_file_sha256")
        ):
            return False, status, "v1 readiness amendment chain is invalid"
    required_gates = (
        readiness_policy.get("required_completion_gates")
        if isinstance(readiness_policy, Mapping)
        else None
    )
    if (
        not isinstance(required_gates, Sequence)
        or isinstance(required_gates, (str, bytes))
        or not required_gates
        or any(not isinstance(gate, str) or not gate for gate in required_gates)
        or len(set(required_gates)) != len(required_gates)
    ):
        return False, status, "v1 readiness policy has no valid frozen gate list"
    policy_schema = str(policy_contract.get("schema_version", ""))
    policy_version = (
        policy_schema.rsplit(".", 1)[-1] if "." in policy_schema else ""
    )
    readiness_schema = (
        str(readiness_policy.get("schema_version", ""))
        if isinstance(readiness_policy, Mapping)
        else ""
    )
    if (
        not policy_version
        or not readiness_schema.endswith(f".{policy_version}")
        or not str(readiness.get("schema_version", "")).endswith(
            f".{policy_version}"
        )
    ):
        return False, status, "readiness manifest/policy schema versions disagree"
    try:
        recovery_manifest = json.loads(v1_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, status, f"invalid v1 recovery manifest: {type(exc).__name__}"
    accepted_schemas = (
        readiness_policy.get("accepted_v1_manifest_schema_versions")
        if isinstance(readiness_policy, Mapping)
        else None
    )
    if (
        not isinstance(recovery_manifest, Mapping)
        or not isinstance(accepted_schemas, Sequence)
        or isinstance(accepted_schemas, (str, bytes))
        or not accepted_schemas
        or readiness_policy.get(
            "current_run_requires_hash_verified_recovery_lineage"
        )
        is not True
        or recovery_manifest.get("schema_version") not in accepted_schemas
        or recovery_manifest.get("run_id") != readiness_policy.get("recovery_run_id")
        or str(recovery_manifest.get("status", "")).upper() != "COMPLETE"
    ):
        return False, status, "v1 evidence is not the policy-accepted recovery lineage"
    recovery_requirements = readiness_policy.get("recovery_requirements")
    if not isinstance(recovery_requirements, Mapping):
        return False, status, "frozen v1 recovery requirements are absent"
    for key in (
        "formal_readiness_allowed",
        "terminal",
        "actionable_pending_count",
        "integrity_failures",
        "task_failures",
        "locked_2025_metrics_consumed",
        "locked_2025_metrics_materialized",
        "locked_2025_policy_decisions_consumed",
    ):
        if key in recovery_requirements and recovery_manifest.get(
            key
        ) != recovery_requirements.get(key):
            return False, status, f"v1 recovery requirement changed: {key}"
    recovery_counts = recovery_manifest.get("counts")
    if not isinstance(recovery_counts, Mapping):
        return False, status, "v1 recovery task counts are absent"
    for key in ("tasks_requested", "tasks_completed", "tasks_failed", "failed_seeds"):
        if key in recovery_requirements and recovery_counts.get(
            key
        ) != recovery_requirements.get(key):
            return False, status, f"v1 recovery count changed: {key}"
    if (
        len(gates) != len(required_gates)
        or set(gates) != set(required_gates)
        or any(gates.get(gate) is not True for gate in required_gates)
    ):
        return False, status, "v1 readiness gates differ from the frozen policy"
    required_disclosure = policy_contract.get("locked_test_exposure_disclosure")
    chronology_scope = policy_contract.get("audited_chronology_scope")
    if (
        required_disclosure in (None, "")
        or readiness.get("locked_test_exposure_disclosure") != required_disclosure
        or readiness.get("locked_2025_metrics_consumed") is not False
        or readiness.get("locked_2025_metrics_materialized") is not False
        or readiness.get("evaluation_split") != policy_contract.get("evaluation_split")
        or readiness.get("evaluation_calendar_years")
        != policy_contract.get("evaluation_calendar_years")
        or readiness.get("evaluation_year_timezone")
        != policy_contract.get("evaluation_year_timezone")
        or not isinstance(chronology_scope, Mapping)
        or readiness.get("audited_chronology_scope") != chronology_scope.get("claim")
        or readiness.get("forecast_training_seed_roles")
        != policy_contract.get("forecast_training_seed_roles")
        or readiness.get("all_candidate_replay_seed_slots")
        != policy_contract.get("all_candidate_replay_seed_slots")
        or readiness.get("performance_gate_applied") is not False
        or readiness.get("performance_results_may_update_policy") is not False
        or readiness.get("common_da_gate_retraining_required") is not True
    ):
        return False, status, "v1 readiness locked-test disclosure boundary changed"
    if payload.get("v1_readiness_policy_sha256") != policy_sha:
        return False, status, "pipeline/readiness frozen policy hashes disagree"
    stages = payload.get("stages", {})
    readiness_stage = (
        stages.get("v1_dart_readiness", {}) if isinstance(stages, Mapping) else {}
    )
    if not isinstance(readiness_stage, Mapping) or readiness_stage.get("passed") is not True:
        return False, status, "pipeline does not retain the passed pre-DART readiness audit"
    try:
        selector = json.loads(
            resolved["dart_value_select_manifest"].read_text(encoding="utf-8")
        )
    except Exception as exc:
        return False, status, f"invalid selector-owned DART manifest: {type(exc).__name__}"
    if not isinstance(selector, Mapping):
        return False, status, "selector-owned DART manifest is not an object"
    selector_contract, selector_error = _dart_manifest_partial_mode(selector)
    selector_status = _status_token(selector.get("status"))
    expected_selector_status = partial_status if candidate_pool_contract is not None else "complete"
    expected_pool_sha = (
        candidate_pool_contract.get("contract_sha256")
        if candidate_pool_contract is not None
        else None
    )
    observed_pool_sha = (
        selector_contract.get("contract_sha256")
        if selector_contract is not None
        else None
    )
    if (
        selector_error is not None
        or selector_status != expected_selector_status
        or observed_pool_sha != expected_pool_sha
        or (
            candidate_pool_contract is not None
            and (
                selector.get("completed") is not True
                or selector.get("downstream_ready") is not True
            )
        )
    ):
        return False, status, "selector-owned DART manifest does not preserve candidate admission"
    valid, reason = _verify_dart_final_outputs(
        resolved["dart_value_select_manifest"], selector
    )
    if not valid:
        return False, status, reason
    for output_name in (
        "dart_replay_manifest",
        "dart_statistical_selection_manifest",
    ):
        try:
            downstream = json.loads(resolved[output_name].read_text(encoding="utf-8"))
        except Exception as exc:
            return False, status, f"invalid {output_name}: {type(exc).__name__}"
        if not isinstance(downstream, Mapping):
            return False, status, f"{output_name} is not an object"
        downstream_contract, downstream_error = _dart_manifest_partial_mode(
            downstream,
            complete_with_partial_contract_allowed=True,
        )
        downstream_pool_sha = (
            downstream_contract.get("contract_sha256")
            if downstream_contract is not None
            else None
        )
        if (
            downstream_error is not None
            or _status_token(downstream.get("status")) != "complete"
            or downstream_pool_sha != expected_pool_sha
        ):
            return False, status, f"{output_name} loses the candidate-pool contract"
    return (
        True,
        status,
        (
            "pipeline hash-pins selector outputs and auxiliary-only pre-DART v1 readiness"
            if candidate_pool_contract is None
            else "pipeline hash-pins the unique PARTIAL_PAPER-capped eight-model evidence chain"
        ),
    )


def _inspect_dart_manifest(
    path: Path,
    payload: Mapping[str, Any],
    canonical_name: str,
) -> tuple[bool, str | None, str]:
    status = _status_token(payload.get("status"))
    partial_status = _status_token(PARTIAL_CANDIDATE_STATUS)
    if _contains_prohibited_dart_marker(payload):
        return False, status, "demo/synthetic/prototype DART evidence is forbidden"
    if payload.get("schema_version") == "canonical_paper_stage_manifest_v1":
        sources = payload.get("sources")
        wrapper_contract, wrapper_error = _dart_manifest_partial_mode(payload)
        wrapper_status_admitted = (
            status == "complete" and wrapper_contract is None
        ) or (
            status == partial_status and wrapper_contract is not None
        )
        if (
            wrapper_error is not None
            or not wrapper_status_admitted
            or payload.get("completed") is not True
            or not isinstance(sources, list)
            or not sources
            or not payload.get("claim_boundary")
        ):
            return False, status, "canonical DART manifest lacks completion/provenance metadata"
        root = _project_root_for(path)
        source_names: set[str] = set()
        pipeline_contract: dict[str, Any] | None = None
        recorded_partial_source_shas: set[str] = set()
        for record in sources:
            if not isinstance(record, Mapping) or record.get("valid") is not True:
                return False, status, "canonical DART manifest has invalid source provenance"
            source = Path(str(record.get("path", "")))
            if not source.is_absolute():
                source = root / source
            if not source.is_file() or str(record.get("sha256", "")) != _sha256(source):
                return False, status, "canonical DART manifest source hash changed"
            source_names.add(source.name)
            if source.suffix.lower() == ".json":
                try:
                    source_payload = json.loads(source.read_text(encoding="utf-8"))
                except Exception:
                    source_payload = None
                if isinstance(source_payload, Mapping):
                    source_contract, source_error = _dart_manifest_partial_mode(
                        source_payload,
                        complete_with_partial_contract_allowed=True,
                    )
                    if source_error is not None:
                        return False, status, (
                            "canonical DART manifest has invalid partial source: "
                            f"{source_error}"
                        )
                    if source_contract is not None:
                        recorded_partial_source_shas.add(
                            str(source_contract["contract_sha256"])
                        )
            if source.name == "dart_pipeline_manifest.json":
                try:
                    pipeline = json.loads(source.read_text(encoding="utf-8"))
                except Exception as exc:
                    return False, status, f"invalid recorded DART pipeline: {type(exc).__name__}"
                valid, _, reason = _verify_dart_pipeline_manifest(source, pipeline)
                if not valid:
                    return False, status, reason
                pipeline_contract, pipeline_error = _dart_manifest_partial_mode(
                    pipeline
                )
                if pipeline_error is not None:
                    return False, status, pipeline_error
        if canonical_name == "dart_value_select_manifest.json":
            readiness = payload.get("pre_dart_v1_readiness_evidence", {})
            if (
                not isinstance(readiness, Mapping)
                or readiness.get("role") != "AUXILIARY_NOT_DA_GATE_ELIGIBLE"
                or readiness.get("selection_eligible") is not False
                or not {
                    "dart_pipeline_manifest.json",
                    "dart_value_select_manifest.json",
                }.issubset(source_names)
            ):
                return False, status, "canonical DART manifest mislabels v1 readiness evidence"
        wrapper_pool_sha = (
            wrapper_contract.get("contract_sha256")
            if wrapper_contract is not None
            else None
        )
        pipeline_pool_sha = (
            pipeline_contract.get("contract_sha256")
            if pipeline_contract is not None
            else None
        )
        if "dart_pipeline_manifest.json" in source_names and (
            wrapper_pool_sha != pipeline_pool_sha
        ):
            return False, status, "canonical wrapper loses the DART candidate-pool contract"
        if wrapper_contract is not None and recorded_partial_source_shas != {
            str(wrapper_pool_sha)
        }:
            return False, status, "canonical partial wrapper does not pin its partial source contract"
        if wrapper_contract is None and recorded_partial_source_shas:
            return False, status, "canonical COMPLETE wrapper hides a partial source contract"
        return True, status, (
            "complete canonical DART provenance wrapper"
            if wrapper_contract is None
            else "valid PARTIAL_PAPER-capped canonical DART provenance wrapper"
        )
    if (
        canonical_name == "dart_value_select_manifest.json"
        and path.name == "dart_pipeline_manifest.json"
    ):
        return _verify_dart_pipeline_manifest(path, payload)
    complete_with_partial_contract = canonical_name in {
        "dart_replay_manifest.json",
        "dart_statistical_selection_manifest.json",
    }
    source_contract, source_error = _dart_manifest_partial_mode(
        payload,
        complete_with_partial_contract_allowed=complete_with_partial_contract,
    )
    source_status_admitted = (
        status == "complete"
        and (source_contract is None or complete_with_partial_contract)
    ) or (
        status == partial_status and source_contract is not None
    )
    if (
        source_error is not None
        or not source_status_admitted
        or payload.get("completed") is False
    ):
        return False, status, "formal DART manifest is not an admitted complete/partial artifact"
    if canonical_name == "dart_value_select_manifest.json":
        chronology = payload.get("chronology", {})
        if not isinstance(chronology, Mapping) or (
            chronology.get("selection_years") != [2022, 2023, 2024]
            or chronology.get("locked_test_year") != 2025
            or chronology.get("selection_frozen_before_locked_test") is not True
            or chronology.get("selection_updated_from_2025") is not False
        ):
            return False, status, "DART final manifest violates the locked chronology"
        valid, reason = _verify_dart_final_outputs(path, payload)
        return valid, status, reason
    if canonical_name == "dart_replay_manifest.json":
        experiments = payload.get("experiments", {})
        if (
            not isinstance(experiments, Mapping)
            or set(experiments) != {"A1", "A2", "A3"}
            or payload.get("common_exact_three_seed_panel") is not True
            or payload.get("counterfactual") is not True
            or payload.get("realized_commercial_profit_claim") is not False
        ):
            return False, status, "DART replay manifest lacks the A1/A2/A3 counterfactual contract"
    else:
        if (
            payload.get("selection_years") != [2022, 2023, 2024]
            or payload.get("locked_test_year") != 2025
            or payload.get("bootstrap_blocks_days") != [7, 14, 28]
            or payload.get("MCS_before_maximum_LCB") is not True
            or payload.get("selection_updated_from_2025") is not False
        ):
            return False, status, "DART statistical manifest violates OOS/LCB/MCS gates"
    valid, reason = _verify_dart_source_listed_by_final(path)
    return valid, status, reason


def _status_token(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower().replace(" ", "_")


def _inspect_json(path: Path, spec: ArtifactSpec) -> tuple[bool, str | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, None, f"invalid JSON: {type(exc).__name__}"
    if not isinstance(payload, Mapping) or not payload:
        return False, None, "JSON artifact is empty or not an object"
    canonical_name = Path(spec.relative_path).name
    if canonical_name in _DART_MANIFEST_NAMES:
        return _inspect_dart_manifest(path, payload, canonical_name)
    status = _status_token(payload.get("status"))
    if path.name == "final_manifest.json" and payload.get("schema_version") == FINAL_MANIFEST_SCHEMA_VERSION:
        return True, status, "final manifest is structurally materialized; its blocked status is reported separately"
    if path.name == "integrity_audit.json" and payload.get("schema_version") == INVENTORY_SCHEMA_VERSION:
        return True, status, "integrity audit truthfully records current blockers"
    if payload.get("synthetic_fallback_used") is True or payload.get("synthetic_fallback") is True:
        return False, status, "synthetic fallback cannot satisfy a paper artifact"
    if payload.get("paper_results_available") is False and spec.category == "manifest":
        return False, status, "manifest explicitly states paper results are unavailable"
    if status in _BAD_STATUS_TOKENS or (
        status is not None and any(token in status for token in _BAD_STATUS_TOKENS)
    ):
        return False, status, f"machine status is {payload.get('status')}"
    if payload.get("completed") is False:
        return False, status, "completed=false"
    if status is not None and (
        status in _GOOD_STATUS_TOKENS
        or any(status.startswith(token) for token in _GOOD_STATUS_TOKENS)
    ):
        return True, status, "machine status is complete"
    # Source inventories and evidence dictionaries legitimately have no run status.
    if any(
        key in payload
        for key in (
            "schema_version",
            "inventory",
            "sources",
            "tasks",
            "calibration_id",
            "source_relationship",
        )
    ):
        return True, status, "nonempty structured evidence artifact"
    return False, status, "JSON has no machine completion or recognized evidence schema"


def _inspect_csv(path: Path, spec: ArtifactSpec) -> tuple[bool, str | None, str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        return False, None, f"unreadable CSV: {type(exc).__name__}"
    lowered = text.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False, None, "CSV contains a placeholder/demo/pending marker"
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except Exception as exc:
        return False, None, f"invalid CSV: {type(exc).__name__}"
    if not rows:
        return False, None, "CSV has no data rows"
    forbidden = {"actual_savings", "realized_revenue", "grid_benefit"}
    columns = set(rows[0])
    if forbidden & columns:
        return False, None, f"forbidden unlabeled claim columns: {sorted(forbidden & columns)}"
    if Path(spec.relative_path).name == "counterfactual_optimization_main.csv":
        required_any = {
            "counterfactual_cost_difference_vs_baseline",
            "counterfactual",
            "evidence_role",
        }
        if not required_any <= columns:
            return False, None, "optimization main table lacks counterfactual boundary columns"
    return True, None, f"CSV contains {len(rows)} non-placeholder rows"


def _inspect_artifact(path: Path, spec: ArtifactSpec) -> tuple[bool, str | None, str]:
    if not path.is_file():
        return False, None, "file does not exist"
    suffix = path.suffix.lower()
    canonical_table_names = set(
        _FORECAST_TABLE_NAMES
        + _SCENARIO_TABLE_NAMES
        + _OPTIMIZATION_TABLE_NAMES
        + _DART_TABLE_NAMES
    )
    if (
        suffix == ".csv"
        and spec.category == "table"
        and path.name in canonical_table_names
    ):
        from .paper_tables_reports import validate_table_bundle

        valid, reason, _ = validate_table_bundle(path)
        return valid, None, reason
    if spec.category == "report" and path.name in _REPORT_NAMES:
        from .paper_tables_reports import validate_report_bundle

        valid, reason, payload = validate_report_bundle(path)
        return valid, _status_token(payload.get("status")), reason
    if suffix == ".json":
        return _inspect_json(path, spec)
    if suffix == ".csv":
        return _inspect_csv(path, spec)
    if suffix == ".png":
        from .paper_figures import validate_figure_bundle

        valid, reason, _ = validate_figure_bundle(path)
        return valid, None, reason
    if suffix in {".md", ".html"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        valid = len(text.strip()) > 40 and not any(
            marker in lowered for marker in _PLACEHOLDER_MARKERS
        )
        return valid, None, "non-placeholder report" if valid else "report is empty or pending"
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            rows = int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            try:
                import pandas as pd

                rows = len(pd.read_parquet(path))
            except Exception as exc:
                return False, None, f"unreadable Parquet: {type(exc).__name__}"
        return rows > 0, None, f"Parquet contains {rows} rows"
    return path.stat().st_size > 0, None, "nonempty source file"


def _source_record(root: Path, path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    valid, content_status, reason = _inspect_artifact(path, spec)
    return {
        "path": _relative(root, path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "valid": valid,
        "content_status": content_status,
        "inspection": reason,
    }


def _resolve_sources(
    root: Path,
    spec: ArtifactSpec,
    branch_results: Mapping[str, Any],
    declared_paths: Sequence[Path],
) -> tuple[list[Path], Mapping[str, Any] | None]:
    explicit = _explicit_mapping(branch_results, spec)
    patterns: list[str] = []
    if explicit is not None:
        source_value = explicit.get("sources", explicit.get("source", explicit.get("path")))
        if isinstance(source_value, (str, Path)):
            patterns.append(str(source_value))
        elif isinstance(source_value, (list, tuple)):
            patterns.extend(str(item) for item in source_value)
    patterns.extend(DEFAULT_SOURCE_CANDIDATES.get(spec.relative_path, ()))
    resolved: list[Path] = []
    seen: set[str] = set()
    for value in patterns:
        candidate = Path(value)
        if candidate.is_absolute() and not any(character in value for character in "*?["):
            candidates = [candidate] if candidate.is_file() else []
        else:
            candidates = _expand_patterns(root, (value,))
        for path in candidates:
            key = str(path.resolve()).lower()
            if key not in seen:
                resolved.append(path)
                seen.add(key)
    aliases = set(SOURCE_BASENAME_ALIASES.get(Path(spec.relative_path).name, ()))
    if aliases:
        for path in declared_paths:
            if path.name in aliases:
                key = str(path.resolve()).lower()
                if key not in seen:
                    resolved.append(path)
                    seen.add(key)
    return resolved, explicit


def _flatten_scalars(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, nested in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(nested, Mapping):
            output.update(_flatten_scalars(nested, name))
        elif isinstance(nested, (str, int, float, bool)) or nested is None:
            output[name] = nested
    return output


def _metadata_columns(spec: ArtifactSpec, source: Path) -> dict[str, Any]:
    return {
        "source_artifact": source.as_posix(),
        "source_artifact_sha256": _sha256(source),
        "evidence_role": spec.evidence_role,
        "source_relationship": spec.source_relationship,
        "measurement_relationship": "not_measured_by_this_study",
        "claim_boundary": spec.claim_boundary,
    }


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot materialize an empty canonical table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    temporary.replace(path)
    return path


def _parquet_to_csv(source: Path, target: Path, spec: ArtifactSpec) -> Path:
    import pandas as pd

    frame = pd.read_parquet(source)
    if frame.empty:
        raise ValueError(f"cannot materialize {target.name} from empty {source}")
    for key, value in _metadata_columns(spec, source).items():
        frame[key] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    return target


def _forecast_rows(metrics_path: Path, task: str, spec: ArtifactSpec) -> list[dict[str, Any]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    task_payload = payload.get("tasks", {}).get(task)
    if not isinstance(task_payload, Mapping):
        return []
    metadata = _metadata_columns(spec, metrics_path)
    rows: list[dict[str, Any]] = []
    locked = task_payload.get("locked_test", [])
    if isinstance(locked, list):
        for record in locked:
            if isinstance(record, Mapping):
                rows.append({**_flatten_scalars(record), **metadata})
    for model_name in ("dgpt_e", "cfc_w"):
        model = task_payload.get(model_name)
        if not isinstance(model, Mapping):
            continue
        for split in ("validation", "locked_test"):
            metrics = model.get(split)
            if isinstance(metrics, Mapping):
                rows.append(
                    {
                        "task": task,
                        "model": model_name,
                        "split": split,
                        **_flatten_scalars(metrics),
                        **metadata,
                    }
                )
    return rows


def _calibration_rows(metrics_path: Path, spec: ArtifactSpec) -> list[dict[str, Any]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = _metadata_columns(spec, metrics_path)
    rows: list[dict[str, Any]] = []
    for task, task_payload in payload.get("tasks", {}).items():
        if not isinstance(task_payload, Mapping):
            continue
        for model_name in ("dgpt_e", "cfc_w"):
            model = task_payload.get(model_name)
            if not isinstance(model, Mapping):
                continue
            for split in ("validation", "locked_test"):
                metrics = model.get(split)
                if not isinstance(metrics, Mapping):
                    continue
                flattened = _flatten_scalars(metrics)
                calibration = {
                    key: value
                    for key, value in flattened.items()
                    if any(
                        token in key.lower()
                        for token in ("coverage", "interval", "quantile", "calibr")
                    )
                }
                if calibration:
                    rows.append(
                        {
                            "task": task,
                            "model": model_name,
                            "split": split,
                            **calibration,
                            **metadata,
                        }
                    )
    return rows


def _materialize_table(
    root: Path,
    spec: ArtifactSpec,
    sources: Sequence[Path],
) -> tuple[Path | None, str | None]:
    target = root / spec.relative_path
    valid, _, _ = _inspect_artifact(target, spec)
    if valid:
        return target, "existing_canonical"
    usable = [source for source in sources if _inspect_artifact(source, spec)[0]]
    if not usable:
        return None, None
    source = usable[0]
    name = target.name
    try:
        if source.suffix.lower() == target.suffix.lower():
            _atomic_copy(source, target)
            return target, "byte_for_byte_alias"
        if source.suffix.lower() == ".parquet" and target.suffix.lower() == ".csv":
            _parquet_to_csv(source, target, spec)
            return target, "parquet_to_csv_with_evidence_metadata"
        if source.suffix.lower() == ".json" and target.suffix.lower() == ".csv":
            task_by_name = {
                "nyiso_da_forecast_main.csv": "nyiso_da",
                "nyiso_rt_forecast_main.csv": "nyiso_rt",
                "workload_forecast_main.csv": "nlr_workload",
            }
            if name in task_by_name:
                rows = _forecast_rows(source, task_by_name[name], spec)
            elif name == "calibration_summary.csv":
                rows = _calibration_rows(source, spec)
            else:
                rows = []
            if rows:
                _write_rows_csv(target, rows)
                return target, "json_metrics_to_csv_with_evidence_metadata"
    except Exception:
        return None, None
    return None, None


def _desired_manifest_status(
    branch_results: Mapping[str, Any],
    spec: ArtifactSpec,
    explicit: Mapping[str, Any] | None,
) -> str | None:
    statuses = branch_results.get("manifest_statuses", {})
    if isinstance(statuses, Mapping):
        for key in (
            spec.relative_path,
            Path(spec.relative_path).name,
            Path(spec.relative_path).stem,
            spec.artifact_id,
        ):
            if key in statuses:
                return str(statuses[key])
    if explicit is not None and explicit.get("status") is not None:
        return str(explicit["status"])
    return None


def _materialize_manifest(
    root: Path,
    spec: ArtifactSpec,
    sources: Sequence[Path],
    branch_results: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
    optional_statuses: Sequence[Mapping[str, Any]],
) -> tuple[Path, str]:
    target = root / spec.relative_path
    valid, _, _ = _inspect_artifact(target, spec)
    if valid and target.name != "final_manifest.json":
        return target, "existing_canonical"

    desired = _desired_manifest_status(branch_results, spec, explicit)
    desired_token = _status_token(desired)
    source_records = [_source_record(root, source, spec) for source in sources]
    dart_partial_contract: dict[str, Any] | None = None
    dart_partial_contract_conflict = False
    if target.name in _DART_MANIFEST_NAMES:
        source_contracts: list[dict[str, Any]] = []
        for source in sources:
            try:
                source_payload = json.loads(source.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(source_payload, Mapping):
                continue
            try:
                source_contract = _dart_partial_contract(source_payload)
            except (TypeError, ValueError):
                source_contract = None
            if source_contract is not None:
                source_contracts.append(source_contract)
        if source_contracts:
            contract_shas = {
                str(contract["contract_sha256"]) for contract in source_contracts
            }
            if len(contract_shas) == 1:
                dart_partial_contract = source_contracts[0]
            else:
                dart_partial_contract_conflict = True
    complete_override = desired_token in _GOOD_STATUS_TOKENS or (
        desired_token is not None
        and any(desired_token.startswith(token) for token in _GOOD_STATUS_TOKENS)
    )
    source_complete = bool(source_records) and all(
        record["valid"] or complete_override for record in source_records
    )
    if target.name in _DART_MANIFEST_NAMES:
        source_complete = bool(source_records) and all(
            record["valid"] for record in source_records
        )
        source_complete = source_complete and not dart_partial_contract_conflict
    if target.name == "dart_value_select_manifest.json":
        source_names = {source.name for source in sources}
        source_complete = source_complete and {
            "dart_pipeline_manifest.json",
            "dart_value_select_manifest.json",
        }.issubset(source_names)
    if target.name in _DART_MANIFEST_NAMES and source_complete:
        # DART sources have a stronger, source-specific COMPLETE/hash/chronology
        # inspection above, so a separate manual status override is unnecessary.
        complete_override = True

    # Preserve a complete upstream manifest byte-for-byte when it already has
    # the canonical semantics; composite manifests use a provenance wrapper.
    copyable = {
        "source_manifest.json",
        "referenced_data_progress_manifest.json",
        "nlr295_inventory_manifest.json",
        "timezone_manifest.json",
        "source_overlap_manifest.json",
        "model_manifest.json",
        "physical_calibration_manifest.json",
        "digital_twin_transfer_manifest.json",
        "xcel_counterfactual_manifest.json",
    }
    valid_sources = [
        source
        for source, record in zip(sources, source_records)
        if record["valid"]
    ]
    if target.is_file() and not any(
        source.resolve() != target.resolve() for source in valid_sources
    ):
        return target, "existing_incomplete_canonical_preserved"
    if target.name in copyable and valid_sources:
        source = valid_sources[0]
        if source.resolve() != target.resolve():
            _atomic_copy(source, target)
        return target, "byte_for_byte_manifest_alias"

    status = (
        PARTIAL_CANDIDATE_STATUS
        if complete_override and source_complete and dart_partial_contract is not None
        else "COMPLETE"
        if complete_override and source_complete
        else (
        "PARTIAL_SOURCE_MAPPING" if source_records else STATUS_MISSING_MANDATORY
        )
    )
    reason = None
    if status not in {"COMPLETE", PARTIAL_CANDIDATE_STATUS}:
        reason = (
            str(explicit.get("reason"))
            if explicit is not None and explicit.get("reason")
            else "Required source evidence is absent or lacks a COMPLETE machine gate."
        )
    payload = {
        "schema_version": "canonical_paper_stage_manifest_v1",
        "artifact_id": spec.artifact_id,
        "canonical_path": spec.relative_path,
        "status": status,
        "completed": status in {"COMPLETE", PARTIAL_CANDIDATE_STATUS},
        "generated_at_utc": _utc_now(),
        "requirement": spec.requirement,
        "sources": source_records,
        "source_hashes": {
            record["path"]: record["sha256"] for record in source_records
        },
        "evidence_role": spec.evidence_role,
        "source_relationship": spec.source_relationship,
        "measurement_relationship": "not_measured_by_this_study",
        "claim_boundary": spec.claim_boundary,
        "reason": reason,
        "optional_robustness_sources": list(optional_statuses),
        "synthetic_fallback_used": False,
    }
    if dart_partial_contract is not None:
        payload["candidate_pool_contract"] = dart_partial_contract
        payload["terminal_status_cap"] = PARTIAL_CANDIDATE_TERMINAL_CAP
    if target.name == "dart_value_select_manifest.json":
        payload["pre_dart_v1_readiness_evidence"] = {
            "role": "AUXILIARY_NOT_DA_GATE_ELIGIBLE",
            "selection_eligible": False,
            "use": "pretraining_readiness_and_frozen_policy_gate_only",
            "verified_via": "dart_pipeline_manifest.json",
        }
    _atomic_json(target, payload)
    return target, "generated_provenance_wrapper"


def _optional_source_records(
    root: Path,
    branch_results: Mapping[str, Any],
    specs: Sequence[ArtifactSpec],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    configured = branch_results.get("optional_sources", {})
    configured = configured if isinstance(configured, Mapping) else {}
    records: list[dict[str, Any]] = []
    derivations: dict[str, dict[str, Any]] = {}
    for spec in specs:
        source_id = spec.artifact_id.split(":", 1)[1]
        value = configured.get(source_id, {})
        if isinstance(value, (str, Path)):
            value = {"source": str(value)}
        if not isinstance(value, Mapping):
            value = {}
        source_value = value.get("source", value.get("path"))
        source_path: Path | None = None
        if source_value:
            source_path = Path(str(source_value))
            if not source_path.is_absolute():
                source_path = root / source_path
        target = root / spec.relative_path
        if source_path is not None and source_path.is_file():
            source_record = _source_record(root, source_path, spec)
            status = _status_token(value.get("status"))
            complete = source_record["valid"] and (
                status is None
                or status in _GOOD_STATUS_TOKENS
                or any(status.startswith(token) for token in _GOOD_STATUS_TOKENS)
            )
            if complete:
                payload = {
                    "schema_version": "optional_robustness_source_status_v1",
                    "source": source_id,
                    "status": "COMPLETE",
                    "completed": True,
                    "source_artifact": source_record,
                    "evidence_role": "optional_robustness",
                    "source_relationship": spec.source_relationship,
                    "claim_boundary": spec.claim_boundary,
                }
                _atomic_json(target, payload)
                record_status = STATUS_MATERIALIZED
                reason = None
                derivations[spec.relative_path] = {
                    "materialization": "optional_source_status_wrapper",
                    "sources": [source_record],
                }
            else:
                record_status = STATUS_SKIPPED_OPTIONAL
                reason = str(value.get("reason", source_record["inspection"]))
        else:
            record_status = STATUS_SKIPPED_OPTIONAL
            reason = str(
                value.get(
                    "reason",
                    "Optional source was not provided; absence does not block the primary paper path.",
                )
            )
        if record_status == STATUS_SKIPPED_OPTIONAL:
            _atomic_json(
                target,
                {
                    "schema_version": "optional_robustness_source_status_v1",
                    "source": source_id,
                    "status": STATUS_SKIPPED_OPTIONAL,
                    "completed": False,
                    "reason": reason,
                    "evidence_role": "optional_robustness",
                    "source_relationship": spec.source_relationship,
                    "claim_boundary": spec.claim_boundary,
                },
            )
        records.append(
            {
                "artifact_id": spec.artifact_id,
                "source": source_id,
                "status": record_status,
                "reason": reason,
                "path": spec.relative_path,
            }
        )
    return records, derivations


def _artifact_record(
    root: Path,
    spec: ArtifactSpec,
    derivation: Mapping[str, Any] | None = None,
    *,
    optional_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = root / spec.relative_path
    valid, content_status, inspection = _inspect_artifact(path, spec)
    if (derivation or {}).get("force_missing"):
        valid = False
        inspection = str(
            (derivation or {}).get(
                "reason", "generated status document lacks its mandatory source evidence"
            )
        )
    if spec.requirement == "optional" and optional_status is not None:
        status = str(optional_status["status"])
        if status == STATUS_SKIPPED_OPTIONAL:
            valid = False
    else:
        status = STATUS_MATERIALIZED if valid else STATUS_MISSING_MANDATORY
    source_records = list((derivation or {}).get("sources", []))
    if not source_records and valid:
        source_records = [
            {
                "path": spec.relative_path,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "valid": True,
                "content_status": content_status,
                "inspection": "canonical artifact is its recorded source",
            }
        ]
    return {
        **asdict(spec),
        "status": status,
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
        "bytes": path.stat().st_size if path.is_file() else None,
        "content_status": content_status,
        "inspection": inspection,
        "source_hashes": {
            str(record["path"]): str(record["sha256"])
            for record in source_records
            if record.get("sha256")
        },
        "sources": source_records,
        "materialization": (derivation or {}).get("materialization"),
        "alias_of": (derivation or {}).get("alias_of"),
        "companion_outputs": dict(
            (derivation or {}).get("companion_outputs", {})
        ),
        "source_rows_sha256": (derivation or {}).get("source_rows_sha256"),
        "sample_count": (derivation or {}).get("sample_count"),
        "reason": (
            optional_status.get("reason")
            if optional_status is not None
            else (None if valid else inspection)
        ),
        "measurement_relationship": "not_measured_by_this_study",
    }


def _write_evidence_layer_audit(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    spec: ArtifactSpec,
) -> tuple[Path, dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "artifact_id": record["artifact_id"],
                "canonical_path": record["relative_path"],
                "requirement": record["requirement"],
                "status": record["status"],
                "evidence_role": record["evidence_role"],
                "source_relationship": record["source_relationship"],
                "measurement_relationship": record[
                    "measurement_relationship"
                ],
                "claim_boundary": record["claim_boundary"],
                "artifact_sha256": record["sha256"],
                "artifact_bytes": record["bytes"],
                "source_hashes_json": json.dumps(
                    record["source_hashes"], sort_keys=True
                ),
            }
        )
    from .paper_tables_reports import write_inventory_audit_table

    result = write_inventory_audit_table(root, rows)
    target = root / spec.relative_path
    return target, {
        "materialization": result.get("materialization"),
        "sources": list(result.get("sources", [])),
        "companion_outputs": dict(result.get("companion_outputs", {})),
        "source_rows_sha256": result.get("source_rows_sha256"),
        "sample_count": result.get("sample_count"),
    }


def _markdown_table(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Artifact | Requirement | Status | Evidence role |",
        "|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| `{}` | {} | `{}` | `{}` |".format(
                record["relative_path"],
                record["requirement"],
                record["status"],
                record["evidence_role"],
            )
        )
    return "\n".join(lines)


def _generate_reports(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    derivations: dict[str, dict[str, Any]],
) -> None:
    missing = [
        record
        for record in records
        if record["requirement"] == "mandatory"
        and record["status"] != STATUS_MATERIALIZED
    ]
    skipped = [
        record for record in records if record["status"] == STATUS_SKIPPED_OPTIONAL
    ]
    status = "COMPLETE" if not missing else "BLOCKED_MISSING_MANDATORY"
    generated_at = _utc_now()
    report_root = root / "artifacts/report"
    report_root.mkdir(parents=True, exist_ok=True)
    report_specs = {
        Path(spec.relative_path).name: spec
        for spec in CANONICAL_ARTIFACT_REGISTRY
        if spec.category == "report"
    }

    def generated_by_this_module(path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() == ".json":
            try:
                return (
                    json.loads(path.read_text(encoding="utf-8")).get(
                        "schema_version"
                    )
                    == INVENTORY_SCHEMA_VERSION
                )
            except Exception:
                return False
        text = path.read_text(encoding="utf-8", errors="replace")
        markers = (
            "# Paper Artifact Run Summary",
            "<title>Paper artifact summary</title>",
            "# Paper Artifact Mapping",
            "# Data Architecture Amendment",
            "# Evidence Layer Dictionary",
            "# Source-to-Claim Matrix",
            "# Transfer and Extrapolation Report",
            "# Failed or Skipped Experiments",
        )
        return any(marker in text for marker in markers)

    def write_text_report(
        name: str, text: str, materialization: str, **metadata: Any
    ) -> None:
        relative_path = f"artifacts/report/{name}"
        target = report_root / name
        spec = report_specs[name]
        if _inspect_artifact(target, spec)[0] and not generated_by_this_module(target):
            derivations.setdefault(
                relative_path,
                {
                    "materialization": "existing_canonical",
                    "sources": [_source_record(root, target, spec)],
                },
            )
            return
        _atomic_text(target, text)
        derivations[relative_path] = {
            "materialization": materialization,
            "sources": [],
            **metadata,
        }

    def write_json_report(
        name: str, payload: Mapping[str, Any], materialization: str
    ) -> None:
        relative_path = f"artifacts/report/{name}"
        target = report_root / name
        spec = report_specs[name]
        if _inspect_artifact(target, spec)[0] and not generated_by_this_module(target):
            derivations.setdefault(
                relative_path,
                {
                    "materialization": "existing_canonical",
                    "sources": [_source_record(root, target, spec)],
                },
            )
            return
        _atomic_json(target, payload)
        derivations[relative_path] = {
            "materialization": materialization,
            "sources": [],
        }

    summary = (
        "# Paper Artifact Run Summary\n\n"
        f"- Inventory status: `{status}`\n"
        f"- Mandatory artifact count: `{sum(r['requirement'] == 'mandatory' for r in records)}`\n"
        f"- Missing mandatory count: `{len(missing)}`\n"
        f"- Optional source skips: `{len(skipped)}`\n"
        "- Claim boundary: all market and VPP outputs are counterfactual simulations; "
        "no realized profit, observed demand-response saving, or demonstrated grid benefit is claimed.\n"
    )
    write_text_report("run_summary.md", summary, "generated_inventory_summary")
    summary_html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Paper artifact summary</title></head>"
        "<body><h1>Paper Artifact Run Summary</h1>"
        f"<p>Status: <code>{html.escape(status)}</code></p>"
        f"<p>Missing mandatory artifacts: {len(missing)}</p>"
        f"<p>Optional source skips: {len(skipped)}</p>"
        "<p>All market and VPP outputs are counterfactual simulations, not realized returns or observed grid benefits.</p>"
        "</body></html>"
    )
    write_text_report(
        "run_summary.html", summary_html, "generated_inventory_summary_html"
    )

    write_json_report(
        "integrity_audit.json",
        {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "status": status,
            "completed": not missing,
            "generated_at_utc": generated_at,
            "mandatory_count": sum(
                record["requirement"] == "mandatory" for record in records
            ),
            "missing_mandatory_count": len(missing),
            "missing_mandatory": [
                {
                    "artifact_id": record["artifact_id"],
                    "path": record["relative_path"],
                    "reason": record["reason"],
                }
                for record in missing
            ],
            "optional_skips": skipped,
            "forbidden_claims_present": False,
            "claim_boundary": (
                "No actual-NLR-in-NYISO, realized-profit, observed-DR-savings, "
                "or demonstrated-grid-benefit claim is authorized by this inventory."
            ),
        },
        "generated_integrity_audit",
    )

    mapping = (
        "# Paper Artifact Mapping\n\n"
        "Canonical paths follow revised specification lines 1939-2060.\n\n"
        + _markdown_table(records)
        + "\n"
    )
    write_text_report("paper_mapping.md", mapping, "generated_registry_mapping")

    architecture = """# Data Architecture Amendment

- NLR, SRRL, Xcel, NYISO, NASA POWER, and optional workload traces remain independent source domains.
- Dates are not row-wise joined to create a false synchronized operator-level observation table.
- NLR physical calibration is transferred to four NYISO-zone twins only for counterfactual co-simulation.
- Xcel outputs are tariff-defined counterfactuals with zero observed event calls.
- Reconstructed weather and replayed workload are labeled separately from source-level observed records.
"""
    write_text_report(
        "data_architecture_amendment.md",
        architecture,
        "generated_method_boundary_report",
    )

    dictionary = """# Evidence Layer Dictionary

- `observed_nlr`: source-level NLR record; not an investigator measurement.
- `observed_nyiso_market`: source-level NYISO market record.
- `observed_srrl`: source-level SRRL record.
- `derived`: deterministic transformation of hashed source records.
- `reconstructed`: reconstructed exogenous series, including NASA POWER weather.
- `replayed`: a source trace replayed in a non-joint counterfactual date frame.
- `transferred_nyiso_twin`: NLR-calibrated parameters transferred to a NYISO-zone twin.
- `xcel_counterfactual`: published tariff-rule calculation with no observed event response.
- `optional_robustness`: independently evaluated optional source.
- `oracle`: perfect-information lower bound, excluded from deployable claims.
- `counterfactual`: simulated VPP/market outcome; never a realized return or observed grid benefit.
"""
    write_text_report(
        "evidence_layer_dictionary.md",
        dictionary,
        "generated_controlled_vocabulary",
    )

    matrix_lines = [
        "# Source-to-Claim Matrix",
        "",
        "| Artifact | Evidence role | Source relationship | Claim boundary |",
        "|---|---|---|---|",
    ]
    for record in records:
        matrix_lines.append(
            "| `{}` | `{}` | `{}` | {} |".format(
                record["relative_path"],
                record["evidence_role"],
                record["source_relationship"],
                str(record["claim_boundary"]).replace("|", "\\|"),
            )
        )
    write_text_report(
        "source_to_claim_matrix.md",
        "\n".join(matrix_lines) + "\n",
        "generated_source_claim_mapping",
    )

    transfer_sources = []
    for record in records:
        if "transfer" in record["artifact_id"] and record["status"] == STATUS_MATERIALIZED:
            transfer_sources.extend(record["sources"])
    if transfer_sources:
        transfer_lines = [
            "# Transfer and Extrapolation Report",
            "",
            "This report maps the hashed transfer evidence. It does not assert that NLR operated in NYISO.",
            "",
        ]
        for source in transfer_sources:
            transfer_lines.append(
                f"- `{source['path']}` — SHA-256 `{source['sha256']}`"
            )
        force_missing = False
    else:
        transfer_lines = [
            "# Transfer and Extrapolation Report",
            "",
            "Required transfer evidence is unavailable; no transfer-support claim is made.",
        ]
        force_missing = True
    write_text_report(
        "transfer_and_extrapolation_report.md",
        "\n".join(transfer_lines) + "\n",
        "generated_transfer_source_mapping",
        sources=transfer_sources,
        force_missing=force_missing,
        reason="No complete transfer evidence was mapped.",
    )

    failure_lines = ["# Failed or Skipped Experiments", ""]
    if not missing and not skipped:
        failure_lines.append("No missing mandatory artifacts or optional-source skips were recorded.")
    for record in missing:
        failure_lines.append(
            f"- `MISSING_MANDATORY` — `{record['relative_path']}`: {record['reason']}"
        )
    for record in skipped:
        failure_lines.append(
            f"- `SKIPPED_OPTIONAL_SOURCE` — `{record['relative_path']}`: {record['reason']}"
        )
    write_text_report(
        "failed_or_skipped_experiments.md",
        "\n".join(failure_lines) + "\n",
        "generated_failure_and_skip_ledger",
    )


def _custom_optional_specs(branch_results: Mapping[str, Any]) -> tuple[ArtifactSpec, ...]:
    configured = branch_results.get("optional_artifacts", ())
    if not isinstance(configured, (list, tuple)):
        raise TypeError("optional_artifacts must be a list of mappings")
    specs = []
    for index, record in enumerate(configured):
        if not isinstance(record, Mapping):
            raise TypeError("each optional_artifacts item must be a mapping")
        artifact_id = str(record.get("artifact_id", f"custom_optional_{index}"))
        relative_path = str(
            record.get(
                "relative_path", f"artifacts/optional/{artifact_id}/source_status.json"
            )
        ).replace("\\", "/")
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError("optional artifact paths must remain inside the project root")
        specs.append(
            ArtifactSpec(
                artifact_id=f"optional_source:{artifact_id}",
                category="optional_source",
                relative_path=relative_path,
                requirement="optional",
                evidence_role=str(record.get("evidence_role", "optional_robustness")),
                source_relationship=str(
                    record.get(
                        "source_relationship", "third_party_referenced_public_data"
                    )
                ),
                claim_boundary=str(
                    record.get(
                        "claim_boundary",
                        "Independent optional robustness source; not part of primary training.",
                    )
                ),
                specification_lines=str(record.get("specification_lines", "59,949-951")),
            )
        )
    return tuple(specs)


def _materialize_generic_alias(
    root: Path,
    spec: ArtifactSpec,
    sources: Sequence[Path],
) -> tuple[Path | None, str | None, Path | None]:
    target = root / spec.relative_path
    valid, _, _ = _inspect_artifact(target, spec)
    if valid:
        return target, "existing_canonical", target
    for source in sources:
        if source.resolve() == target.resolve():
            continue
        source_valid, _, _ = _inspect_artifact(source, spec)
        if source_valid and source.suffix.lower() == target.suffix.lower():
            _atomic_copy(source, target)
            return target, "byte_for_byte_alias", source
    return None, None, None


def finalize_artifact_inventory(
    root: Path | str,
    branch_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize and inventory the paper's canonical evidence artifacts.

    This function never creates numeric table rows or figures without a real,
    hashed source artifact.  Missing required items remain
    ``MISSING_MANDATORY`` and prevent ``COMPLETE``.  Optional source absence is
    recorded as ``SKIPPED_OPTIONAL_SOURCE`` and is non-blocking.
    """

    project_root = Path(root).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    results = dict(branch_results or {})
    custom_optional = _custom_optional_specs(results)
    optional_specs = DEFAULT_OPTIONAL_SOURCE_SPECS + custom_optional
    # Allow custom optional records to use the same optional_sources interface.
    if custom_optional:
        optional_sources = dict(results.get("optional_sources", {}))
        for record in results.get("optional_artifacts", ()):
            artifact_id = str(record.get("artifact_id"))
            optional_sources.setdefault(artifact_id, dict(record))
        results["optional_sources"] = optional_sources

    declared_paths = _collect_declared_paths(results, project_root)
    optional_statuses, derivations = _optional_source_records(
        project_root, results, optional_specs
    )
    optional_by_id = {
        record["artifact_id"]: record for record in optional_statuses
    }
    specs = CANONICAL_ARTIFACT_REGISTRY + optional_specs
    spec_by_path = {spec.relative_path: spec for spec in specs}
    figure_results: dict[str, dict[str, Any]] | None = None
    table_results: dict[str, dict[str, Any]] | None = None

    for spec in CANONICAL_ARTIFACT_REGISTRY:
        if Path(spec.relative_path).name == "final_manifest.json":
            continue
        sources, explicit = _resolve_sources(
            project_root, spec, results, declared_paths
        )
        source_records = [_source_record(project_root, source, spec) for source in sources]
        if spec.category == "manifest":
            target, mode = _materialize_manifest(
                project_root,
                spec,
                sources,
                results,
                explicit,
                optional_statuses,
            )
            derivations[spec.relative_path] = {
                "materialization": mode,
                "sources": source_records,
            }
        elif spec.category == "table" and Path(spec.relative_path).name != "evidence_layer_audit.csv":
            if table_results is None:
                from .paper_tables_reports import generate_paper_tables

                table_results = generate_paper_tables(project_root, results)
            table_result = table_results.get(spec.relative_path, {})
            if table_result.get("status") == STATUS_MATERIALIZED:
                derivations[spec.relative_path] = {
                    "materialization": table_result.get("materialization"),
                    "sources": list(table_result.get("sources", [])),
                    "companion_outputs": dict(
                        table_result.get("companion_outputs", {})
                    ),
                    "source_rows_sha256": table_result.get(
                        "source_rows_sha256"
                    ),
                    "sample_count": table_result.get("sample_count"),
                }
            else:
                derivations[spec.relative_path] = {
                    "force_missing": True,
                    "reason": table_result.get(
                        "reason", "table input mapping is incomplete"
                    ),
                    "sources": list(table_result.get("sources", [])),
                }
        elif spec.category == "figure":
            if figure_results is None:
                from .paper_figures import generate_paper_figures

                figure_results = generate_paper_figures(project_root, results)
            figure_result = figure_results.get(spec.relative_path, {})
            if figure_result.get("status") == STATUS_MATERIALIZED:
                derivations[spec.relative_path] = {
                    "materialization": figure_result.get("materialization"),
                    "sources": list(figure_result.get("sources", [])),
                    "companion_outputs": dict(
                        figure_result.get("companion_outputs", {})
                    ),
                    "source_rows_sha256": figure_result.get(
                        "source_rows_sha256"
                    ),
                    "sample_count": figure_result.get("sample_count"),
                }
            else:
                derivations[spec.relative_path] = {
                    "force_missing": True,
                    "reason": figure_result.get(
                        "reason", "figure input mapping is incomplete"
                    ),
                    "sources": list(figure_result.get("sources", [])),
                }
        elif spec.category == "report":
            # Reports are generated after all non-report records have been
            # assessed so their sidecars pin the exact hashed evidence set.
            derivations[spec.relative_path] = {
                "force_missing": True,
                "reason": "deferred until canonical evidence records are assessed",
                "sources": [],
            }

    def assess() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for spec in specs:
            records.append(
                _artifact_record(
                    project_root,
                    spec,
                    derivations.get(spec.relative_path),
                    optional_status=optional_by_id.get(spec.artifact_id),
                )
            )
        return records

    records = assess()
    from .paper_tables_reports import generate_paper_reports

    report_results = generate_paper_reports(project_root, records, results)
    for spec in CANONICAL_ARTIFACT_REGISTRY:
        if spec.category != "report":
            continue
        result = report_results.get(spec.relative_path, {})
        if result.get("status") == STATUS_MATERIALIZED:
            derivations[spec.relative_path] = {
                "materialization": result.get("materialization"),
                "sources": list(result.get("sources", [])),
                "companion_outputs": dict(result.get("companion_outputs", {})),
                "source_rows_sha256": result.get("source_rows_sha256"),
                "sample_count": result.get("sample_count"),
            }
        else:
            derivations[spec.relative_path] = {
                "force_missing": True,
                "reason": result.get("reason", "report source mapping is incomplete"),
                "sources": list(result.get("sources", [])),
            }
    records = assess()

    evidence_spec = spec_by_path["artifacts/tables/evidence_layer_audit.csv"]
    audit_inputs = [
        record
        for record in records
        if record["relative_path"]
        not in {
            "artifacts/tables/evidence_layer_audit.csv",
            "artifacts/manifests/final_manifest.json",
        }
    ]
    _, audit_derivation = _write_evidence_layer_audit(
        project_root, audit_inputs, evidence_spec
    )
    derivations[evidence_spec.relative_path] = audit_derivation
    records = assess()

    final_spec = spec_by_path["artifacts/manifests/final_manifest.json"]
    prefinal_missing = [
        record
        for record in records
        if record["requirement"] == "mandatory"
        and record["relative_path"] != final_spec.relative_path
        and record["status"] != STATUS_MATERIALIZED
    ]
    final_status = "COMPLETE" if not prefinal_missing else "BLOCKED_MISSING_MANDATORY"
    final_path = project_root / final_spec.relative_path
    final_payload = {
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
        "status": final_status,
        "completed": not prefinal_missing,
        "generated_at_utc": _utc_now(),
        "canonical_artifact_count": len(CANONICAL_ARTIFACT_REGISTRY),
        "optional_artifact_count": len(optional_specs),
        "missing_mandatory_count": len(prefinal_missing),
        "missing_mandatory": [
            {
                "artifact_id": record["artifact_id"],
                "path": record["relative_path"],
                "reason": record["reason"],
            }
            for record in prefinal_missing
        ],
        "paper_artifacts": [
            {
                "artifact_id": record["artifact_id"],
                "path": record["relative_path"],
                "status": record["status"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "source_hashes": record["source_hashes"],
                "evidence_role": record["evidence_role"],
                "source_relationship": record["source_relationship"],
                "claim_boundary": record["claim_boundary"],
                "companion_outputs": record.get("companion_outputs", {}),
                "source_rows_sha256": record.get("source_rows_sha256"),
                "sample_count": record.get("sample_count"),
            }
            for record in records
            if record["relative_path"] != final_spec.relative_path
        ],
        "self_hash_recorded_in": "artifacts/manifests/paper_artifact_inventory.json",
        "claim_boundary": (
            "No actual-NLR-in-NYISO, realized-profit, observed demand-response savings, "
            "or demonstrated-grid-benefit claim is authorized."
        ),
    }
    _atomic_json(final_path, final_payload)
    derivations[final_spec.relative_path] = {
        "materialization": "generated_final_hash_manifest",
        "sources": [],
    }
    records = assess()
    missing_mandatory = [
        record
        for record in records
        if record["requirement"] == "mandatory"
        and record["status"] != STATUS_MATERIALIZED
    ]
    overall_status = (
        "COMPLETE" if not missing_mandatory else "BLOCKED_MISSING_MANDATORY"
    )
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "status": overall_status,
        "completed": not missing_mandatory,
        "generated_at_utc": _utc_now(),
        "authoritative_specification_lines": [
            "1939-2060",
            "2188-2197",
            "2264-2266",
        ],
        "canonical_registry_count": len(CANONICAL_ARTIFACT_REGISTRY),
        "optional_registry_count": len(optional_specs),
        "mandatory_count": sum(
            record["requirement"] == "mandatory" for record in records
        ),
        "materialized_mandatory_count": sum(
            record["requirement"] == "mandatory"
            and record["status"] == STATUS_MATERIALIZED
            for record in records
        ),
        "missing_mandatory_count": len(missing_mandatory),
        "optional_skipped_count": sum(
            record["status"] == STATUS_SKIPPED_OPTIONAL for record in records
        ),
        "artifacts": records,
        "final_manifest": {
            "path": final_spec.relative_path,
            "sha256": _sha256(final_path),
            "bytes": final_path.stat().st_size,
        },
        "gate_rule": (
            "COMPLETE is forbidden while any mandatory registry item is not MATERIALIZED; "
            "optional SKIPPED_OPTIONAL_SOURCE records are non-blocking."
        ),
    }
    inventory_path = project_root / "artifacts/manifests/paper_artifact_inventory.json"
    _atomic_json(inventory_path, inventory)
    pending_path = project_root / "artifacts/manifests/artifact_pending_and_skips.json"
    _atomic_json(
        pending_path,
        {
            "schema_version": "paper_artifact_pending_and_skips_v1",
            "status": overall_status,
            "missing_mandatory": missing_mandatory,
            "skipped_optional_sources": [
                record
                for record in records
                if record["status"] == STATUS_SKIPPED_OPTIONAL
            ],
        },
    )
    inventory["inventory_path"] = str(inventory_path)
    inventory["pending_and_skips_path"] = str(pending_path)
    return inventory


__all__ = [
    "ArtifactSpec",
    "CANONICAL_ARTIFACT_REGISTRY",
    "DEFAULT_OPTIONAL_SOURCE_SPECS",
    "FINAL_MANIFEST_SCHEMA_VERSION",
    "INVENTORY_SCHEMA_VERSION",
    "STATUS_INVALID",
    "STATUS_MATERIALIZED",
    "STATUS_MISSING_MANDATORY",
    "STATUS_SKIPPED_OPTIONAL",
    "finalize_artifact_inventory",
]
