from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# The backend, seed, and style are part of the scientific artifact contract.
FIGURE_SCHEMA_VERSION = "paper_figure_source_v1"
FIGURE_RENDERER_VERSION = "paper_figures_v1"
FIGURE_SEED = 20260809
FIGURE_DPI = 300

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"

COLORS = {
    "navy": "#0F4D92",
    "blue": "#3775BA",
    "teal": "#42949E",
    "violet": "#7C6CCF",
    "gold": "#C28B00",
    "red": "#B64342",
    "green": "#2E7D4F",
    "gray": "#767676",
    "light": "#CFCECE",
    "dark": "#272727",
}
MARKERS = ("o", "s", "^", "D", "v", "P")
LINESTYLES = ("-", "--", "-.", ":")

REFERENCE_BOUNDARY = (
    "Referenced public-data evidence only; observed values are source records, "
    "not measurements made by this study."
)
NLR_BOUNDARY = (
    "Derived from referenced NLR/SRRL records; no NYISO deployment or market "
    "operation is represented."
)
NYISO_BOUNDARY = (
    "Forecast evaluation on referenced NYISO records; no facility operation or "
    "realized market return is claimed."
)
WORKLOAD_BOUNDARY = (
    "NLR workload-trace evidence only; replayed intervals are not additional "
    "independent observations."
)
TRANSFER_BOUNDARY = (
    "NLR-calibrated parameter transfer to counterfactual NYISO twins; no NLR "
    "facility was operated in NYISO."
)
COUNTERFACTUAL_BOUNDARY = (
    "Counterfactual co-simulation only; not realized profit, observed grid "
    "benefit, or an actual NLR-in-NYISO operation."
)
DART_COUNTERFACTUAL_BOUNDARY = (
    "DART accuracy and constrained data-center decision value are evaluated on "
    "referenced NYISO prices through counterfactual replay; values are not "
    "virtual-trader P&L, realized commercial profit, or observed grid benefit."
)


@dataclass(frozen=True)
class FigureDefinition:
    name: str
    renderer: str
    title: str
    conclusion: str
    archetype: str
    source_patterns: tuple[str, ...]
    evidence_role: str
    source_relationship: str
    claim_boundary: str

    @property
    def relative_path(self) -> str:
        return f"artifacts/figures/{self.name}"


def _definition(
    name: str,
    renderer: str,
    title: str,
    conclusion: str,
    patterns: Sequence[str],
    *,
    boundary: str = REFERENCE_BOUNDARY,
    evidence_role: str = "derived",
    relationship: str = "derived_from_third_party_referenced_public_data",
    archetype: str = "quantitative_grid",
) -> FigureDefinition:
    return FigureDefinition(
        name=name,
        renderer=renderer,
        title=title,
        conclusion=conclusion,
        archetype=archetype,
        source_patterns=tuple(patterns),
        evidence_role=evidence_role,
        source_relationship=relationship,
        claim_boundary=boundary,
    )


FIGURE_INPUT_REGISTRY: tuple[FigureDefinition, ...] = (
    _definition(
        "data_timeline.png",
        "timeline",
        "Chronological evidence and locked evaluation windows",
        "Training, calibration, and locked-test windows remain chronologically separated by data domain.",
        (
            "artifacts/real_forecasting/*/manifests/split_manifest.json",
            "artifacts/research/physical/run_id=*/physical_split_manifest.json",
            "artifacts/manifests/split_manifest.json",
        ),
    ),
    _definition(
        "data_missingness.png",
        "missingness",
        "Source-data missingness",
        "Missingness is quantified from source row counts rather than silently zero-filled.",
        (
            "artifacts/real_forecasting/*/manifests/input_audits.json",
            "artifacts/research/physical/run_id=*/source_overlap_manifest.json",
            "artifacts/manifests/data_quality_manifest.json",
        ),
    ),
    _definition(
        "nlr_physical_representative_week.png",
        "physical_week",
        "NLR physical representative week",
        "A median-load source week preserves the observed relationship between IT, facility power, and PUE.",
        ("data/processed/research/physical/run_id=*/nlr300_facility_15m.parquet",),
        boundary=NLR_BOUNDARY,
        archetype="time_series_grid",
    ),
    _definition(
        "nyiso_counterfactual_representative_week.png",
        "counterfactual_week",
        "NYISO counterfactual representative week",
        "Causal rolling schedules, grid imports, and local prices can be inspected over a representative counterfactual week.",
        (
            "artifacts/optimization_branch/*/representative_week_dispatch.csv",
            "artifacts/optimization_branch/*/rolling_checkpoints/*.json",
            "artifacts/research/optimization/*/representative_week_dispatch.csv",
            "artifacts/research/optimization/*/rolling_checkpoints/*.json",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        archetype="time_series_grid",
    ),
    _definition(
        "nlr_physical_calibration.png",
        "physical_calibration",
        "Held-out NLR physical calibration",
        "The calibrated physical twin is compared with preregistered uncalibrated models on held-out NLR records.",
        (
            "artifacts/tables/nlr_twin_validation.csv",
            "artifacts/research/physical/run_id=*/nlr_physical_validation_metrics.parquet",
        ),
        boundary=NLR_BOUNDARY,
    ),
    _definition(
        "pue_component_closure.png",
        "component_closure",
        "PUE component closure",
        "Facility power closes against the documented component sum with explicitly retained residuals.",
        (
            "data/processed/research/physical/run_id=*/nlr300_facility_15m.parquet",
            "artifacts/tables/pue_component_closure.csv",
        ),
        boundary=NLR_BOUNDARY,
    ),
    _definition(
        "twin_transfer_support.png",
        "transfer_support",
        "Digital-twin transfer support",
        "Climate support distance and physical-parameter uncertainty make NYISO twin extrapolation visible.",
        (
            "artifacts/tables/digital_twin_transfer.csv",
            "artifacts/research/physical/run_id=*/digital_twin_transfer_parameters.parquet",
        ),
        boundary=TRANSFER_BOUNDARY,
        evidence_role="transferred_nyiso_twin",
    ),
    _definition(
        "nyiso_da_calibration.png",
        "calibration_da",
        "NYISO day-ahead calibration",
        "Locked-test day-ahead interval coverage is assessed against nominal coverage with interval width visible.",
        (
            "artifacts/real_forecasting/*/predictions/nyiso_da*quantile*.parquet",
            "artifacts/tables/calibration_summary.csv",
        ),
        boundary=NYISO_BOUNDARY,
    ),
    _definition(
        "nyiso_rt_calibration.png",
        "calibration_rt",
        "NYISO real-time calibration",
        "Locked-test real-time interval coverage is assessed against nominal coverage with interval width visible.",
        (
            "artifacts/real_forecasting/*/predictions/nyiso_rt*quantile*.parquet",
            "artifacts/tables/calibration_summary.csv",
        ),
        boundary=NYISO_BOUNDARY,
    ),
    _definition(
        "workload_calibration.png",
        "calibration_workload",
        "NLR workload calibration",
        "Held-out workload interval coverage and width are reported without treating replayed timestamps as new observations.",
        (
            "artifacts/real_forecasting/*/predictions/*workload*calibrat*.parquet",
            "artifacts/real_forecasting/*/predictions/*workload*cfc*.parquet",
            "artifacts/tables/calibration_summary.csv",
        ),
        boundary=WORKLOAD_BOUNDARY,
    ),
    _definition(
        "price_spike_case.png",
        "price_spike",
        "Locked-test NYISO price-spike case",
        "The largest locked-test absolute price event is shown without clipping negative prices or positive spikes.",
        (
            "artifacts/real_forecasting/*/predictions/nyiso_rt*quantile*.parquet",
            "artifacts/real_forecasting/*/predictions/nyiso_rt_predictions.parquet",
            "artifacts/real_forecasting/*/predictions/nyiso_da*quantile*.parquet",
        ),
        boundary=NYISO_BOUNDARY,
        archetype="event_case",
    ),
    _definition(
        "workload_burst_case.png",
        "workload_burst",
        "Held-out NLR workload-burst case",
        "The largest held-out workload burst is compared with issue-time forecasts on its original scale.",
        (
            "artifacts/real_forecasting/*/predictions/*workload*cfc*.parquet",
            "artifacts/real_forecasting/*/predictions/nlr_workload_predictions.parquet",
        ),
        boundary=WORKLOAD_BOUNDARY,
        archetype="event_case",
    ),
    _definition(
        "scenario_dependence.png",
        "scenario_dependence",
        "Scenario dependence diagnostics",
        "Rank dependence and joint-extreme mass are retained after scenario construction and reduction.",
        (
            "artifacts/tables/scenario_dependence.csv",
            "artifacts/optimization_branch/*/scenario_diagnostics.json",
            "artifacts/research/optimization/*/scenario_diagnostics.json",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "dependence_stress.png",
        "dependence_stress",
        "Dependence-stress counterfactual outcomes",
        "Counterfactual expected cost and CVaR are evaluated across dependence and stress regimes.",
        (
            "artifacts/tables/dependence_stress.csv",
            "artifacts/optimization_branch/*/optimization_results.csv",
            "artifacts/research/optimization/*/optimization_results.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "risk_frontier.png",
        "risk_frontier",
        "Counterfactual cost-risk frontier",
        "Risk-weight choices expose the counterfactual trade-off between expected cost and empirical CVaR.",
        (
            "artifacts/tables/risk_frontier.csv",
            "artifacts/optimization_branch/*/optimization_sensitivity.csv",
            "artifacts/optimization_branch/*/optimization_results.csv",
            "artifacts/research/optimization/*/optimization_sensitivity.csv",
            "artifacts/research/optimization/*/optimization_results.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "policy_comparison.png",
        "policy_comparison",
        "Counterfactual policy comparison",
        "Expected cost and CVaR are compared across preregistered B0-B8 policies under matched counterfactual cells.",
        (
            "artifacts/tables/counterfactual_optimization_main.csv",
            "artifacts/optimization_branch/*/optimization_results.csv",
            "artifacts/research/optimization/*/optimization_results.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "typical_day_dispatch.png",
        "typical_dispatch",
        "Typical-day counterfactual dispatch",
        "A median-grid-energy day exposes causal grid, PV, battery, workload, and state-of-charge trajectories.",
        (
            "artifacts/optimization_branch/*/typical_day_dispatch.csv",
            "artifacts/optimization_branch/*/rolling_checkpoints/*.json",
            "artifacts/research/optimization/*/typical_day_dispatch.csv",
            "artifacts/research/optimization/*/rolling_checkpoints/*.json",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        archetype="time_series_grid",
    ),
    _definition(
        "grid_rebound.png",
        "grid_rebound",
        "Counterfactual grid rebound",
        "Ramp and concentration constraints expose their counterfactual rebound and peak-import consequences.",
        (
            "artifacts/tables/optimization_grid_facing_proxies.csv",
            "artifacts/optimization_branch/*/optimization_results.csv",
            "artifacts/research/optimization/*/optimization_results.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "sensitivity_flexibility.png",
        "sensitivity_flexibility",
        "Flexibility sensitivity",
        "Counterfactual cost and risk responses are traced across preregistered flexibility levels.",
        (
            "artifacts/tables/optimization_sensitivity.csv",
            "artifacts/optimization_branch/*/optimization_sensitivity.csv",
            "artifacts/research/optimization/*/optimization_sensitivity.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "sensitivity_bandwidth.png",
        "sensitivity_bandwidth",
        "Migration-bandwidth sensitivity",
        "Counterfactual cost and risk responses are traced across the preregistered migration-bandwidth grid.",
        (
            "artifacts/tables/optimization_sensitivity.csv",
            "artifacts/optimization_branch/*/optimization_sensitivity.csv",
            "artifacts/research/optimization/*/optimization_sensitivity.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "sensitivity_storage.png",
        "sensitivity_storage",
        "Storage-duration sensitivity",
        "Counterfactual cost and risk responses are traced across the preregistered battery-duration grid.",
        (
            "artifacts/tables/optimization_sensitivity.csv",
            "artifacts/optimization_branch/*/optimization_sensitivity.csv",
            "artifacts/research/optimization/*/optimization_sensitivity.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "solver_scaling.png",
        "solver_scaling",
        "Optimization solver scaling",
        "Solve time, problem size, and termination status are reported across formal counterfactual cells.",
        (
            "artifacts/tables/solver_performance.csv",
            "artifacts/optimization_branch/*/optimization_results.csv",
            "artifacts/research/optimization/*/optimization_results.csv",
        ),
        boundary=COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
    ),
    _definition(
        "dart_accuracy_value_contradiction.png",
        "dart_accuracy_value_contradiction",
        "DART accuracy and decision-value contradiction",
        "Candidate forecast accuracy rank need not equal constrained data-center operational-value rank.",
        (
            "artifacts/tables/dart_regime_value.csv",
            "artifacts/tables/dart_probability_quality.csv",
            "artifacts/tables/dart_decision_value.csv",
        ),
        boundary=DART_COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        archetype="decision_value_scatter",
    ),
    _definition(
        "dart_value_selection_lcb.png",
        "dart_value_selection_lcb",
        "DART-ValueSelect lower confidence bounds",
        "Selection is based on the preregistered lower confidence bound of paired incremental operational value, with 7-, 14-, and 28-day block sensitivity retained.",
        ("artifacts/tables/dart_selection_methods.csv",),
        boundary=DART_COUNTERFACTUAL_BOUNDARY,
        evidence_role="counterfactual",
        relationship="counterfactual_using_third_party_referenced_public_data",
        archetype="selection_lcb",
    ),
)


class FigureInputError(ValueError):
    pass


@dataclass
class LoadedSource:
    path: Path
    sha256: str
    bytes: int
    frame: pd.DataFrame | None
    payload: Any


@dataclass
class PlotResult:
    figure: Any
    plot_rows: list[dict[str, Any]]
    used_sources: list[LoadedSource]
    selection: Mapping[str, Any]
    caption: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "synthetic"}


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


def _records(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    return [
        {str(key): _jsonable(value) for key, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_synthetic_frame(frame: pd.DataFrame) -> bool:
    for column in ("synthetic", "synthetic_fixture", "demo", "prototype_only"):
        if column in frame and frame[column].map(_truthy).any():
            return True
    for column in ("evidence_role", "status", "source_record_type"):
        if column in frame:
            values = frame[column].astype(str).str.lower()
            if values.str.contains("demo|prototype_only|synthetic_fallback", regex=True).any():
                return True
    return False


def _verify_embedded_hashes(root: Path, frame: pd.DataFrame) -> None:
    if not {"source_artifact", "source_artifact_sha256"}.issubset(frame.columns):
        return
    pairs = frame[["source_artifact", "source_artifact_sha256"]].dropna().drop_duplicates()
    if len(pairs) > 100:
        raise FigureInputError("input has more than 100 distinct embedded source hashes")
    for source_value, expected_value in pairs.itertuples(index=False, name=None):
        source = Path(str(source_value))
        if not source.is_absolute():
            source = root / source
        expected = str(expected_value).strip().lower()
        if len(expected) != 64 or not source.is_file() or _sha256(source) != expected:
            raise FigureInputError(f"embedded source hash is invalid for {source_value}")


def _verify_dart_table_sidecar(root: Path, path: Path, digest: str) -> None:
    from .paper_tables_reports import validate_table_bundle

    valid, reason, validated = validate_table_bundle(path)
    if not valid:
        raise FigureInputError(f"invalid DART canonical table bundle: {reason}")
    sidecar_path = Path(str(path) + ".source.json")
    if not sidecar_path.is_file():
        raise FigureInputError(f"DART canonical table lacks source sidecar: {path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FigureInputError(f"invalid DART table sidecar: {type(exc).__name__}") from exc
    if sidecar.get("completed") is not True or sidecar.get("status") != "COMPLETE":
        raise FigureInputError("DART table sidecar is incomplete")
    if str(sidecar.get("output", {}).get("sha256", "")) != digest:
        raise FigureInputError("DART table hash disagrees with its sidecar")
    if not isinstance(sidecar.get("denominator_contract"), Mapping) or not sidecar[
        "denominator_contract"
    ]:
        raise FigureInputError("DART table sidecar lacks denominator semantics")
    if validated.get("provenance_contract", {}).get(
        "demo_synthetic_partial_sources_accepted"
    ) is not False:
        raise FigureInputError("DART table sidecar lacks strict provenance gates")
    for record in sidecar.get("source_files", []):
        if not isinstance(record, Mapping):
            raise FigureInputError("DART table sidecar has malformed source provenance")
        source = Path(str(record.get("path", "")))
        if not source.is_absolute():
            source = root / source
        if not source.is_file() or _sha256(source) != str(record.get("sha256", "")):
            raise FigureInputError("DART table source provenance hash changed")


def _load_source(
    root: Path,
    path: Path,
    *,
    expected_sha256: str | None,
    allow_test_fixtures: bool,
) -> LoadedSource:
    path = path.resolve()
    if not path.is_file():
        raise FigureInputError(f"source file is missing: {path}")
    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise FigureInputError(f"declared SHA-256 does not match {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.is_file():
        declared = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
        if len(declared) != 64 or declared != digest:
            raise FigureInputError(f"hash sidecar does not match {path}")
    if not allow_test_fixtures and "artifacts/demo/" in path.as_posix().lower():
        raise FigureInputError("demo artifacts cannot source formal paper figures")
    frame: pd.DataFrame | None = None
    payload: Any = None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            frame = pd.json_normalize(payload)
        elif isinstance(payload, Mapping):
            status = str(payload.get("status", "")).strip().upper()
            if status in {"FAILED", "INVALID", "PARTIAL", "IN_PROGRESS", "MISSING_MANDATORY"}:
                raise FigureInputError(f"JSON source status is {status}")
            if payload.get("completed") is False and status:
                raise FigureInputError("JSON source has completed=false")
    else:
        raise FigureInputError(f"unsupported figure input: {path.suffix}")
    if frame is not None:
        if frame.empty:
            raise FigureInputError(f"source table is empty: {path}")
        if not allow_test_fixtures and _is_synthetic_frame(frame):
            raise FigureInputError("synthetic/demo rows cannot source formal paper figures")
        _verify_embedded_hashes(root, frame)
        if not allow_test_fixtures and path.name.startswith("dart_"):
            _verify_dart_table_sidecar(root, path, digest)
    return LoadedSource(path, digest, path.stat().st_size, frame, payload)


def _explicit_source_entries(
    branch_results: Mapping[str, Any], definition: FigureDefinition
) -> list[tuple[str, str | None]]:
    mapping = branch_results.get("figure_inputs", {})
    if not isinstance(mapping, Mapping):
        return []
    value = None
    for key in (definition.name, Path(definition.name).stem, definition.relative_path):
        if key in mapping:
            value = mapping[key]
            break
    if value is None:
        return []
    entries: list[tuple[str, str | None]] = []
    if isinstance(value, (str, Path)):
        return [(str(value), None)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        return [(str(item), None) for item in value]
    if not isinstance(value, Mapping):
        raise FigureInputError(f"invalid figure input mapping for {definition.name}")
    sources = value.get("sources", value.get("source", value.get("path")))
    expected = value.get("sha256")
    source_hashes = value.get("source_hashes", {})
    if isinstance(sources, (str, Path)):
        sources = [sources]
    if not isinstance(sources, Sequence):
        raise FigureInputError(f"figure mapping has no sources for {definition.name}")
    for item in sources:
        item_text = str(item)
        item_hash = None
        if isinstance(source_hashes, Mapping):
            item_hash = source_hashes.get(item_text, source_hashes.get(Path(item_text).name))
        if item_hash is None and len(sources) == 1:
            item_hash = expected
        entries.append((item_text, str(item_hash).lower() if item_hash else None))
    return entries


def _resolve_sources(
    root: Path,
    definition: FigureDefinition,
    branch_results: Mapping[str, Any],
) -> list[tuple[Path, str | None]]:
    explicit = _explicit_source_entries(branch_results, definition)
    entries = explicit or [(pattern, None) for pattern in definition.source_patterns]
    resolved: list[tuple[Path, str | None]] = []
    seen: set[str] = set()
    for value, expected in entries:
        candidate = Path(value)
        if candidate.is_absolute() and not any(token in value for token in "*?["):
            paths = [candidate] if candidate.is_file() else []
        else:
            paths = sorted(root.glob(value), key=lambda item: item.as_posix())
        for path in paths:
            key = str(path.resolve()).lower()
            if key not in seen:
                resolved.append((path, expected))
                seen.add(key)
    return resolved


def _column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _required_columns(
    frame: pd.DataFrame, mapping: Mapping[str, Sequence[str]]
) -> dict[str, str] | None:
    resolved: dict[str, str] = {}
    for logical, aliases in mapping.items():
        column = _column(frame, aliases)
        if column is None:
            return None
        resolved[logical] = column
    return resolved


def _find_frame(
    sources: Sequence[LoadedSource],
    mapping: Mapping[str, Sequence[str]],
) -> tuple[LoadedSource, pd.DataFrame, dict[str, str]]:
    for source in sources:
        if source.frame is None:
            continue
        columns = _required_columns(source.frame, mapping)
        if columns is not None:
            return source, source.frame.copy(), columns
    required = ", ".join(mapping)
    raise FigureInputError(f"no source contains required semantic columns: {required}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _style_axis(ax: Any) -> None:
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
    ax.tick_params(width=0.6, length=3)


def _panel(ax: Any, label: str) -> None:
    ax.text(
        -0.08,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _figure(width: float = 7.2, height: float = 4.2, **kwargs: Any) -> Any:
    return plt.figure(figsize=(width, height), constrained_layout=True, **kwargs)


def _timeline_rows(sources: Sequence[LoadedSource]) -> tuple[list[dict[str, Any]], list[LoadedSource]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    direct_mapping = {
        "dataset": ("dataset", "task", "source", "domain"),
        "split": ("split", "partition", "phase"),
        "start": ("start_time", "start_utc", "target_min_timestamp_utc", "start"),
        "end": ("end_time", "end_utc", "target_max_timestamp_utc", "end"),
    }
    for source in sources:
        contributed = False
        if source.frame is not None:
            columns = _required_columns(source.frame, direct_mapping)
            if columns is not None:
                for record in source.frame.to_dict(orient="records"):
                    rows.append(
                        {
                            "dataset": str(record[columns["dataset"]]),
                            "split": str(record[columns["split"]]),
                            "start": record[columns["start"]],
                            "end": record[columns["end"]],
                            "row_count": record.get("row_count"),
                            "source": _relative(Path.cwd(), source.path),
                        }
                    )
                contributed = True
        payload = source.payload
        if isinstance(payload, Mapping):
            tasks = payload.get("tasks")
            if isinstance(tasks, Mapping):
                for task, task_payload in tasks.items():
                    splits = task_payload.get("splits", {}) if isinstance(task_payload, Mapping) else {}
                    if not isinstance(splits, Mapping):
                        continue
                    for split, split_payload in splits.items():
                        if not isinstance(split_payload, Mapping):
                            continue
                        start = split_payload.get(
                            "target_min_timestamp_utc",
                            split_payload.get("prediction_origin_min_utc"),
                        )
                        end = split_payload.get(
                            "target_max_timestamp_utc",
                            split_payload.get("prediction_origin_max_utc"),
                        )
                        if start is not None and end is not None:
                            rows.append(
                                {
                                    "dataset": str(task),
                                    "split": str(split),
                                    "start": start,
                                    "end": end,
                                    "row_count": split_payload.get("row_count"),
                                    "source": _relative(Path.cwd(), source.path),
                                }
                            )
                            contributed = True
            for dataset, split_payload in payload.items():
                if not isinstance(split_payload, Mapping):
                    continue
                start = split_payload.get("support_start_utc")
                end = split_payload.get("support_end_utc")
                train_end = split_payload.get("train_end_exclusive")
                validation_end = split_payload.get("validation_end_exclusive")
                if not all((start, end, train_end, validation_end)):
                    continue
                counts = split_payload.get("row_counts", {})
                for split, left, right in (
                    ("train", start, train_end),
                    ("validation", train_end, validation_end),
                    ("locked_test", validation_end, end),
                ):
                    rows.append(
                        {
                            "dataset": str(dataset),
                            "split": split,
                            "start": left,
                            "end": right,
                            "row_count": counts.get(split) if isinstance(counts, Mapping) else None,
                            "source": _relative(Path.cwd(), source.path),
                        }
                    )
                contributed = True
        if contributed:
            used.append(source)
    return rows, used


def _render_timeline(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    rows, used = _timeline_rows(sources)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FigureInputError("timeline inputs contain no split start/end records")
    frame["start"] = _time(frame["start"])
    frame["end"] = _time(frame["end"])
    frame = frame.dropna(subset=["start", "end"])
    frame = frame.loc[frame["end"] > frame["start"]].copy()
    if frame.empty:
        raise FigureInputError("timeline has no valid chronological intervals")
    frame = frame.sort_values(["dataset", "start", "split"]).drop_duplicates(
        ["dataset", "split", "start", "end"]
    )
    datasets = list(dict.fromkeys(frame["dataset"].astype(str)))
    palette = {"train": COLORS["navy"], "validation": COLORS["gold"], "locked_test": COLORS["red"]}
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.42 * len(datasets) + 1.5)), constrained_layout=True)
    for y, dataset in enumerate(datasets):
        subset = frame.loc[frame["dataset"].astype(str).eq(dataset)]
        for _, row in subset.iterrows():
            left = row["start"]
            width_days = (row["end"] - left).total_seconds() / 86400.0
            ax.barh(
                y,
                width_days,
                left=left,
                height=0.55,
                color=palette.get(str(row["split"]).lower(), COLORS["teal"]),
                edgecolor="white",
                linewidth=0.5,
                label=str(row["split"]),
            )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncol=min(3, len(unique)), loc="upper center")
    ax.set_yticks(range(len(datasets)), datasets)
    ax.invert_yaxis()
    ax.set_xlabel("Calendar time (UTC)")
    ax.set_title(definition.title)
    _style_axis(ax)
    plot_rows = _records(frame, ["dataset", "split", "start", "end", "row_count"])
    return PlotResult(
        fig,
        plot_rows,
        used,
        {"interval_count": len(frame), "dataset_count": len(datasets)},
        f"Chronological windows; n={len(frame)} domain-split intervals.",
    )


def _walk_missingness(value: Any, path: tuple[str, ...], output: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        raw = value.get("raw_rows", value.get("row_count"))
        valid = value.get("valid_timestamp_rows", value.get("valid_rows"))
        missing = value.get("missing_rows", value.get("null_timestamp_rows"))
        try:
            raw_value = float(raw) if raw is not None else None
            if valid is not None and raw_value and raw_value > 0:
                missing_value = max(raw_value - float(valid), 0.0)
            elif missing is not None and raw_value and raw_value > 0:
                missing_value = max(float(missing), 0.0)
            else:
                missing_value = None
            if missing_value is not None:
                output.append(
                    {
                        "dataset": "/".join(path[:2] or ("source",)),
                        "missing_rows": missing_value,
                        "row_count": raw_value,
                    }
                )
        except (TypeError, ValueError):
            pass
        for key, nested in value.items():
            _walk_missingness(nested, path + (str(key),), output)
    elif isinstance(value, list):
        for nested in value:
            _walk_missingness(nested, path, output)


def _render_missingness(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    for source in sources:
        count_before = len(rows)
        if source.frame is not None:
            dataset = _column(source.frame, ("dataset", "source", "task", "field"))
            fraction = _column(source.frame, ("missing_fraction", "missingness", "null_fraction"))
            missing = _column(source.frame, ("missing_rows", "null_rows"))
            total = _column(source.frame, ("row_count", "raw_rows", "total_rows"))
            if dataset and fraction:
                for label, value in zip(source.frame[dataset], _numeric(source.frame[fraction])):
                    rows.append({"dataset": str(label), "missing_fraction": value, "row_count": None})
            elif dataset and missing and total:
                for label, miss, n in zip(
                    source.frame[dataset], _numeric(source.frame[missing]), _numeric(source.frame[total])
                ):
                    if pd.notna(n) and n > 0:
                        rows.append({"dataset": str(label), "missing_fraction": float(miss / n), "row_count": float(n)})
        if source.payload is not None:
            nested: list[dict[str, Any]] = []
            _walk_missingness(source.payload, (), nested)
            for row in nested:
                rows.append(
                    {
                        **row,
                        "missing_fraction": row["missing_rows"] / row["row_count"],
                    }
                )
        if len(rows) > count_before:
            used.append(source)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FigureInputError("missingness inputs contain no raw/valid or missing-fraction fields")
    frame["missing_fraction"] = _numeric(frame["missing_fraction"])
    frame = frame.dropna(subset=["missing_fraction"])
    frame = frame.loc[frame["missing_fraction"].between(0, 1)].copy()
    if frame.empty:
        raise FigureInputError("missingness fractions are outside [0,1]")
    grouped = (
        frame.groupby("dataset", as_index=False)
        .agg(missing_fraction=("missing_fraction", "mean"), source_records=("missing_fraction", "size"))
        .sort_values("missing_fraction", ascending=False)
        .head(20)
        .sort_values("missing_fraction")
    )
    fig, ax = plt.subplots(figsize=(7.2, max(3.1, 0.28 * len(grouped) + 1.4)), constrained_layout=True)
    ax.barh(grouped["dataset"], 100.0 * grouped["missing_fraction"], color=COLORS["blue"], edgecolor=COLORS["dark"], linewidth=0.4)
    ax.set_xlabel("Missing source rows (%)")
    ax.set_title(definition.title)
    _style_axis(ax)
    return PlotResult(
        fig,
        _records(grouped),
        used,
        {"displayed_group_count": len(grouped), "source_record_count": len(frame)},
        f"Source-row missingness; n={len(frame)} count records.",
    )


def _median_week(frame: pd.DataFrame, timestamp: str, value: str) -> pd.DataFrame:
    working = frame.copy()
    working["_timestamp"] = _time(working[timestamp])
    working["_value"] = _numeric(working[value])
    working = working.dropna(subset=["_timestamp", "_value"]).sort_values("_timestamp")
    if working.empty:
        raise FigureInputError("time-series source has no finite timestamp/value rows")
    working["_week"] = (
        working["_timestamp"].dt.floor("D")
        - pd.to_timedelta(working["_timestamp"].dt.weekday, unit="D")
    )
    summary = working.groupby("_week").agg(rows=("_value", "size"), mean_value=("_value", "mean"))
    eligible = summary.loc[summary["rows"] >= 48]
    if eligible.empty:
        raise FigureInputError("no week has at least 48 observed intervals")
    median_value = float(eligible["mean_value"].median())
    selected_week = (
        eligible.assign(distance=(eligible["mean_value"] - median_value).abs())
        .sort_values(["distance", "_week"])
        .index[0]
    )
    selected = working.loc[working["_week"].eq(selected_week)].copy()
    if (selected["_timestamp"].max() - selected["_timestamp"].min()) < pd.Timedelta(days=5):
        raise FigureInputError("selected representative week covers fewer than six calendar days")
    return selected


def _render_physical_week(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    mapping = {
        "timestamp": ("timestamp_utc", "timestamp", "ts"),
        "facility": ("facility_power_kw", "observed_facility_kw", "facility_kw"),
        "it": ("it_power_kw", "observed_it_power_kw", "it_kw"),
        "pue": ("pue", "observed_pue"),
    }
    source, frame, columns = _find_frame(sources, mapping)
    selected = _median_week(frame, columns["timestamp"], columns["facility"])
    selected["facility_power_kw"] = _numeric(selected[columns["facility"]])
    selected["it_power_kw"] = _numeric(selected[columns["it"]])
    selected["pue_value"] = _numeric(selected[columns["pue"]])
    selected = selected.dropna(subset=["facility_power_kw", "it_power_kw", "pue_value"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.plot(selected["_timestamp"], selected["facility_power_kw"], color=COLORS["navy"], lw=1.1, label="Facility power")
    ax.plot(selected["_timestamp"], selected["it_power_kw"], color=COLORS["teal"], lw=1.0, ls="--", label="IT power")
    ax.set_ylabel("Power (kW)")
    ax2 = ax.twinx()
    ax2.plot(selected["_timestamp"], selected["pue_value"], color=COLORS["gold"], lw=0.9, ls=":", label="PUE")
    ax2.set_ylabel("PUE")
    handles = ax.get_lines() + ax2.get_lines()
    ax.legend(handles, [line.get_label() for line in handles], ncol=3, loc="upper center")
    ax.set_xlabel("UTC")
    ax.set_title(definition.title)
    _style_axis(ax)
    rows = selected[["_timestamp", "facility_power_kw", "it_power_kw", "pue_value"]].rename(columns={"_timestamp": "timestamp_utc"})
    return PlotResult(
        fig,
        _records(rows),
        [source],
        {"week_start_utc": str(selected["_week"].iloc[0]), "interval_count": len(selected)},
        f"Median-load source week; n={len(selected)} intervals.",
    )


def _sum_array(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=float)
        return float(np.nansum(array)) if array.size else 0.0
    except (TypeError, ValueError):
        return None


def _dispatch_rows(sources: Sequence[LoadedSource]) -> tuple[pd.DataFrame, list[LoadedSource]]:
    rows: list[dict[str, Any]] = []
    used: list[LoadedSource] = []
    tabular = {
        "timestamp": ("timestamp_utc", "timestamp", "interval_time"),
        "grid": ("grid_import_mw", "simulated_grid_import_mw", "grid_mw"),
        "schedule": ("day_ahead_schedule_mw", "da_schedule_mw", "schedule_mw"),
        "price": ("rt_price_usd_per_mwh", "mean_rt_price_usd_per_mwh", "price_usd_per_mwh"),
    }
    for source in sources:
        before = len(rows)
        if source.frame is not None:
            columns = _required_columns(source.frame, tabular)
            if columns is not None:
                optional = {
                    logical: _column(source.frame, aliases)
                    for logical, aliases in {
                        "pv": ("pv_used_mw", "pv_available_mw", "pv_mw"),
                        "charge": ("battery_charge_mw", "charge_mw"),
                        "discharge": ("battery_discharge_mw", "discharge_mw"),
                        "soc": ("soc_mwh", "soc_end_mwh", "battery_soc_mwh"),
                        "work": ("work_work_units", "completed_work_units", "work_units"),
                        "policy": ("policy_id", "policy"),
                    }.items()
                }
                for record in source.frame.to_dict(orient="records"):
                    row = {
                        "timestamp_utc": record[columns["timestamp"]],
                        "grid_import_mw": record[columns["grid"]],
                        "day_ahead_schedule_mw": record[columns["schedule"]],
                        "rt_price_usd_per_mwh": record[columns["price"]],
                    }
                    for logical, column in optional.items():
                        row[logical] = record.get(column) if column else None
                    rows.append(row)
        payload = source.payload
        if isinstance(payload, Mapping) and isinstance(payload.get("intervals"), list):
            if payload.get("status") != "COMPLETE" or payload.get("future_realized_targets_used") is not False:
                continue
            policy = payload.get("policy_id", payload.get("policy"))
            for interval in payload["intervals"]:
                if not isinstance(interval, Mapping):
                    continue
                dispatch = interval.get("dispatch", {})
                revealed = interval.get("revealed", {})
                service = interval.get("service", {})
                if not isinstance(dispatch, Mapping) or not isinstance(revealed, Mapping):
                    continue
                rows.append(
                    {
                        "timestamp_utc": interval.get("timestamp_utc"),
                        "grid_import_mw": _sum_array(dispatch.get("grid_import_mw")),
                        "day_ahead_schedule_mw": _sum_array(dispatch.get("day_ahead_schedule_mw")),
                        "rt_price_usd_per_mwh": revealed.get("mean_rt_price_usd_per_mwh"),
                        "pv": _sum_array(dispatch.get("pv_used_mw")),
                        "charge": _sum_array(dispatch.get("battery_charge_mw")),
                        "discharge": _sum_array(dispatch.get("battery_discharge_mw")),
                        "soc": _sum_array(dispatch.get("soc_end_mwh")),
                        "work": service.get("completed_work_units") if isinstance(service, Mapping) else None,
                        "policy": policy,
                    }
                )
        if len(rows) > before:
            used.append(source)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FigureInputError("rolling sources contain no interval dispatch rows")
    frame["timestamp_utc"] = _time(frame["timestamp_utc"])
    for column in (
        "grid_import_mw",
        "day_ahead_schedule_mw",
        "rt_price_usd_per_mwh",
        "pv",
        "charge",
        "discharge",
        "soc",
        "work",
    ):
        if column in frame:
            frame[column] = _numeric(frame[column])
    return frame.dropna(subset=["timestamp_utc", "grid_import_mw", "day_ahead_schedule_mw", "rt_price_usd_per_mwh"]).sort_values("timestamp_utc"), used


def _render_counterfactual_week(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    frame, used = _dispatch_rows(sources)
    frame["_week"] = frame["timestamp_utc"].dt.floor("D") - pd.to_timedelta(frame["timestamp_utc"].dt.weekday, unit="D")
    summary = frame.groupby("_week").agg(days=("timestamp_utc", lambda x: x.dt.floor("D").nunique()), mean_grid=("grid_import_mw", "mean"))
    eligible = summary.loc[summary["days"] >= 7]
    if eligible.empty:
        raise FigureInputError("counterfactual representative week requires seven completed operating days")
    median_grid = float(eligible["mean_grid"].median())
    selected_week = eligible.assign(distance=(eligible["mean_grid"] - median_grid).abs()).sort_values(["distance", "_week"]).index[0]
    selected = frame.loc[frame["_week"].eq(selected_week)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    axes[0].plot(selected["timestamp_utc"], selected["grid_import_mw"], color=COLORS["navy"], lw=1.0, label="Grid import")
    axes[0].plot(selected["timestamp_utc"], selected["day_ahead_schedule_mw"], color=COLORS["gold"], ls="--", lw=0.9, label="DA schedule")
    if "pv" in selected and selected["pv"].notna().any():
        axes[0].plot(selected["timestamp_utc"], selected["pv"], color=COLORS["green"], ls=":", lw=0.9, label="PV used")
    axes[0].set_ylabel("Power (MW)")
    axes[0].legend(ncol=3)
    axes[1].plot(selected["timestamp_utc"], selected["rt_price_usd_per_mwh"], color=COLORS["red"], lw=0.9)
    axes[1].axhline(0, color=COLORS["gray"], lw=0.6)
    axes[1].set_ylabel("RT price\n(USD MWh$^{-1}$)")
    axes[1].set_xlabel("UTC")
    axes[0].set_title(definition.title)
    for label, ax in zip(("a", "b"), axes):
        _panel(ax, label)
        _style_axis(ax)
    columns = [column for column in ("timestamp_utc", "grid_import_mw", "day_ahead_schedule_mw", "rt_price_usd_per_mwh", "pv", "charge", "discharge", "soc", "work", "policy") if column in selected]
    return PlotResult(
        fig,
        _records(selected, columns),
        used,
        {"week_start_utc": str(selected_week), "interval_count": len(selected)},
        f"Median-grid-energy counterfactual week; n={len(selected)} rolling intervals.",
    )


def _render_physical_calibration(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    mapping = {
        "model": ("model", "model_name"),
        "split": ("split", "partition"),
        "rmse": ("facility_rmse_kw", "rmse_kw", "facility_rmse"),
    }
    source, frame, columns = _find_frame(sources, mapping)
    frame["rmse_kw"] = _numeric(frame[columns["rmse"]])
    frame["model_label"] = frame[columns["model"]].astype(str)
    frame["split_label"] = frame[columns["split"]].astype(str)
    locked = frame.loc[frame["split_label"].str.lower().isin({"locked_test", "test", "held_out", "held-out"})].copy()
    if locked.empty:
        raise FigureInputError("physical calibration has no locked-test rows")
    locked = locked.dropna(subset=["rmse_kw"])
    grouped = locked.groupby("model_label", as_index=False).agg(rmse_kw=("rmse_kw", "mean"), source_rows=("rmse_kw", "size"))
    if grouped["model_label"].nunique() < 2:
        raise FigureInputError("physical calibration requires at least two held-out models")
    pue_col = _column(locked, ("pue_mae", "pue_error"))
    if pue_col:
        locked["pue_mae_value"] = _numeric(locked[pue_col])
        pue = locked.groupby("model_label", as_index=False).agg(pue_mae=("pue_mae_value", "mean"))
        grouped = grouped.merge(pue, on="model_label", how="left")
    fig, axes = plt.subplots(1, 2 if pue_col else 1, figsize=(7.2, 3.6), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    colors = [COLORS["blue"] if "calibrat" in label.lower() else COLORS["light"] for label in grouped["model_label"]]
    axes_array[0].bar(grouped["model_label"], grouped["rmse_kw"], color=colors, edgecolor=COLORS["dark"], linewidth=0.4)
    axes_array[0].set_ylabel("Facility RMSE (kW)")
    axes_array[0].tick_params(axis="x", rotation=25)
    _style_axis(axes_array[0])
    _panel(axes_array[0], "a")
    if pue_col:
        axes_array[1].bar(grouped["model_label"], grouped["pue_mae"], color=colors, edgecolor=COLORS["dark"], linewidth=0.4)
        axes_array[1].set_ylabel("PUE MAE")
        axes_array[1].tick_params(axis="x", rotation=25)
        _style_axis(axes_array[1])
        _panel(axes_array[1], "b")
    fig.suptitle(definition.title)
    return PlotResult(fig, _records(grouped), [source], {"held_out_row_count": len(locked), "model_count": len(grouped)}, f"Locked-test physical calibration; n={len(locked)} metric rows.")


def _render_component_closure(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    mapping = {
        "facility": ("facility_power_kw", "observed_facility_kw"),
        "component": ("component_power_with_documented_residual_kw", "component_power_kw", "component_sum_kw"),
    }
    source, frame, columns = _find_frame(sources, mapping)
    working = pd.DataFrame(
        {
            "facility_power_kw": _numeric(frame[columns["facility"]]),
            "component_power_kw": _numeric(frame[columns["component"]]),
        }
    ).dropna()
    if len(working) < 20:
        raise FigureInputError("component closure requires at least 20 paired intervals")
    working["closure_residual_kw"] = working["facility_power_kw"] - working["component_power_kw"]
    if len(working) > 2000:
        indices = np.linspace(0, len(working) - 1, 2000, dtype=int)
        plotted = working.iloc[indices].copy()
    else:
        plotted = working
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    axes[0].scatter(plotted["component_power_kw"], plotted["facility_power_kw"], s=5, alpha=0.35, color=COLORS["blue"], rasterized=True)
    low = float(min(plotted["component_power_kw"].min(), plotted["facility_power_kw"].min()))
    high = float(max(plotted["component_power_kw"].max(), plotted["facility_power_kw"].max()))
    axes[0].plot([low, high], [low, high], color=COLORS["dark"], ls="--", lw=0.8)
    axes[0].set_xlabel("Component sum (kW)")
    axes[0].set_ylabel("Facility power (kW)")
    axes[1].hist(plotted["closure_residual_kw"], bins=30, color=COLORS["teal"], edgecolor="white")
    axes[1].axvline(0, color=COLORS["dark"], ls="--", lw=0.8)
    axes[1].set_xlabel("Closure residual (kW)")
    axes[1].set_ylabel("Intervals")
    for label, ax in zip(("a", "b"), axes):
        _panel(ax, label)
        _style_axis(ax)
    fig.suptitle(definition.title)
    return PlotResult(fig, _records(plotted), [source], {"paired_interval_count": len(working), "displayed_interval_count": len(plotted)}, f"Component closure; n={len(working)} paired source intervals.")


def _render_transfer_support(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    mapping = {
        "site": ("site_id", "zone", "site"),
        "distance": ("climate_support_distance", "support_distance"),
        "outside": ("out_of_support_fraction", "climate_extrapolation_rate"),
        "median": ("physical_multiplier_median", "multiplier_median"),
        "low": ("physical_multiplier_p05", "multiplier_p05", "multiplier_low"),
        "high": ("physical_multiplier_p95", "multiplier_p95", "multiplier_high"),
    }
    source, frame, columns = _find_frame(sources, mapping)
    working = pd.DataFrame(
        {
            "site": frame[columns["site"]].astype(str),
            "climate_support_distance": _numeric(frame[columns["distance"]]),
            "out_of_support_fraction": _numeric(frame[columns["outside"]]),
            "physical_multiplier_median": _numeric(frame[columns["median"]]),
            "physical_multiplier_p05": _numeric(frame[columns["low"]]),
            "physical_multiplier_p95": _numeric(frame[columns["high"]]),
        }
    ).dropna()
    if working.empty:
        raise FigureInputError("transfer support has no complete site rows")
    working = working.sort_values("site")
    y = np.arange(len(working))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(3.2, 0.42 * len(working) + 1.5)), constrained_layout=True)
    lower = working["physical_multiplier_median"] - working["physical_multiplier_p05"]
    upper = working["physical_multiplier_p95"] - working["physical_multiplier_median"]
    axes[0].errorbar(working["physical_multiplier_median"], y, xerr=np.vstack([lower, upper]), fmt="o", color=COLORS["navy"], capsize=3)
    axes[0].axvline(1.0, color=COLORS["gray"], ls="--", lw=0.7)
    axes[0].set_yticks(y, working["site"])
    axes[0].set_xlabel("Physical multiplier (P05-P95)")
    axes[1].barh(y - 0.14, working["climate_support_distance"], height=0.28, color=COLORS["violet"], label="Climate distance")
    axes[1].barh(y + 0.14, working["out_of_support_fraction"], height=0.28, color=COLORS["gold"], label="Outside support")
    axes[1].set_yticks(y, working["site"])
    axes[1].set_xlabel("Support diagnostic")
    axes[1].legend()
    for label, ax in zip(("a", "b"), axes):
        _panel(ax, label)
        _style_axis(ax)
    fig.suptitle(definition.title)
    return PlotResult(fig, _records(working), [source], {"site_count": len(working)}, f"Twin-transfer support; n={len(working)} counterfactual sites.")


def _calibration_rows(
    definition: FigureDefinition,
    sources: Sequence[LoadedSource],
) -> tuple[pd.DataFrame, LoadedSource]:
    task = {
        "calibration_da": "nyiso_da",
        "calibration_rt": "nyiso_rt",
        "calibration_workload": "nlr_workload",
    }[definition.renderer]
    for source in sources:
        if source.frame is None:
            continue
        frame = source.frame.copy()
        task_column = _column(frame, ("task", "forecast_task"))
        if task_column:
            frame = frame.loc[frame[task_column].astype(str).str.lower().eq(task)].copy()
        if frame.empty:
            continue
        split_column = _column(frame, ("split", "partition"))
        if split_column:
            locked = frame.loc[
                frame[split_column].astype(str).str.lower().isin(
                    {"locked_test", "test", "held_out", "held-out"}
                )
            ].copy()
            if not locked.empty:
                frame = locked
        observed = _column(
            frame,
            ("observed_target", "actual", "y_true", "observed_job_count"),
        )
        q05 = _column(frame, ("q05", "lower_90", "lower_95", "prediction_lower_90"))
        q10 = _column(frame, ("q10", "lower_80", "prediction_lower_80"))
        q90 = _column(frame, ("q90", "upper_80", "prediction_upper_80"))
        q95 = _column(frame, ("q95", "upper_90", "upper_95", "prediction_upper_90"))
        if observed and q05 and q95:
            group_columns = [
                column
                for column in (
                    split_column,
                    _column(frame, ("zone", "cohort")),
                    _column(frame, ("model", "model_class")),
                )
                if column
            ]
            if not group_columns:
                frame["_group"] = task
                group_columns = ["_group"]
            rows: list[dict[str, Any]] = []
            for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
                key_values = keys if isinstance(keys, tuple) else (keys,)
                label = ":".join(str(value) for value in key_values)
                observed_values = _numeric(group[observed])
                lower90 = _numeric(group[q05])
                upper90 = _numeric(group[q95])
                valid90 = observed_values.notna() & lower90.notna() & upper90.notna()
                if valid90.any():
                    rows.append(
                        {
                            "group": label,
                            "nominal_coverage": 0.90,
                            "empirical_coverage": float(
                                (
                                    (observed_values[valid90] >= lower90[valid90])
                                    & (observed_values[valid90] <= upper90[valid90])
                                ).mean()
                            ),
                            "mean_interval_width": float(
                                (upper90[valid90] - lower90[valid90]).mean()
                            ),
                            "n": int(valid90.sum()),
                        }
                    )
                if q10 and q90:
                    lower80 = _numeric(group[q10])
                    upper80 = _numeric(group[q90])
                    valid80 = observed_values.notna() & lower80.notna() & upper80.notna()
                    if valid80.any():
                        rows.append(
                            {
                                "group": label,
                                "nominal_coverage": 0.80,
                                "empirical_coverage": float(
                                    (
                                        (observed_values[valid80] >= lower80[valid80])
                                        & (observed_values[valid80] <= upper80[valid80])
                                    ).mean()
                                ),
                                "mean_interval_width": float(
                                    (upper80[valid80] - lower80[valid80]).mean()
                                ),
                                "n": int(valid80.sum()),
                            }
                        )
            result = pd.DataFrame(rows)
            if not result.empty:
                return result, source
        nominal = _column(frame, ("nominal_coverage", "nominal", "target_coverage"))
        empirical = _column(frame, ("empirical_coverage", "coverage", "observed_coverage"))
        if nominal and empirical:
            result = pd.DataFrame(
                {
                    "group": (
                        frame[_column(frame, ("model", "group", "zone", "split"))].astype(str)
                        if _column(frame, ("model", "group", "zone", "split"))
                        else task
                    ),
                    "nominal_coverage": _numeric(frame[nominal]),
                    "empirical_coverage": _numeric(frame[empirical]),
                    "mean_interval_width": _numeric(
                        frame[_column(frame, ("mean_interval_width", "mean_width", "interval_width"))]
                    )
                    if _column(frame, ("mean_interval_width", "mean_width", "interval_width"))
                    else np.nan,
                    "n": _numeric(frame[_column(frame, ("n", "sample_count", "row_count"))])
                    if _column(frame, ("n", "sample_count", "row_count"))
                    else len(frame),
                }
            ).dropna(subset=["nominal_coverage", "empirical_coverage"])
            if not result.empty:
                return result, source
        wide_rows: list[dict[str, Any]] = []
        group_column = _column(frame, ("model", "group", "zone", "split"))
        width_column = _column(frame, ("mean_width_90", "mean_interval_width", "mean_width"))
        n_column = _column(frame, ("n", "sample_count", "row_count"))
        for nominal_level, aliases in (
            (0.80, ("coverage_80", "empirical_coverage_80")),
            (0.90, ("coverage_90", "empirical_coverage_90")),
        ):
            coverage = _column(frame, aliases)
            if coverage is None:
                continue
            for index, value in _numeric(frame[coverage]).items():
                if pd.isna(value):
                    continue
                wide_rows.append(
                    {
                        "group": str(frame.loc[index, group_column]) if group_column else task,
                        "nominal_coverage": nominal_level,
                        "empirical_coverage": float(value),
                        "mean_interval_width": (
                            float(frame.loc[index, width_column])
                            if width_column and pd.notna(frame.loc[index, width_column])
                            else np.nan
                        ),
                        "n": (
                            int(frame.loc[index, n_column])
                            if n_column and pd.notna(frame.loc[index, n_column])
                            else None
                        ),
                    }
                )
        if wide_rows:
            return pd.DataFrame(wide_rows), source
    raise FigureInputError(f"no complete interval-calibration fields for {task}")


def _render_calibration(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    frame, source = _calibration_rows(definition, sources)
    frame["nominal_coverage"] = _numeric(frame["nominal_coverage"])
    frame["empirical_coverage"] = _numeric(frame["empirical_coverage"])
    frame["mean_interval_width"] = _numeric(frame["mean_interval_width"])
    frame = frame.dropna(subset=["nominal_coverage", "empirical_coverage"])
    frame = frame.loc[
        frame["nominal_coverage"].between(0, 1)
        & frame["empirical_coverage"].between(0, 1)
    ].copy()
    if frame.empty:
        raise FigureInputError("calibration coverage is outside [0,1]")
    has_width = frame["mean_interval_width"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_width else 1, figsize=(7.2, 3.4), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for index, (group, group_frame) in enumerate(frame.groupby("group", sort=True)):
        ordered = group_frame.sort_values("nominal_coverage")
        axes_array[0].plot(
            ordered["nominal_coverage"],
            ordered["empirical_coverage"],
            marker=MARKERS[index % len(MARKERS)],
            ls=LINESTYLES[index % len(LINESTYLES)],
            color=(COLORS["navy"], COLORS["teal"], COLORS["violet"], COLORS["gold"])[index % 4],
            label=str(group),
        )
    axes_array[0].plot([0, 1], [0, 1], color=COLORS["gray"], ls="--", lw=0.7, label="Nominal")
    axes_array[0].set_xlim(0.72, 0.98)
    axes_array[0].set_ylim(0.72, 0.98)
    axes_array[0].set_xlabel("Nominal coverage")
    axes_array[0].set_ylabel("Empirical coverage")
    axes_array[0].legend(fontsize=6)
    _panel(axes_array[0], "a")
    _style_axis(axes_array[0])
    if has_width:
        width_rows = frame.dropna(subset=["mean_interval_width"]).copy()
        labels = width_rows["group"].astype(str) + " / " + width_rows["nominal_coverage"].map(lambda value: f"{value:.0%}")
        axes_array[1].bar(labels, width_rows["mean_interval_width"], color=COLORS["blue"], edgecolor=COLORS["dark"], linewidth=0.4)
        axes_array[1].set_ylabel("Mean interval width")
        axes_array[1].tick_params(axis="x", rotation=30)
        _panel(axes_array[1], "b")
        _style_axis(axes_array[1])
    fig.suptitle(definition.title)
    sample_count = int(_numeric(frame["n"]).dropna().sum()) if "n" in frame else len(frame)
    return PlotResult(
        fig,
        _records(frame),
        [source],
        {"calibration_point_count": len(frame), "reported_sample_count": sample_count},
        f"Coverage calibration; reported n={sample_count} target observations across displayed groups.",
    )


def _locked_subset(frame: pd.DataFrame) -> pd.DataFrame:
    split = _column(frame, ("split", "partition"))
    if split:
        selected = frame.loc[
            frame[split].astype(str).str.lower().isin(
                {"locked_test", "test", "held_out", "held-out"}
            )
        ].copy()
        if not selected.empty:
            return selected
    return frame.copy()


def _event_frame(
    sources: Sequence[LoadedSource],
    *,
    workload: bool,
) -> tuple[LoadedSource, pd.DataFrame, dict[str, str], list[str]]:
    required = {
        "timestamp": ("timestamp_utc", "timestamp", "target_timestamp_utc"),
        "observed": (
            ("observed_job_count", "observed_nodes_requested", "observed_target", "actual")
            if workload
            else ("observed_target", "actual_price", "price_usd_per_mwh", "actual")
        ),
    }
    for source in sources:
        if source.frame is None:
            continue
        columns = _required_columns(source.frame, required)
        if columns is None:
            continue
        predictions = [
            column
            for column in (
                _column(source.frame, ("predicted_job_count", "predicted_nodes_requested")) if workload else None,
                _column(source.frame, ("q50", "median_prediction")),
                _column(source.frame, ("advanced_lightgbm", "prediction", "forecast")),
                _column(source.frame, ("ridge_autoregression",)),
                _column(source.frame, ("baseline_seasonal",)),
            )
            if column is not None
        ]
        predictions = list(dict.fromkeys(predictions))
        if predictions:
            return source, _locked_subset(source.frame), columns, predictions[:3]
    raise FigureInputError("event-case source lacks timestamp, observed target, and prediction columns")


def _render_event_case(
    definition: FigureDefinition,
    sources: Sequence[LoadedSource],
    *,
    workload: bool,
) -> PlotResult:
    source, frame, columns, prediction_columns = _event_frame(sources, workload=workload)
    working = frame.copy()
    working["_timestamp"] = _time(working[columns["timestamp"]])
    working["_observed"] = _numeric(working[columns["observed"]])
    zone = _column(working, ("zone", "cohort"))
    working = working.dropna(subset=["_timestamp", "_observed"]).sort_values("_timestamp")
    if working.empty:
        raise FigureInputError("event-case source has no finite held-out target rows")
    peak_index = working["_observed"].idxmax() if workload else working["_observed"].abs().idxmax()
    if zone:
        peak_zone = working.loc[peak_index, zone]
        working = working.loc[working[zone].eq(peak_zone)].copy().sort_values("_timestamp")
        peak_index = working["_observed"].idxmax() if workload else working["_observed"].abs().idxmax()
    position = working.index.get_loc(peak_index)
    if not isinstance(position, (int, np.integer)):
        position = int(np.asarray(position).flat[0])
    radius = 48 if workload else 96
    selected = working.iloc[max(0, position - radius) : min(len(working), position + radius + 1)].copy()
    if len(selected) < 12:
        raise FigureInputError("event window has fewer than 12 target observations")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(selected["_timestamp"], selected["_observed"], color=COLORS["dark"], lw=1.4, label="Observed")
    plotted_columns = ["_timestamp", "_observed"]
    for index, column in enumerate(prediction_columns):
        selected[column] = _numeric(selected[column])
        if selected[column].notna().any():
            ax.plot(
                selected["_timestamp"],
                selected[column],
                color=(COLORS["navy"], COLORS["teal"], COLORS["violet"])[index % 3],
                ls=LINESTYLES[index % len(LINESTYLES)],
                lw=0.9,
                label=column.replace("_", " "),
            )
            plotted_columns.append(column)
    qlow = _column(selected, ("q05", "lower_90", "prediction_lower_90"))
    qhigh = _column(selected, ("q95", "upper_90", "prediction_upper_90"))
    if qlow and qhigh:
        selected[qlow] = _numeric(selected[qlow])
        selected[qhigh] = _numeric(selected[qhigh])
        valid = selected[qlow].notna() & selected[qhigh].notna()
        if valid.any():
            ax.fill_between(selected.loc[valid, "_timestamp"], selected.loc[valid, qlow], selected.loc[valid, qhigh], color=COLORS["blue"], alpha=0.15, linewidth=0, label="90% interval")
            plotted_columns.extend([qlow, qhigh])
    if not workload:
        ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.7)
        ax.set_ylabel("Price (USD MWh$^{-1}$)")
    else:
        ax.set_ylabel("Submitted workload")
    ax.axvline(working.loc[peak_index, "_timestamp"], color=COLORS["red"], ls=":", lw=0.8)
    ax.set_xlabel("UTC")
    ax.set_title(definition.title)
    ax.legend(ncol=min(4, len(ax.get_legend_handles_labels()[0])), fontsize=6)
    _style_axis(ax)
    rows = selected[plotted_columns].rename(columns={"_timestamp": "timestamp_utc", "_observed": "observed"})
    return PlotResult(
        fig,
        _records(rows),
        [source],
        {"event_timestamp_utc": str(working.loc[peak_index, "_timestamp"]), "window_row_count": len(selected), "selection": "maximum_observed" if workload else "maximum_absolute_observed"},
        f"Deterministically selected held-out event window; n={len(selected)} observations.",
    )


def _render_scenario_dependence(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    long_mapping = {
        "left": ("variable_x", "variable_1", "left_variable"),
        "right": ("variable_y", "variable_2", "right_variable"),
        "correlation": ("rank_correlation", "spearman_correlation", "correlation"),
    }
    try:
        source, frame, columns = _find_frame(sources, long_mapping)
    except FigureInputError:
        source, frame, columns = _find_frame(
            sources,
            {
                "regime": ("regime", "dependence_regime"),
                "correlation": ("mean_pairwise_rank_correlation", "rank_correlation"),
                "extreme": ("all_four_upper_quintile_probability", "joint_extreme_probability", "extreme_probability"),
            },
        )
        working = pd.DataFrame(
            {
                "regime": frame[columns["regime"]].astype(str),
                "mean_pairwise_rank_correlation": _numeric(frame[columns["correlation"]]),
                "joint_extreme_probability": _numeric(frame[columns["extreme"]]),
            }
        ).dropna()
        grouped = working.groupby("regime", as_index=False).mean(numeric_only=True).sort_values("regime")
        if len(grouped) < 2:
            raise FigureInputError("scenario dependence requires at least two regimes")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
        axes[0].bar(grouped["regime"], grouped["mean_pairwise_rank_correlation"], color=COLORS["blue"])
        axes[0].set_ylabel("Mean pairwise rank correlation")
        axes[1].bar(grouped["regime"], grouped["joint_extreme_probability"], color=COLORS["red"])
        axes[1].set_ylabel("Joint upper-quintile probability")
        for label, ax in zip(("a", "b"), axes):
            ax.tick_params(axis="x", rotation=25)
            _panel(ax, label)
            _style_axis(ax)
        fig.suptitle(definition.title)
        return PlotResult(fig, _records(grouped), [source], {"regime_count": len(grouped), "source_row_count": len(working)}, f"Scenario dependence diagnostics; n={len(working)} scenario-build records.")
    working = pd.DataFrame(
        {
            "variable_x": frame[columns["left"]].astype(str),
            "variable_y": frame[columns["right"]].astype(str),
            "rank_correlation": _numeric(frame[columns["correlation"]]),
        }
    ).dropna()
    variables = sorted(set(working["variable_x"]) | set(working["variable_y"]))
    if len(variables) < 2:
        raise FigureInputError("scenario dependence matrix has fewer than two variables")
    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    for row in working.itertuples(index=False):
        matrix.loc[row.variable_x, row.variable_y] = row.rank_correlation
        matrix.loc[row.variable_y, row.variable_x] = row.rank_correlation
    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(variables)), variables, rotation=30, ha="right")
    ax.set_yticks(range(len(variables)), variables)
    fig.colorbar(image, ax=ax, label="Rank correlation", shrink=0.8)
    ax.set_title(definition.title)
    return PlotResult(fig, _records(working), [source], {"variable_count": len(variables), "pair_count": len(working)}, f"Scenario rank-dependence matrix; n={len(working)} variable pairs.")


def _optimization_source(
    sources: Sequence[LoadedSource],
    required: Mapping[str, Sequence[str]],
) -> tuple[LoadedSource, pd.DataFrame, dict[str, str]]:
    source, frame, columns = _find_frame(sources, required)
    counterfactual = _column(frame, ("counterfactual",))
    if counterfactual and not frame[counterfactual].map(_truthy).all():
        raise FigureInputError("optimization figure input contains non-counterfactual rows")
    status = _column(frame, ("termination_condition", "solver_status", "status"))
    if status:
        accepted = frame[status].astype(str).str.lower().isin(
            {"optimal", "ok", "success", "complete", "completed", "completed_counterfactual_cells"}
        )
        if accepted.any():
            frame = frame.loc[accepted].copy()
    if frame.empty:
        raise FigureInputError("optimization input has no completed rows")
    return source, frame, columns


def _render_dependence_stress(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    source, frame, columns = _optimization_source(
        sources,
        {
            "regime": ("regime", "dependence_regime", "stress_regime"),
            "cost": ("expected_total_cost_usd", "expected_total_cost_usd_mean", "counterfactual_expected_cost_usd"),
            "cvar": ("cvar_total_cost_usd", "cvar_total_cost_usd_mean", "empirical_cvar_95_usd"),
        },
    )
    policy = _column(frame, ("policy_id", "policy"))
    if policy and frame[policy].astype(str).eq("B6").any():
        frame = frame.loc[frame[policy].astype(str).eq("B6")].copy()
    working = pd.DataFrame(
        {
            "regime": frame[columns["regime"]].astype(str),
            "expected_cost_usd": _numeric(frame[columns["cost"]]),
            "cvar_95_usd": _numeric(frame[columns["cvar"]]),
        }
    ).dropna()
    grouped = working.groupby("regime", as_index=False).agg(
        expected_cost_usd=("expected_cost_usd", "mean"),
        cvar_95_usd=("cvar_95_usd", "mean"),
        cell_count=("expected_cost_usd", "size"),
    ).sort_values("regime")
    if len(grouped) < 2:
        raise FigureInputError("dependence stress requires at least two regimes")
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    ax.plot(x, grouped["expected_cost_usd"], marker="o", color=COLORS["navy"], label="Expected cost")
    ax.plot(x, grouped["cvar_95_usd"], marker="s", ls="--", color=COLORS["red"], label="CVaR 95%")
    ax.set_xticks(x, grouped["regime"], rotation=25, ha="right")
    ax.set_ylabel("Counterfactual cost (USD)")
    ax.set_title(definition.title)
    ax.legend()
    _style_axis(ax)
    return PlotResult(fig, _records(grouped), [source], {"regime_count": len(grouped), "cell_count": len(working)}, f"Dependence-stress outcomes; n={len(working)} completed matched cells.")


def _parse_variant_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for nested in value.values():
            result = _parse_variant_value(nested)
            if result is not None:
                return result
        return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if decoded != value:
                return _parse_variant_value(decoded)
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _render_risk_frontier(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    source, frame, columns = _optimization_source(
        sources,
        {
            "cost": ("expected_total_cost_usd", "expected_total_cost_usd_mean", "counterfactual_expected_cost_usd"),
            "cvar": ("cvar_total_cost_usd", "cvar_total_cost_usd_mean", "empirical_cvar_95_usd"),
        },
    )
    dimension = _column(frame, ("sensitivity_dimension", "dimension"))
    if dimension and frame[dimension].astype(str).eq("risk_weight_frontier").any():
        frame = frame.loc[frame[dimension].astype(str).eq("risk_weight_frontier")].copy()
    risk = _column(frame, ("risk_weight", "risk_lambda", "lambda", "variant_value", "cvar_weight"))
    if risk is None:
        raise FigureInputError("risk frontier input has no risk-weight column")
    working = pd.DataFrame(
        {
            "risk_weight": frame[risk].map(_parse_variant_value),
            "expected_cost_usd": _numeric(frame[columns["cost"]]),
            "cvar_95_usd": _numeric(frame[columns["cvar"]]),
        }
    ).dropna()
    grouped = working.groupby("risk_weight", as_index=False).agg(
        expected_cost_usd=("expected_cost_usd", "mean"),
        cvar_95_usd=("cvar_95_usd", "mean"),
        cell_count=("expected_cost_usd", "size"),
    ).sort_values("risk_weight")
    if len(grouped) < 3:
        raise FigureInputError("risk frontier requires at least three distinct risk weights")
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    scatter = ax.scatter(grouped["expected_cost_usd"], grouped["cvar_95_usd"], c=grouped["risk_weight"], cmap="viridis", s=35, edgecolor=COLORS["dark"], linewidth=0.4)
    ax.plot(grouped["expected_cost_usd"], grouped["cvar_95_usd"], color=COLORS["gray"], lw=0.7, zorder=0)
    for row in grouped.itertuples(index=False):
        ax.annotate(f"{row.risk_weight:g}", (row.expected_cost_usd, row.cvar_95_usd), xytext=(3, 3), textcoords="offset points", fontsize=6)
    fig.colorbar(scatter, ax=ax, label="Risk weight")
    ax.set_xlabel("Expected counterfactual cost (USD)")
    ax.set_ylabel("Empirical CVaR 95% (USD)")
    ax.set_title(definition.title)
    _style_axis(ax)
    return PlotResult(fig, _records(grouped), [source], {"risk_weight_count": len(grouped), "cell_count": len(working)}, f"Cost-risk frontier; n={len(working)} completed counterfactual cells.")


def _render_policy_comparison(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    source, frame, columns = _optimization_source(
        sources,
        {
            "policy": ("policy_id", "policy"),
            "cost": ("expected_total_cost_usd", "expected_total_cost_usd_mean", "counterfactual_expected_cost_usd"),
            "cvar": ("cvar_total_cost_usd", "cvar_total_cost_usd_mean", "empirical_cvar_95_usd"),
        },
    )
    working = pd.DataFrame(
        {
            "policy": frame[columns["policy"]].astype(str),
            "expected_cost_usd": _numeric(frame[columns["cost"]]),
            "cvar_95_usd": _numeric(frame[columns["cvar"]]),
        }
    ).dropna()
    sla = _column(frame, ("expected_sla_shortfall_work_units", "expected_sla_shortfall_work_units_mean", "deadline_missed_work_units"))
    if sla:
        working["sla_shortfall_work_units"] = _numeric(frame.loc[working.index, sla])
    grouped = working.groupby("policy", as_index=False).agg(
        expected_cost_usd=("expected_cost_usd", "mean"),
        cvar_95_usd=("cvar_95_usd", "mean"),
        cell_count=("expected_cost_usd", "size"),
        **({"sla_shortfall_work_units": ("sla_shortfall_work_units", "mean")} if "sla_shortfall_work_units" in working else {}),
    ).sort_values("policy")
    if len(grouped) < 2:
        raise FigureInputError("policy comparison requires at least two policies")
    x = np.arange(len(grouped))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    ax.bar(x - width / 2, grouped["expected_cost_usd"], width, color=COLORS["blue"], edgecolor=COLORS["dark"], linewidth=0.4, label="Expected cost")
    ax.bar(x + width / 2, grouped["cvar_95_usd"], width, color=COLORS["red"], edgecolor=COLORS["dark"], linewidth=0.4, label="CVaR 95%")
    ax.set_xticks(x, grouped["policy"])
    ax.set_ylabel("Counterfactual cost (USD)")
    ax.set_title(definition.title)
    ax.legend(ncol=2)
    _style_axis(ax)
    return PlotResult(fig, _records(grouped), [source], {"policy_count": len(grouped), "cell_count": len(working)}, f"Matched policy comparison; n={len(working)} completed counterfactual cells.")


def _render_typical_dispatch(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    frame, used = _dispatch_rows(sources)
    frame["_date"] = frame["timestamp_utc"].dt.floor("D")
    summary = frame.groupby("_date").agg(rows=("grid_import_mw", "size"), grid_energy_proxy=("grid_import_mw", "sum"))
    eligible = summary.loc[summary["rows"] >= 12]
    if eligible.empty:
        raise FigureInputError("typical dispatch requires a completed operating day")
    median_energy = float(eligible["grid_energy_proxy"].median())
    selected_date = eligible.assign(distance=(eligible["grid_energy_proxy"] - median_energy).abs()).sort_values(["distance", "_date"]).index[0]
    selected = frame.loc[frame["_date"].eq(selected_date)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), sharex=True, constrained_layout=True)
    axes[0].plot(selected["timestamp_utc"], selected["grid_import_mw"], color=COLORS["navy"], lw=1.1, label="Grid import")
    axes[0].plot(selected["timestamp_utc"], selected["day_ahead_schedule_mw"], color=COLORS["gold"], ls="--", lw=0.9, label="DA schedule")
    if "pv" in selected and selected["pv"].notna().any():
        axes[0].plot(selected["timestamp_utc"], selected["pv"], color=COLORS["green"], ls=":", lw=0.9, label="PV used")
    axes[0].set_ylabel("Power (MW)")
    axes[0].legend(ncol=3)
    if {"charge", "discharge"}.issubset(selected.columns) and (selected["charge"].notna().any() or selected["discharge"].notna().any()):
        selected["battery_net_discharge_mw"] = selected["discharge"].fillna(0) - selected["charge"].fillna(0)
        axes[1].plot(selected["timestamp_utc"], selected["battery_net_discharge_mw"], color=COLORS["violet"], lw=0.9, label="Battery net discharge")
    if "soc" in selected and selected["soc"].notna().any():
        axes[1].plot(selected["timestamp_utc"], selected["soc"], color=COLORS["teal"], ls="--", lw=0.9, label="SOC")
    axes[1].axhline(0, color=COLORS["gray"], lw=0.6)
    axes[1].set_ylabel("Battery (MW or MWh)")
    axes[1].set_xlabel("UTC")
    axes[1].legend(ncol=2)
    axes[0].set_title(definition.title)
    for label, ax in zip(("a", "b"), axes):
        _panel(ax, label)
        _style_axis(ax)
    columns = [column for column in ("timestamp_utc", "grid_import_mw", "day_ahead_schedule_mw", "rt_price_usd_per_mwh", "pv", "charge", "discharge", "battery_net_discharge_mw", "soc", "work", "policy") if column in selected]
    return PlotResult(fig, _records(selected, columns), used, {"operating_date": str(selected_date.date()), "interval_count": len(selected), "selection": "median_grid_energy_day"}, f"Typical counterfactual dispatch day; n={len(selected)} causal intervals.")


def _render_grid_rebound(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    source, frame, columns = _optimization_source(
        sources,
        {
            "variant": ("constraint_variant", "policy_id", "policy", "variant"),
            "rebound": ("rebound_peak_mw", "rebound_after_low_price_intervals_mw", "grid_rebound_mw"),
        },
    )
    peak = _column(frame, ("peak_import_mw", "simulated_grid_import_peak", "maximum_grid_import_mw"))
    working = pd.DataFrame(
        {
            "variant": frame[columns["variant"]].astype(str),
            "rebound_mw": _numeric(frame[columns["rebound"]]),
            "peak_import_mw": _numeric(frame[peak]) if peak else np.nan,
        }
    ).dropna(subset=["rebound_mw"])
    grouped = working.groupby("variant", as_index=False).agg(
        rebound_mw=("rebound_mw", "mean"),
        peak_import_mw=("peak_import_mw", "mean"),
        cell_count=("rebound_mw", "size"),
    ).sort_values("variant")
    if len(grouped) < 2:
        raise FigureInputError("grid rebound requires at least two policy/constraint variants")
    fig, axes = plt.subplots(1, 2 if grouped["peak_import_mw"].notna().any() else 1, figsize=(7.2, 3.4), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    axes_array[0].bar(grouped["variant"], grouped["rebound_mw"], color=COLORS["red"])
    axes_array[0].set_ylabel("Rebound after low-price intervals (MW)")
    axes_array[0].tick_params(axis="x", rotation=25)
    _panel(axes_array[0], "a")
    _style_axis(axes_array[0])
    if len(axes_array) > 1:
        axes_array[1].bar(grouped["variant"], grouped["peak_import_mw"], color=COLORS["blue"])
        axes_array[1].set_ylabel("Simulated grid-import peak (MW)")
        axes_array[1].tick_params(axis="x", rotation=25)
        _panel(axes_array[1], "b")
        _style_axis(axes_array[1])
    fig.suptitle(definition.title)
    return PlotResult(fig, _records(grouped), [source], {"variant_count": len(grouped), "cell_count": len(working)}, f"Grid-facing counterfactual proxies; n={len(working)} completed cells.")


def _sensitivity_dimension(renderer: str) -> tuple[set[str], str]:
    if renderer == "sensitivity_flexibility":
        return {"flexible_share", "deadline_steps", "migration_eligibility"}, "Flexibility level"
    if renderer == "sensitivity_bandwidth":
        return {"migration_bandwidth"}, "Migration-bandwidth multiplier"
    return {"battery_duration"}, "Battery duration (h)"


def _render_sensitivity(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    source, frame, columns = _optimization_source(
        sources,
        {
            "dimension": ("sensitivity_dimension", "dimension"),
            "value": ("variant_value", "sensitivity_value", "level_value"),
            "cost": ("expected_total_cost_usd", "expected_total_cost_usd_mean", "counterfactual_expected_cost_usd"),
            "cvar": ("cvar_total_cost_usd", "cvar_total_cost_usd_mean", "empirical_cvar_95_usd"),
        },
    )
    dimensions, xlabel = _sensitivity_dimension(definition.renderer)
    frame = frame.loc[frame[columns["dimension"]].astype(str).isin(dimensions)].copy()
    working = pd.DataFrame(
        {
            "dimension": frame[columns["dimension"]].astype(str),
            "value": frame[columns["value"]].map(_parse_variant_value),
            "expected_cost_usd": _numeric(frame[columns["cost"]]),
            "cvar_95_usd": _numeric(frame[columns["cvar"]]),
        }
    ).dropna()
    grouped = working.groupby(["dimension", "value"], as_index=False).agg(
        expected_cost_usd=("expected_cost_usd", "mean"),
        cvar_95_usd=("cvar_95_usd", "mean"),
        cell_count=("expected_cost_usd", "size"),
    ).sort_values(["dimension", "value"])
    if grouped["value"].nunique() < 3:
        raise FigureInputError(f"{definition.name} requires at least three sensitivity levels")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    for index, (dimension, subset) in enumerate(grouped.groupby("dimension", sort=True)):
        ax.plot(subset["value"], subset["expected_cost_usd"], marker=MARKERS[index % len(MARKERS)], ls=LINESTYLES[index % len(LINESTYLES)], color=(COLORS["navy"], COLORS["teal"], COLORS["violet"])[index % 3], label=f"{dimension}: expected")
        ax.plot(subset["value"], subset["cvar_95_usd"], marker=MARKERS[(index + 1) % len(MARKERS)], ls="--", color=(COLORS["red"], COLORS["gold"], COLORS["gray"])[index % 3], label=f"{dimension}: CVaR")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counterfactual cost (USD)")
    ax.set_title(definition.title)
    ax.legend(fontsize=6, ncol=2)
    _style_axis(ax)
    return PlotResult(fig, _records(grouped), [source], {"dimension_count": grouped["dimension"].nunique(), "level_count": len(grouped), "cell_count": len(working)}, f"One-factor-at-a-time sensitivity; n={len(working)} completed cells.")


def _render_solver_scaling(definition: FigureDefinition, sources: Sequence[LoadedSource]) -> PlotResult:
    source, frame, columns = _find_frame(
        sources,
        {
            "time": ("solve_time_seconds", "solver_time_seconds", "runtime_seconds"),
        },
    )
    x_column = _column(frame, ("number_of_variables", "variable_count", "reduced_scenario_count", "scenario_count", "number_of_constraints"))
    if x_column is None:
        raise FigureInputError("solver scaling input lacks problem-size or scenario-count column")
    working = pd.DataFrame(
        {
            "problem_size": _numeric(frame[x_column]),
            "solve_time_seconds": _numeric(frame[columns["time"]]),
            "termination_condition": frame[_column(frame, ("termination_condition", "solver_status", "status"))].astype(str)
            if _column(frame, ("termination_condition", "solver_status", "status"))
            else "unreported",
            "mip_gap": _numeric(frame[_column(frame, ("mip_gap", "solver_gap"))])
            if _column(frame, ("mip_gap", "solver_gap"))
            else np.nan,
        }
    ).dropna(subset=["problem_size", "solve_time_seconds"])
    if len(working) < 3 or working["problem_size"].nunique() < 2:
        raise FigureInputError("solver scaling requires at least three cells and two problem sizes")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for index, (status, subset) in enumerate(working.groupby("termination_condition", sort=True)):
        axes[0].scatter(subset["problem_size"], subset["solve_time_seconds"], marker=MARKERS[index % len(MARKERS)], color=(COLORS["navy"], COLORS["red"], COLORS["gray"])[index % 3], s=20, alpha=0.75, label=status)
    axes[0].set_xlabel(x_column.replace("_", " "))
    axes[0].set_ylabel("Solve time (s)")
    axes[0].legend(fontsize=6)
    grouped = working.groupby("problem_size", as_index=False).agg(median_solve_time_seconds=("solve_time_seconds", "median"), p90_solve_time_seconds=("solve_time_seconds", lambda values: float(np.quantile(values, 0.9))), cell_count=("solve_time_seconds", "size"))
    axes[1].plot(grouped["problem_size"], grouped["median_solve_time_seconds"], marker="o", color=COLORS["teal"], label="Median")
    axes[1].plot(grouped["problem_size"], grouped["p90_solve_time_seconds"], marker="s", ls="--", color=COLORS["gold"], label="P90")
    axes[1].set_xlabel(x_column.replace("_", " "))
    axes[1].set_ylabel("Solve time (s)")
    axes[1].legend()
    for label, ax in zip(("a", "b"), axes):
        _panel(ax, label)
        _style_axis(ax)
    fig.suptitle(definition.title)
    return PlotResult(fig, _records(working), [source], {"cell_count": len(working), "problem_size_count": working["problem_size"].nunique()}, f"Solver scaling; n={len(working)} formal optimization cells, including reported termination states.")


def _render_dart_accuracy_value_contradiction(
    definition: FigureDefinition,
    sources: Sequence[LoadedSource],
) -> PlotResult:
    direct_source: LoadedSource | None = None
    direct: pd.DataFrame | None = None
    for source in sources:
        frame = source.frame
        if frame is None:
            continue
        candidate = _column(frame, ("candidate_id", "model"))
        accuracy = _column(
            frame,
            (
                "accuracy_metric_value",
                "mean_crps",
                "crps",
                "mean_mae",
                "mae",
            ),
        )
        value = _column(
            frame,
            (
                "mean_incremental_operational_value_usd",
                "incremental_operational_value_usd",
                "mean_daily_incremental_operational_value",
            ),
        )
        if candidate and accuracy and value:
            subset = frame.copy()
            experiment = _column(subset, ("experiment_id",))
            phase = _column(subset, ("phase", "evaluation_window"))
            metric = _column(subset, ("accuracy_metric",))
            if experiment:
                a3 = subset[experiment].astype(str).str.upper().eq("A3")
                if a3.any():
                    subset = subset.loc[a3]
            if phase:
                locked = subset[phase].astype(str).str.contains(
                    "locked|2025", case=False, regex=True
                )
                if locked.any():
                    subset = subset.loc[locked]
            if metric:
                crps = subset[metric].astype(str).str.lower().eq("crps")
                if crps.any():
                    subset = subset.loc[crps]
            direct = pd.DataFrame(
                {
                    "candidate_id": subset[candidate].astype(str),
                    "accuracy": _numeric(subset[accuracy]),
                    "incremental_operational_value_usd": _numeric(subset[value]),
                }
            ).dropna()
            if len(direct) >= 2 and direct["candidate_id"].nunique() >= 2:
                direct_source = source
                accuracy_name = accuracy
                break
    used_sources: list[LoadedSource]
    if direct is None or direct_source is None:
        probability_source, probability, probability_columns = _find_frame(
            sources,
            {
                "candidate": ("candidate_id", "model"),
                "accuracy": ("crps", "mean_crps", "mae", "mean_mae"),
            },
        )
        decision_source, decision, decision_columns = _find_frame(
            sources,
            {
                "candidate": ("candidate_id", "model"),
                "value": (
                    "incremental_operational_value_usd",
                    "mean_incremental_operational_value_usd",
                    "mean_daily_incremental_operational_value",
                ),
            },
        )
        for frame in (probability, decision):
            phase = _column(frame, ("phase", "evaluation_window"))
            if phase:
                locked = frame[phase].astype(str).str.contains(
                    "locked|2025", case=False, regex=True
                )
                if locked.any():
                    frame.drop(frame.index[~locked], inplace=True)
        accuracy_rows = pd.DataFrame(
            {
                "candidate_id": probability[
                    probability_columns["candidate"]
                ].astype(str),
                "accuracy": _numeric(probability[probability_columns["accuracy"]]),
            }
        ).dropna()
        value_rows = pd.DataFrame(
            {
                "candidate_id": decision[decision_columns["candidate"]].astype(str),
                "incremental_operational_value_usd": _numeric(
                    decision[decision_columns["value"]]
                ),
            }
        ).dropna()
        direct = (
            accuracy_rows.groupby("candidate_id", as_index=False)["accuracy"]
            .mean()
            .merge(
                value_rows.groupby("candidate_id", as_index=False)[
                    "incremental_operational_value_usd"
                ].mean(),
                on="candidate_id",
                how="inner",
            )
        )
        direct_source = probability_source
        used_sources = list(
            {
                str(source.path): source
                for source in (probability_source, decision_source)
            }.values()
        )
        accuracy_name = probability_columns["accuracy"]
    else:
        used_sources = [direct_source]
    working = direct.groupby("candidate_id", as_index=False).agg(
        accuracy=("accuracy", "mean"),
        incremental_operational_value_usd=(
            "incremental_operational_value_usd",
            "mean",
        ),
    )
    if len(working) < 2:
        raise FigureInputError("DART contradiction requires at least two candidates")
    working["accuracy_rank"] = working["accuracy"].rank(method="min", ascending=True)
    working["decision_value_rank"] = working[
        "incremental_operational_value_usd"
    ].rank(method="min", ascending=False)
    working["rank_contradiction"] = (
        working["accuracy_rank"] != working["decision_value_rank"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    colors = [
        COLORS["red"] if contradiction else COLORS["navy"]
        for contradiction in working["rank_contradiction"]
    ]
    ax.scatter(
        working["accuracy"],
        working["incremental_operational_value_usd"],
        c=colors,
        marker="o",
        s=38,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    for row in working.itertuples(index=False):
        ax.annotate(
            row.candidate_id,
            (row.accuracy, row.incremental_operational_value_usd),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"{accuracy_name.replace('_', ' ')} (lower is better)")
    ax.set_ylabel("Counterfactual incremental operational value (USD)")
    ax.set_title(definition.title)
    _style_axis(ax)
    return PlotResult(
        fig,
        _records(working),
        used_sources,
        {
            "candidate_count": len(working),
            "accuracy_metric": accuracy_name,
            "value_metric": "incremental_operational_value_usd",
            "denominator_contract": {
                "accuracy": "scored_locked_test_DART_events",
                "value": "baseline_operating_cost_minus_candidate_operating_cost",
            },
            "rank_contradiction_count": int(working["rank_contradiction"].sum()),
        },
        (
            "Referenced-price accuracy versus constrained data-center counterfactual "
            f"operational value; n={len(working)} candidates."
        ),
    )


def _render_dart_value_selection_lcb(
    definition: FigureDefinition,
    sources: Sequence[LoadedSource],
) -> PlotResult:
    source, frame, columns = _find_frame(
        sources,
        {
            "candidate": ("candidate_id", "model"),
            "lcb7": ("lcb_7_day", "lcb_incremental_operational_value_usd"),
        },
    )
    method = _column(frame, ("selection_method", "method"))
    if method:
        primary = frame[method].astype(str).eq("DART-ValueSelect")
        if primary.any():
            frame = frame.loc[primary].copy()
    block_columns = {
        7: columns["lcb7"],
        14: _column(frame, ("lcb_14_day",)),
        28: _column(frame, ("lcb_28_day",)),
    }
    if any(column is None for column in block_columns.values()):
        raise FigureInputError("DART LCB figure requires 7-, 14-, and 28-day block LCBs")
    selected_column = _column(frame, ("selected",))
    selected_model_column = _column(frame, ("selected_model",))
    rows: list[dict[str, Any]] = []
    for _, raw in frame.iterrows():
        candidate = str(raw[columns["candidate"]])
        selected = (
            _truthy(raw[selected_column])
            if selected_column
            else bool(
                selected_model_column
                and str(raw[selected_model_column]) == candidate
            )
        )
        for block, column in block_columns.items():
            numeric = pd.to_numeric(pd.Series([raw[column]]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                rows.append(
                    {
                        "candidate_id": candidate,
                        "block_days": int(block),
                        "lcb_incremental_operational_value_usd": float(numeric),
                        "selected": selected,
                    }
                )
    working = pd.DataFrame(rows)
    if (
        working.empty
        or working["candidate_id"].nunique() < 2
        or set(working["block_days"]) != {7, 14, 28}
    ):
        raise FigureInputError(
            "DART LCB figure requires two candidates and complete 7/14/28-day sensitivity"
        )
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    candidates = sorted(working["candidate_id"].unique())
    base = np.arange(len(candidates), dtype=float)
    offsets = {7: -0.20, 14: 0.0, 28: 0.20}
    for index, block in enumerate((7, 14, 28)):
        subset = working.loc[working["block_days"].eq(block)].set_index(
            "candidate_id"
        )
        values = [subset.loc[candidate, "lcb_incremental_operational_value_usd"] for candidate in candidates]
        ax.plot(
            base + offsets[block],
            values,
            marker=MARKERS[index],
            linestyle="none",
            color=(COLORS["navy"], COLORS["teal"], COLORS["gold"])[index],
            label=f"{block}-day blocks",
        )
    selected = set(working.loc[working["selected"], "candidate_id"])
    for position, candidate in enumerate(candidates):
        if candidate in selected:
            ax.axvspan(position - 0.35, position + 0.35, color=COLORS["blue"], alpha=0.08)
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.7, linestyle="--")
    ax.set_xticks(base, candidates, rotation=20, ha="right")
    ax.set_ylabel("LCB of incremental operational value (USD)")
    ax.set_title(definition.title)
    ax.legend(fontsize=6, ncol=3)
    _style_axis(ax)
    return PlotResult(
        fig,
        _records(working),
        [source],
        {
            "candidate_count": len(candidates),
            "bootstrap_blocks_days": [7, 14, 28],
            "selected_candidates": sorted(selected),
            "denominator_contract": {
                "lcb": "paired_daily_incremental_operational_value_vs_frozen_baseline",
                "bootstrap": "7_14_28_day_moving_blocks_on_2022_2024_rolling_OOS",
            },
        },
        (
            "Selection-period lower confidence bounds under the preregistered "
            f"block-length sensitivity; n={len(candidates)} candidates."
        ),
    )


RENDERERS: dict[str, Callable[[FigureDefinition, Sequence[LoadedSource]], PlotResult]] = {
    "timeline": _render_timeline,
    "missingness": _render_missingness,
    "physical_week": _render_physical_week,
    "counterfactual_week": _render_counterfactual_week,
    "physical_calibration": _render_physical_calibration,
    "component_closure": _render_component_closure,
    "transfer_support": _render_transfer_support,
    "calibration_da": _render_calibration,
    "calibration_rt": _render_calibration,
    "calibration_workload": _render_calibration,
    "price_spike": lambda definition, sources: _render_event_case(
        definition, sources, workload=False
    ),
    "workload_burst": lambda definition, sources: _render_event_case(
        definition, sources, workload=True
    ),
    "scenario_dependence": _render_scenario_dependence,
    "dependence_stress": _render_dependence_stress,
    "risk_frontier": _render_risk_frontier,
    "policy_comparison": _render_policy_comparison,
    "typical_dispatch": _render_typical_dispatch,
    "grid_rebound": _render_grid_rebound,
    "sensitivity_flexibility": _render_sensitivity,
    "sensitivity_bandwidth": _render_sensitivity,
    "sensitivity_storage": _render_sensitivity,
    "solver_scaling": _render_solver_scaling,
    "dart_accuracy_value_contradiction": _render_dart_accuracy_value_contradiction,
    "dart_value_selection_lcb": _render_dart_value_selection_lcb,
}


def _bundle_paths(png_path: Path) -> tuple[Path, Path]:
    return png_path.with_suffix(".pdf"), png_path.with_suffix(".source.json")


def _valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 100 or not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
            return image.width > 0 and image.height > 0
    except Exception:
        return False


def validate_figure_bundle(
    png_path: Path | str,
    *,
    root: Path | str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate PNG/PDF/source-sidecar integrity and every recorded source hash."""

    path = Path(png_path).resolve()
    project_root = Path(root).resolve() if root is not None else path.parents[2]
    pdf_path, sidecar_path = _bundle_paths(path)
    if not _valid_png(path):
        return False, "PNG is missing or invalid", None
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 100:
        return False, "PDF companion is missing or empty", None
    if not pdf_path.read_bytes().startswith(b"%PDF"):
        return False, "PDF companion has an invalid signature", None
    if not sidecar_path.is_file():
        return False, "source sidecar is missing", None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid source sidecar: {type(exc).__name__}", None
    if payload.get("schema_version") != FIGURE_SCHEMA_VERSION or payload.get("status") != "COMPLETE":
        return False, "source sidecar schema/status is incomplete", payload
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        return False, "source sidecar has no output hashes", payload
    expected_png = outputs.get("png", {}).get("sha256") if isinstance(outputs.get("png"), Mapping) else None
    expected_pdf = outputs.get("pdf", {}).get("sha256") if isinstance(outputs.get("pdf"), Mapping) else None
    if expected_png != _sha256(path) or expected_pdf != _sha256(pdf_path):
        return False, "PNG/PDF hash does not match the source sidecar", payload
    rows = payload.get("source_rows_used_for_plot")
    if not isinstance(rows, list) or not rows:
        return False, "source sidecar has no exact plotted rows", payload
    if payload.get("source_rows_sha256") != _json_sha256(rows):
        return False, "plotted-row hash does not match the source sidecar", payload
    sources = payload.get("source_files")
    if not isinstance(sources, list) or not sources:
        return False, "source sidecar has no source files", payload
    for source in sources:
        if not isinstance(source, Mapping):
            return False, "malformed source-file record", payload
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_absolute():
            source_path = project_root / source_path
        if not source_path.is_file() or source.get("sha256") != _sha256(source_path):
            return False, f"source hash no longer matches: {source.get('path')}", payload
        bundle_sidecar = source.get("source_bundle_sidecar")
        if isinstance(bundle_sidecar, Mapping):
            bundle_path = Path(str(bundle_sidecar.get("path", "")))
            if not bundle_path.is_absolute():
                bundle_path = project_root / bundle_path
            if (
                not bundle_path.is_file()
                or bundle_sidecar.get("sha256") != _sha256(bundle_path)
            ):
                return False, "DART table source-sidecar hash no longer matches", payload
    if not payload.get("claim_boundary") or not payload.get("evidence_role"):
        return False, "claim/evidence metadata is missing", payload
    if path.name.startswith("dart_"):
        denominator = payload.get("denominator_contract")
        provenance = payload.get("provenance_contract")
        if not isinstance(denominator, Mapping) or not denominator:
            return False, "DART figure sidecar lacks denominator semantics", payload
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("demo_synthetic_partial_sources_accepted") is not False
            or provenance.get("source_hashes_verified_at_write") is not True
            or provenance.get("source_rows_sha256") != payload.get("source_rows_sha256")
        ):
            return False, "DART figure sidecar lacks strict provenance gates", payload
    return True, "validated PNG/PDF/source-sidecar bundle", payload


def _candidate_fingerprint(root: Path, candidates: Sequence[tuple[Path, str | None]]) -> str:
    rows = []
    for path, expected in candidates:
        if path.is_file():
            rows.append(
                {
                    "path": _relative(root, path),
                    "sha256": _sha256(path),
                    "declared_sha256": expected,
                    "bytes": path.stat().st_size,
                }
            )
    return _json_sha256(rows)


def _source_records(root: Path, sources: Sequence[LoadedSource]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        record = {
            "path": _relative(root, source.path),
            "sha256": source.sha256,
            "bytes": source.bytes,
            "row_count": (
                len(source.frame)
                if source.frame is not None
                else (
                    len(source.payload)
                    if isinstance(source.payload, list)
                    else len(source.payload.get("intervals", []))
                    if isinstance(source.payload, Mapping)
                    else None
                )
            ),
            "valid": True,
            "content_status": "source_hash_verified",
            "inspection": "source file and any declared/embedded hashes validated",
        }
        if source.path.name.startswith("dart_") and source.path.suffix == ".csv":
            table_sidecar = Path(str(source.path) + ".source.json")
            if table_sidecar.is_file():
                record["source_bundle_sidecar"] = {
                    "path": _relative(root, table_sidecar),
                    "sha256": _sha256(table_sidecar),
                    "bytes": table_sidecar.stat().st_size,
                }
        records.append(record)
    return records


def _save_figure_bundle(
    root: Path,
    definition: FigureDefinition,
    result: PlotResult,
    *,
    candidate_fingerprint: str,
) -> dict[str, Any]:
    if not result.plot_rows:
        raise FigureInputError("renderer produced no exact plotted rows")
    if not result.used_sources:
        raise FigureInputError("renderer did not identify a source file")
    sample_count = len(result.plot_rows)
    for key in (
        "reported_sample_count",
        "paired_interval_count",
        "cell_count",
        "source_record_count",
        "source_row_count",
        "held_out_row_count",
        "interval_count",
        "site_count",
        "pair_count",
    ):
        value = result.selection.get(key)
        try:
            if value is not None and int(value) > 0:
                sample_count = int(value)
                break
        except (TypeError, ValueError):
            continue
    target = root / definition.relative_path
    pdf_path, sidecar_path = _bundle_paths(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    png_tmp = target.with_name(target.name + ".tmp")
    pdf_tmp = pdf_path.with_name(pdf_path.name + ".tmp")
    try:
        result.figure.text(
            0.995,
            0.002,
            f"n={sample_count}; exact rows and definitions in source sidecar",
            ha="right",
            va="bottom",
            fontsize=5.2,
            color=COLORS["gray"],
        )
        result.figure.savefig(
            png_tmp,
            format="png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            metadata={
                "Software": FIGURE_RENDERER_VERSION,
                "Title": definition.title,
                "Description": definition.claim_boundary,
            },
        )
        result.figure.savefig(
            pdf_tmp,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Title": definition.title,
                "Subject": definition.claim_boundary,
                "Creator": FIGURE_RENDERER_VERSION,
                "CreationDate": None,
                "ModDate": None,
            },
        )
        png_tmp.replace(target)
        pdf_tmp.replace(pdf_path)
    finally:
        plt.close(result.figure)
        for temporary in (png_tmp, pdf_tmp):
            if temporary.exists():
                temporary.unlink()
    source_files = _source_records(root, result.used_sources)
    source_rows = [_jsonable(row) for row in result.plot_rows]
    source_rows_sha = _json_sha256(source_rows)
    selection_payload = _jsonable(result.selection)
    payload = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "renderer_version": FIGURE_RENDERER_VERSION,
        "status": "COMPLETE",
        "generated_at_utc": _utc_now(),
        "figure": definition.relative_path,
        "pdf": _relative(root, pdf_path),
        "backend": "Python matplotlib Agg",
        "seed": FIGURE_SEED,
        "style": {
            "font_family": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font_size_pt": 7,
            "dpi": FIGURE_DPI,
            "pdf_fonttype": 42,
            "color_plus_linestyle_and_marker_encoding": True,
        },
        "figure_contract": {
            "core_conclusion": definition.conclusion,
            "archetype": definition.archetype,
            "target_output": "double-column publication-readable PNG plus editable-text PDF",
            "reviewer_risk": "Claims must not exceed the recorded evidence role, source relationship, or claim boundary.",
        },
        "caption": result.caption,
        "sample_count": sample_count,
        "source_row_selection": selection_payload,
        "source_rows_used_for_plot": source_rows,
        "source_rows_sha256": source_rows_sha,
        "source_files": source_files,
        "candidate_fingerprint": candidate_fingerprint,
        "denominator_contract": (
            selection_payload.get("denominator_contract", {})
            if isinstance(selection_payload, Mapping)
            else {}
        ),
        "provenance_contract": {
            "source_hashes_verified_at_write": True,
            "source_rows_sha256": source_rows_sha,
            "candidate_fingerprint": candidate_fingerprint,
            "demo_synthetic_partial_sources_accepted": False,
        },
        "evidence_role": definition.evidence_role,
        "source_relationship": definition.source_relationship,
        "measurement_relationship": "not_measured_by_this_study",
        "claim_boundary": definition.claim_boundary,
        "outputs": {
            "png": {
                "path": definition.relative_path,
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
            },
            "pdf": {
                "path": _relative(root, pdf_path),
                "sha256": _sha256(pdf_path),
                "bytes": pdf_path.stat().st_size,
            },
        },
    }
    _atomic_json(sidecar_path, payload)
    valid, reason, checked = validate_figure_bundle(target, root=root)
    if not valid:
        raise RuntimeError(f"newly written figure bundle failed validation: {reason}")
    return checked or payload


def _result_from_payload(
    root: Path,
    definition: FigureDefinition,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    target = root / definition.relative_path
    pdf_path, sidecar_path = _bundle_paths(target)
    sources = [
        {
            **dict(source),
            "valid": True,
            "content_status": "source_hash_verified",
            "inspection": "validated by figure source sidecar",
        }
        for source in payload.get("source_files", [])
        if isinstance(source, Mapping)
    ]
    return {
        "status": "MATERIALIZED",
        "relative_path": definition.relative_path,
        "reason": None,
        "sources": sources,
        "materialization": "data_driven_png_pdf_with_source_sidecar",
        "companion_outputs": {
            "pdf": {
                "path": _relative(root, pdf_path),
                "sha256": _sha256(pdf_path),
                "bytes": pdf_path.stat().st_size,
            },
            "source_sidecar": {
                "path": _relative(root, sidecar_path),
                "sha256": _sha256(sidecar_path),
                "bytes": sidecar_path.stat().st_size,
            },
        },
        "source_rows_sha256": payload.get("source_rows_sha256"),
        "sample_count": payload.get("sample_count"),
        "claim_boundary": definition.claim_boundary,
    }


def generate_paper_figures(
    root: Path | str,
    branch_results: Mapping[str, Any] | None = None,
    *,
    names: Sequence[str] | None = None,
    allow_test_fixtures: bool = False,
) -> dict[str, dict[str, Any]]:
    """Generate only source-backed canonical paper figures.

    Production calls must leave ``allow_test_fixtures`` at its default.  Missing
    inputs, invalid hashes, incomplete rolling checkpoints, demo rows, and
    absent required semantic columns are reported without creating an image.
    """

    project_root = Path(root).resolve()
    results = dict(branch_results or {})
    requested = set(names) if names is not None else None
    output: dict[str, dict[str, Any]] = {}
    for definition in FIGURE_INPUT_REGISTRY:
        if requested is not None and definition.name not in requested and definition.relative_path not in requested:
            continue
        target = project_root / definition.relative_path
        explicit = bool(_explicit_source_entries(results, definition))
        try:
            candidates = _resolve_sources(project_root, definition, results)
            fingerprint = _candidate_fingerprint(project_root, candidates)
            valid_existing, _, existing_payload = validate_figure_bundle(target, root=project_root)
            if (
                valid_existing
                and existing_payload is not None
                and existing_payload.get("candidate_fingerprint") == fingerprint
            ):
                output[definition.relative_path] = _result_from_payload(
                    project_root, definition, existing_payload
                )
                continue
            if not candidates:
                raise FigureInputError("no mapped source files exist")
            loaded: list[LoadedSource] = []
            errors: list[str] = []
            for path, expected_sha in candidates:
                try:
                    loaded.append(
                        _load_source(
                            project_root,
                            path,
                            expected_sha256=expected_sha,
                            allow_test_fixtures=allow_test_fixtures,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{_relative(project_root, path)}: {type(exc).__name__}: {exc}")
            if explicit and errors:
                raise FigureInputError("; ".join(errors))
            if not loaded:
                raise FigureInputError("; ".join(errors) or "no valid source files")
            renderer = RENDERERS[definition.renderer]
            plotted = renderer(definition, loaded)
            payload = _save_figure_bundle(
                project_root,
                definition,
                plotted,
                candidate_fingerprint=fingerprint,
            )
            output[definition.relative_path] = _result_from_payload(
                project_root, definition, payload
            )
        except Exception as exc:
            output[definition.relative_path] = {
                "status": "MISSING_MANDATORY",
                "relative_path": definition.relative_path,
                "reason": f"{type(exc).__name__}: {exc}",
                "sources": [],
                "materialization": None,
                "companion_outputs": {},
                "claim_boundary": definition.claim_boundary,
            }
    return output


__all__ = [
    "FIGURE_DPI",
    "FIGURE_INPUT_REGISTRY",
    "FIGURE_RENDERER_VERSION",
    "FIGURE_SCHEMA_VERSION",
    "FIGURE_SEED",
    "FigureDefinition",
    "FigureInputError",
    "generate_paper_figures",
    "validate_figure_bundle",
]
