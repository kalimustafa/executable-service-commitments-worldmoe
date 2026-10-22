"""Selection-only DART replay and terminal analysis wiring.

This module deliberately ends at the frozen 2022--2024 selector.  It never
opens the locked 2025 panel and it never invokes the main optimization branch.
Scoped pilots may exercise the same replay adapter, but only the exact frozen
conditional selection panel is admitted to MCS/LCB analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
import json
from pathlib import Path
from typing import Any

from dcvpp.core import atomic_json, atomic_text, load_yaml, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_conditional_settlement_protocol import (
    apply_conditional_settlement_overlay,
    load_conditional_settlement_protocol,
)
from dcvpp.experiments.dart_forecast_oos import FREEZE_FINGERPRINT_KEYS
from dcvpp.experiments.dart_partial_candidate import (
    PARTIAL_CANDIDATE_STATUS,
    validate_partial_candidate_pool_contract,
)
from dcvpp.experiments.dart_replay import DART_VALUE_SELECT_SEEDS, EXPERIMENT_IDS
from dcvpp.experiments.dart_value_select import DARTPhaseEvidence, DARTValueSelectExperiment


SCHEMA_VERSION = "dcvpp.dart_selection_only.v1"
PARTIAL_PAPER = "PARTIAL_PAPER"
BOUNDED_AMENDMENT_RELATIVE = Path(
    "artifacts/dart_value_select/protocol_amendments/"
    "bounded_gap_600s_v1/pre_result_amendment_manifest.json"
)
BOUNDED_COMPATIBILITY_V2_RELATIVE = Path(
    "artifacts/dart_value_select/protocol_amendments/"
    "bounded_gap_600s_v2/pre_result_implementation_compatibility_amendment.json"
)
CONFIG_RELATIVE = Path("configs/dart_value_select.yaml")
CONDITIONAL_PROTOCOL_RELATIVE = Path(
    "artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1/"
    "conditional_settlement_protocol_manifest.json"
)
SELECTION_TRAINING_RELATIVE = Path(
    "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos/"
    "selection_stage_manifest.json"
)
READY_INPUT_RELATIVE = Path(
    "artifacts/dart_value_select/counterfactual_inputs/v1/"
    "selection_config_ready_input_block.json"
)
PHYSICAL_RELATIVE = Path(
    "artifacts/research/physical/run_id=paper-physical-v1/"
    "frozen_physical_artifact.json"
)
FORMAL_PREFLIGHT_RELATIVE = Path(
    "artifacts/dart_value_select/run_control/conditional_selection_replay_v1/"
    "formal_preflight/selection_formal_preflight_manifest.json"
)
FORMAL_REPLAY_RELATIVE = Path(
    "artifacts/dart_value_select/run_id=paper-dart-value-select-v4/replay/"
    "run_id=paper-dart-replay-v4/selection"
)
FORMAL_SELECTOR_RELATIVE = Path(
    "artifacts/dart_value_select/run_id=paper-dart-value-select-v4/selection_only"
)


class SelectionOnlyProtocolError(RuntimeError):
    """Raised when selection-only execution would cross a frozen boundary."""


def _read(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionOnlyProtocolError(f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionOnlyProtocolError(f"{label} must be a JSON mapping")
    return value


def _relative(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SelectionOnlyProtocolError(f"artifact escapes repository: {path}") from exc


def _assert_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise SelectionOnlyProtocolError(
            f"{label} changed: expected {expected!r}, observed {actual!r}"
        )


def validate_bounded_gap_amendment(
    *,
    repository_root: str | Path,
    config_path: str | Path | None = None,
    amendment_path: str | Path | None = None,
    compatibility_amendment_path: str | Path | None = None,
    candidate_ids: Sequence[str] | None = None,
    dependency_candidate_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Fail closed unless the pre-result 600-second amendment is hash-pinned.

    The amendment owns scientific solver/value semantics.  Runtime performance
    implementations (preflight/sharding/cache) are pinned independently by
    their own manifests and are not allowed to alter this contract.
    """

    root = Path(repository_root).resolve()
    config_file = Path(config_path) if config_path is not None else root / CONFIG_RELATIVE
    amendment_file = (
        Path(amendment_path)
        if amendment_path is not None
        else root / BOUNDED_AMENDMENT_RELATIVE
    )
    compatibility_file = (
        Path(compatibility_amendment_path)
        if compatibility_amendment_path is not None
        else root / BOUNDED_COMPATIBILITY_V2_RELATIVE
    )
    if not config_file.is_absolute():
        config_file = root / config_file
    if not amendment_file.is_absolute():
        amendment_file = root / amendment_file
    if not compatibility_file.is_absolute():
        compatibility_file = root / compatibility_file
    amendment = _read(amendment_file, label="bounded-gap amendment")
    compatibility = _read(
        compatibility_file, label="bounded-gap implementation compatibility amendment"
    )
    config = load_yaml(config_file)

    _assert_equal(
        amendment.get("schema_version"),
        "dcvpp.dart_bounded_gap_pre_result_amendment.v1",
        label="bounded-gap amendment schema",
    )
    _assert_equal(amendment.get("status"), "FROZEN_PRE_RESULT", label="amendment status")
    _assert_equal(
        amendment.get("launcher_must_hash_pin_this_manifest_and_amended_config"),
        True,
        label="launcher hash-pin requirement",
    )
    _assert_equal(
        amendment.get("formal_A1_A2_results_existed_before_amendment"),
        False,
        label="pre-amendment A1/A2 result exposure",
    )
    _assert_equal(
        int(amendment.get("formal_A1_A2_cells_started_before_amendment", -1)),
        0,
        label="pre-amendment A1/A2 cell count",
    )
    _assert_equal(amendment.get("locked_2025_opened"), False, label="locked-2025 state")
    _assert_equal(
        compatibility.get("schema_version"),
        "dcvpp.dart_bounded_gap_pre_result_implementation_compatibility_amendment.v2",
        label="compatibility amendment schema",
    )
    _assert_equal(
        compatibility.get("status"), "FROZEN_PRE_RESULT", label="compatibility status"
    )
    _assert_equal(
        compatibility.get("parent_bounded_gap_amendment", {}).get("sha256"),
        sha256_file(amendment_file),
        label="compatibility parent amendment SHA-256",
    )
    config_sha = sha256_file(config_file)
    _assert_equal(
        amendment.get("amended_config_sha256"), config_sha, label="amended config SHA-256"
    )
    _assert_equal(
        compatibility.get("amended_config", {}).get("sha256"),
        config_sha,
        label="compatibility config SHA-256",
    )

    solver = amendment.get("solver_policy", {})
    _assert_equal(solver.get("time_limit_seconds"), 600, label="solver time limit")
    _assert_equal(solver.get("mip_relative_gap_target"), 0.0, label="solver target gap")
    _assert_equal(
        solver.get("maximum_accepted_reported_relative_mip_gap"),
        0.05,
        label="accepted MIP-gap cap",
    )
    _assert_equal(solver.get("incumbent_required"), True, label="incumbent requirement")
    _assert_equal(
        solver.get("finite_reported_gap_required"), True, label="finite-gap requirement"
    )
    value = amendment.get("value_semantics", {})
    _assert_equal(
        value.get("primary_A1_A2"),
        "exact_ex_post_settlement_cost_of_executed_incumbent_actions",
        label="primary A1/A2 value semantics",
    )

    runtime = config.get("optimization_runtime", {})
    feasibility = config.get("feasibility", {})
    _assert_equal(runtime.get("time_limit_seconds"), 600, label="config time limit")
    _assert_equal(runtime.get("mip_relative_gap"), 0.0, label="config target gap")
    _assert_equal(runtime.get("solver_threads"), 1, label="config solver threads")
    _assert_equal(
        feasibility.get("maximum_solver_relative_mip_gap"), 0.05, label="config gap cap"
    )
    _assert_equal(
        feasibility.get("require_solver_incumbent"), True, label="config incumbent rule"
    )
    _assert_equal(
        feasibility.get("require_finite_reported_solver_gap"),
        True,
        label="config finite-gap rule",
    )

    frozen_candidates = tuple(
        map(str, config.get("candidate_contract", {}).get("frozen_deployable_pool", ()))
    )
    evaluated = (
        frozen_candidates if candidate_ids is None else tuple(map(str, candidate_ids))
    )
    dependencies = tuple(map(str, dependency_candidate_ids))
    if not evaluated or len(set(evaluated)) != len(evaluated):
        raise SelectionOnlyProtocolError("evaluated candidate_ids must be nonempty and unique")
    if len(set(dependencies)) != len(dependencies):
        raise SelectionOnlyProtocolError("dependency_candidate_ids must be unique")
    if not set(evaluated).issubset(frozen_candidates):
        raise SelectionOnlyProtocolError("evaluated candidates escape the frozen pool")
    if not set(dependencies).issubset(frozen_candidates):
        raise SelectionOnlyProtocolError("dependency candidates escape the frozen pool")
    baseline_source = str(
        config.get("decision_pipeline_contract", {})
        .get("preregistered_baseline", {})
        .get("forecast_source", "")
    )
    if dependencies and dependencies != (baseline_source,):
        raise SelectionOnlyProtocolError(
            "selection replay dependency candidates must be the preregistered baseline source"
        )

    compatibility_change = compatibility.get("implementation_change", {})
    allowed_changed_path = str(compatibility_change.get("path", ""))
    _assert_equal(
        compatibility_change.get("prior_sha256"),
        amendment.get("implementation_sha256", {}).get(allowed_changed_path),
        label="compatibility prior implementation SHA-256",
    )
    _assert_equal(
        compatibility_change.get("amended_sha256"),
        sha256_file(root / allowed_changed_path),
        label="compatibility amended implementation SHA-256",
    )
    _assert_equal(
        compatibility_change.get("dependency_candidate_can_enter_selection_ranking"),
        False,
        label="baseline dependency ranking role",
    )
    _assert_equal(
        compatibility_change.get("dependency_candidate_can_create_A1_A2_candidate_cells"),
        False,
        label="baseline dependency candidate-cell role",
    )
    unchanged = compatibility.get("unchanged_scientific_contract", {})
    _assert_equal(unchanged.get("solver_time_limit_seconds"), 600, label="v2 time limit")
    _assert_equal(
        unchanged.get("maximum_accepted_reported_relative_mip_gap"),
        0.05,
        label="v2 gap cap",
    )
    _assert_equal(
        unchanged.get("primary_A1_A2_value_semantics"),
        value.get("primary_A1_A2"),
        label="v2 value semantics",
    )
    pre_result = compatibility.get("pre_result_state", {})
    _assert_equal(pre_result.get("formal_A1_A2_cells_started"), 0, label="v2 formal cells")
    _assert_equal(pre_result.get("locked_2025_opened"), False, label="v2 locked state")
    _assert_equal(
        pre_result.get("main_optimization_started"), False, label="v2 main optimization state"
    )

    for relative, expected_sha in amendment.get("implementation_sha256", {}).items():
        implementation = root / str(relative)
        effective_expected = (
            compatibility_change["amended_sha256"]
            if str(relative) == allowed_changed_path
            else expected_sha
        )
        if not implementation.is_file() or sha256_file(implementation) != effective_expected:
            raise SelectionOnlyProtocolError(
                f"bounded-gap scientific implementation pin changed: {relative}"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "HASH_PINNED",
        "amendment_path": _relative(amendment_file, root=root),
        "amendment_sha256": sha256_file(amendment_file),
        "compatibility_amendment_path": _relative(compatibility_file, root=root),
        "compatibility_amendment_sha256": sha256_file(compatibility_file),
        "config_path": _relative(config_file, root=root),
        "config_sha256": config_sha,
        "prior_config_sha256": amendment["prior_frozen_config_sha256"],
        "time_limit_seconds": 600,
        "target_mip_relative_gap": 0.0,
        "maximum_accepted_relative_mip_gap": 0.05,
        "primary_value_semantics": value["primary_A1_A2"],
        "performance_runtime_pin_is_separate": True,
        "scientific_contract_changed_by_runtime_optimization": False,
        "implementation_compatibility_change": compatibility_change["allowed_change"],
        "locked_2025_opened": False,
        "scope_roles": {
            "candidate_ids": list(evaluated),
            "candidate_role": "evaluated_A1_A2_cells_and_selection_ranking",
            "dependency_candidate_ids": list(dependencies),
            "dependency_role": "strict_common_baseline_source_only_not_ranked",
            "dependency_records_create_candidate_cells": False,
            "dependency_records_enter_winner_selection": False,
        },
    }


def load_selection_only_inputs(
    *, repository_root: str | Path
) -> dict[str, Any]:
    """Load and validate the immutable selection-side inputs only."""

    root = Path(repository_root).resolve()
    config_for_scope = load_yaml(root / CONFIG_RELATIVE)
    training_path = root / SELECTION_TRAINING_RELATIVE
    ready_path = root / READY_INPUT_RELATIVE
    physical_path = root / PHYSICAL_RELATIVE
    conditional_path = root / CONDITIONAL_PROTOCOL_RELATIVE
    preflight_path = root / FORMAL_PREFLIGHT_RELATIVE
    training = _read(training_path, label="selection training manifest")
    ready = _read(ready_path, label="selection real-input block")
    physical = _read(physical_path, label="frozen physical artifact")
    preflight = _read(preflight_path, label="formal selection preflight")
    conditional = load_conditional_settlement_protocol(
        conditional_path, repository_root=root
    )
    pool = validate_partial_candidate_pool_contract(training.get("candidate_pool_contract"))
    protocol_pins = validate_bounded_gap_amendment(
        repository_root=root,
        candidate_ids=tuple(map(str, pool["eligible_candidates"])),
        dependency_candidate_ids=(
            str(
                config_for_scope.get("decision_pipeline_contract", {})
                .get("preregistered_baseline", {})
                .get("forecast_source", "")
            ),
        ),
    )
    config = config_for_scope
    config = apply_conditional_settlement_overlay(
        config, conditional, repository_root=root
    )
    input_key = "dart_real_evidence_inputs"
    input_block = ready.get(input_key)
    if not isinstance(input_block, Mapping):
        raise SelectionOnlyProtocolError("selection real-input block is absent")
    config[input_key] = dict(input_block)

    _assert_equal(training.get("status"), PARTIAL_CANDIDATE_STATUS, label="training status")
    _assert_equal(
        ready.get("status"),
        "SELECTION_READY_LOCKED_TEST_CLOSED",
        label="selection input readiness",
    )
    _assert_equal(ready.get("locked_test_opened"), False, label="ready-input locked state")
    _assert_equal(conditional.get("terminal_paper_cap"), PARTIAL_PAPER, label="terminal cap")
    _assert_equal(
        conditional.get("protocol_timing", {}).get("locked_2025_result_opened_by_this_step"),
        False,
        label="conditional-protocol locked state",
    )
    selection_dates = tuple(
        map(
            str,
            config["real_evidence_contract"]["operating_dates_by_phase"]["selection"],
        )
    )
    expected_candidates = tuple(map(str, pool["eligible_candidates"]))
    contract = preflight.get("contract", {})
    _assert_equal(preflight.get("status"), "COMPLETE", label="formal preflight status")
    _assert_equal(
        tuple(map(str, contract.get("candidate_ids", ()))),
        expected_candidates,
        label="formal preflight candidates",
    )
    _assert_equal(
        int(contract.get("operating_date_count", -1)),
        len(selection_dates),
        label="formal preflight date count",
    )
    _assert_equal(
        contract.get("operating_dates_sha256"),
        sha256_json(list(selection_dates)),
        label="formal preflight date SHA-256",
    )
    if not set(map(lambda value: int(str(value)[:4]), selection_dates)) == {2022, 2023, 2024}:
        raise SelectionOnlyProtocolError("formal selection dates do not cover 2022--2024")
    if tuple(config["randomization_contract"]["study_seeds"]) != DART_VALUE_SELECT_SEEDS:
        raise SelectionOnlyProtocolError("selection replay must use the exact three seeds")

    return {
        "root": root,
        "protocol_pins": protocol_pins,
        "config": config,
        "selection_training": training,
        "candidate_pool_contract": pool,
        "candidate_ids": expected_candidates,
        "operating_dates": selection_dates,
        "ready": ready,
        "physical": physical,
        "preflight_manifest_path": preflight_path,
        "preflight_manifest": preflight,
        "conditional_protocol_path": conditional_path,
        "conditional_protocol": conditional,
        "source_pins": {
            "selection_training_manifest": {
                "path": _relative(training_path, root=root),
                "sha256": sha256_file(training_path),
            },
            "selection_ready_input": {
                "path": _relative(ready_path, root=root),
                "sha256": sha256_file(ready_path),
            },
            "physical_artifact": {
                "path": _relative(physical_path, root=root),
                "sha256": sha256_file(physical_path),
            },
            "conditional_protocol": {
                "path": _relative(conditional_path, root=root),
                "sha256": sha256_file(conditional_path),
                "fingerprint": conditional["manifest_fingerprint"],
            },
            "formal_preflight": {
                "path": _relative(preflight_path, root=root),
                "sha256": sha256_file(preflight_path),
                "fingerprint": preflight["manifest_fingerprint"],
            },
        },
    }


def _normalize_evidence(value: DARTPhaseEvidence | Mapping[str, Any]) -> DARTPhaseEvidence:
    if isinstance(value, DARTPhaseEvidence):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("selection replay result must be evidence or a mapping")
    if str(value.get("status", "")).upper() != "COMPLETE":
        raise SelectionOnlyProtocolError(
            f"selection replay is not complete: {value.get('status', 'UNKNOWN')}"
        )
    raw = value.get("evidence", value.get("phase_evidence", value))
    if isinstance(raw, DARTPhaseEvidence):
        return raw
    if isinstance(raw, Mapping):
        return DARTPhaseEvidence.from_dict(raw)
    raise SelectionOnlyProtocolError("completed replay result does not contain phase evidence")


def _freeze_dependencies(
    *,
    config: Mapping[str, Any],
    selection_training: Mapping[str, Any],
    evidence: DARTPhaseEvidence,
    source_pins: Mapping[str, Any],
    protocol_pins: Mapping[str, Any],
) -> dict[str, str]:
    inputs = selection_training.get("input_fingerprints", {})
    training_sha = selection_training.get("training_config_sha256")
    if not isinstance(inputs, Mapping) or not training_sha:
        raise SelectionOnlyProtocolError("selection training fingerprints are incomplete")
    values = {
        "data_through_2024": sha256_json(
            {
                "referenced_nyiso_partitions": inputs.get("referenced_nyiso_partitions"),
                "source_evidence": evidence.source_evidence_sha256,
                "conditional_protocol": source_pins["conditional_protocol"],
            }
        ),
        "feature_contract": sha256_json(config.get("forecast_issue_contract", {})),
        "model_hyperparameters": str(training_sha),
        "calibration_rule": sha256_json(
            {
                "probabilistic_credibility": config.get("probabilistic_credibility", {}),
                "actionability": config.get("actionability", {}),
                "joint_coupling": selection_training.get("phase_evidence", {}).get(
                    "joint_coupling_manifest_sha256"
                ),
            }
        ),
        "selection_rule": sha256_json(
            {
                key: config.get(key, {})
                for key in (
                    "feasibility",
                    "bootstrap",
                    "model_confidence_set",
                    "selection",
                    "aggregation_contract",
                    "selection_freeze_contract",
                )
            }
        ),
        "scenario_optimizer_user_contract": sha256_json(
            {
                "decision_pipeline_contract": config.get("decision_pipeline_contract", {}),
                "identical_replay": config.get("identical_replay", {}),
                "physical": source_pins["physical_artifact"],
                "ready_input": source_pins["selection_ready_input"],
                "bounded_amendment": protocol_pins,
            }
        ),
        "code": sha256_json(
            {
                "selection_engine": sha256_file(Path(__file__).with_name("dart_value_select.py")),
                "selection_only_wiring": sha256_file(Path(__file__).resolve()),
            }
        ),
    }
    if set(values) != set(FREEZE_FINGERPRINT_KEYS) or any(
        not isinstance(value, str) or len(value) != 64 for value in values.values()
    ):
        raise SelectionOnlyProtocolError("selection freeze dependency fingerprint is invalid")
    return values


def _selector_comparison_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    selectors = analysis.get("frozen_selector_candidates", {})
    results = analysis.get("selection_results_by_experiment", {})
    a1 = results.get("A1", {}) if isinstance(results, Mapping) else {}
    a2 = results.get("A2", {}) if isinstance(results, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for name in (
        "MAE-Select",
        "DART-ValueSelect",
        "CRPS-Select",
        "q99-tail-F1-Select",
        "mean-validation-value-Select",
    ):
        selector = selectors.get(name, {}) if isinstance(selectors, Mapping) else {}
        winners = list(map(str, selector.get("winner_set", ())))
        rows.append(
            {
                "selector": name,
                "comparison_role": (
                    "PRIMARY_MAE_SELECT_VS_DART_VALUE_SELECT"
                    if name in {"MAE-Select", "DART-ValueSelect"}
                    else "SUPPLEMENTARY_DIAGNOSTIC"
                ),
                "status": selector.get("status"),
                "selected_model": selector.get("selected_model"),
                "winner_set": json.dumps(winners, separators=(",", ":")),
                "winner_set_size": len(winners),
                "selection_basis": selector.get("selection_basis"),
                "a1_noninferiority_required": name == "DART-ValueSelect",
                "a1_status": a1.get("status") if name == "DART-ValueSelect" else None,
                "a2_status": a2.get("status") if name == "DART-ValueSelect" else None,
                "a2_mcs_set": (
                    json.dumps(list(a2.get("mcs_set", ())), separators=(",", ":"))
                    if name == "DART-ValueSelect"
                    else None
                ),
                "primary_7_day_lcb_by_model": (
                    json.dumps(dict(a2.get("lcb_by_model", {})), sort_keys=True, separators=(",", ":"))
                    if name == "DART-ValueSelect"
                    else None
                ),
                "derived_from_same_selection_replay": True,
                "derived_without_new_solves": True,
                "locked_2025_used_for_selection": False,
                "scientific_winner_set_collapsed": False,
            }
        )
    return rows


def write_selection_method_comparison(
    *, analysis: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    destination = Path(output_root).resolve()
    rows = _selector_comparison_rows(analysis)
    json_path = destination / "tables/selection_method_comparison.json"
    csv_path = destination / "tables/selection_method_comparison.csv"
    atomic_json(
        json_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "primary_comparison": ["MAE-Select", "DART-ValueSelect"],
            "supplementary_selectors": [
                "CRPS-Select",
                "q99-tail-F1-Select",
                "mean-validation-value-Select",
            ],
            "derived_without_new_solves": True,
            "locked_2025_used": False,
            "rows": rows,
        },
    )
    fieldnames = list(rows[0])
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    lines.append(buffer.getvalue())
    atomic_text(csv_path, "".join(lines))
    return {
        "json_path": json_path.as_posix(),
        "json_sha256": sha256_file(json_path),
        "csv_path": csv_path.as_posix(),
        "csv_sha256": sha256_file(csv_path),
        "row_count": len(rows),
    }


def finalize_completed_selection_replay(
    *,
    repository_root: str | Path,
    replay_result: DARTPhaseEvidence | Mapping[str, Any],
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run existing ingest/analyze/MCS/LCB/A1 gate, then stop before 2025."""

    bundle = load_selection_only_inputs(repository_root=repository_root)
    root: Path = bundle["root"]
    destination = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / FORMAL_SELECTOR_RELATIVE).resolve()
    )
    evidence = _normalize_evidence(replay_result)
    if evidence.phase != "selection":
        raise SelectionOnlyProtocolError("selection-only finalizer rejects locked-test evidence")
    _assert_equal(
        tuple(evidence.operating_dates),
        tuple(bundle["operating_dates"]),
        label="formal conditional selection date panel",
    )
    for experiment in EXPERIMENT_IDS:
        candidates = tuple(
            sorted(candidate.candidate_id for candidate in evidence.candidates_by_experiment[experiment])
        )
        _assert_equal(
            candidates,
            tuple(sorted(bundle["candidate_ids"])),
            label=f"{experiment} formal candidate pool",
        )
    _assert_equal(
        evidence.candidate_pool_contract.get("contract_sha256")
        if evidence.candidate_pool_contract
        else None,
        bundle["candidate_pool_contract"]["contract_sha256"],
        label="formal replay candidate contract",
    )

    experiment = DARTValueSelectExperiment(
        destination,
        bundle["config"],
        contract_profile="paper",
        candidate_pool_contract=bundle["candidate_pool_contract"],
    )
    experiment.initialize()
    ingestion = experiment.ingest_selection_evidence(evidence)
    analysis = experiment.analyze_selection()
    dependencies = _freeze_dependencies(
        config=bundle["config"],
        selection_training=bundle["selection_training"],
        evidence=evidence,
        source_pins=bundle["source_pins"],
        protocol_pins=bundle["protocol_pins"],
    )
    freeze = experiment.freeze_selection(dependencies)
    comparison = write_selection_method_comparison(
        analysis=analysis, output_root=destination
    )
    manifest_path = destination / "manifests/selection_only_terminal_manifest.json"
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "status": PARTIAL_PAPER,
        "terminal_status": PARTIAL_PAPER,
        "terminal": True,
        "claim_scope": "conditional_decision_value_on_fully_observed_settlement_days",
        "selection_status": "COMPLETE_AND_FROZEN_BEFORE_2025",
        "locked_2025_opened": False,
        "locked_2025_run_count": 0,
        "main_optimization_started": False,
        "main_optimization_result_consumed": False,
        "selection_only": True,
        "candidate_ids": list(bundle["candidate_ids"]),
        "operating_date_count": len(bundle["operating_dates"]),
        "study_seeds": list(DART_VALUE_SELECT_SEEDS),
        "expected_replay_cell_count": (
            len(bundle["operating_dates"])
            * len(bundle["candidate_ids"])
            * len(EXPERIMENT_IDS)
            * len(DART_VALUE_SELECT_SEEDS)
        ),
        "bounded_gap_protocol": bundle["protocol_pins"],
        "performance_runtime_pin": bundle["source_pins"]["formal_preflight"],
        "scientific_contract_changed_by_performance_runtime": False,
        "selection_evidence": {
            "path": _relative(experiment.selection_evidence_path, root=root),
            "sha256": sha256_file(experiment.selection_evidence_path),
            "fingerprint": evidence.fingerprint,
        },
        "selection_analysis": {
            "path": _relative(experiment.selection_analysis_path, root=root),
            "sha256": sha256_file(experiment.selection_analysis_path),
            "analysis_sha256": analysis["analysis_sha256"],
        },
        "selection_freeze": {
            "path": _relative(experiment.selection_freeze_path, root=root),
            "sha256": sha256_file(experiment.selection_freeze_path),
            "freeze_fingerprint": freeze["freeze_fingerprint"],
        },
        "selection_method_comparison": {
            key: (
                _relative(Path(value), root=root) if key.endswith("_path") else value
            )
            for key, value in comparison.items()
        },
        "A1": analysis["selection_results_by_experiment"]["A1"],
        "A2": analysis["selection_results_by_experiment"]["A2"],
        "frozen_selectors": analysis["frozen_selector_candidates"],
        "source_pins": bundle["source_pins"],
        "terminal_reason": (
            "2022-2024 conditional selection is complete; locked 2025 remains sealed "
            "and the paper is capped by the pre-result conditional-settlement protocol."
        ),
        "created_at_utc": utc_now(),
    }
    terminal["terminal_fingerprint"] = sha256_json(terminal)
    atomic_json(manifest_path, terminal)
    return {
        "status": PARTIAL_PAPER,
        "terminal": True,
        "locked_2025_opened": False,
        "main_optimization_started": False,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "comparison": comparison,
        "ingestion": ingestion,
        "analysis": analysis,
        "freeze": freeze,
    }


def run_formal_selection_only(
    *,
    repository_root: str | Path,
    replay_builder: Callable[..., DARTPhaseEvidence | Mapping[str, Any]],
    runtime_context_factory: Callable[[Mapping[str, Any], Path], Any],
    replay_output_root: str | Path | None = None,
    selector_output_root: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the full formal selection replay under an injected runtime context.

    ``runtime_context_factory`` owns only performance seams (certified
    preflight, process sharding, exact cache reuse).  The replay call and the
    post-replay scientific analysis remain unchanged.
    """

    bundle = load_selection_only_inputs(repository_root=repository_root)
    root: Path = bundle["root"]
    replay_root = (
        Path(replay_output_root).resolve()
        if replay_output_root is not None
        else (root / FORMAL_REPLAY_RELATIVE).resolve()
    )
    control_root = (
        root
        / "artifacts/dart_value_select/run_control/conditional_selection_replay_v1/formal_run"
    )
    launch_path = control_root / "selection_only_launch_manifest.json"
    launch = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "phase": "selection",
        "selection_only": True,
        "candidate_ids": list(bundle["candidate_ids"]),
        "operating_date_count": len(bundle["operating_dates"]),
        "study_seeds": list(DART_VALUE_SELECT_SEEDS),
        "expected_replay_cell_count": (
            len(bundle["operating_dates"])
            * len(bundle["candidate_ids"])
            * len(EXPERIMENT_IDS)
            * len(DART_VALUE_SELECT_SEEDS)
        ),
        "bounded_gap_protocol": bundle["protocol_pins"],
        "formal_preflight": bundle["source_pins"]["formal_preflight"],
        "locked_2025_opened": False,
        "main_optimization_started": False,
        "started_at_utc": utc_now(),
    }
    atomic_json(launch_path, launch)
    try:
        with runtime_context_factory(bundle, control_root):
            replay = replay_builder(
                root=root,
                dart_forecast_root=Path(bundle["selection_training"]["stage_root"]),
                physical_result={"frozen_artifact": bundle["physical"]},
                forecast_result={
                    "dart_real_evidence_inputs": bundle["ready"][
                        "dart_real_evidence_inputs"
                    ]
                },
                config=bundle["config"],
                phase="selection",
                output_root=replay_root,
                resume=resume,
                candidate_pool_contract=bundle["candidate_pool_contract"],
            )
        final = finalize_completed_selection_replay(
            repository_root=root,
            replay_result=replay,
            output_root=selector_output_root,
        )
        launch.update(
            {
                "status": PARTIAL_PAPER,
                "terminal": True,
                "finished_at_utc": utc_now(),
                "selection_terminal_manifest": {
                    "path": _relative(Path(final["manifest_path"]), root=root),
                    "sha256": final["manifest_sha256"],
                },
            }
        )
        atomic_json(launch_path, launch)
        return final
    except BaseException as exc:
        launch.update(
            {
                "status": "PARTIAL",
                "terminal": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "locked_2025_opened": False,
                "main_optimization_started": False,
                "finished_at_utc": utc_now(),
            }
        )
        atomic_json(launch_path, launch)
        raise


__all__ = [
    "BOUNDED_AMENDMENT_RELATIVE",
    "PARTIAL_PAPER",
    "SCHEMA_VERSION",
    "SelectionOnlyProtocolError",
    "finalize_completed_selection_replay",
    "load_selection_only_inputs",
    "run_formal_selection_only",
    "validate_bounded_gap_amendment",
    "write_selection_method_comparison",
]
