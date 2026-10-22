"""Persistent composition layer for the paper DART-ValueSelect branch.

The lower-level modules deliberately separate forecast training, real replay
evidence, and statistical selection.  This module fixes their order, keeps
their run IDs stable across resumes, and exposes one fail-closed result to the
paper orchestrator.  It never substitutes fixture or synthetic evidence.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dcvpp.core import atomic_json, load_yaml, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_forecast_oos import (
    CANDIDATE_POOL,
    FREEZE_FINGERPRINT_KEYS,
)
from dcvpp.experiments.dart_partial_candidate import (
    PARTIAL_CANDIDATE_STATUS,
    PARTIAL_CANDIDATE_TERMINAL_CAP,
    run_partial_candidate_freeze_stage,
    run_partial_candidate_locked_stage,
    run_partial_candidate_selection_stage,
    validate_partial_candidate_pool_contract,
)
from dcvpp.experiments.dart_replay import (
    A1_ALLOWED_TO_DIFFER,
    A2_ALLOWED_TO_DIFFER,
    DART_VALUE_SELECT_SEEDS,
)
from dcvpp.experiments.dart_training import run_dart_forecast_training_stage
from dcvpp.experiments.dart_value_select import (
    DARTPhaseEvidence,
    DARTValueSelectExperiment,
)


DART_PIPELINE_SCHEMA_VERSION = "dart_paper_pipeline_v1"
DART_PIPELINE_COMPLETE = "COMPLETE"
DART_PIPELINE_PARTIAL = "PARTIAL"

_REAL_EVIDENCE_CORE_GATE_KEYS = (
    "external_historical_nyiso_replay",
    "first_stage_frozen_before_replay",
    "causal_recourse_no_future_reveal",
    "counterfactual_labels",
    "a1_identity_exact",
    "a2_allowed_changes_exact",
    "three_layers_complete",
    "no_future_or_locked_leakage",
)
_REAL_EVIDENCE_ADDITIONAL_GATE_KEYS = (
    "external_replay_causal_provenance_complete",
    "zero_gap_solver_feasibility_rule_applied",
    "ratio_of_sums_metric_aggregation_complete",
    "layer1_layer2_diagnostics_emitted",
)
_EVALUATED_HYPOTHESIS_STATUSES = {
    "SUPPORTED",
    "NOT_SUPPORTED",
    "INCONCLUSIVE",
    "INCONCLUSIVE_FROZEN_WINNER_SET",
    "INCONCLUSIVE_INSUFFICIENT_RANK_VARIATION",
    "INCONCLUSIVE_MISSING_ACTIONABILITY_METRICS",
}
_H4D_TERMINAL_STATUSES = {
    *_EVALUATED_HYPOTHESIS_STATUSES,
    "SKIPPED_OPTIONAL_EXTENSION",
}
_REQUIRED_PIPELINE_OUTPUT_KEYS = {
    "v1_readiness_manifest",
    "selection_training_manifest",
    "forecast_freeze_manifest",
    "locked_training_manifest",
    "selection_real_evidence_manifest",
    "locked_real_evidence_manifest",
    "selection_metric_aggregates",
    "selection_diagnostic_availability",
    "locked_metric_aggregates",
    "locked_diagnostic_availability",
    "selection_evidence",
    "selection_analysis",
    "selection_freeze",
    "locked_test_ledger",
    "locked_test_evidence",
    "dart_value_select_manifest",
    "dart_replay_manifest",
    "dart_statistical_selection_manifest",
}

ReplayBuilder = Callable[..., DARTPhaseEvidence | Mapping[str, Any]]
V1ReadinessEvaluator = Callable[..., Mapping[str, Any]]


def _selection_candidate_pool_contract(
    selection_training: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Admit either the intact nine-model pool or the sole frozen partial case."""

    status = str(selection_training.get("status", "")).upper()
    if status == DART_PIPELINE_COMPLETE:
        if (
            selection_training.get("downstream_ready") is not True
            or int(selection_training.get("failed_unit_count", -1) or 0) != 0
        ):
            raise ValueError("complete selection stage is not an intact nine-model run")
        supplied_contract = selection_training.get("candidate_pool_contract")
        if supplied_contract is not None:
            if not isinstance(supplied_contract, Mapping):
                raise ValueError("complete selection candidate contract is malformed")
            attempted = tuple(map(str, supplied_contract.get("attempted_candidates", ())))
            eligible = tuple(map(str, supplied_contract.get("eligible_candidates", ())))
            canonical = tuple(candidate.name for candidate in CANDIDATE_POOL)
            if attempted != canonical or eligible != canonical:
                raise ValueError("complete selection candidate contract is not the intact pool")
        return None
    if (
        status != PARTIAL_CANDIDATE_STATUS
        or selection_training.get("downstream_ready") is not False
        or selection_training.get("partial_candidate_downstream_ready") is not True
        or selection_training.get("terminal_status_cap")
        != PARTIAL_CANDIDATE_TERMINAL_CAP
        or int(selection_training.get("expected_unit_count", -1)) != 126
        or int(selection_training.get("completed_unit_count", -1)) != 123
        or int(selection_training.get("eligible_expected_unit_count", -1)) != 108
        or int(selection_training.get("eligible_completed_unit_count", -1)) != 108
        or int(selection_training.get("failed_unit_count", -1)) != 3
        or int(selection_training.get("date_drop_count", -1)) != 0
        or selection_training.get("common_date_panel_preserved") is not True
    ):
        raise ValueError("selection stage is neither complete nor the unique partial candidate case")
    return validate_partial_candidate_pool_contract(
        selection_training.get("candidate_pool_contract"),
        failed_units=selection_training.get("failed_units"),
        require_failed_units=True,
    )


def _effective_candidate_names(
    candidate_pool_contract: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if candidate_pool_contract is None:
        return tuple(candidate.name for candidate in CANDIDATE_POOL)
    normalized = validate_partial_candidate_pool_contract(candidate_pool_contract)
    return tuple(map(str, normalized["eligible_candidates"]))


def _required_real_evidence_gate_keys() -> tuple[str, ...]:
    """Consume the bridge contract while retaining newly mandatory hard gates."""

    try:
        from dcvpp.experiments.dart_real_evidence import REQUIRED_GATE_KEYS

        declared = tuple(map(str, REQUIRED_GATE_KEYS))
    except (ImportError, AttributeError, TypeError):
        declared = ()
    return tuple(
        dict.fromkeys(
            (
                *_REAL_EVIDENCE_CORE_GATE_KEYS,
                *declared,
                *_REAL_EVIDENCE_ADDITIONAL_GATE_KEYS,
            )
        )
    )


def _resolve(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        value = value.get("path") or value.get("local_path")
    if value in (None, ""):
        return None
    root = Path(root).resolve()
    path = Path(str(value))
    if ".." in path.parts:
        raise ValueError("evidence path contains traversal")
    if path.is_absolute():
        raise ValueError(
            "absolute evidence path is outside the repository-relative contract"
        )
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence path resolves outside the repository") from exc
    return candidate


def _trusted_repository_path(root: Path, value: Any) -> Path:
    """Resolve a caller-owned path while still enforcing repository containment.

    Public evidence records must pass through :func:`_resolve` and remain
    relative.  A small number of internal call sites already hold a resolved
    ``Path``; accepting those here avoids serializing an absolute path back into
    an evidence manifest.
    """

    root = Path(root).resolve()
    path = Path(value)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("trusted path resolves outside the repository") from exc
    return candidate


def _internal_artifact_path(root: Path, value: Any) -> Path | None:
    """Resolve a producer-owned path while proving repository containment."""

    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        value = value.get("path") or value.get("local_path")
    if value in (None, ""):
        return None
    return _trusted_repository_path(root, value)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _date_panel(start: date, end: date) -> tuple[str, ...]:
    values: list[str] = []
    current = start
    while current <= end:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


SELECTION_OPERATING_DATES = _date_panel(date(2022, 1, 1), date(2024, 12, 31))
LOCKED_OPERATING_DATES = _date_panel(date(2025, 1, 1), date(2025, 12, 31))


def _path_sha(root: Path, value: Any) -> str | None:
    path = _resolve(root, value)
    return sha256_file(path) if path is not None and path.is_file() else None


def _result_manifest_identity(root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    paths: dict[str, str] = {}
    for key in (
        "manifest_path",
        "run_manifest",
        "extension_run_manifest",
        "frozen_day_index",
    ):
        digest = _path_sha(root, result.get(key))
        if digest is not None:
            paths[key] = digest
    outputs = result.get("outputs")
    if isinstance(outputs, Mapping):
        for key, value in sorted(outputs.items()):
            digest = _path_sha(root, value)
            if digest is not None:
                paths[f"outputs.{key}"] = digest
    return {
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "artifact_sha256": paths,
    }


def _code_fingerprint(root: Path) -> str:
    relative_paths = (
        "src/dcvpp/experiments/dart_pipeline.py",
        "src/dcvpp/experiments/dart_partial_candidate.py",
        "src/dcvpp/experiments/dart_checkpoint_replay_compat.py",
        "src/dcvpp/experiments/dart_training.py",
        "src/dcvpp/experiments/v1_dart_readiness.py",
        "src/dcvpp/experiments/dart_forecast_oos.py",
        "src/dcvpp/experiments/dart_real_evidence.py",
        "src/dcvpp/experiments/dart_replay.py",
        "src/dcvpp/experiments/dart_value_select.py",
        "src/dcvpp/evaluation/dart_metrics.py",
        "src/dcvpp/evaluation/action_weighted_crps.py",
        "src/dcvpp/evaluation/profit_evaluator.py",
        "src/dcvpp/evaluation/dart_value_select.py",
        "src/dcvpp/optimization/stochastic_milp.py",
        "src/dcvpp/optimization/replay_engine.py",
        "src/dcvpp/statistics/block_bootstrap.py",
        "src/dcvpp/statistics/model_confidence_set.py",
    )
    records: list[dict[str, str]] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"DART implementation file is absent: {path}")
        records.append({"path": relative, "sha256": sha256_file(path)})
    return sha256_json(records)


def _anchor_only_contract(config: Mapping[str, Any]) -> dict[str, Any] | None:
    readiness = config.get("pre_dart_v1_readiness")
    amendment = (
        readiness.get("anchor_only_checkpoint_amendment")
        if isinstance(readiness, Mapping)
        else None
    )
    if not isinstance(amendment, Mapping) or amendment.get("enabled") is not True:
        return None
    return dict(amendment)


def _json_self_hash_is_valid(
    payload: Mapping[str, Any],
    *,
    field: str = "manifest_sha256",
    accept_utf8_unescaped: bool = False,
) -> bool:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        return False
    core = {key: value for key, value in payload.items() if key != field}
    if claimed == sha256_json(core):
        return True
    if not accept_utf8_unescaped:
        return False
    producer_payload = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return claimed == hashlib.sha256(producer_payload).hexdigest()


def _anchor_expected_track_count(config: Mapping[str, Any]) -> int:
    chronology = config.get("chronology", {})
    randomization = config.get("randomization_contract", {})
    roles = (
        randomization.get("forecast_training_seed_roles", {})
        if isinstance(randomization, Mapping)
        else {}
    )
    years = (
        chronology.get("selection_years", ())
        if isinstance(chronology, Mapping)
        else ()
    )
    seeds = (
        randomization.get("study_seeds", ())
        if isinstance(randomization, Mapping)
        else ()
    )
    neural = roles.get("neural_three_fit", ()) if isinstance(roles, Mapping) else ()
    # The common-gate contract has exactly the separately trained DA and RT heads.
    return len(tuple(years)) * 2 * len(tuple(neural)) * len(tuple(seeds))


def _selected_epochs_off_anchor(value: Any, *, interval: int) -> list[int]:
    failures: list[int] = []
    selected_keys = {
        "selected_epoch",
        "provisional_selected_epoch",
        "final_selected_epoch",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in selected_keys and item not in (None, ""):
                try:
                    epoch = int(item)
                except (TypeError, ValueError):
                    failures.append(-1)
                else:
                    if epoch <= 0 or epoch % interval != 0:
                        failures.append(epoch)
            else:
                failures.extend(_selected_epochs_off_anchor(item, interval=interval))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            failures.extend(_selected_epochs_off_anchor(item, interval=interval))
    return failures


def audit_anchor_training_lineage_evidence(
    repository_root: Path,
    *,
    dart_config: Mapping[str, Any],
    selection_training: Mapping[str, Any],
    expected_forecast_run_id: str,
    readiness_anchor_evidence: Mapping[str, Any] | None = None,
    required: bool | None = None,
) -> dict[str, Any]:
    """Verify the v3 -> v4 overlay and per-epoch archive before any replay."""

    root = Path(repository_root).resolve()
    failures: list[str] = []
    paths: dict[str, Path] = {}
    amendment = _anchor_only_contract(dart_config)
    if amendment is None or required is False:
        return {"required": False, "passed": True, "failures": [], "paths": {}}
    destination_run_id = str(amendment.get("destination_run_id", ""))
    if expected_forecast_run_id != destination_run_id:
        failures.append("formal DART forecast run ID differs from the frozen v4 destination")
    if str(selection_training.get("run_id", "")) != destination_run_id:
        failures.append("selection stage does not belong to the frozen v4 forecast run")
    try:
        candidate_pool_contract = _selection_candidate_pool_contract(
            selection_training
        )
    except (TypeError, ValueError) as exc:
        candidate_pool_contract = None
        failures.append(f"forecast-v4 selection admission failed: {exc}")

    checkpoint = dart_config.get("checkpoint_selection", {})
    grid = checkpoint.get("candidate_epoch_grid", {}) if isinstance(checkpoint, Mapping) else {}
    training = checkpoint.get("neural_training", {}) if isinstance(checkpoint, Mapping) else {}
    try:
        interval = int(grid.get("anchor_interval_epochs", 0))
    except (TypeError, ValueError):
        interval = 0
    if (
        interval != 5
        or grid.get("positive_epoch_required") is not True
        or grid.get("exact_anchor_required_for_all_three_seeds") is not True
        or grid.get("between_anchor_replay_forbidden") is not True
        or grid.get("selected_checkpoint_source")
        != "exact_persisted_anchor_payload_no_replay_training"
        or training.get("v4_every_completed_epoch_full_immutable_checkpoint_required")
        is not True
        or int(training.get("v4_full_epoch_checkpoint_interval_epochs", 0) or 0) != 1
        or training.get("v4_full_epoch_checkpoint_must_be_hash_pinned_and_append_only")
        is not True
    ):
        failures.append("v12 five-epoch selection/full-epoch archive contract changed")
    off_anchor = _selected_epochs_off_anchor(selection_training, interval=max(interval, 1))
    if off_anchor:
        failures.append(f"formal selection contains non-anchor epochs: {off_anchor[:5]}")

    stage_root_value = selection_training.get("stage_root")
    try:
        stage_root = _trusted_repository_path(root, stage_root_value)
    except (TypeError, ValueError):
        stage_root = root / "__invalid_v4_stage_root__"
    expected_stage_root = (
        root
        / "artifacts/dart_value_select"
        / f"run_id={destination_run_id}"
        / "forecast_oos"
    ).resolve()
    if stage_root != expected_stage_root:
        failures.append("selection stage root differs from the frozen forecast-v4 namespace")

    snapshot_path = _resolve(root, amendment.get("source_snapshot_manifest_path"))
    snapshot = _load_json(snapshot_path)
    if snapshot_path is not None:
        paths["v3_anchor_snapshot_manifest"] = snapshot_path
    evidence = dict(readiness_anchor_evidence or {})
    if snapshot is None:
        failures.append("v3 immutable anchor snapshot is absent")
    else:
        snapshot_sha = sha256_file(snapshot_path)
        if (
            snapshot.get("source_run_id") != amendment.get("source_run_id")
            or snapshot.get("destination_run_id") != destination_run_id
            or int(snapshot.get("record_count", -1))
            != int(amendment.get("source_anchor_record_count", -2))
            or int(snapshot.get("total_bytes", -1))
            != int(amendment.get("source_anchor_total_bytes", -2))
            or snapshot.get("records_sha256")
            != amendment.get("source_anchor_records_sha256")
            or snapshot.get("completed_oos_unit_count") != 0
            or snapshot.get("copy_hardlink_or_source_rewrite_performed") is not False
            or not _json_self_hash_is_valid(snapshot)
            or (
                amendment.get("source_snapshot_manifest_file_sha256")
                not in (None, snapshot_sha)
            )
            or (
                amendment.get("source_snapshot_manifest_byte_count")
                not in (None, snapshot_path.stat().st_size)
            )
            or (
                amendment.get("source_snapshot_manifest_sha256")
                not in (None, snapshot.get("manifest_sha256"))
            )
        ):
            failures.append("v3 immutable anchor snapshot contract/hash changed")
        if evidence and (
            evidence.get("snapshot_manifest_file_sha256") != snapshot_sha
            or evidence.get("snapshot_manifest_sha256")
            != snapshot.get("manifest_sha256")
            or evidence.get("records_sha256") != snapshot.get("records_sha256")
            or int(evidence.get("record_count", -1)) != int(snapshot.get("record_count", -2))
            or int(evidence.get("total_bytes", -1)) != int(snapshot.get("total_bytes", -2))
        ):
            failures.append("v8 readiness anchor evidence differs from the source snapshot")

    import_path = stage_root / "recovery_overlay/v3_anchor_import_ledger.json"
    import_ledger = _load_json(import_path)
    paths["v4_anchor_import_ledger"] = import_path
    expected_tracks = _anchor_expected_track_count(dart_config)
    overlay = selection_training.get("anchor_overlay_recovery")
    if not isinstance(overlay, Mapping):
        failures.append("forecast-v4 selection stage omits anchor overlay recovery")
    else:
        try:
            overlay_snapshot = _trusted_repository_path(
                root, overlay.get("snapshot_manifest_path")
            )
            overlay_import = _trusted_repository_path(
                root, overlay.get("import_ledger_path")
            )
            overlay_scheduler = _trusted_repository_path(
                root, overlay.get("scheduler_state_path")
            )
        except (TypeError, ValueError):
            overlay_snapshot = overlay_import = overlay_scheduler = root / "__invalid__"
        if (
            overlay.get("status") != "V4_OVERLAY_PREPARED_NO_TRAINING_LAUNCHED"
            or overlay_snapshot != snapshot_path
            or overlay.get("snapshot_manifest_sha256")
            != (
                sha256_file(snapshot_path)
                if snapshot_path is not None and snapshot_path.is_file()
                else None
            )
            or overlay_import != import_path
            or overlay.get("import_ledger_file_sha256")
            != (sha256_file(import_path) if import_path.is_file() else None)
            or int(overlay.get("imported_highest_seed_state_count", -1))
            != expected_tracks
            or overlay_scheduler
            != stage_root / "checkpoint_scheduler/hybrid_scheduler_state.json"
            or not overlay_scheduler.is_file()
            or overlay.get("scheduler_state_file_sha256")
            != (
                sha256_file(overlay_scheduler)
                if overlay_scheduler.is_file()
                else None
            )
            or overlay.get("training_launched_by_prepare") is not False
            or overlay.get("v3_files_moved_modified_or_deleted") != 0
        ):
            failures.append("forecast-v4 stage anchor overlay evidence/hash changed")
    if import_ledger is None:
        failures.append("v4 anchor import ledger is absent")
    elif (
        import_ledger.get("status") != "V4_OVERLAY_IMPORTED_NO_TRAINING_LAUNCHED"
        # The immutable v3->v4 compatibility producer hashes JSON with
        # ensure_ascii=False.  Preserve that producer-native contract for
        # ledgers containing Unicode repository paths; do not rewrite history.
        or not _json_self_hash_is_valid(
            import_ledger, accept_utf8_unescaped=True
        )
        or import_ledger.get("v3_snapshot_manifest_sha256")
        != (
            sha256_file(snapshot_path)
            if snapshot_path is not None and snapshot_path.is_file()
            else None
        )
        or int(import_ledger.get("imported_highest_seed_state_count", -1))
        != expected_tracks
        or len(import_ledger.get("imports", ())) != expected_tracks
        or import_ledger.get("v3_files_moved_modified_or_deleted") != 0
        or import_ledger.get(
            "training_not_launched_during_import",
            import_ledger.get("training_launched") is False,
        )
        is not True
    ):
        failures.append("v4 anchor import ledger contract/hash changed")

    inventory_path = stage_root / "recovery_overlay/v4_epoch_archive_inventory.json"
    inventory = _load_json(inventory_path)
    paths["v4_epoch_archive_inventory"] = inventory_path
    if inventory is None:
        failures.append("v4 full-epoch archive inventory is absent")
    else:
        track_records = inventory.get("track_manifests")
        if (
            inventory.get("status") != "COMPLETE"
            or not _json_self_hash_is_valid(inventory)
            or inventory.get("every_completed_epoch_full_state_persisted") is not True
            or inventory.get("formal_selection_candidates_divisible_by_five_only")
            is not True
            or int(inventory.get("track_count", -1)) != expected_tracks
            or not isinstance(track_records, Sequence)
            or len(track_records) != expected_tracks
            or inventory.get("track_manifests_sha256") != sha256_json(track_records)
        ):
            failures.append("v4 full-epoch archive inventory contract/hash changed")
        elif isinstance(track_records, Sequence):
            for record in track_records:
                if not isinstance(record, Mapping):
                    failures.append("v4 full-epoch archive inventory record is malformed")
                    break
                try:
                    track_path = _trusted_repository_path(root, record.get("path"))
                except (TypeError, ValueError):
                    failures.append("v4 full-epoch archive manifest escapes the repository")
                    break
                track = _load_json(track_path)
                if (
                    track is None
                    or record.get("sha256") != sha256_file(track_path)
                    or not _json_self_hash_is_valid(track)
                    or track.get("status") != "COMPLETE_THROUGH_CURRENT_EPOCH"
                    or track.get("every_completed_epoch_full_state_persisted") is not True
                    or track.get("formal_selection_candidates_divisible_by_five_only")
                    is not True
                ):
                    failures.append("a v4 full-epoch archive track hash/contract changed")
                    break
                for epoch_record in track.get("records", ()):
                    try:
                        epoch = int(epoch_record.get("epoch", -1))
                    except (AttributeError, TypeError, ValueError):
                        failures.append("a v4 archived epoch record is malformed")
                        break
                    if bool(epoch_record.get("formal_selection_candidate")) != (
                        epoch > 0 and epoch % interval == 0
                    ):
                        failures.append("a v4 archived epoch has an invalid selection role")
                        break
                if failures and failures[-1].startswith("a v4 archived epoch"):
                    break
    compatibility = selection_training.get("anchor_only_compatibility")
    if (
        not isinstance(compatibility, Mapping)
        or compatibility.get("schema_version") != "dcvpp.dart_anchor_only_overlay.v1"
        or compatibility.get("v3_snapshot_manifest_sha256")
        != (
            sha256_file(snapshot_path)
            if snapshot_path is not None and snapshot_path.is_file()
            else None
        )
        or compatibility.get(
            "provisional_candidate_grid_changed_to_exact_five_epoch_anchors"
        )
        is not True
        or compatibility.get("arbitrary_non_anchor_replay_forbidden") is not True
        or compatibility.get("v3_files_moved_modified_or_deleted") != 0
    ):
        failures.append("forecast-v4 anchor-only compatibility evidence changed")
    embedded_inventory = selection_training.get("v4_full_epoch_archive_inventory")
    if (
        not isinstance(embedded_inventory, Mapping)
        or embedded_inventory.get("path") != inventory_path.as_posix()
        or embedded_inventory.get("sha256")
        != (sha256_file(inventory_path) if inventory_path.is_file() else None)
        or embedded_inventory.get("manifest_sha256")
        != (inventory or {}).get("manifest_sha256")
        or embedded_inventory.get("track_manifests_sha256")
        != (inventory or {}).get("track_manifests_sha256")
    ):
        failures.append("forecast-v4 stage does not pin its full-epoch archive inventory")
    return {
        "required": True,
        "passed": not failures,
        "failures": failures,
        "paths": paths,
        "source_run_id": amendment.get("source_run_id"),
        "destination_run_id": destination_run_id,
        "expected_archive_track_count": expected_tracks,
        "candidate_pool_contract": candidate_pool_contract,
    }


def _freeze_dependencies(
    root: Path,
    dart_config: Mapping[str, Any],
    selection_training: Mapping[str, Any],
    selection_evidence: DARTPhaseEvidence,
    physical_result: Mapping[str, Any],
    forecast_result: Mapping[str, Any],
    candidate_pool_contract: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    input_fingerprints = selection_training.get("input_fingerprints", {})
    training_config_sha = selection_training.get("training_config_sha256")
    if not isinstance(input_fingerprints, Mapping) or not training_config_sha:
        raise ValueError("selection training lacks frozen input/training fingerprints")
    values = {
        "data_through_2024": sha256_json(
            {
                "referenced_nyiso_partitions": input_fingerprints.get(
                    "referenced_nyiso_partitions"
                ),
                "selection_source_evidence": selection_evidence.source_evidence_sha256,
            }
        ),
        "feature_contract": sha256_json(dart_config.get("forecast_issue_contract", {})),
        "model_hyperparameters": str(training_config_sha),
        "calibration_rule": sha256_json(
            {
                "probabilistic_credibility": dart_config.get(
                    "probabilistic_credibility", {}
                ),
                "actionability": dart_config.get("actionability", {}),
                "joint_coupling": selection_training.get("phase_evidence", {}).get(
                    "joint_coupling_manifest_sha256"
                )
                if isinstance(selection_training.get("phase_evidence"), Mapping)
                else None,
            }
        ),
        "selection_rule": sha256_json(
            {
                "v1_testing_selection_policy": input_fingerprints.get(
                    "v1_testing_selection_policy"
                ),
                "rules": {
                    key: dart_config.get(key, {})
                    for key in (
                        "feasibility",
                        "bootstrap",
                        "model_confidence_set",
                        "selection",
                        "contextual_selector",
                        "aggregation_contract",
                        "selection_freeze_contract",
                        "pre_dart_v1_readiness",
                        "locked_test_exposure_disclosure",
                    )
                },
                "candidate_pool_contract": candidate_pool_contract,
            }
        ),
        "scenario_optimizer_user_contract": sha256_json(
            {
                "decision_pipeline_contract": dart_config.get(
                    "decision_pipeline_contract", {}
                ),
                "identical_replay": dart_config.get("identical_replay", {}),
                "physical": _result_manifest_identity(root, physical_result),
                "forecast": _result_manifest_identity(root, forecast_result),
            }
        ),
        "code": _code_fingerprint(root),
    }
    missing = set(FREEZE_FINGERPRINT_KEYS).difference(values)
    if missing or any(len(str(value)) != 64 for value in values.values()):
        raise ValueError(f"invalid DART freeze dependencies: missing={sorted(missing)}")
    return values


def _default_replay_builder(**kwargs: Any) -> DARTPhaseEvidence | Mapping[str, Any]:
    from dcvpp.experiments.dart_real_evidence import run_real_dart_phase_evidence

    # The composition seam uses the descriptive ``repository_root`` keyword,
    # while the concrete evidence builder keeps its public parameter named
    # ``root``.  Keep custom test/build callbacks stable and translate only at
    # the default production boundary.
    forwarded = dict(kwargs)
    forwarded["root"] = forwarded.pop("repository_root")
    repository_root = Path(forwarded["root"]).resolve()
    value = run_real_dart_phase_evidence(**forwarded)
    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    # The concrete builder returns internal absolute paths for direct callers.
    # Persist only repository-relative paths at the public pipeline boundary.
    for key in (
        "manifest_path",
        "metric_aggregates_path",
        "diagnostic_availability_path",
    ):
        raw = normalized.get(key)
        if raw in (None, ""):
            continue
        path = Path(str(raw)).resolve()
        try:
            normalized[key] = path.relative_to(repository_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"real replay output escapes the repository: {key}") from exc
    return normalized


def _default_v1_readiness_evaluator(**kwargs: Any) -> Mapping[str, Any]:
    from dcvpp.experiments.v1_dart_readiness import run_v1_dart_readiness_audit

    return run_v1_dart_readiness_audit(**kwargs)


def audit_v1_dart_readiness_evidence(
    repository_root: Path,
    result: Mapping[str, Any] | None,
    *,
    expected_v1_manifest: Path | None = None,
) -> dict[str, Any]:
    """Verify the immutable-v1 strategy audit before formal DART training."""

    root = Path(repository_root).resolve()
    supplied = dict(result or {})
    manifest_path = _resolve(root, supplied.get("manifest_path"))
    manifest = _load_json(manifest_path)
    failures: list[str] = []
    if manifest is None:
        failures.append("v1 DART-readiness manifest is absent or invalid")
        manifest = {}
    actual_manifest_sha = (
        sha256_file(manifest_path)
        if manifest_path is not None and manifest_path.is_file()
        else None
    )
    declared_manifest_sha = supplied.get(
        "manifest_file_sha256", supplied.get("manifest_sha256")
    )
    if declared_manifest_sha and declared_manifest_sha != actual_manifest_sha:
        failures.append("v1 DART-readiness manifest hash changed")
    manifest_fingerprint = manifest.get("manifest_fingerprint")
    manifest_core = {
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    }
    if manifest_fingerprint != sha256_json(manifest_core):
        failures.append("v1 DART-readiness internal manifest fingerprint changed")
    if str(manifest.get("status", supplied.get("status", ""))).upper() != "COMPLETE":
        failures.append("v1 DART-readiness status is not COMPLETE")
    policy_sha = manifest.get(
        "frozen_policy_sha256", supplied.get("frozen_policy_sha256")
    )
    if (
        not isinstance(policy_sha, str)
        or len(policy_sha) != 64
        or any(character not in "0123456789abcdef" for character in policy_sha.lower())
    ):
        failures.append("v1 testing/selection policy hash is absent or invalid")
    policy_path = _resolve(root, manifest.get("policy_freeze_path"))
    policy = _load_json(policy_path)
    required_gate_names: tuple[str, ...] = ()
    required_disclosure: Any = None
    frozen_config_sha: str | None = None
    frozen_evaluator_sha: str | None = None
    frozen_readiness_policy: Mapping[str, Any] | None = None
    frozen_policy_schema: str | None = None
    if policy is None:
        failures.append("v1 testing/selection policy-freeze file is absent")
    else:
        policy_contract = policy.get("contract")
        if not isinstance(policy_contract, Mapping):
            failures.append("v1 testing/selection policy-freeze contract is malformed")
        else:
            frozen_policy_schema = str(policy_contract.get("schema_version", ""))
            if (
                policy.get("frozen_policy_sha256") != policy_sha
                or sha256_json(policy_contract) != policy_sha
            ):
                failures.append("v1 testing/selection frozen policy content changed")
            policy_sections = policy_contract.get("policy_sections")
            frozen_config_sha = str(policy_contract.get("config_sha256", ""))
            frozen_config_path = _resolve(root, policy_contract.get("config_path"))
            if (
                len(frozen_config_sha) != 64
                or frozen_config_path is None
                or not frozen_config_path.is_file()
                or sha256_file(frozen_config_path) != frozen_config_sha
            ):
                failures.append("v1 frozen DART config bytes changed")
            frozen_evaluator_sha = str(
                policy_contract.get("evaluator_code_sha256", "")
            )
            evaluator_path = Path(__file__).with_name("v1_dart_readiness.py")
            if (
                len(frozen_evaluator_sha) != 64
                or not evaluator_path.is_file()
                or sha256_file(evaluator_path) != frozen_evaluator_sha
            ):
                failures.append("v1 readiness evaluator differs from the frozen policy")
            readiness_policy = (
                policy_sections.get("pre_dart_v1_readiness")
                if isinstance(policy_sections, Mapping)
                else None
            )
            if isinstance(readiness_policy, Mapping):
                frozen_readiness_policy = readiness_policy
            configured_gates = (
                readiness_policy.get("required_completion_gates")
                if isinstance(readiness_policy, Mapping)
                else None
            )
            if (
                not isinstance(configured_gates, Sequence)
                or isinstance(configured_gates, (str, bytes))
                or not configured_gates
                or any(not isinstance(key, str) or not key for key in configured_gates)
                or len(set(configured_gates)) != len(configured_gates)
            ):
                failures.append(
                    "v1 testing/selection policy has no valid frozen readiness-gate list"
                )
            else:
                required_gate_names = tuple(configured_gates)
            required_disclosure = policy_contract.get(
                "locked_test_exposure_disclosure"
            )
            if required_disclosure in (None, ""):
                failures.append(
                    "v1 testing/selection policy has no locked-test exposure disclosure"
                )
            post_result_amendment = (
                policy.get(
                    "post_result_structural_missingness_and_three_seed_amendment"
                )
                is True
            )
            validation_informed_method_amendment = (
                policy.get("validation_informed_selection_method_amendment") is True
            )
            hierarchical_checkpoint_amendment = (
                policy.get("hierarchical_checkpoint_selection_amendment") is True
            )
            anchor_checkpoint_amendment = (
                policy.get("anchor_only_checkpoint_amendment") is True
            )
            if (
                policy.get("frozen_before_v1_complete_manifest") is not True
                and not (post_result_amendment or anchor_checkpoint_amendment)
            ):
                failures.append("v1 testing/selection policy was not frozen before v1")
            chronology_scope = policy_contract.get("audited_chronology_scope")
            if (
                manifest.get("evaluation_split")
                != policy_contract.get("evaluation_split")
                or manifest.get("evaluation_calendar_years")
                != policy_contract.get("evaluation_calendar_years")
                or manifest.get("evaluation_year_timezone")
                != policy_contract.get("evaluation_year_timezone")
                or manifest.get("locked_2025_metrics_consumed")
                != policy_contract.get("locked_2025_metrics_consumed")
                or manifest.get("locked_2025_metrics_materialized")
                != policy_contract.get("locked_2025_metrics_materialized")
                or not isinstance(chronology_scope, Mapping)
                or manifest.get("audited_chronology_scope")
                != chronology_scope.get("claim")
                or manifest.get("forecast_training_seed_roles")
                != policy_contract.get("forecast_training_seed_roles")
                or manifest.get("all_candidate_replay_seed_slots")
                != policy_contract.get("all_candidate_replay_seed_slots")
            ):
                failures.append(
                    "v1 readiness evidence differs from the frozen policy scope"
                )
        expected_policy_file_sha = manifest.get("policy_freeze_file_sha256")
        if (
            not isinstance(expected_policy_file_sha, str)
            or expected_policy_file_sha != sha256_file(policy_path)
        ):
            failures.append("v1 testing/selection policy-freeze bytes changed")
        pre_result_amendment = policy.get("pre_result_amendment")
        post_result_amendment = (
            policy.get("post_result_structural_missingness_and_three_seed_amendment")
            is True
        )
        validation_informed_method_amendment = (
            policy.get("validation_informed_selection_method_amendment") is True
        )
        anchor_checkpoint_amendment = (
            policy.get("anchor_only_checkpoint_amendment") is True
        )
        if not isinstance(pre_result_amendment, bool):
            failures.append("v1 policy pre-result amendment flag is absent")
        elif (
            pre_result_amendment
            or post_result_amendment
            or validation_informed_method_amendment
            or anchor_checkpoint_amendment
        ):
            superseded = policy.get("superseded_freeze")
            if not isinstance(superseded, Mapping):
                failures.append("v1 superseded policy-freeze record is absent")
            else:
                superseded_path = _resolve(root, superseded.get("path"))
                if superseded_path is None or not superseded_path.is_file():
                    failures.append("v1 superseded policy-freeze bytes are absent")
                elif superseded.get("file_sha256") != sha256_file(superseded_path):
                    failures.append("v1 superseded policy-freeze bytes changed")
            ledger_path = _resolve(root, manifest.get("amendment_ledger_path"))
            ledger = _load_json(ledger_path)
            if ledger is None:
                failures.append("v1 readiness amendment ledger is absent")
            else:
                declared_ledger_sha = manifest.get("amendment_ledger_sha256")
                if declared_ledger_sha != sha256_file(ledger_path):
                    failures.append("v1 readiness amendment ledger bytes changed")
                ledger_fingerprint = ledger.get("ledger_fingerprint")
                ledger_core = {
                    key: value
                    for key, value in ledger.items()
                    if key != "ledger_fingerprint"
                }
                replacement = ledger.get("replacement")
                if ledger_fingerprint != sha256_json(ledger_core):
                    failures.append("v1 readiness amendment ledger fingerprint changed")
                ledger_status = str(ledger.get("status", "")).upper()
                anchor_contract = (
                    frozen_readiness_policy.get("anchor_only_checkpoint_amendment", {})
                    if isinstance(frozen_readiness_policy, Mapping)
                    else {}
                )
                anchor_status = str(
                    anchor_contract.get(
                        "amendment_status",
                        "AMENDED_PRE_OOS_ANCHOR_ONLY_CHECKPOINT_SELECTION_AFTER_FAIL_CLOSED_REPLAY_DEFECT",
                    )
                ).upper()
                anchor_valid = (
                    anchor_checkpoint_amendment
                    and isinstance(anchor_contract, Mapping)
                    and anchor_contract.get("enabled") is True
                    and ledger_status == anchor_status
                    and ledger.get("performance_independent") is True
                    and int(ledger.get("formal_dart_completed_oos_units", -1)) == 0
                    and ledger.get("formal_dart_replay_started") is False
                    and ledger.get("locked_2025_dart_metrics_consumed") is False
                    and ledger.get("market_replay_milp_action_or_value_results_consumed")
                    is False
                    and ledger.get("anchor_import_evidence")
                    == manifest.get("anchor_import_evidence")
                )
                post_result_valid = (
                    post_result_amendment
                    and ledger_status
                    in {
                        "AMENDED_POST_RESULT_STRUCTURAL_MISSINGNESS_AND_THREE_SEED_PANEL",
                        "AMENDED_PRE_FORMAL_DART_CHECKPOINT_SELECTION",
                    }
                    and ledger.get("amended_before_v1_complete_manifest") is False
                    and (
                        ledger.get("performance_independent") is True
                        or (
                            hierarchical_checkpoint_amendment
                            and ledger_status
                            == "AMENDED_PRE_FORMAL_DART_CHECKPOINT_SELECTION"
                            and ledger.get(
                                "hierarchical_checkpoint_selection_validation_informed"
                            )
                            is True
                        )
                    )
                    and ledger.get("confirmatory_study_seeds")
                    == policy_contract.get("all_candidate_replay_seed_slots")
                    and ledger.get("legacy_excluded_seeds") == [20260908, 20260918]
                    and isinstance(frozen_readiness_policy, Mapping)
                    and frozen_readiness_policy.get(
                        "post_recovery_structural_missingness_amendment", {}
                    ).get("enabled")
                    is True
                    and frozen_readiness_policy.get(
                        "post_recovery_structural_missingness_amendment", {}
                    ).get("three_seed_amendment_enabled")
                    is True
                    and frozen_readiness_policy.get(
                        "post_recovery_structural_missingness_amendment", {}
                    ).get("performance_independent")
                    is True
                    and (
                        not validation_informed_method_amendment
                        or (
                            ledger.get(
                                "selection_method_amendment_validation_informed"
                            )
                            is True
                            and ledger.get("locked_2025_dart_metrics_consumed")
                            is False
                            and ledger.get("formal_dart_training_or_replay_existed")
                            is False
                            and isinstance(frozen_readiness_policy, Mapping)
                            and frozen_readiness_policy.get(
                                "validation_informed_selection_method_amendment",
                                {},
                            ).get("enabled")
                            is True
                        )
                    )
                )
                pre_result_valid = (
                    pre_result_amendment
                    and str(ledger.get("status", "")).upper() == "AMENDED_PRE_RESULT"
                    and ledger.get("amended_before_v1_complete_manifest") is True
                )
                if (
                    not (pre_result_valid or post_result_valid or anchor_valid)
                    or ledger.get("superseded") != superseded
                    or not isinstance(replacement, Mapping)
                    or replacement.get("frozen_policy_sha256") != policy_sha
                    or replacement.get("file_sha256") != expected_policy_file_sha
                ):
                    failures.append("v1 readiness amendment chain is invalid")
    readiness_outputs = manifest.get("outputs")
    if not isinstance(readiness_outputs, Sequence) or not readiness_outputs:
        failures.append("v1 DART-readiness hashed output ledger is absent")
    else:
        for record in readiness_outputs:
            if not isinstance(record, Mapping):
                failures.append("v1 DART-readiness output record is malformed")
                continue
            output_path = _resolve(root, record)
            if output_path is None or not output_path.is_file():
                failures.append("v1 DART-readiness output is absent")
            elif record.get("sha256") != sha256_file(output_path):
                failures.append("v1 DART-readiness output hash changed")
    if (
        manifest.get(
            "formal_dart_training_allowed",
            supplied.get("formal_dart_training_allowed"),
        )
        is not True
    ):
        failures.append("v1 readiness did not authorize formal DART training")
    if (
        manifest.get(
            "native_rt_cross_head_label",
            supplied.get("native_rt_cross_head_label"),
        )
        != "AUXILIARY_NOT_DA_GATE_ELIGIBLE"
    ):
        failures.append("v1 native-RT cross-head results lack the auxiliary-only label")
    if (
        manifest.get(
            "native_rt_selection_eligible",
            supplied.get("native_rt_selection_eligible"),
        )
        is not False
    ):
        failures.append("v1 native-RT cross-head results remain selection-eligible")
    if manifest.get("evaluation_split") != "validation":
        failures.append("v1 readiness was not evaluated on validation only")
    if manifest.get("evaluation_calendar_years") != [2024]:
        failures.append(
            "v1 readiness did not use the frozen 2024 validation projection"
        )
    if manifest.get("locked_2025_metrics_consumed") is not False:
        failures.append(
            "v1 readiness consumed or failed to exclude locked 2025 metrics"
        )
    if manifest.get("locked_2025_metrics_materialized") is not False:
        failures.append("v1 readiness materialized or failed to exclude locked 2025 metrics")
    if (
        manifest.get("performance_gate_applied") is not False
        or manifest.get("performance_results_may_update_policy") is not False
    ):
        failures.append("v1 readiness improperly used performance as a policy gate")
    if manifest.get("common_da_gate_retraining_required") is not True:
        failures.append("v1 readiness did not require common DA-gate retraining")
    if isinstance(frozen_readiness_policy, Mapping):
        formal_readiness_run_id = frozen_readiness_policy.get("formal_run_id")
        if formal_readiness_run_id not in (None, "") and manifest.get(
            "run_id"
        ) != formal_readiness_run_id:
            failures.append("v1 readiness run ID differs from the frozen formal run")
        frozen_anchor_contract = frozen_readiness_policy.get(
            "anchor_only_checkpoint_amendment"
        )
        if (
            isinstance(frozen_anchor_contract, Mapping)
            and frozen_anchor_contract.get("enabled") is True
        ):
            anchor_evidence = manifest.get("anchor_import_evidence")
            if not isinstance(anchor_evidence, Mapping):
                failures.append("v8 readiness anchor-import evidence is absent")
            else:
                anchor_path = _resolve(
                    root, anchor_evidence.get("snapshot_manifest_path")
                )
                anchor_snapshot = _load_json(anchor_path)
                if (
                    anchor_snapshot is None
                    or anchor_evidence.get("source_run_id")
                    != frozen_anchor_contract.get("source_run_id")
                    or anchor_evidence.get("destination_run_id")
                    != frozen_anchor_contract.get("destination_run_id")
                    or anchor_evidence.get("snapshot_manifest_file_sha256")
                    != sha256_file(anchor_path)
                    or anchor_evidence.get("snapshot_manifest_sha256")
                    != anchor_snapshot.get("manifest_sha256")
                    or anchor_evidence.get("records_sha256")
                    != frozen_anchor_contract.get("source_anchor_records_sha256")
                    or int(anchor_evidence.get("record_count", -1))
                    != int(frozen_anchor_contract.get("source_anchor_record_count", -2))
                    or int(anchor_evidence.get("total_bytes", -1))
                    != int(frozen_anchor_contract.get("source_anchor_total_bytes", -2))
                    or anchor_evidence.get("completed_oos_unit_count") != 0
                    or anchor_evidence.get("copy_hardlink_or_source_rewrite_performed")
                    is not False
                ):
                    failures.append("v8 readiness anchor-import evidence/hash changed")
    if manifest.get("locked_test_exposure_disclosure") != required_disclosure:
        failures.append("prior auxiliary v1 2025 exposure disclosure is absent")
    gates = manifest.get("completion_gates", supplied.get("completion_gates"))
    if not isinstance(gates, Mapping):
        failures.append("v1 DART-readiness completion gates are absent")
        gates = {}
    if len(gates) != len(required_gate_names) or set(gates) != set(required_gate_names):
        failures.append("v1 DART-readiness gates differ from the frozen policy")
    for key in required_gate_names:
        if gates.get(key) is not True:
            failures.append(f"v1 DART-readiness gate is not true: {key}")
    if expected_v1_manifest is not None:
        candidate = Path(expected_v1_manifest)
        v1_path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            v1_path.relative_to(root)
        except ValueError:
            failures.append("immutable v1 run manifest escapes the repository")
            v1_path = root / "__invalid_external_v1_manifest__"
        if not v1_path.is_file():
            failures.append("immutable v1 run manifest is absent")
        else:
            expected_sha = sha256_file(v1_path)
            observed_sha = manifest.get(
                "v1_forecast_manifest_sha256",
                supplied.get("v1_forecast_manifest_sha256"),
            )
            if observed_sha != expected_sha:
                failures.append(
                    "readiness evidence does not pin the immutable v1 bytes"
                )
    if expected_v1_manifest is not None:
        candidate = Path(expected_v1_manifest)
        recovery_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        try:
            recovery_path.relative_to(root)
        except ValueError:
            recovery_path = None
    else:
        recovery_path = _resolve(root, manifest.get("v1_forecast_manifest_path"))
    recovery_manifest = _load_json(recovery_path)
    if frozen_readiness_policy is None:
        failures.append("frozen pre-DART readiness policy is absent")
    else:
        readiness_schema = str(frozen_readiness_policy.get("schema_version", ""))
        policy_version = (
            frozen_policy_schema.rsplit(".", 1)[-1]
            if frozen_policy_schema and "." in frozen_policy_schema
            else ""
        )
        if (
            not policy_version
            or not readiness_schema.endswith(f".{policy_version}")
            or not str(manifest.get("schema_version", "")).endswith(
                f".{policy_version}"
            )
        ):
            failures.append("readiness manifest/policy schema versions disagree")
        accepted_schemas = frozen_readiness_policy.get(
            "accepted_v1_manifest_schema_versions"
        )
        if (
            not isinstance(accepted_schemas, Sequence)
            or isinstance(accepted_schemas, (str, bytes))
            or not accepted_schemas
            or frozen_readiness_policy.get(
                "current_run_requires_hash_verified_recovery_lineage"
            )
            is not True
            or recovery_manifest is None
            or recovery_manifest.get("schema_version") not in accepted_schemas
            or recovery_manifest.get("run_id")
            != frozen_readiness_policy.get("recovery_run_id")
            or str(recovery_manifest.get("status", "")).upper() != "COMPLETE"
        ):
            failures.append(
                "immutable v1 evidence is not the policy-accepted recovery lineage"
            )
        requirements = frozen_readiness_policy.get("recovery_requirements")
        if isinstance(requirements, Mapping) and recovery_manifest is not None:
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
                if key in requirements and recovery_manifest.get(key) != requirements.get(
                    key
                ):
                    failures.append(f"v1 recovery requirement changed: {key}")
            recovery_counts = recovery_manifest.get("counts")
            if not isinstance(recovery_counts, Mapping):
                failures.append("v1 recovery task counts are absent")
            else:
                for key in (
                    "tasks_requested",
                    "tasks_completed",
                    "tasks_failed",
                    "failed_seeds",
                ):
                    if key in requirements and recovery_counts.get(key) != requirements.get(
                        key
                    ):
                        failures.append(f"v1 recovery count changed: {key}")
        else:
            failures.append("frozen v1 recovery requirements are absent")
    return {
        "passed": not failures,
        "failures": failures,
        "manifest_path": _relative(root, manifest_path) if manifest_path else None,
        "manifest_file_sha256": actual_manifest_sha,
        "frozen_policy_sha256": policy_sha,
        "frozen_dart_config_sha256": frozen_config_sha,
        "frozen_readiness_evaluator_sha256": frozen_evaluator_sha,
        "native_rt_cross_head_label": manifest.get(
            "native_rt_cross_head_label",
            supplied.get("native_rt_cross_head_label"),
        ),
        "native_rt_selection_eligible": manifest.get(
            "native_rt_selection_eligible",
            supplied.get("native_rt_selection_eligible"),
        ),
        "completion_gates": dict(gates),
        "required_completion_gates": list(required_gate_names),
        "locked_test_exposure_disclosure": manifest.get(
            "locked_test_exposure_disclosure"
        ),
        "anchor_import_evidence": manifest.get("anchor_import_evidence"),
    }


def run_v1_dart_readiness_stage(
    repository_root: Path,
    *,
    v1_manifest_path: Path,
    dart_config_path: Path,
    run_id: str,
    resume: bool = True,
    evaluator: V1ReadinessEvaluator | None = None,
) -> dict[str, Any]:
    """Run the strategy/readiness audit without opening any formal DART fit."""

    root = Path(repository_root).resolve()
    output_root = (
        root / "artifacts" / "dart_value_select" / "v1_readiness" / f"run_id={run_id}"
    )
    value = (evaluator or _default_v1_readiness_evaluator)(
        repository_root=root,
        v1_manifest_path=Path(v1_manifest_path).resolve(),
        dart_config_path=Path(dart_config_path).resolve(),
        output_root=output_root,
        run_id=run_id,
        resume=resume,
    )
    if not isinstance(value, Mapping):
        raise TypeError("v1 DART-readiness evaluator did not return a mapping")
    result = dict(value)
    audit = audit_v1_dart_readiness_evidence(
        root,
        result,
        expected_v1_manifest=Path(v1_manifest_path),
    )
    result["readiness_audit"] = audit
    result["downstream_ready"] = audit["passed"]
    if not audit["passed"]:
        result["status"] = "PARTIAL"
    return result


def _normalize_replay_result(
    value: DARTPhaseEvidence | Mapping[str, Any],
    *,
    phase: str,
    candidate_pool_contract: Mapping[str, Any] | None = None,
) -> tuple[DARTPhaseEvidence, dict[str, Any]]:
    if isinstance(value, DARTPhaseEvidence):
        evidence = value
        metadata: dict[str, Any] = {
            "status": "UNVERIFIED_DIRECT_EVIDENCE",
            "phase": phase,
            "evidence_fingerprint": evidence.fingerprint,
            "actionable_pending_count": 0,
        }
    elif isinstance(value, Mapping):
        supplied = dict(value)
        raw = supplied.get("evidence", supplied.get("phase_evidence"))
        if isinstance(raw, DARTPhaseEvidence):
            evidence = raw
        elif isinstance(raw, Mapping):
            evidence = DARTPhaseEvidence.from_dict(raw)
        elif "replay_results" in supplied and "metric_records" in supplied:
            evidence = DARTPhaseEvidence.from_dict(supplied)
        else:
            raise TypeError("real DART replay stage did not return phase evidence")
        status = str(supplied.get("status", "COMPLETE")).upper()
        if status != "COMPLETE":
            raise RuntimeError(f"real DART {phase} replay stage is {status}")
        metadata = _replay_stage_snapshot(supplied)
        metadata["evidence_fingerprint"] = evidence.fingerprint
    else:
        raise TypeError("real DART replay builder returned an unsupported type")
    if evidence.phase != phase:
        raise ValueError(
            f"real DART evidence phase is {evidence.phase}, expected {phase}"
        )
    expected_pool_sha = (
        validate_partial_candidate_pool_contract(candidate_pool_contract).get(
            "contract_sha256"
        )
        if candidate_pool_contract is not None
        else None
    )
    observed_pool = evidence.candidate_pool_contract
    if expected_pool_sha is None:
        if observed_pool is not None:
            raise ValueError("full replay unexpectedly carries a partial candidate pool")
    else:
        observed_pool_sha = validate_partial_candidate_pool_contract(
            observed_pool
        ).get("contract_sha256")
        if observed_pool_sha != expected_pool_sha:
            raise ValueError("real replay candidate pool differs from selection admission")
    return evidence, metadata


def _replay_stage_snapshot(value: Any) -> dict[str, Any]:
    """Retain resumability/provenance without embedding a large dataclass."""

    if not isinstance(value, Mapping):
        return {"status": "UNVERIFIED_DIRECT_EVIDENCE"}
    snapshot: dict[str, Any] = {}
    for key in (
        "status",
        "phase",
        "manifest_path",
        "manifest_sha256",
        "completion_gates",
        "actionable_pending_count",
        "synthetic_fallback_used",
        "v1_2025_metrics_consumed",
        "candidate_pool_contract",
        "terminal_status_cap",
    ):
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, Path):
            item = item.as_posix()
        elif isinstance(item, Mapping):
            item = dict(item)
        snapshot[key] = item
    return snapshot


def _manifest_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _write_pipeline_manifest(
    root: Path,
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = dict(payload)
    manifest["schema_version"] = DART_PIPELINE_SCHEMA_VERSION
    manifest["updated_at_utc"] = utc_now()
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = sha256_json(core)
    atomic_json(path, manifest)
    return manifest


def _early_result(
    root: Path,
    manifest_path: Path,
    *,
    run_ids: Mapping[str, str],
    stage: str,
    stages: Mapping[str, Any],
    error: Exception | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    pending_candidates: list[int] = []
    for stage_value in stages.values():
        if not isinstance(stage_value, Mapping):
            continue
        for key in ("actionable_pending_count", "pending_count", "failed_unit_count"):
            try:
                value = int(stage_value.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                pending_candidates.append(value)
    payload = {
        "status": DART_PIPELINE_PARTIAL,
        "completed": False,
        "downstream_ready": False,
        "paper_results_available": bool(stages),
        "synthetic_fallback_used": False,
        "failed_stage": stage,
        "reason": reason or (str(error) if error is not None else "stage incomplete"),
        "error_type": type(error).__name__ if error is not None else None,
        "traceback": traceback.format_exc() if error is not None else None,
        "run_ids": dict(run_ids),
        "stages": dict(stages),
        # Preserve a resumable state when a lower-level training/replay stage
        # has an explicit work queue.  Integrity-gate failures with no queue
        # remain terminal PARTIAL evidence instead of spinning indefinitely.
        "actionable_pending_count": max(pending_candidates, default=0),
    }
    manifest = _write_pipeline_manifest(root, manifest_path, payload)
    return {
        **manifest,
        "manifest_path": _relative(root, manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
    }


def run_dart_value_select_pipeline(
    repository_root: Path,
    *,
    dart_config_path: Path,
    forecast_run_id: str,
    replay_run_id: str,
    value_select_run_id: str,
    physical_result: Mapping[str, Any],
    forecast_result: Mapping[str, Any],
    v1_readiness_result: Mapping[str, Any],
    resume: bool = True,
    replay_builder: ReplayBuilder | None = None,
) -> dict[str, Any]:
    """Run/resume the complete selection-before-lock DART composition."""

    root = Path(repository_root).resolve()
    config_path = Path(dart_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    artifacts = root / "artifacts"
    selector_root = artifacts / "dart_value_select" / f"run_id={value_select_run_id}"
    replay_root = selector_root / "replay" / f"run_id={replay_run_id}"
    manifest_path = selector_root / "manifests" / "dart_pipeline_manifest.json"
    run_ids = {
        "dart_forecast": str(forecast_run_id),
        "dart_replay": str(replay_run_id),
        "dart_value_select": str(value_select_run_id),
    }
    stages: dict[str, Any] = {}
    builder = replay_builder or _default_replay_builder
    try:
        dart_config = load_yaml(config_path)
        v1_manifest_value = forecast_result.get("run_manifest")
        if v1_manifest_value in (None, ""):
            raise FileNotFoundError("immutable v1 run manifest path is absent")
        v1_manifest_path = _trusted_repository_path(root, v1_manifest_value)
        readiness_audit = audit_v1_dart_readiness_evidence(
            root,
            v1_readiness_result,
            expected_v1_manifest=v1_manifest_path,
        )
        stages["v1_dart_readiness"] = readiness_audit
        if not readiness_audit["passed"]:
            return _early_result(
                root,
                manifest_path,
                run_ids=run_ids,
                stage="v1_dart_readiness",
                stages=stages,
                reason="; ".join(readiness_audit["failures"]),
            )
        if readiness_audit.get("frozen_dart_config_sha256") != sha256_file(
            config_path
        ):
            return _early_result(
                root,
                manifest_path,
                run_ids=run_ids,
                stage="v1_dart_readiness",
                stages=stages,
                reason="formal DART config differs from the frozen readiness policy",
            )
        training_kwargs = {
            "phase": "selection",
            "resume": resume,
            "input_fingerprints": {
                "immutable_v1_manifest": sha256_file(v1_manifest_path),
                "v1_readiness_manifest": str(
                    readiness_audit["manifest_file_sha256"]
                ),
                "v1_testing_selection_policy": str(
                    readiness_audit["frozen_policy_sha256"]
                ),
            },
        }
        selection_training = run_partial_candidate_selection_stage(
            root,
            artifacts,
            config_path,
            forecast_run_id,
            stage_runner=run_dart_forecast_training_stage,
            **training_kwargs,
        )
        stages["selection_training"] = selection_training
        try:
            candidate_pool_contract = _selection_candidate_pool_contract(
                selection_training
            )
        except (TypeError, ValueError) as exc:
            return _early_result(
                root,
                manifest_path,
                run_ids=run_ids,
                stage="selection_training",
                stages=stages,
                reason=f"2022-2024 DART forecast training is inadmissible: {exc}",
            )
        experiment = DARTValueSelectExperiment(
            selector_root,
            config_path,
            candidate_pool_contract=candidate_pool_contract,
        )
        experiment.initialize()

        anchor_training_audit = audit_anchor_training_lineage_evidence(
            root,
            dart_config=dart_config,
            selection_training=selection_training,
            expected_forecast_run_id=str(forecast_run_id),
            readiness_anchor_evidence=readiness_audit.get(
                "anchor_import_evidence"
            ),
            required=isinstance(
                readiness_audit.get("anchor_import_evidence"), Mapping
            ),
        )
        stages["anchor_training_lineage"] = anchor_training_audit
        if not anchor_training_audit["passed"]:
            return _early_result(
                root,
                manifest_path,
                run_ids=run_ids,
                stage="anchor_training_lineage",
                stages=stages,
                reason="; ".join(anchor_training_audit["failures"]),
            )

        training_root = Path(str(selection_training["stage_root"])).resolve()
        selection_raw = builder(
            repository_root=root,
            dart_forecast_root=training_root,
            physical_result=physical_result,
            forecast_result=forecast_result,
            config=dart_config,
            phase="selection",
            output_root=replay_root / "selection",
            resume=resume,
            **(
                {"candidate_pool_contract": candidate_pool_contract}
                if candidate_pool_contract is not None
                else {}
            ),
        )
        stages["selection_replay"] = _replay_stage_snapshot(selection_raw)
        selection_evidence, selection_replay = _normalize_replay_result(
            selection_raw,
            phase="selection",
            candidate_pool_contract=candidate_pool_contract,
        )
        stages["selection_replay"] = selection_replay
        replay_passed, replay_failures = _real_gate_audit(
            selection_replay,
            phase="selection",
            root=root,
            candidate_pool_contract=candidate_pool_contract,
        )
        if not replay_passed:
            raise RuntimeError("; ".join(replay_failures))
        if tuple(selection_evidence.operating_dates) != SELECTION_OPERATING_DATES:
            raise ValueError("selection evidence must contain the exact 1096-day panel")
        stages["selection_evidence"] = experiment.ingest_selection_evidence(
            selection_evidence
        )
        stages["selection_analysis"] = experiment.analyze_selection()

        dependencies = _freeze_dependencies(
            root,
            dart_config,
            selection_training,
            selection_evidence,
            physical_result,
            forecast_result,
            candidate_pool_contract,
        )
        stages["selection_freeze"] = experiment.freeze_selection(dependencies)
        freeze_runner = (
            run_partial_candidate_freeze_stage
            if candidate_pool_contract is not None
            else run_dart_forecast_training_stage
        )
        forecast_freeze = freeze_runner(
            root, artifacts, config_path, forecast_run_id,
            phase="freeze", resume=resume,
            dependency_fingerprints=dependencies,
        )
        stages["forecast_freeze"] = forecast_freeze
        if str(forecast_freeze.get("status", "")).upper() != "COMPLETE":
            return _early_result(
                root,
                manifest_path,
                run_ids=run_ids,
                stage="forecast_freeze",
                stages=stages,
                reason="pre-2025 DART forecast freeze is incomplete",
            )

        stages["locked_test_open"] = experiment.begin_locked_test(
            LOCKED_OPERATING_DATES,
            dependency_fingerprints=dependencies,
            resume=resume,
        )
        locked_runner = (
            run_partial_candidate_locked_stage
            if candidate_pool_contract is not None
            else run_dart_forecast_training_stage
        )
        locked_training = locked_runner(
            root, artifacts, config_path, forecast_run_id,
            phase="locked_test", resume=resume,
            dependency_fingerprints=dependencies,
        )
        stages["locked_training"] = locked_training
        if str(locked_training.get("status", "")).upper() != "COMPLETE":
            return _early_result(
                root,
                manifest_path,
                run_ids=run_ids,
                stage="locked_training",
                stages=stages,
                reason="once-only 2025 DART forecast training is incomplete",
            )

        locked_raw = builder(
            repository_root=root,
            dart_forecast_root=Path(str(locked_training["stage_root"])).resolve(),
            physical_result=physical_result,
            forecast_result=forecast_result,
            config=dart_config,
            phase="locked_test",
            output_root=replay_root / "locked_test",
            resume=resume,
            **(
                {"candidate_pool_contract": candidate_pool_contract}
                if candidate_pool_contract is not None
                else {}
            ),
        )
        stages["locked_replay"] = _replay_stage_snapshot(locked_raw)
        locked_evidence, locked_replay = _normalize_replay_result(
            locked_raw,
            phase="locked_test",
            candidate_pool_contract=candidate_pool_contract,
        )
        stages["locked_replay"] = locked_replay
        replay_passed, replay_failures = _real_gate_audit(
            locked_replay,
            phase="locked_test",
            root=root,
            frozen_dependencies=dependencies,
            candidate_pool_contract=candidate_pool_contract,
        )
        if not replay_passed:
            raise RuntimeError("; ".join(replay_failures))
        if tuple(locked_evidence.operating_dates) != LOCKED_OPERATING_DATES:
            raise ValueError("locked evidence must contain the exact 365-day panel")
        if dict(locked_evidence.frozen_dependency_fingerprints) != dependencies:
            raise ValueError(
                "locked DART evidence does not carry the frozen dependencies"
            )
        final = experiment.complete_locked_test(locked_evidence)
        stages["finalization"] = final

        selection_checkpoint = experiment.selection_evidence_path
        locked_checkpoint = experiment.locked_evidence_path
        replay_manifest = selector_root / "manifests" / "dart_replay_manifest.json"
        statistics_manifest = (
            selector_root / "manifests" / "dart_statistical_selection_manifest.json"
        )
        readiness_manifest_path = _resolve(root, readiness_audit["manifest_path"])
        if readiness_manifest_path is None:
            raise FileNotFoundError("audited v1 readiness manifest path is absent")
        selection_real_manifest = _resolve(root, selection_replay.get("manifest_path"))
        locked_real_manifest = _resolve(root, locked_replay.get("manifest_path"))
        if selection_real_manifest is None or locked_real_manifest is None:
            raise FileNotFoundError("real DART replay manifests are absent")
        selection_real_payload = _load_json(selection_real_manifest) or {}
        locked_real_payload = _load_json(locked_real_manifest) or {}
        replay_diagnostics = {
            "selection_metric_aggregates": _resolve(
                root, selection_real_payload.get("metric_aggregates_path")
            ),
            "selection_diagnostic_availability": _resolve(
                root, selection_real_payload.get("diagnostic_availability_path")
            ),
            "locked_metric_aggregates": _resolve(
                root, locked_real_payload.get("metric_aggregates_path")
            ),
            "locked_diagnostic_availability": _resolve(
                root, locked_real_payload.get("diagnostic_availability_path")
            ),
        }
        if any(path is None for path in replay_diagnostics.values()):
            raise FileNotFoundError("DART replay aggregate/diagnostic artifacts are absent")
        outputs = {
            "v1_readiness_manifest": readiness_manifest_path,
            "selection_training_manifest": training_root
            / "selection_stage_manifest.json",
            "forecast_freeze_manifest": training_root / "freeze_stage_manifest.json",
            "locked_training_manifest": training_root
            / "locked_test_stage_manifest.json",
            "selection_real_evidence_manifest": selection_real_manifest,
            "locked_real_evidence_manifest": locked_real_manifest,
            "selection_evidence": selection_checkpoint,
            "selection_analysis": experiment.selection_analysis_path,
            "selection_freeze": experiment.selection_freeze_path,
            "locked_test_ledger": experiment.locked_ledger_path,
            "locked_test_evidence": locked_checkpoint,
            "dart_value_select_manifest": experiment.final_manifest_path,
            "dart_replay_manifest": replay_manifest,
            "dart_statistical_selection_manifest": statistics_manifest,
            **replay_diagnostics,
        }
        outputs.update(anchor_training_audit.get("paths", {}))
        missing_outputs = [str(path) for path in outputs.values() if not path.is_file()]
        if missing_outputs:
            raise FileNotFoundError(f"DART final outputs are absent: {missing_outputs}")
        manifest = _write_pipeline_manifest(
            root,
            manifest_path,
            {
                "status": (
                    PARTIAL_CANDIDATE_STATUS
                    if candidate_pool_contract is not None
                    else DART_PIPELINE_COMPLETE
                ),
                "completed": True,
                "downstream_ready": True,
                "paper_results_available": True,
                "synthetic_fallback_used": False,
                "counterfactual": True,
                "realized_commercial_profit_claim": False,
                "run_ids": run_ids,
                "config_path": _relative(root, config_path),
                "config_sha256": sha256_file(config_path),
                "selection_operating_date_count": len(SELECTION_OPERATING_DATES),
                "locked_operating_date_count": len(LOCKED_OPERATING_DATES),
                "candidate_count": len(
                    _effective_candidate_names(candidate_pool_contract)
                ),
                "attempted_candidate_count": len(CANDIDATE_POOL),
                "candidate_pool_contract": candidate_pool_contract,
                "terminal_status_cap": (
                    PARTIAL_CANDIDATE_TERMINAL_CAP
                    if candidate_pool_contract is not None
                    else None
                ),
                "experiment_count": 2,
                "seed_count": len(DART_VALUE_SELECT_SEEDS),
                "freeze_dependency_fingerprints": dependencies,
                "v1_readiness_policy_sha256": readiness_audit["frozen_policy_sha256"],
                "anchor_training_lineage_required": anchor_training_audit.get(
                    "required", False
                ),
                "stages": stages,
                "outputs": {
                    key: _manifest_record(root, path) for key, path in outputs.items()
                },
                "actionable_pending_count": 0,
            },
        )
        result = {
            **manifest,
            "manifest_path": _relative(root, manifest_path),
            "manifest_file_sha256": sha256_file(manifest_path),
            "output_root": _relative(root, selector_root),
        }
        audit = audit_dart_pipeline_evidence(root, result)
        if not audit["passed"]:
            return _early_result(
                root,
                manifest_path,
                run_ids=run_ids,
                stage="completion_audit",
                stages=stages,
                reason="; ".join(audit["failures"]),
            )
        return result
    except Exception as exc:
        return _early_result(
            root,
            manifest_path,
            run_ids=run_ids,
            stage="exception",
            stages=stages,
            error=exc,
        )


def attach_dart_overlay_to_forecast(
    repository_root: Path,
    forecast_result: Mapping[str, Any],
    dart_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach only audited frozen DART evidence to the optimization input."""

    root = Path(repository_root).resolve()
    audit = audit_dart_pipeline_evidence(root, dart_result)
    if not audit["passed"]:
        raise ValueError(
            "DART overlay is not admissible: " + "; ".join(audit["failures"])
        )
    pipeline_path = _resolve(root, dart_result.get("manifest_path"))
    if pipeline_path is None or not pipeline_path.is_file():
        raise FileNotFoundError("audited DART pipeline manifest is absent")
    pipeline = _load_json(pipeline_path) or {}
    pipeline_status = str(pipeline.get("status", "")).upper()
    candidate_pool_contract = audit.get("candidate_pool_contract")
    final_path = _resolve(
        root,
        pipeline.get("outputs", {}).get("dart_value_select_manifest"),
    )
    final = _load_json(final_path) or {}
    output_root = _resolve(root, dart_result.get("output_root"))
    if output_root is None:
        output_root = pipeline_path.parents[1]
    overlay_path = output_root / "manifests" / "dart_overlay_bridge_manifest.json"
    base_manifest_sha = _path_sha(root, forecast_result.get("run_manifest"))
    overlay = {
        "schema_version": "dart_forecast_overlay_v1",
        "status": "COMPLETE",
        "optimization_allowed": True,
        "base_forecast_manifest_sha256": base_manifest_sha,
        "dart_pipeline_manifest": _manifest_record(root, pipeline_path),
        "dart_value_select_manifest": (
            _manifest_record(root, final_path)
            if final_path is not None and final_path.is_file()
            else None
        ),
        "frozen_selectors": final.get("frozen_selectors"),
        "selection_updated_from_2025": final.get("chronology", {}).get(
            "selection_updated_from_2025"
        ),
        "locked_test_run_count": final.get("chronology", {}).get(
            "locked_test_run_count"
        ),
        "counterfactual": True,
        "source_dart_status": pipeline_status,
        "candidate_pool_contract": candidate_pool_contract,
        "terminal_status_cap": audit.get("terminal_status_cap"),
        "created_at_utc": utc_now(),
    }
    atomic_json(overlay_path, overlay)
    enriched = deepcopy(dict(forecast_result))
    enriched["dart_value_select"] = {
        "status": pipeline_status,
        "downstream_ready": True,
        "pipeline_manifest": _relative(root, pipeline_path),
        "pipeline_manifest_sha256": sha256_file(pipeline_path),
        "overlay_manifest": _relative(root, overlay_path),
        "overlay_manifest_sha256": sha256_file(overlay_path),
        "frozen_selectors": final.get("frozen_selectors"),
        "counterfactual": True,
        "candidate_pool_contract": candidate_pool_contract,
        "terminal_status_cap": audit.get("terminal_status_cap"),
    }
    enriched["dart_overlay_manifest"] = _relative(root, overlay_path)
    enriched["dart_overlay_manifest_sha256"] = sha256_file(overlay_path)
    return enriched


def _sidecar_sign_audit(
    root: Path,
    stage_manifest: Mapping[str, Any],
) -> tuple[bool, list[str], int]:
    failures: list[str] = []
    checked = 0
    phase_evidence = stage_manifest.get("phase_evidence")
    if isinstance(phase_evidence, Mapping) and isinstance(
        phase_evidence.get("evidence"), Mapping
    ):
        # The locked-test adapter returns its once-only ledger, with collected
        # forecast evidence nested under ``evidence``.  Selection returns the
        # collected evidence directly.
        phase_evidence = phase_evidence["evidence"]
    dart_records = (
        phase_evidence.get("dart_artifacts", [])
        if isinstance(phase_evidence, Mapping)
        else []
    )
    if not dart_records:
        failures.append("DART forecast stage has no serialized DART artifacts")
    for record in dart_records if isinstance(dart_records, Sequence) else []:
        if not isinstance(record, Mapping):
            failures.append("DART artifact record is malformed")
            continue
        path = _resolve(root, record.get("path"))
        if path is None or not path.is_file():
            failures.append(f"DART artifact is absent: {record.get('path')}")
            continue
        if record.get("sha256") != sha256_file(path):
            failures.append(f"DART artifact hash changed: {path}")
            continue
        sidecar = path.with_suffix(".json")
        manifest = _load_json(sidecar)
        if manifest is None:
            failures.append(f"DART sign sidecar is absent: {sidecar}")
            continue
        checked += 1
        if manifest.get("sign_gate_after_generation") != "PASSED":
            failures.append(f"generation sign gate failed: {sidecar}")
        if manifest.get("sign_gate_after_serialization") != "PASSED":
            failures.append(f"serialization sign gate failed: {sidecar}")
        if manifest.get("ambiguous_rt_spread_present") is not False:
            failures.append(f"ambiguous rt_spread alias survives: {sidecar}")
        contract = manifest.get("contract", {})
        if not isinstance(contract, Mapping):
            failures.append(f"DART sidecar contract is malformed: {sidecar}")
        else:
            if contract.get("formula") != "draw-wise da_lbmp - rt_lbmp":
                failures.append(f"canonical DART formula changed: {sidecar}")
            if contract.get("inverse_formula") != "rt_lbmp - da_lbmp":
                failures.append(f"inverse DART formula changed: {sidecar}")
            if contract.get("locked_targets_used_for_coupling") is not False:
                failures.append(f"locked target entered coupling: {sidecar}")
    return not failures, failures, checked


def _real_gate_audit(
    stage: Any,
    *,
    phase: str,
    root: Path | None = None,
    frozen_dependencies: Mapping[str, str] | None = None,
    candidate_pool_contract: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    if not isinstance(stage, Mapping):
        return False, [f"{phase} real replay stage metadata is absent"]
    failures: list[str] = []
    manifest: Mapping[str, Any] = {}
    if root is not None:
        manifest_path = _resolve(root, stage.get("manifest_path"))
        loaded = _load_json(manifest_path)
        if loaded is None:
            failures.append(f"{phase} real replay manifest is absent or invalid")
        else:
            manifest = loaded
            expected_sha = stage.get("manifest_sha256")
            if not expected_sha or expected_sha != sha256_file(manifest_path):
                failures.append(f"{phase} real replay manifest hash changed")
            evidence_path = _internal_artifact_path(root, manifest.get("evidence_path"))
            if evidence_path is None or not evidence_path.is_file():
                failures.append(f"{phase} real phase-evidence file is absent")
            elif manifest.get("evidence_file_sha256") != sha256_file(evidence_path):
                failures.append(f"{phase} real phase-evidence hash changed")
            inventory_path = _internal_artifact_path(
                root, manifest.get("source_inventory_path")
            )
            inventory = _load_json(inventory_path)
            if inventory is None or str(inventory.get("status", "")).upper() != "COMPLETE":
                failures.append(f"{phase} OOS source inventory is absent or incomplete")
            elif inventory.get("inventory_sha256") != manifest.get(
                "source_inventory_sha256"
            ):
                failures.append(f"{phase} OOS source inventory fingerprint changed")
            if frozen_dependencies is not None:
                observed_dependencies = manifest.get(
                    "frozen_dependency_fingerprints"
                )
                if not isinstance(observed_dependencies, Mapping) or dict(
                    observed_dependencies
                ) != dict(frozen_dependencies):
                    failures.append(
                        f"{phase} real replay dependencies differ from the pre-2025 freeze"
                    )
            aggregate_path = _internal_artifact_path(
                root, manifest.get("metric_aggregates_path")
            )
            aggregate = _load_json(aggregate_path)
            if aggregate is None:
                failures.append(f"{phase} ratio-of-sums aggregate is absent")
            else:
                aggregate_core = {
                    key: value
                    for key, value in aggregate.items()
                    if key != "aggregation_sha256"
                }
                expected_candidates = _effective_candidate_names(
                    candidate_pool_contract
                )
                if (
                    manifest.get("metric_aggregates_file_sha256")
                    != sha256_file(aggregate_path)
                    or manifest.get("metric_aggregates_sha256")
                    != aggregate.get("aggregation_sha256")
                    or aggregate.get("aggregation_sha256")
                    != sha256_json(aggregate_core)
                    or aggregate.get("mean_of_ratios_used") is not False
                    or aggregate.get("candidate_count") != len(expected_candidates)
                ):
                    failures.append(
                        f"{phase} ratio-of-sums aggregate fingerprint changed"
                    )
            diagnostics_path = _internal_artifact_path(
                root, manifest.get("diagnostic_availability_path")
            )
            diagnostics = _load_json(diagnostics_path)
            if diagnostics is None:
                failures.append(f"{phase} diagnostic-availability artifact is absent")
            else:
                unavailable = diagnostics.get("unavailable_records")
                if (
                    manifest.get("diagnostic_availability_sha256")
                    != sha256_file(diagnostics_path)
                    or str(diagnostics.get("status", "")).upper()
                    not in {"COMPLETE", "COMPLETE_WITH_EXPLICIT_NULLS"}
                    or diagnostics.get("invented_or_zero_imputed_metrics") is not False
                    or not isinstance(unavailable, list)
                    or diagnostics.get("unavailable_record_count") != len(unavailable)
                    or manifest.get("unsupported_or_undefined_metric_record_count")
                    != len(unavailable)
                ):
                    failures.append(
                        f"{phase} diagnostic-availability fingerprint changed"
                    )
    effective = manifest or stage
    if str(effective.get("status", "")).upper() != "COMPLETE":
        failures.append(f"{phase} real replay status is not COMPLETE")
    if effective.get("phase", phase) != phase:
        failures.append(f"{phase} real replay phase label changed")
    if effective.get("synthetic_fallback_used") is not False:
        failures.append(f"{phase} real replay used or failed to exclude synthetic fallback")
    if effective.get("v1_2025_metrics_consumed") is not False:
        failures.append(f"{phase} real replay consumed or failed to exclude v1 2025 metrics")
    if int(effective.get("actionable_pending_count", -1) or 0) != 0:
        failures.append(f"{phase} real replay still has actionable pending work")
    expected_pool_sha = (
        validate_partial_candidate_pool_contract(candidate_pool_contract).get(
            "contract_sha256"
        )
        if candidate_pool_contract is not None
        else None
    )
    observed_pool = effective.get("candidate_pool_contract")
    if expected_pool_sha is None:
        if observed_pool is not None:
            failures.append(f"{phase} full replay carries a partial candidate pool")
    else:
        try:
            observed_pool_sha = validate_partial_candidate_pool_contract(
                observed_pool
            ).get("contract_sha256")
        except (TypeError, ValueError):
            observed_pool_sha = None
        if observed_pool_sha != expected_pool_sha:
            failures.append(f"{phase} replay candidate pool differs from selection")
    gates = effective.get("completion_gates", stage.get("completion_gates"))
    if not isinstance(gates, Mapping):
        failures.append(f"{phase} real replay completion gates are absent")
    else:
        for key in _required_real_evidence_gate_keys():
            if gates.get(key) is not True:
                failures.append(f"{phase} real replay gate is not true: {key}")
        if (
            gates.get("v1_2025_metrics_consumed") is not False
            or gates.get("selected_scenario_used_as_realized") is not False
        ):
            failures.append(f"{phase} real replay claim/leakage boundary changed")
        expected_days = 1096 if phase == "selection" else 365
        expected_cells = (
            expected_days
            * len(_effective_candidate_names(candidate_pool_contract))
            * 2
            * len(DART_VALUE_SELECT_SEEDS)
        )
        if int(gates.get("operating_date_count", -1)) != expected_days:
            failures.append(f"{phase} real replay operating-date gate changed")
        if int(gates.get("panel_cell_count", -1)) != expected_cells:
            failures.append(f"{phase} real replay panel-cell gate changed")
    return not failures, failures


def _selection_result_audit(value: Any, *, experiment_id: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{experiment_id} statistical selection result is absent"]
    failures: list[str] = []
    required = {
        "status",
        "selected_model",
        "winner_set",
        "feasible_models",
        "excluded_by_feasibility",
        "mcs_set",
        "lcb_by_model",
        "robustness_lcb_by_block",
        "tie_break_trace",
        "mcs",
    }
    missing = required.difference(value)
    if missing:
        failures.append(f"{experiment_id} selection fields missing: {sorted(missing)}")
        return failures
    status = str(value.get("status", ""))
    winners = value.get("winner_set")
    if not isinstance(winners, list):
        failures.append(f"{experiment_id} winner_set is not explicit")
        winners = []
    feasible = value.get("feasible_models")
    if not isinstance(feasible, list):
        failures.append(f"{experiment_id} feasibility result is not explicit")
        feasible = []
    if status in {"SELECTED", "STATISTICALLY_INDISTINGUISHABLE_WINNER_SET"}:
        robustness = value.get("robustness_lcb_by_block")
        if not isinstance(robustness, Mapping) or set(map(str, robustness)) != {
            "7",
            "14",
            "28",
        }:
            failures.append(f"{experiment_id} lacks exact 7/14/28-day LCBs")
        if not isinstance(value.get("mcs"), Mapping):
            failures.append(f"{experiment_id} MCS result is absent")
        if not value.get("mcs_set"):
            failures.append(f"{experiment_id} MCS retained set is empty")
        if not value.get("lcb_by_model"):
            failures.append(f"{experiment_id} primary 7-day LCBs are absent")
        if not winners:
            failures.append(f"{experiment_id} winner set is empty")
    elif status in {"NO_FEASIBLE_MODEL", "NO_FEASIBLE_MCS_MEMBER"}:
        if status == "NO_FEASIBLE_MODEL" and feasible:
            failures.append(
                f"{experiment_id} no-feasible status contradicts feasible_models"
            )
        if winners:
            failures.append(f"{experiment_id} no-winner status has a winner set")
        if status == "NO_FEASIBLE_MCS_MEMBER":
            robustness = value.get("robustness_lcb_by_block")
            if not isinstance(robustness, Mapping) or set(map(str, robustness)) != {
                "7",
                "14",
                "28",
            }:
                failures.append(f"{experiment_id} lacks exact 7/14/28-day LCBs")
            if not isinstance(value.get("mcs"), Mapping):
                failures.append(f"{experiment_id} MCS result is absent")
    else:
        failures.append(f"{experiment_id} selection status is not terminal: {status}")
    if isinstance(winners, list):
        selected = value.get("selected_model")
        if len(winners) == 1 and selected != winners[0]:
            failures.append(
                f"{experiment_id} singleton winner set and selected_model disagree"
            )
        if len(winners) > 1 and selected is not None:
            failures.append(
                f"{experiment_id} statistical winner set was forcibly collapsed"
            )
    if experiment_id == "A2":
        gate = value.get("preregistered_gate")
        if (
            not isinstance(gate, Mapping)
            or gate.get("gate_id") != "A1_DART_NONINFERIORITY"
            or float(
                gate.get("epsilon_fraction_of_mean_baseline_daily_cost", -1.0)
            )
            != 0.005
            or gate.get("source_experiment_id") != "A1"
            or gate.get("target_experiment_id") != "A2"
        ):
            failures.append("A2 lacks the frozen 0.5% A1 noninferiority gate")
        if float(value.get("economic_tolerance_fraction", -1.0)) != 0.005:
            failures.append("A2 lacks the frozen 0.5% economic winner tolerance")
    return failures


def audit_dart_pipeline_evidence(
    repository_root: Path,
    dart_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Independently hard-gate the DART branch before COMPLETE_PAPER."""

    root = Path(repository_root).resolve()
    result = dict(dart_result or {})
    failures: list[str] = []
    manifest_path = _resolve(root, result.get("manifest_path"))
    manifest = _load_json(manifest_path)
    if manifest is None:
        failures.append("DART pipeline manifest is absent or invalid")
        manifest = {}
    else:
        expected = result.get("manifest_file_sha256")
        if expected and expected != sha256_file(manifest_path):
            failures.append("DART pipeline manifest hash changed")
        core = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if manifest.get("manifest_sha256") != sha256_json(core):
            failures.append("DART pipeline internal manifest hash changed")
    supplied_status = str(result.get("status", "")).upper()
    manifest_status = str(manifest.get("status", "")).upper()
    if supplied_status and supplied_status != manifest_status:
        failures.append("DART result status differs from its pipeline manifest")
    supplied_pool = result.get("candidate_pool_contract")
    if supplied_pool != manifest.get("candidate_pool_contract"):
        failures.append("DART result candidate pool differs from its pipeline manifest")
    if result.get("terminal_status_cap") != manifest.get("terminal_status_cap"):
        failures.append("DART result terminal cap differs from its pipeline manifest")
    candidate_pool_contract: dict[str, Any] | None = None
    if manifest_status == PARTIAL_CANDIDATE_STATUS:
        try:
            candidate_pool_contract = validate_partial_candidate_pool_contract(
                manifest.get("candidate_pool_contract")
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"DART partial candidate contract is invalid: {exc}")
        if manifest.get("terminal_status_cap") != PARTIAL_CANDIDATE_TERMINAL_CAP:
            failures.append("DART partial candidate run lacks the PARTIAL_PAPER cap")
    elif manifest_status == DART_PIPELINE_COMPLETE:
        if manifest.get("candidate_pool_contract") is not None:
            failures.append("COMPLETE DART pipeline carries a partial candidate contract")
        if manifest.get("terminal_status_cap") not in (None, ""):
            failures.append("COMPLETE DART pipeline carries an unexpected terminal cap")
    else:
        failures.append("DART pipeline status is neither COMPLETE nor the sole admitted partial case")
    expected_candidate_count = (
        len(_effective_candidate_names(candidate_pool_contract))
        if manifest_status in {DART_PIPELINE_COMPLETE, PARTIAL_CANDIDATE_STATUS}
        else len(CANDIDATE_POOL)
    )
    if int(manifest.get("attempted_candidate_count", len(CANDIDATE_POOL))) != len(
        CANDIDATE_POOL
    ):
        failures.append("DART attempted candidate count is not the frozen nine")
    if int(manifest.get("candidate_count", expected_candidate_count)) != expected_candidate_count:
        failures.append("DART eligible candidate count differs from its pool contract")
    if (
        manifest.get("completed") is not True
        or manifest.get("downstream_ready") is not True
    ):
        failures.append("DART pipeline is not downstream-ready")
    if manifest.get("synthetic_fallback_used") is not False:
        failures.append("DART pipeline used or failed to exclude synthetic fallback")

    output_records = manifest.get("outputs")
    outputs: dict[str, Path] = {}
    if not isinstance(output_records, Mapping):
        failures.append("DART pipeline output ledger is absent")
        output_records = {}
    missing_output_keys = _REQUIRED_PIPELINE_OUTPUT_KEYS.difference(output_records)
    if missing_output_keys:
        failures.append(
            f"DART pipeline output ledger is incomplete: {sorted(missing_output_keys)}"
        )
    for key, record in output_records.items():
        if not isinstance(record, Mapping):
            failures.append(f"DART output record is malformed: {key}")
            continue
        path = _resolve(root, record)
        if path is None or not path.is_file():
            failures.append(f"DART output is absent: {key}")
            continue
        if record.get("sha256") != sha256_file(path):
            failures.append(f"DART output hash changed: {key}")
            continue
        outputs[str(key)] = path

    readiness_path = outputs.get("v1_readiness_manifest")
    readiness_audit = audit_v1_dart_readiness_evidence(
        root,
        {
            "manifest_path": (
                _relative(root, readiness_path)
                if readiness_path is not None
                else None
            ),
            "manifest_file_sha256": (
                sha256_file(readiness_path)
                if readiness_path is not None and readiness_path.is_file()
                else None
            ),
        },
    )
    failures.extend(readiness_audit["failures"])
    if manifest.get("v1_readiness_policy_sha256") != readiness_audit.get(
        "frozen_policy_sha256"
    ):
        failures.append("DART pipeline does not pin the audited v1 policy hash")

    selection_training = _load_json(outputs.get("selection_training_manifest")) or {}
    forecast_freeze = _load_json(outputs.get("forecast_freeze_manifest")) or {}
    locked_training = _load_json(outputs.get("locked_training_manifest")) or {}
    training_inputs = selection_training.get("input_fingerprints", {})
    if not isinstance(training_inputs, Mapping) or (
        training_inputs.get("v1_testing_selection_policy")
        != readiness_audit.get("frozen_policy_sha256")
        or training_inputs.get("v1_readiness_manifest")
        != readiness_audit.get("manifest_file_sha256")
    ):
        failures.append(
            "formal DART training did not pin the audited v1 readiness contract"
        )
    try:
        selection_pool = _selection_candidate_pool_contract(selection_training)
    except (TypeError, ValueError) as exc:
        selection_pool = None
        failures.append(f"selection forecast admission failed: {exc}")
    expected_pool_sha = (
        candidate_pool_contract.get("contract_sha256")
        if candidate_pool_contract is not None
        else None
    )
    observed_selection_pool_sha = (
        selection_pool.get("contract_sha256")
        if selection_pool is not None
        else None
    )
    if observed_selection_pool_sha != expected_pool_sha:
        failures.append("pipeline and selection candidate-pool contracts differ")
    if candidate_pool_contract is not None:
        eligible_phase = selection_training.get("phase_evidence", {})
        coverage_contract = (
            eligible_phase.get("coverage_contract", {})
            if isinstance(eligible_phase, Mapping)
            else {}
        )
        prediction_artifacts = (
            eligible_phase.get("prediction_artifacts", ())
            if isinstance(eligible_phase, Mapping)
            else ()
        )
        if (
            eligible_phase.get("evidence_sha256")
            != sha256_json(
                {
                    key: value
                    for key, value in eligible_phase.items()
                    if key != "evidence_sha256"
                }
            )
            or int(eligible_phase.get("eligible_unit_count", -1)) != 108
            or not isinstance(prediction_artifacts, Sequence)
            or len(prediction_artifacts) != 108
            or coverage_contract.get("date_drop_count") != 0
            or coverage_contract.get("common_date_panel_preserved") is not True
            or coverage_contract.get("required_selection_years")
            != [2022, 2023, 2024]
            or coverage_contract.get("eligible_candidates")
            != list(_effective_candidate_names(candidate_pool_contract))
        ):
            failures.append(
                "partial selection does not prove the complete 108-unit eligible OOS panel"
            )
    for label, training in (("freeze", forecast_freeze), ("locked_test", locked_training)):
        if str(training.get("status", "")).upper() != DART_PIPELINE_COMPLETE:
            failures.append(f"{label} forecast training is not COMPLETE")
        if training.get("downstream_ready") is not True:
            failures.append(f"{label} forecast training is not downstream-ready")
        if int(training.get("failed_unit_count", -1) or 0) != 0:
            failures.append(f"{label} forecast training has failed units")
        if candidate_pool_contract is not None:
            try:
                stage_pool_sha = validate_partial_candidate_pool_contract(
                    training.get("candidate_pool_contract")
                ).get("contract_sha256")
            except (TypeError, ValueError):
                stage_pool_sha = None
            if (
                training.get("partial_candidate_downstream_ready") is not True
                or training.get("terminal_status_cap")
                != PARTIAL_CANDIDATE_TERMINAL_CAP
                or stage_pool_sha != expected_pool_sha
            ):
                failures.append(f"{label} does not preserve the admitted partial pool")
        elif training.get("candidate_pool_contract") is not None:
            failures.append(f"{label} full-pool stage carries a partial contract")
    frozen_dependencies = manifest.get("freeze_dependency_fingerprints")
    if not isinstance(frozen_dependencies, Mapping) or (
        set(frozen_dependencies) != set(FREEZE_FINGERPRINT_KEYS)
        or any(len(str(value)) != 64 for value in frozen_dependencies.values())
    ):
        failures.append("DART seven-part freeze dependency ledger is invalid")
        frozen_dependencies = {}
    freeze_phase = forecast_freeze.get("phase_evidence", {})
    freeze_contract = (
        freeze_phase.get("frozen_contract", {})
        if isinstance(freeze_phase, Mapping)
        else {}
    )
    locked_phase = locked_training.get("phase_evidence", {})
    locked_contract = (
        locked_phase.get("locked_contract", {})
        if isinstance(locked_phase, Mapping)
        else {}
    )
    selector_freeze = _load_json(outputs.get("selection_freeze")) or {}
    selector_freeze_contract = selector_freeze.get("frozen_contract", {})
    for label, dependency_record in (
        (
            "forecast freeze",
            freeze_contract.get("dependency_fingerprints")
            if isinstance(freeze_contract, Mapping)
            else None,
        ),
        (
            "locked forecast",
            locked_contract.get("dependency_fingerprints")
            if isinstance(locked_contract, Mapping)
            else None,
        ),
        (
            "selector freeze",
            selector_freeze_contract.get("dependency_fingerprints")
            if isinstance(selector_freeze_contract, Mapping)
            else None,
        ),
    ):
        if dict(dependency_record or {}) != dict(frozen_dependencies):
            failures.append(f"{label} dependencies differ from the pre-2025 freeze")
    sign_counts: dict[str, int] = {}
    for label, training in (
        ("selection", selection_training),
        ("locked_test", locked_training),
    ):
        _passed, sign_failures, checked = _sidecar_sign_audit(root, training)
        failures.extend(f"{label}: {reason}" for reason in sign_failures)
        sign_counts[label] = checked

    config_path = _resolve(root, manifest.get("config_path"))
    actual_config_sha = (
        sha256_file(config_path)
        if config_path is not None and config_path.is_file()
        else None
    )
    if manifest.get("config_sha256") != actual_config_sha:
        failures.append("DART pipeline config bytes changed")
    if readiness_audit.get("frozen_dart_config_sha256") != actual_config_sha:
        failures.append("DART config differs from the frozen readiness policy")
    dart_config = (
        load_yaml(config_path)
        if config_path is not None and config_path.is_file()
        else {}
    )
    anchor_contract = (
        _anchor_only_contract(dart_config)
        if isinstance(dart_config, Mapping)
        else None
    )
    if anchor_contract is not None and isinstance(
        readiness_audit.get("anchor_import_evidence"), Mapping
    ):
        for key in (
            "v3_anchor_snapshot_manifest",
            "v4_anchor_import_ledger",
            "v4_epoch_archive_inventory",
        ):
            if key not in outputs:
                failures.append(f"DART v4 anchor/archive output is absent: {key}")
        run_ids = manifest.get("run_ids", {})
        expected_forecast_id = str(anchor_contract.get("destination_run_id", ""))
        if (
            not isinstance(run_ids, Mapping)
            or run_ids.get("dart_forecast") != expected_forecast_id
            or selection_training.get("run_id") != expected_forecast_id
        ):
            failures.append("DART pipeline/selection run IDs do not pin forecast-v4")
        anchor_audit = audit_anchor_training_lineage_evidence(
            root,
            dart_config=dart_config,
            selection_training=selection_training,
            expected_forecast_run_id=expected_forecast_id,
            readiness_anchor_evidence=readiness_audit.get(
                "anchor_import_evidence"
            ),
            required=True,
        )
        failures.extend(anchor_audit["failures"])
        for key, expected_path in anchor_audit.get("paths", {}).items():
            if outputs.get(key) != expected_path:
                failures.append(f"DART pipeline does not hash-pin audited {key}")
        if manifest.get("anchor_training_lineage_required") is not True:
            failures.append("DART pipeline does not declare the required v4 lineage gate")
    sign = (
        dart_config.get("sign_contract", {}) if isinstance(dart_config, Mapping) else {}
    )
    if not isinstance(sign, Mapping) or (
        sign.get("canonical_dart_field") != "dart_da_minus_rt"
        or sign.get("inverse_spread_field") != "rt_spread_rt_minus_da"
        or sign.get("bare_rt_spread_field_forbidden") is not True
    ):
        failures.append("canonical DART field/no-bare-alias config gate failed")
    issue = (
        dart_config.get("forecast_issue_contract", {})
        if isinstance(dart_config, Mapping)
        else {}
    )
    if not isinstance(issue, Mapping) or (
        issue.get("profile_id") != "COMMON_DA_GATE_DART_V1"
        or issue.get("both_targets_share_identical_information_cutoff") is not True
        or issue.get("later_intraday_rt_updates_forbidden_in_selection_comparison")
        is not True
    ):
        failures.append("common DA-gate forecast issue contract failed")
    aggregation = (
        dart_config.get("aggregation_contract", {})
        if isinstance(dart_config, Mapping)
        else {}
    )
    if not isinstance(aggregation, Mapping) or any(
        aggregation.get(key) is not True
        for key in (
            "exact_common_operating_date_panel_required",
            "silent_inner_join_forbidden",
            "candidate_date_seed_experiment_key_unique",
            "first_average_common_three_seed_values_within_date",
            "then_apply_time_block_inference_across_dates",
        )
    ):
        failures.append(
            "DART common-panel/within-date seed aggregation contract failed"
        )
    selection_freeze_contract = (
        dart_config.get("selection_freeze_contract", {})
        if isinstance(dart_config, Mapping)
        else {}
    )
    if not isinstance(selection_freeze_contract, Mapping) or any(
        selection_freeze_contract.get(key) is not True
        for key in (
            "freeze_before_reading_selection_outcomes",
            "feasibility_precedes_statistical_ranking",
            "mcs_precedes_lcb_selection",
            "daily_three_seed_average_precedes_bootstrap",
            "common_bootstrap_indices_across_candidates",
            "mcs_and_lcb_use_same_exact_daily_panel",
        )
    ):
        failures.append("DART frozen feasibility/MCS/LCB selection contract failed")
    exposure = (
        dart_config.get("locked_test_exposure_disclosure", {})
        if isinstance(dart_config, Mapping)
        else {}
    )
    if not isinstance(exposure, Mapping) or (
        exposure.get("formal_dart_code_may_not_read_v1_2025_metric_values") is not True
        or exposure.get(
            "formal_dart_selector_may_not_consume_v1_2025_predictions_or_metrics"
        )
        is not True
        or exposure.get("original_human_blindness_claimed") is not False
    ):
        failures.append("DART prior-v1-2025 exposure boundary is not frozen")

    selection_payload = _load_json(outputs.get("selection_evidence")) or {}
    locked_payload = _load_json(outputs.get("locked_test_evidence")) or {}
    expected_candidate_ids = sorted(
        _effective_candidate_names(candidate_pool_contract)
    )
    expected_candidate_count = len(expected_candidate_ids)
    panel_expectations = (
        (
            "selection",
            selection_payload,
            SELECTION_OPERATING_DATES,
            1096 * expected_candidate_count * 2 * len(DART_VALUE_SELECT_SEEDS),
        ),
        (
            "locked_test",
            locked_payload,
            LOCKED_OPERATING_DATES,
            365 * expected_candidate_count * 2 * len(DART_VALUE_SELECT_SEEDS),
        ),
    )
    panel_counts: dict[str, Any] = {}
    for phase, payload, expected_dates, expected_cells in panel_expectations:
        validation = payload.get("validation", {})
        if not isinstance(validation, Mapping):
            failures.append(f"{phase} selector validation is absent")
            continue
        dates = tuple(map(str, validation.get("operating_dates", [])))
        if dates != expected_dates:
            failures.append(f"{phase} operating dates are not the exact full panel")
        if int(validation.get("operating_date_count", -1)) != len(expected_dates):
            failures.append(
                f"{phase} operating-date count is not {len(expected_dates)}"
            )
        if int(validation.get("panel_cell_count", -1)) != expected_cells:
            failures.append(
                f"{phase} A1/A2 replay panel cell count is not {expected_cells}"
            )
        expected_metrics = (
            len(expected_dates) * expected_candidate_count * len(DART_VALUE_SELECT_SEEDS)
        )
        if int(validation.get("metric_record_count", -1)) != expected_metrics:
            failures.append(f"{phase} three-layer metric panel is incomplete")
        if (
            sorted(map(str, validation.get("candidate_ids", [])))
            != expected_candidate_ids
        ):
            failures.append(f"{phase} candidate pool changed")
        payload_pool = payload.get("candidate_pool_contract")
        if candidate_pool_contract is None:
            if payload_pool is not None:
                failures.append(f"{phase} selector evidence carries a partial pool")
        else:
            try:
                payload_pool_sha = validate_partial_candidate_pool_contract(
                    payload_pool
                ).get("contract_sha256")
            except (TypeError, ValueError):
                payload_pool_sha = None
            if payload_pool_sha != expected_pool_sha:
                failures.append(f"{phase} selector candidate pool is not hash-pinned")
        if tuple(validation.get("experiments", [])) != ("A1", "A2"):
            failures.append(f"{phase} A1/A2 experiment panel changed")
        if tuple(validation.get("seeds", [])) != DART_VALUE_SELECT_SEEDS:
            failures.append(f"{phase} exact three-seed panel changed")
        if phase == "locked_test":
            observed_dependencies = payload.get("frozen_dependency_fingerprints")
            if not isinstance(observed_dependencies, Mapping) or dict(
                observed_dependencies
            ) != dict(frozen_dependencies):
                failures.append(
                    "locked selector evidence dependencies differ from the pre-2025 freeze"
                )
        identity = validation.get("identity_audits", {})
        if not isinstance(identity, Mapping):
            failures.append(f"{phase} A1/A2 identity audit is absent")
        else:
            for experiment_id, allowed in (
                ("A1", A1_ALLOWED_TO_DIFFER),
                ("A2", A2_ALLOWED_TO_DIFFER),
            ):
                audit = identity.get(experiment_id, {})
                if not isinstance(audit, Mapping) or audit.get("passed") is not True:
                    failures.append(f"{phase} {experiment_id} identity audit failed")
                    continue
                if set(audit.get("allowed_candidate_varying_hashes", [])) != set(
                    allowed
                ):
                    failures.append(
                        f"{phase} {experiment_id} allowed-change set changed"
                    )
                observed = set(audit.get("observed_candidate_varying_hashes", []))
                if not observed.issubset(allowed):
                    failures.append(f"{phase} {experiment_id} has forbidden changes")
        panel_counts[phase] = {
            "operating_dates": len(dates),
            "panel_cells": validation.get("panel_cell_count"),
            "metric_records": validation.get("metric_record_count"),
        }

    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping):
        failures.append("DART pipeline stage evidence is absent")
        stages = {}
    for phase, key in (
        ("selection", "selection_replay"),
        ("locked_test", "locked_replay"),
    ):
        _passed, gate_failures = _real_gate_audit(
            stages.get(key),
            phase=phase,
            root=root,
            frozen_dependencies=(
                frozen_dependencies if phase == "locked_test" else None
            ),
            candidate_pool_contract=candidate_pool_contract,
        )
        failures.extend(gate_failures)

    analysis = _load_json(outputs.get("selection_analysis")) or {}
    layers = analysis.get("three_layers", {})
    if not isinstance(layers, Mapping) or (
        not isinstance(layers.get("probabilistic_credibility"), Mapping)
        or not isinstance(layers.get("actionability"), Mapping)
        or not isinstance(layers.get("decision_value"), Mapping)
        or layers.get("primary_ranking_layer") != "decision_value"
        or layers.get("point_metrics_select_final_winner_alone") is not False
    ):
        failures.append("three-layer DART evaluation is incomplete or mis-ranked")
    selection_results = analysis.get("selection_results_by_experiment", {})
    if not isinstance(selection_results, Mapping):
        failures.append("DART statistical selection results are absent")
    else:
        for experiment_id in ("A1", "A2"):
            failures.extend(
                _selection_result_audit(
                    selection_results.get(experiment_id), experiment_id=experiment_id
                )
            )

    final = _load_json(outputs.get("dart_value_select_manifest")) or {}
    chronology = final.get("chronology", {})
    expected_final_status = (
        PARTIAL_CANDIDATE_STATUS
        if candidate_pool_contract is not None
        else DART_PIPELINE_COMPLETE
    )
    if str(final.get("status", "")).upper() != expected_final_status:
        failures.append("DART-ValueSelect final status differs from candidate admission")
    if candidate_pool_contract is not None:
        try:
            final_pool_sha = validate_partial_candidate_pool_contract(
                final.get("candidate_pool_contract")
            ).get("contract_sha256")
        except (TypeError, ValueError):
            final_pool_sha = None
        if (
            final_pool_sha != expected_pool_sha
            or final.get("terminal_status_cap") != PARTIAL_CANDIDATE_TERMINAL_CAP
        ):
            failures.append("DART-ValueSelect final manifest loses the partial cap")
    if not isinstance(chronology, Mapping) or (
        chronology.get("selection_years") != [2022, 2023, 2024]
        or chronology.get("selection_frozen_before_locked_test") is not True
        or chronology.get("locked_test_year") != 2025
        or chronology.get("locked_test_run_count") != 1
        or chronology.get("selection_updated_from_2025") is not False
    ):
        failures.append("DART once-lock chronology gate failed")
    ledger = _load_json(outputs.get("locked_test_ledger")) or {}
    if (
        str(ledger.get("status", "")).upper() != "COMPLETE"
        or ledger.get("run_count") != 1
        or ledger.get("selection_updated_from_2025") is not False
    ):
        failures.append("DART locked-test once-only ledger failed")

    replay = _load_json(outputs.get("dart_replay_manifest")) or {}
    if str(replay.get("status", "")).upper() != "COMPLETE":
        failures.append("DART replay manifest is not COMPLETE")
    statistics = _load_json(outputs.get("dart_statistical_selection_manifest")) or {}
    if str(statistics.get("status", "")).upper() != "COMPLETE":
        failures.append("DART statistical manifest is not COMPLETE")
    if statistics.get("bootstrap_blocks_days") != [7, 14, 28]:
        failures.append("DART statistical manifest lacks exact 7/14/28-day blocks")
    if statistics.get("primary_block_days") != 7:
        failures.append("DART primary LCB block is not seven days")
    if statistics.get("MCS_before_maximum_LCB") is not True:
        failures.append("DART MCS is not applied before maximum LCB")
    if statistics.get("no_forced_winner") is not True:
        failures.append("DART no-forced-winner rule is absent")
    if statistics.get("selection_updated_from_2025") is not False:
        failures.append("2025 result updated DART selection")
    for label, downstream_manifest in (
        ("replay", replay),
        ("statistics", statistics),
    ):
        downstream_pool = downstream_manifest.get("candidate_pool_contract")
        if candidate_pool_contract is None:
            if downstream_pool is not None:
                failures.append(f"DART {label} manifest carries a partial pool")
        else:
            try:
                downstream_pool_sha = validate_partial_candidate_pool_contract(
                    downstream_pool
                ).get("contract_sha256")
            except (TypeError, ValueError):
                downstream_pool_sha = None
            if (
                downstream_pool_sha != expected_pool_sha
                or downstream_manifest.get("terminal_status_cap")
                != PARTIAL_CANDIDATE_TERMINAL_CAP
            ):
                failures.append(f"DART {label} manifest loses the partial cap")
    selector_contract = statistics.get("selector_comparison_contract", {})
    if (
        not isinstance(selector_contract, Mapping)
        or selector_contract.get("primary_selection_experiment") != "A2"
        or selector_contract.get("preregistered_noninferiority_experiment") != "A1"
        or selector_contract.get("a1_noninferiority_epsilon_fraction") != 0.005
        or selector_contract.get("economic_winner_tolerance_fraction") != 0.005
        or selector_contract.get("primary_comparison")
        != ["MAE-Select", "DART-ValueSelect"]
        or selector_contract.get("winner_set_member_rows_preserved") is not True
        or selector_contract.get("same_locked_A2_replay") is not True
        or selector_contract.get("derived_without_new_solves") is not True
    ):
        failures.append("DART two-method A2/A1 selector contract changed")
    selector_rows = statistics.get("locked_test_selector_rows", [])
    if not isinstance(selector_rows, list) or not selector_rows:
        failures.append("locked two-method selector rows are absent")
        selector_rows = []
    else:
        main_methods = {
            str(row.get("selector"))
            for row in selector_rows
            if isinstance(row, Mapping)
            and row.get("comparison_role")
            == "PRIMARY_MAE_SELECT_VS_DART_VALUE_SELECT"
        }
        if main_methods != {"MAE-Select", "DART-ValueSelect"}:
            failures.append("locked headline selector rows are not exactly MAE and DART")
        if any(
            not isinstance(row, Mapping)
            or row.get("primary_selection_experiment") != "A2"
            or row.get("derived_from_same_locked_A2_replay") is not True
            or row.get("derived_without_new_solves") is not True
            or row.get("selection_updated_from_2025") is not False
            for row in selector_rows
        ):
            failures.append("locked selector rows violate A2/no-new-solve chronology")
    hypotheses = statistics.get("hypotheses", {})
    if not isinstance(hypotheses, Mapping):
        failures.append("H4a-H4d evidence is absent")
        hypotheses = {}
    for hypothesis_id in ("H4a", "H4b", "H4c"):
        value = hypotheses.get(hypothesis_id, {})
        status = str(value.get("status", "")) if isinstance(value, Mapping) else ""
        if status not in _EVALUATED_HYPOTHESIS_STATUSES:
            failures.append(f"{hypothesis_id} was not evaluated")
    h4d = hypotheses.get("H4d", {})
    h4d_status = str(h4d.get("status", "")) if isinstance(h4d, Mapping) else ""
    if h4d_status not in _H4D_TERMINAL_STATUSES:
        failures.append("H4d is neither evaluated nor explicitly skipped")

    h4b = hypotheses.get("H4b", {})
    cssa = h4b.get("cssa", {}) if isinstance(h4b, Mapping) else {}
    if (
        not isinstance(cssa, Mapping)
        or cssa.get("status") != "COMPLETE"
        or cssa.get("primary_comparison")
        != "MAE-Select_vs_DART-ValueSelect"
        or cssa.get("primary_support_uses_only_MAE_comparison") is not True
        or cssa.get("block_days") != [7, 14, 28]
        or cssa.get("derived_from_same_locked_A2_replay") is not True
        or cssa.get("derived_without_new_solves") is not True
    ):
        failures.append("simultaneous locked-A2 CSSA evidence is incomplete")

    return {
        "passed": not failures,
        "failures": failures,
        "manifest_path": _relative(root, manifest_path) if manifest_path else None,
        "panel_counts": panel_counts,
        "sign_sidecars_checked": sign_counts,
        "three_layers_complete": not any("three-layer" in item for item in failures),
        "hypothesis_statuses": {
            key: (
                hypotheses.get(key, {}).get("status")
                if isinstance(hypotheses.get(key), Mapping)
                else None
            )
            for key in ("H4a", "H4b", "H4c", "H4d")
        },
        "selector_comparison": {
            "primary_methods": ["MAE-Select", "DART-ValueSelect"],
            "locked_test_selector_rows": selector_rows,
            "cssa": cssa,
        },
        "dynamic_artifact_inventory_required": True,
        "candidate_pool_contract": candidate_pool_contract,
        "complete_paper_eligible": candidate_pool_contract is None,
        "terminal_status_cap": (
            PARTIAL_CANDIDATE_TERMINAL_CAP
            if candidate_pool_contract is not None
            else None
        ),
    }


__all__ = [
    "DART_PIPELINE_COMPLETE",
    "DART_PIPELINE_PARTIAL",
    "LOCKED_OPERATING_DATES",
    "SELECTION_OPERATING_DATES",
    "attach_dart_overlay_to_forecast",
    "audit_anchor_training_lineage_evidence",
    "audit_dart_pipeline_evidence",
    "audit_v1_dart_readiness_evidence",
    "run_dart_value_select_pipeline",
    "run_v1_dart_readiness_stage",
]
