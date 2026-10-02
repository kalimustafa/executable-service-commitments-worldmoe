"""Hash-pinned cold/warm preflight for formal DART phase replay.

The builder executes the frozen adapter's complete inventory and daily-input
validation once, then stores the certified Python objects in a local binary
cache whose bytes and immutable sources are pinned by an atomic JSON manifest.
The runtime verifies those pins and injects the certified objects at the
adapter's existing seams.  It never changes training artifacts or solver
semantics and it never treats mtime/size as a substitute for SHA-256.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import date
import json
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments import dart_real_evidence as real
from dcvpp.experiments.dart_partial_candidate import (
    candidate_specs_for_contract,
    validate_partial_candidate_pool_contract,
)
from dcvpp.research_branch.contracts import FrozenForecastArtifact


SCHEMA_VERSION = "dcvpp.dart_formal_preflight.v1"
CACHE_SCHEMA_VERSION = "dcvpp.dart_formal_preflight_cache.v1"
SHARED_BASELINE_DEPENDENCY_ROLE = "shared_baseline_dependency_only"


class FormalPreflightError(RuntimeError):
    """Raised when a formal preflight cache is absent or no longer pinned."""


@contextmanager
def _v4_calibration_metadata_compatibility() -> Iterator[None]:
    """Resolve the v4 field rename from the same frozen prediction bytes."""

    original = real._valid_sha256

    def compatible(value: Any, *, label: str) -> str:
        if label.startswith("calibration panel ") and value is None:
            prediction_path = Path(label.removeprefix("calibration panel "))
            if prediction_path.is_file():
                with np.load(prediction_path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata_json"].item()))
                runner = metadata.get("runner_metadata", {})
                renamed = runner.get("deploy_calibration_panel_date_sha256")
                internal = (
                    runner.get("data_audit", {})
                    .get("internal_holdout_panels", {})
                    .get("final_calibration_and_residual_coupling", {})
                    .get("date_sha256")
                )
                if (
                    runner.get("deploy_calibration_panel_role")
                    != "final_calibration_and_residual_coupling_only"
                    or renamed != internal
                ):
                    raise FormalPreflightError(
                        "v4 calibration metadata compatibility identity failed"
                    )
                return original(renamed, label=label)
        return original(value, label=label)

    real._valid_sha256 = compatible
    try:
        yield
    finally:
        real._valid_sha256 = original


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormalPreflightError(f"JSON root is not a mapping: {path}")
    return value


def _resolve(raw: str | Path, *, root: Path, sibling: Path | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    from_root = (root / path).resolve()
    if from_root.exists() or sibling is None:
        return from_root
    return (sibling / path).resolve()


def _relative(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else resolved.as_posix()


def _manifest_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: item for key, item in value.items() if key != "manifest_fingerprint"}
    )


def _normalize_npz_scalars(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    for field in (
        "candidate_independent",
        "selected_scenario_as_realized",
        "locked_test",
        "frozen",
    ):
        value = payload.get(field)
        if isinstance(value, np.ndarray) and value.ndim == 0:
            payload[field] = value.item()
    with np.load(path, allow_pickle=False) as data:
        if "source_fields_json" in data.files:
            payload["source_fields"] = json.loads(str(data["source_fields_json"].item()))
    return payload


def _index_record(
    raw: Any, *, root: Path, label: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if isinstance(raw, Mapping) and "index_path" in raw:
        index_path = _resolve(raw["index_path"], root=root)
    elif isinstance(raw, (str, Path)):
        index_path = _resolve(raw, root=root)
    else:
        raise FormalPreflightError(f"{label} must use a frozen JSON index")
    index = _read(index_path)
    if str(index.get("status", "")).upper() != "COMPLETE":
        raise FormalPreflightError(f"{label} index is not COMPLETE")
    records = real._expand_day_records(raw, root=root, label=label)
    return index_path, index, records


def _source_pin(path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FormalPreflightError(f"source pin is absent: {path}")
    return {
        "path": _relative(path, root=root),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _inventory_source_paths(
    inventory: real.OOSInventory, *, root: Path, stage_root: Path
) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for record in inventory.records:
        paths.update((record.artifact_path.resolve(), record.manifest_path.resolve()))
        for target in ("da_lbmp", "rt_lbmp_at_da_gate"):
            unit_root = (
                stage_root
                / "units"
                / record.origin_name
                / target
                / record.candidate_name
                / record.seed_slot_name
            )
            checkpoint_path = unit_root / "unit_checkpoint.json"
            checkpoint = _read(checkpoint_path)
            paths.update(
                (
                    checkpoint_path.resolve(),
                    (unit_root / "quantile_predictions.npz").resolve(),
                )
            )
            model = checkpoint.get("runner_model_checkpoint", {})
            paths.add(
                real._resolve_path(
                    model["path"], root=root, sibling=unit_root
                ).resolve()
            )
        coupling_root = (
            stage_root
            / "joint_uniforms"
            / record.origin_name
            / record.candidate_name
            / record.seed_slot_name
        )
        paths.update(
            (
                (coupling_root / "paired_uniforms.npz").resolve(),
                (coupling_root / "coupling_manifest.json").resolve(),
            )
        )
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _input_source_paths(
    *,
    index_paths: Sequence[Path],
    record_groups: Sequence[Mapping[str, Any]],
    root: Path,
) -> tuple[Path, ...]:
    paths = {path.resolve() for path in index_paths}
    for records in record_groups:
        for raw in records.values():
            if not isinstance(raw, Mapping) or "path" not in raw:
                raise FormalPreflightError("formal daily input record lacks a path")
            paths.add(_resolve(raw["path"], root=root).resolve())
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _dump_cache(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(dict(value), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        raise FormalPreflightError(f"formal preflight cache is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise FormalPreflightError("formal preflight cache schema changed")
    return value


def prepare_formal_preflight(
    *,
    repository_root: str | Path,
    stage_root: str | Path,
    phase: str,
    forecast_result: Mapping[str, Any],
    config: Mapping[str, Any],
    candidate_pool_contract: Mapping[str, Any],
    output_root: str | Path,
    candidate_names: Sequence[str] | None = None,
    operating_dates: Sequence[str] | None = None,
    scope_id: str = "full_phase",
    shared_baseline_dependency_name: str | None = None,
) -> dict[str, Any]:
    """Cold-certify the complete formal inventory and daily input panel."""

    root = Path(repository_root).resolve()
    stage = Path(stage_root).resolve()
    destination = Path(output_root).resolve()
    if phase not in {"selection", "locked_test"}:
        raise ValueError("phase must be selection or locked_test")
    normalized_pool = validate_partial_candidate_pool_contract(candidate_pool_contract)
    full_candidate_specs = candidate_specs_for_contract(normalized_pool)
    requested_candidates = (
        tuple(spec.name for spec in full_candidate_specs)
        if candidate_names is None
        else tuple(map(str, candidate_names))
    )
    if not requested_candidates or len(set(requested_candidates)) != len(
        requested_candidates
    ):
        raise FormalPreflightError("candidate_names must be nonempty and unique")
    candidate_specs = tuple(
        spec for spec in full_candidate_specs if spec.name in requested_candidates
    )
    if tuple(spec.name for spec in candidate_specs) != requested_candidates:
        raise FormalPreflightError("candidate_names are not an ordered eligible subset")
    real_contract = config.get("real_evidence_contract", {})
    if not isinstance(real_contract, Mapping):
        raise FormalPreflightError("real_evidence_contract must be a mapping")
    inputs = real._input_block(config, forecast_result)

    requested_dates = (
        None
        if operating_dates is None
        else tuple(sorted(map(str, operating_dates)))
    )
    if requested_dates is not None:
        if not requested_dates or len(set(requested_dates)) != len(requested_dates):
            raise FormalPreflightError("operating_dates must be nonempty and unique")
        requested_years = {date.fromisoformat(day).year for day in requested_dates}
        configured = real_contract.get("operating_dates_by_phase", {}).get(phase)
        if configured is not None and set(requested_dates).difference(map(str, configured)):
            raise FormalPreflightError(
                "operating_dates are outside the frozen phase date overlay"
            )
    else:
        requested_years = None

    started = perf_counter()
    inventory_started = perf_counter()
    original_phase_origins = real._phase_origins

    def scoped_phase_origins(value: str) -> tuple[Any, ...]:
        origins = original_phase_origins(value)
        if value != phase or requested_years is None:
            return origins
        scoped = tuple(
            origin for origin in origins if int(origin.forecast_year) in requested_years
        )
        if {int(origin.forecast_year) for origin in scoped} != requested_years:
            raise FormalPreflightError("scoped dates lack an exact frozen OOS origin")
        return scoped

    try:
        real._phase_origins = scoped_phase_origins
        with _v4_calibration_metadata_compatibility():
            inventory = real._discover_oos_inventory(
                repository_root=root,
                stage_root=stage,
                phase=phase,
                real_contract=real_contract,
                candidate_specs=candidate_specs,
            )
    finally:
        real._phase_origins = original_phase_origins
    inventory_seconds = perf_counter() - inventory_started

    dependency_inventory: real.OOSInventory | None = None
    dependency_spec: Any | None = None
    if shared_baseline_dependency_name is not None:
        dependency_specs = tuple(
            spec
            for spec in full_candidate_specs
            if spec.name == str(shared_baseline_dependency_name)
        )
        if len(dependency_specs) != 1:
            raise FormalPreflightError(
                "shared baseline dependency is not uniquely eligible"
            )
        if dependency_specs[0].name in requested_candidates:
            raise FormalPreflightError(
                "shared baseline dependency must not be a scoped candidate"
            )
        dependency_spec = dependency_specs[0]
        try:
            real._phase_origins = scoped_phase_origins
            with _v4_calibration_metadata_compatibility():
                dependency_inventory = real._discover_oos_inventory(
                    repository_root=root,
                    stage_root=stage,
                    phase=phase,
                    real_contract=real_contract,
                    candidate_specs=dependency_specs,
                )
        finally:
            real._phase_origins = original_phase_origins
    dates = (
        real._selected_dates(inventory, phase=phase, real_contract=real_contract)
        if requested_dates is None
        else requested_dates
    )
    if set(dates).difference(inventory.dates):
        raise FormalPreflightError(
            "operating_dates are not a nonempty unique subset of the frozen phase panel"
        )
    if dependency_inventory is not None and set(dates).difference(
        dependency_inventory.dates
    ):
        raise FormalPreflightError(
            "shared baseline dependency lacks scoped operating dates"
        )

    issue_index_path, issue_index, issue_records = _index_record(
        inputs.get("issue_time_ensembles_by_date"),
        root=root,
        label="issue_time_ensembles_by_date",
    )
    settlement_index_path, settlement_index, settlement_records = _index_record(
        inputs.get("nyiso_replay_by_date"), root=root, label="nyiso_replay_by_date"
    )
    forcing_index_path, forcing_index, forcing_records = _index_record(
        inputs.get("replay_forcing_by_date"), root=root, label="replay_forcing_by_date"
    )
    action_index_path, action_index, action_records = _index_record(
        inputs.get("actionability_by_date"), root=root, label="actionability_by_date"
    )

    # One path+digest load services every logical store.  This is especially
    # important because issue, forcing, and actionability share one NPZ/day.
    payload_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
    original_load_npz = real._load_npz_record
    original_forecast_load = real.FrozenForecastStore._load_npz

    def cached_payload(
        record: Mapping[str, Any], *, root: Path, label: str
    ) -> dict[str, Any]:
        resolved = real._resolve_path(record["path"], root=root)
        key = (resolved.as_posix(), record.get("sha256"))
        if key not in payload_cache:
            loaded = original_load_npz(record, root=root, label=label)
            payload_cache[key] = _normalize_npz_scalars(loaded, resolved)
        return payload_cache[key]

    def forecast_load(
        store: Any, path: Path, record: Mapping[str, Any]
    ) -> Any:
        loaded = cached_payload(
            {"path": path, "sha256": record.get("sha256")},
            root=store._root,
            label=f"frozen forecast {path}",
        )
        source_hashes = dict(loaded.get("source_hashes", record.get("source_hashes", {})))
        source_hashes["frozen_day_npz"] = str(loaded["artifact_sha256"])
        return FrozenForecastArtifact.from_mapping(
            {
                "issue_time_utc": str(loaded["issue_time_utc"].item()),
                "operating_date": str(loaded["operating_date"].item()),
                "forecast_id": str(loaded["forecast_id"].item()),
                "timestamps_utc": loaded["timestamps_utc"],
                "da_price": loaded["da_price"],
                "rt_price_pool": loaded["rt_price_pool"],
                "workload_pool": loaded["workload_pool"],
                "pv_pool": loaded["pv_pool"],
                "zones": tuple(map(str, loaded["zones"].tolist())),
                "source_hashes": source_hashes,
                "locked_test": bool(loaded.get("locked_test", True)),
            }
        )

    preflight_started = perf_counter()
    try:
        real._load_npz_record = cached_payload
        real.FrozenForecastStore._load_npz = forecast_load
        issue_store = real._issue_store(
            repository_root=root, forecast_result=forecast_result, inputs=inputs
        )
        settlement_store = real._ExternalInputStore(
            repository_root=root,
            phase=phase,
            records=settlement_records,
            nyiso_data_root=(
                None
                if inputs.get("referenced_nyiso_data_root") is None
                else _resolve(inputs["referenced_nyiso_data_root"], root=root)
            ),
        )
        forcing_store = real._ForcingStore(
            repository_root=root, records=forcing_records
        )
        action_store = real._ActionabilityStore(
            repository_root=root, records=action_records
        )
        issue, settlement, forcing, actionability, thresholds = real._preflight_inputs(
            dates,
            inventory=inventory,
            issue_store=issue_store,
            settlement_store=settlement_store,
            forcing_store=forcing_store,
            action_store=action_store,
            threshold_records=dict(inputs.get("training_thresholds_by_year", {})),
        )
    finally:
        real._load_npz_record = original_load_npz
        real.FrozenForecastStore._load_npz = original_forecast_load
    preflight_seconds = perf_counter() - preflight_started

    cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": phase,
        "dates": tuple(dates),
        "inventory": inventory,
        "shared_baseline_dependency_inventory": dependency_inventory,
        "issue": issue,
        "settlement": settlement,
        "forcing": forcing,
        "actionability": actionability,
        "thresholds": thresholds,
    }
    cache_path = destination / f"{phase}_formal_preflight_cache.pkl"
    _dump_cache(cache_path, cache)

    inventory_paths = _inventory_source_paths(
        inventory, root=root, stage_root=stage
    )
    if dependency_inventory is not None:
        inventory_paths = tuple(
            sorted(
                set(inventory_paths).union(
                    _inventory_source_paths(
                        dependency_inventory, root=root, stage_root=stage
                    )
                ),
                key=lambda item: item.as_posix(),
            )
        )
    scoped_issue_records = {day: issue_records[day] for day in dates}
    scoped_settlement_records = {day: settlement_records[day] for day in dates}
    scoped_forcing_records = {day: forcing_records[day] for day in dates}
    scoped_action_records = {day: action_records[day] for day in dates}
    input_paths = _input_source_paths(
        index_paths=(
            issue_index_path,
            settlement_index_path,
            forcing_index_path,
            action_index_path,
        ),
        record_groups=(
            scoped_issue_records,
            scoped_settlement_records,
            scoped_forcing_records,
            scoped_action_records,
        ),
        root=root,
    )
    source_paths = tuple(
        sorted(set(inventory_paths).union(input_paths), key=lambda item: item.as_posix())
    )
    source_pins = [_source_pin(path, root=root) for path in source_paths]
    index_contract = {
        "issue": {
            "path": _relative(issue_index_path, root=root),
            "sha256": sha256_file(issue_index_path),
            "records_sha256": issue_index.get("records_sha256"),
        },
        "settlement": {
            "path": _relative(settlement_index_path, root=root),
            "sha256": sha256_file(settlement_index_path),
            "records_sha256": settlement_index.get(
                "index_fingerprint", settlement_index.get("records_sha256")
            ),
        },
        "forcing": {
            "path": _relative(forcing_index_path, root=root),
            "sha256": sha256_file(forcing_index_path),
            "records_sha256": forcing_index.get("records_sha256"),
        },
        "actionability": {
            "path": _relative(action_index_path, root=root),
            "sha256": sha256_file(action_index_path),
            "records_sha256": action_index.get("records_sha256"),
        },
    }
    cache_pin = _source_pin(cache_path, root=root)
    contract = {
        "repository_root": root.as_posix(),
        "stage_root": stage.as_posix(),
        "phase": phase,
        "scope_id": str(scope_id),
        "full_phase_scope": candidate_names is None and operating_dates is None,
        "candidate_pool_contract_sha256": normalized_pool["contract_sha256"],
        "candidate_ids": [spec.name for spec in candidate_specs],
        "shared_baseline_dependency": (
            None
            if dependency_inventory is None or dependency_spec is None
            else {
                "candidate_id": dependency_spec.name,
                "seed_slot_ids": list(dependency_spec.seed_slots),
                "role": SHARED_BASELINE_DEPENDENCY_ROLE,
                "inventory_fingerprint": dependency_inventory.fingerprint,
                "inventory_record_count": len(dependency_inventory.records),
                "excluded_from_candidate_ids": True,
                "excluded_from_candidate_inventory": True,
                "excluded_from_panel_and_ranking": True,
            }
        ),
        "inventory_fingerprint": inventory.fingerprint,
        "operating_dates_sha256": sha256_json(list(dates)),
        "operating_date_count": len(dates),
        "expected_intervals_per_day": int(real_contract.get("expected_intervals_per_day", 96)),
        "input_indexes": index_contract,
        "real_adapter_code_sha256": sha256_file(Path(real.__file__).resolve()),
        "runtime_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "formal_evidence_eligible": True,
        "formal_full_phase_evidence_eligible": (
            candidate_names is None and operating_dates is None
        ),
        "formal_scoped_pilot_evidence_eligible": (
            candidate_names is not None or operating_dates is not None
        ),
        "cold_certification_complete": True,
        "solver_invocation_count": 0,
        "contract": contract,
        "contract_sha256": sha256_json(contract),
        "cache": cache_pin,
        "source_pins": source_pins,
        "source_pins_sha256": sha256_json(source_pins),
        "validation": {
            "inventory_record_count": len(inventory.records),
            "operating_date_count": len(dates),
            "daily_issue_count": len(issue),
            "daily_settlement_count": len(settlement),
            "daily_forcing_count": len(forcing),
            "daily_actionability_count": len(actionability),
            "threshold_years": sorted(thresholds),
            "unique_daily_npz_load_count": len(payload_cache),
            "inventory_all_leakage_gates_passed": inventory.all_leakage_gates_passed,
            "shared_baseline_dependency_record_count": (
                0
                if dependency_inventory is None
                else len(dependency_inventory.records)
            ),
            "shared_baseline_dependency_all_leakage_gates_passed": (
                None
                if dependency_inventory is None
                else dependency_inventory.all_leakage_gates_passed
            ),
            "candidate_panel_cell_count": len(dates)
            * len(candidate_specs)
            * 2
            * len(real.DART_VALUE_SELECT_SEEDS),
        },
        "timing": {
            "inventory_seconds": max(inventory_seconds, 0.0),
            "daily_input_preflight_seconds": max(preflight_seconds, 0.0),
            "total_seconds": max(perf_counter() - started, 0.0),
        },
        "created_at_utc": utc_now(),
    }
    manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)
    manifest_path = destination / f"{phase}_formal_preflight_manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        "status": "COMPLETE",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "cache_path": cache_path.as_posix(),
        "validation": manifest["validation"],
        "timing": manifest["timing"],
    }


def load_formal_preflight(
    manifest_path: str | Path, *, repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail-closed load of one hash-pinned formal preflight cache."""

    root = Path(repository_root).resolve()
    path = _resolve(manifest_path, root=root)
    manifest = _read(path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("formal_evidence_eligible") is not True
        or manifest.get("cold_certification_complete") is not True
        or manifest.get("manifest_fingerprint") != _manifest_fingerprint(manifest)
        or manifest.get("contract_sha256") != sha256_json(manifest.get("contract"))
        or manifest.get("source_pins_sha256") != sha256_json(manifest.get("source_pins"))
    ):
        raise FormalPreflightError("formal preflight manifest identity is invalid")
    if manifest["contract"].get("runtime_code_sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise FormalPreflightError("formal preflight runtime code changed")
    if manifest["contract"].get("real_adapter_code_sha256") != sha256_file(
        Path(real.__file__).resolve()
    ):
        raise FormalPreflightError("formal real-evidence adapter code changed")
    for source in manifest["source_pins"]:
        source_path = _resolve(source["path"], root=root)
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(source["byte_count"])
            or sha256_file(source_path) != source["sha256"]
        ):
            raise FormalPreflightError(f"formal source pin changed: {source_path}")
    cache_record = manifest["cache"]
    cache_path = _resolve(cache_record["path"], root=root)
    if (
        not cache_path.is_file()
        or cache_path.stat().st_size != int(cache_record["byte_count"])
        or sha256_file(cache_path) != cache_record["sha256"]
    ):
        raise FormalPreflightError("formal preflight binary cache changed")
    cache = _load_cache(cache_path)
    contract = manifest["contract"]
    dependency_contract = contract.get("shared_baseline_dependency")
    dependency_inventory = cache.get("shared_baseline_dependency_inventory")
    if (
        cache.get("phase") != contract["phase"]
        or sha256_json(list(cache.get("dates", ())))
        != contract["operating_dates_sha256"]
        or cache["inventory"].fingerprint != contract["inventory_fingerprint"]
        or len(cache["inventory"].records)
        != int(manifest["validation"]["inventory_record_count"])
    ):
        raise FormalPreflightError("formal preflight binary cache contract changed")
    if dependency_contract is None:
        if dependency_inventory is not None:
            raise FormalPreflightError(
                "uncertified shared baseline dependency appeared in cache"
            )
    elif (
        dependency_contract.get("role") != SHARED_BASELINE_DEPENDENCY_ROLE
        or dependency_contract.get("excluded_from_candidate_ids") is not True
        or dependency_contract.get("excluded_from_candidate_inventory") is not True
        or dependency_contract.get("excluded_from_panel_and_ranking") is not True
        or dependency_contract.get("candidate_id") in contract["candidate_ids"]
        or dependency_inventory is None
        or dependency_inventory.fingerprint
        != dependency_contract.get("inventory_fingerprint")
        or len(dependency_inventory.records)
        != int(dependency_contract.get("inventory_record_count", -1))
        or any(
            record.candidate_name != dependency_contract.get("candidate_id")
            for record in dependency_inventory.records
        )
    ):
        raise FormalPreflightError(
            "shared baseline dependency contract changed"
        )
    return manifest, cache


@contextmanager
def formal_preflight_runtime(
    *,
    repository_root: str | Path,
    manifest_path: str | Path,
    runtime_ledger_path: str | Path | None = None,
    expected_scope_id: str | None = None,
    expected_candidate_ids: Sequence[str] | None = None,
    expected_operating_dates: Sequence[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Inject a certified formal inventory/input cache at existing seams."""

    root = Path(repository_root).resolve()
    load_started = perf_counter()
    manifest, cache = load_formal_preflight(manifest_path, repository_root=root)
    if (
        expected_scope_id is not None
        and manifest["contract"].get("scope_id") != expected_scope_id
    ):
        raise FormalPreflightError("formal preflight scope_id differs")
    if expected_candidate_ids is not None and list(map(str, expected_candidate_ids)) != list(
        manifest["contract"].get("candidate_ids", ())
    ):
        raise FormalPreflightError("formal preflight candidate scope differs")
    if expected_operating_dates is not None and tuple(
        sorted(map(str, expected_operating_dates))
    ) != tuple(cache["dates"]):
        raise FormalPreflightError("formal preflight operating-date scope differs")
    load_seconds = perf_counter() - load_started
    original_discover = real._discover_oos_inventory
    original_selected = real._selected_dates
    original_preflight = real._preflight_inputs
    original_candidate_specs = real.candidate_specs_for_contract
    calls = 0
    state = {
        "schema_version": "dcvpp.dart_formal_preflight_runtime.v1",
        "status": "ACTIVE",
        "formal_evidence_eligible": True,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "manifest_sha256": sha256_file(_resolve(manifest_path, root=root)),
        "phase": manifest["contract"]["phase"],
        "scope_id": manifest["contract"]["scope_id"],
        "candidate_ids": list(manifest["contract"]["candidate_ids"]),
        "shared_baseline_dependency": manifest["contract"].get(
            "shared_baseline_dependency"
        ),
        "operating_date_count": len(cache["dates"]),
        "warm_load_seconds": max(load_seconds, 0.0),
        "activated_at_utc": utc_now(),
    }

    def discover(**kwargs: Any) -> real.OOSInventory:
        nonlocal calls
        calls += 1
        if str(kwargs.get("phase")) != manifest["contract"]["phase"]:
            raise FormalPreflightError("formal preflight phase differs from runtime call")
        specs = kwargs.get("candidate_specs", ())
        requested_ids = [spec.name for spec in specs]
        if Path(kwargs.get("stage_root", "")).resolve().as_posix() != manifest["contract"][
            "stage_root"
        ]:
            raise FormalPreflightError("formal preflight stage root differs")
        if requested_ids == manifest["contract"]["candidate_ids"]:
            return cache["inventory"]
        dependency = manifest["contract"].get("shared_baseline_dependency")
        if (
            dependency is not None
            and requested_ids == [dependency["candidate_id"]]
            and dependency.get("role") == SHARED_BASELINE_DEPENDENCY_ROLE
        ):
            return cache["shared_baseline_dependency_inventory"]
        raise FormalPreflightError("formal preflight candidate pool differs")

    def candidate_specs(
        value: Mapping[str, Any] | None,
    ) -> tuple[Any, ...]:
        normalized = validate_partial_candidate_pool_contract(value)
        if normalized["contract_sha256"] != manifest["contract"][
            "candidate_pool_contract_sha256"
        ]:
            raise FormalPreflightError("formal candidate-pool contract differs")
        eligible = set(manifest["contract"]["candidate_ids"])
        specs = tuple(
            spec
            for spec in original_candidate_specs(normalized)
            if spec.name in eligible
        )
        if [spec.name for spec in specs] != manifest["contract"]["candidate_ids"]:
            raise FormalPreflightError("formal candidate scope cannot be resolved")
        return specs

    def selected(
        inventory: real.OOSInventory, *, phase: str, real_contract: Mapping[str, Any]
    ) -> tuple[str, ...]:
        if inventory.fingerprint != manifest["contract"]["inventory_fingerprint"]:
            raise FormalPreflightError("formal runtime received a different inventory")
        configured = real_contract.get("operating_dates_by_phase", {}).get(phase)
        expected = tuple(cache["dates"])
        configured_dates = None if configured is None else set(map(str, configured))
        if manifest["contract"].get("full_phase_scope"):
            if configured_dates is not None and tuple(sorted(configured_dates)) != expected:
                raise FormalPreflightError("formal operating-date overlay differs from preflight")
        elif configured_dates is not None and set(expected).difference(configured_dates):
            raise FormalPreflightError("scoped dates are outside the formal date overlay")
        return expected

    def preflight(
        dates: Sequence[str], **_kwargs: Any
    ) -> tuple[dict[str, Any], ...]:
        if tuple(map(str, dates)) != tuple(cache["dates"]):
            raise FormalPreflightError("formal preflight date panel differs")
        return (
            cache["issue"],
            cache["settlement"],
            cache["forcing"],
            cache["actionability"],
            cache["thresholds"],
        )

    real.candidate_specs_for_contract = candidate_specs
    real._discover_oos_inventory = discover
    real._selected_dates = selected
    real._preflight_inputs = preflight
    if runtime_ledger_path is not None:
        atomic_json(_resolve(runtime_ledger_path, root=root), state)
    try:
        yield state
    finally:
        real.candidate_specs_for_contract = original_candidate_specs
        real._discover_oos_inventory = original_discover
        real._selected_dates = original_selected
        real._preflight_inputs = original_preflight
        state.update(
            {
                "status": "INACTIVE",
                "adapter_call_count": calls,
                "deactivated_at_utc": utc_now(),
            }
        )
        if runtime_ledger_path is not None:
            atomic_json(_resolve(runtime_ledger_path, root=root), state)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "FormalPreflightError",
    "SCHEMA_VERSION",
    "SHARED_BASELINE_DEPENDENCY_ROLE",
    "formal_preflight_runtime",
    "load_formal_preflight",
    "prepare_formal_preflight",
]
