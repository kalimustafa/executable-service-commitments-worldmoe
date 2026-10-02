"""Scoped, non-formal acceleration for one real DART replay probe.

The frozen real-evidence adapter deliberately validates the complete phase.
That is the correct formal behaviour, but it makes a one-cell runtime probe
pay for every candidate and annual origin before a solver is called.  This
module narrows only an explicitly isolated probe to one date/candidate and
persists the resulting *cold-certified* inventory.  A warm load re-hashes the
same source files but never decompresses their large NPZ arrays.

Nothing in this module is eligible as formal A1/A2 evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

import numpy as np

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments import dart_real_evidence as real
from dcvpp.experiments.dart_forecast_oos import ORIGINS
from dcvpp.experiments.dart_partial_candidate import candidate_specs_for_contract
from dcvpp.research_branch.contracts import FrozenForecastArtifact


SCHEMA_VERSION = "dcvpp.dart_probe_preflight_runtime.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not a mapping: {path}")
    return value


def _source_paths(
    record: real.OOSArtifactRecord, *, repository_root: Path, stage_root: Path
) -> tuple[Path, ...]:
    """Return every byte source whose digest was checked by cold discovery."""

    paths: set[Path] = {record.artifact_path.resolve(), record.manifest_path.resolve()}
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
        paths.add(checkpoint_path.resolve())
        paths.add((unit_root / "quantile_predictions.npz").resolve())
        model = checkpoint.get("runner_model_checkpoint", {})
        paths.add(
            real._resolve_path(
                model["path"], root=repository_root, sibling=unit_root
            ).resolve()
        )
    coupling_root = (
        stage_root
        / "joint_uniforms"
        / record.origin_name
        / record.candidate_name
        / record.seed_slot_name
    )
    paths.add((coupling_root / "paired_uniforms.npz").resolve())
    paths.add((coupling_root / "coupling_manifest.json").resolve())
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _source_records(
    inventory: real.OOSInventory, *, repository_root: Path, stage_root: Path
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in inventory.records:
        for path in _source_paths(
            record, repository_root=repository_root, stage_root=stage_root
        ):
            if not path.is_file():
                raise FileNotFoundError(f"probe inventory source is absent: {path}")
            output.append(
                {
                    "path": path.as_posix(),
                    "byte_count": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    unique = {item["path"]: item for item in output}
    return [unique[key] for key in sorted(unique)]


def _serialize_inventory(inventory: real.OOSInventory) -> dict[str, Any]:
    records = []
    for record in inventory.records:
        payload = asdict(record)
        payload["artifact_path"] = record.artifact_path.as_posix()
        payload["manifest_path"] = record.manifest_path.as_posix()
        payload["dates"] = list(record.dates)
        payload["timestamps_by_date"] = {
            day: list(values) for day, values in record.timestamps_by_date.items()
        }
        records.append(payload)
    return {
        "records": records,
        "dates": list(inventory.dates),
        "timestamps_by_date": {
            day: list(values) for day, values in inventory.timestamps_by_date.items()
        },
        "fingerprint": inventory.fingerprint,
        "all_leakage_gates_passed": inventory.all_leakage_gates_passed,
    }


def _deserialize_inventory(raw: Mapping[str, Any]) -> real.OOSInventory:
    records = []
    for item in raw["records"]:
        payload = dict(item)
        payload["artifact_path"] = Path(payload["artifact_path"])
        payload["manifest_path"] = Path(payload["manifest_path"])
        payload["dates"] = tuple(map(str, payload["dates"]))
        payload["timestamps_by_date"] = {
            str(day): tuple(map(str, values))
            for day, values in payload["timestamps_by_date"].items()
        }
        records.append(real.OOSArtifactRecord(**payload))
    record_tuple = tuple(records)
    inventory = real.OOSInventory(
        records=record_tuple,
        records_by_key={record.key: record for record in record_tuple},
        dates=tuple(map(str, raw["dates"])),
        timestamps_by_date={
            str(day): tuple(map(str, values))
            for day, values in raw["timestamps_by_date"].items()
        },
        fingerprint=str(raw["fingerprint"]),
        all_leakage_gates_passed=bool(raw["all_leakage_gates_passed"]),
    )
    serialized = [
        record.to_dict() for record in sorted(inventory.records, key=lambda item: item.key)
    ]
    if sha256_json(serialized) != inventory.fingerprint:
        raise RuntimeError("cached probe inventory fingerprint changed")
    return inventory


def _cache_contract(
    *,
    stage_root: Path,
    phase: str,
    operating_date: str,
    candidate_name: str,
    candidate_pool_contract: Mapping[str, Any],
    expected_intervals: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "isolated_probe_only_not_formal_a1_a2_evidence",
        "stage_root": stage_root.resolve().as_posix(),
        "phase": phase,
        "operating_date": operating_date,
        "candidate_name": candidate_name,
        "candidate_pool_contract_sha256": sha256_json(candidate_pool_contract),
        "expected_intervals_per_day": int(expected_intervals),
        "real_adapter_code_sha256": sha256_file(Path(real.__file__).resolve()),
        "runtime_code_sha256": sha256_file(Path(__file__).resolve()),
    }


def _load_warm_inventory(
    path: Path, *, expected_contract: Mapping[str, Any]
) -> real.OOSInventory | None:
    if not path.is_file():
        return None
    cached = _read(path)
    if (
        cached.get("status") != "COLD_CERTIFIED"
        or cached.get("contract_sha256") != sha256_json(expected_contract)
        or cached.get("payload_sha256") != sha256_json(cached.get("inventory"))
    ):
        return None
    for source in cached.get("sources", []):
        source_path = Path(source["path"])
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(source["byte_count"])
            or sha256_file(source_path) != source["sha256"]
        ):
            return None
    return _deserialize_inventory(cached["inventory"])


@contextmanager
def focused_probe_preflight_runtime(
    *,
    repository_root: Path,
    stage_root: Path,
    phase: str,
    operating_date: str,
    candidate_name: str,
    candidate_pool_contract: Mapping[str, Any],
    cache_path: Path,
    ledger_path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Narrow and cache one probe without changing the formal adapter files."""

    if phase != "selection":
        raise ValueError("focused probe runtime currently permits selection only")
    year = int(operating_date[:4])
    origins = tuple(
        origin
        for origin in ORIGINS
        if origin.role == "selection_oos" and origin.forecast_year == year
    )
    if len(origins) != 1:
        raise ValueError(f"no unique selection origin exists for {operating_date}")
    full_specs = candidate_specs_for_contract(candidate_pool_contract)
    focused_specs = tuple(spec for spec in full_specs if spec.name == candidate_name)
    if len(focused_specs) != 1:
        raise ValueError(f"candidate is not uniquely eligible: {candidate_name}")

    original_phase_origins = real._phase_origins
    original_candidate_specs = real.candidate_specs_for_contract
    original_selected_dates = real._selected_dates
    original_discover = real._discover_oos_inventory
    original_load_npz = real._load_npz_record
    original_forecast_load = real.FrozenForecastStore._load_npz
    payload_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ACTIVE",
        "scope": "ISOLATED_PROBE_ONLY",
        "formal_a1_a2_evidence_eligible": False,
        "operating_date": operating_date,
        "candidate_name": candidate_name,
        "cache_path": cache_path.resolve().as_posix(),
        "activated_at_utc": utc_now(),
        "inventory_cache_status": "NOT_USED",
    }

    def phase_origins(value: str) -> tuple[Any, ...]:
        if value != phase:
            raise RuntimeError("focused probe refuses a different phase")
        return origins

    def specs(_contract: Mapping[str, Any] | None) -> tuple[Any, ...]:
        return focused_specs

    def dates(
        inventory: real.OOSInventory, *, phase: str, real_contract: Mapping[str, Any]
    ) -> tuple[str, ...]:
        if operating_date not in inventory.dates:
            raise RuntimeError("focused operating date is absent from certified inventory")
        return (operating_date,)

    def discover(**kwargs: Any) -> real.OOSInventory:
        contract = _cache_contract(
            stage_root=stage_root,
            phase=phase,
            operating_date=operating_date,
            candidate_name=candidate_name,
            candidate_pool_contract=candidate_pool_contract,
            expected_intervals=int(
                kwargs.get("real_contract", {}).get("expected_intervals_per_day", 96)
            ),
        )
        started = perf_counter()
        inventory = _load_warm_inventory(cache_path, expected_contract=contract)
        if inventory is None:
            inventory = original_discover(**kwargs)
            inventory_payload = _serialize_inventory(inventory)
            sources = _source_records(
                inventory,
                repository_root=repository_root,
                stage_root=stage_root,
            )
            atomic_json(
                cache_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "COLD_CERTIFIED",
                    "scope": contract["scope"],
                    "contract": contract,
                    "contract_sha256": sha256_json(contract),
                    "sources": sources,
                    "sources_sha256": sha256_json(sources),
                    "inventory": inventory_payload,
                    "payload_sha256": sha256_json(inventory_payload),
                    "certified_at_utc": utc_now(),
                },
            )
            state["inventory_cache_status"] = "COLD_BUILT"
        else:
            state["inventory_cache_status"] = "WARM_HIT"
        state["inventory_elapsed_seconds"] = max(perf_counter() - started, 0.0)
        state["inventory_record_count"] = len(inventory.records)
        if ledger_path is not None:
            atomic_json(ledger_path, state)
        return inventory

    def cached_payload(
        record: Mapping[str, Any], *, root: Path, label: str
    ) -> dict[str, Any]:
        resolved = real._resolve_path(record["path"], root=root)
        key = (resolved.as_posix(), record.get("sha256"))
        if key not in payload_cache:
            payload = original_load_npz(record, root=root, label=label)
            # Counterfactual daily bundles combine three logical stores.  The
            # frozen loader keeps NumPy scalars, while downstream provenance
            # gates intentionally use Python-bool identity checks.
            for field in (
                "candidate_independent",
                "selected_scenario_as_realized",
                "locked_test",
                "frozen",
            ):
                value = payload.get(field)
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    payload[field] = value.item()
            with np.load(resolved, allow_pickle=False) as data:
                if "source_fields_json" in data.files:
                    payload["source_fields"] = json.loads(
                        str(data["source_fields_json"].item())
                    )
            payload_cache[key] = payload
        return payload_cache[key]

    def forecast_load(
        store: Any, path: Path, record: Mapping[str, Any]
    ) -> FrozenForecastArtifact:
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

    real._phase_origins = phase_origins
    real.candidate_specs_for_contract = specs
    real._selected_dates = dates
    real._discover_oos_inventory = discover
    real._load_npz_record = cached_payload
    real.FrozenForecastStore._load_npz = forecast_load
    if ledger_path is not None:
        atomic_json(ledger_path, state)
    try:
        yield state
    finally:
        real._phase_origins = original_phase_origins
        real.candidate_specs_for_contract = original_candidate_specs
        real._selected_dates = original_selected_dates
        real._discover_oos_inventory = original_discover
        real._load_npz_record = original_load_npz
        real.FrozenForecastStore._load_npz = original_forecast_load
        state["status"] = "INACTIVE"
        state["deactivated_at_utc"] = utc_now()
        state["shared_daily_npz_unique_load_count"] = len(payload_cache)
        if ledger_path is not None:
            atomic_json(ledger_path, state)


__all__ = ["SCHEMA_VERSION", "focused_probe_preflight_runtime"]
