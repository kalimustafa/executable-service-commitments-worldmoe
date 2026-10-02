from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .contracts import (
    FrozenForecastArtifact,
    FrozenPhysicalArtifact,
    NYISO_ZONES,
    ScenarioBundle,
)
from .hypotheses import aggregate_hypotheses
from .optimization import OptimizationFailure, POLICIES, save_optimization_result, solve_policy
from .scenarios import (
    DEFAULT_REGIMES,
    SCENARIO_SCHEMA_VERSION,
    DependenceRegime,
    build_scenario_bundle,
    mix_nominal_and_stress,
    scenario_diagnostics,
)
from .selection import select_primary_days
from .sensitivities import (
    REQUIRED_DIMENSIONS,
    apply_ofat_variant,
    build_ofat_variants,
    load_support_distances,
    variants_fingerprint,
)
from .statistics import (
    _moving_block_interval,
    _newey_west_test,
    build_statistical_inference,
)
from .xcel_counterfactual import (
    build_xcel_counterfactual_outputs,
    build_xcel_local_tariff_replay,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _sha256_json(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkCell:
    operating_date: str
    policy_id: str
    seed: int
    regime: str
    reduced_count: int
    constraint_variant: str = "base"
    phase: str = "primary_96_day"
    sensitivity_dimension: str = "reference"
    variant_id: str = "reference"
    variant_value: Any = "reference"

    @property
    def key(self) -> str:
        raw = (
            f"date={self.operating_date}__policy={self.policy_id}__seed={self.seed}__"
            f"regime={self.regime}__sensitivity={self.reduced_count}__variant={self.constraint_variant}"
        )
        if self.variant_id != "reference":
            full_identity = raw + f"__ofat={self.variant_id}"
            raw = (
                f"date={self.operating_date}__policy={self.policy_id}__seed={self.seed}__"
                f"ofat={hashlib.sha256(full_identity.encode('utf-8')).hexdigest()[:20]}"
            )
        return re.sub(r"[^A-Za-z0-9_.=-]+", "_", raw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "operating_date": self.operating_date,
            "policy_id": self.policy_id,
            "seed": self.seed,
            "regime": self.regime,
            "reduced_count": self.reduced_count,
            "constraint_variant": self.constraint_variant,
            "phase": self.phase,
            "sensitivity_dimension": self.sensitivity_dimension,
            "variant_id": self.variant_id,
            "variant_value": self.variant_value,
            "checkpoint_key": self.key,
        }


class CheckpointStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, cell: WorkCell) -> Path:
        return self.directory / f"{cell.key}.json"

    def read(self, cell: WorkCell) -> dict[str, Any] | None:
        path = self.path(cell)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def start(self, cell: WorkCell, fingerprint: str) -> dict[str, Any]:
        record = self.read(cell) or {
            "schema_version": "optimization_cell_checkpoint_v1",
            "cell": cell.as_dict(),
            "attempts": [],
        }
        if record.get("current_status") == "RUNNING" and record["attempts"]:
            interrupted_at = _utc_now()
            previous_attempt = record["attempts"][-1]
            if previous_attempt.get("status") == "RUNNING":
                previous_attempt.update(
                    {
                        "status": "INTERRUPTED",
                        "finished_at_utc": interrupted_at,
                        "diagnostics": {
                            "reason": "checkpoint was RUNNING when the cell was resumed"
                        },
                    }
                )
        record.update(
            {
                "current_status": "RUNNING",
                "fingerprint": fingerprint,
                "updated_at_utc": _utc_now(),
            }
        )
        record["attempts"].append(
            {
                "attempt_number": len(record["attempts"]) + 1,
                "started_at_utc": record["updated_at_utc"],
                "status": "RUNNING",
                "fingerprint": fingerprint,
            }
        )
        _atomic_json(self.path(cell), record)
        return record

    def finish(
        self,
        cell: WorkCell,
        record: dict[str, Any],
        *,
        status: str,
        result_path: Path | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        result_sha256 = (
            _file_sha256(result_path)
            if status == "SUCCESS" and result_path is not None and result_path.is_file()
            else None
        )
        record["current_status"] = status
        record["updated_at_utc"] = now
        record["result_path"] = str(result_path) if result_path else None
        record["result_sha256"] = result_sha256
        if diagnostics is not None:
            record["diagnostics"] = dict(diagnostics)
        attempt = record["attempts"][-1]
        attempt["status"] = status
        attempt["finished_at_utc"] = now
        attempt["result_path"] = str(result_path) if result_path else None
        attempt["result_sha256"] = result_sha256
        if diagnostics is not None:
            attempt["diagnostics"] = dict(diagnostics)
        _atomic_json(self.path(cell), record)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenForecastStore:
    """Small lazy index over per-day NPZ files or in-memory fixture entries."""

    def __init__(self, entries: Mapping[str, Any], root: Path) -> None:
        self._entries = dict(entries)
        self._root = root
        self._cached_date: str | None = None
        self._cached_artifact: FrozenForecastArtifact | None = None

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, operating_date: object) -> bool:
        return operating_date in self._entries

    def _resolve_path(self, raw: str | Path) -> Path:
        path = Path(raw)
        return path.resolve() if path.is_absolute() else (self._root / path).resolve()

    def _load_npz(self, path: Path, record: Mapping[str, Any]) -> FrozenForecastArtifact:
        if not path.exists():
            raise FileNotFoundError(f"frozen forecast day does not exist: {path}")
        actual_hash = _file_sha256(path)
        expected_hash = record.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"frozen forecast day hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            source_hashes = (
                json.loads(str(data["source_hashes_json"].item()))
                if "source_hashes_json" in data.files
                else dict(record.get("source_hashes", {}))
            )
            source_hashes = dict(source_hashes)
            source_hashes["frozen_day_npz"] = actual_hash
            payload = {
                "issue_time_utc": str(data["issue_time_utc"].item()),
                "operating_date": str(data["operating_date"].item()),
                "forecast_id": str(data["forecast_id"].item()),
                "timestamps_utc": data["timestamps_utc"],
                "da_price": data["da_price"],
                "rt_price_pool": data["rt_price_pool"],
                "workload_pool": data["workload_pool"],
                "pv_pool": data["pv_pool"],
                "zones": tuple(map(str, data["zones"].tolist())),
                "source_hashes": source_hashes,
                "locked_test": bool(data["locked_test"].item()) if "locked_test" in data.files else True,
            }
        return FrozenForecastArtifact.from_mapping(payload)

    def __getitem__(self, operating_date: str) -> FrozenForecastArtifact:
        if operating_date == self._cached_date and self._cached_artifact is not None:
            return self._cached_artifact
        raw = self._entries[operating_date]
        if isinstance(raw, FrozenForecastArtifact):
            artifact = raw
        elif isinstance(raw, (str, Path)):
            artifact = self._load_npz(self._resolve_path(raw), {})
        elif isinstance(raw, Mapping) and "path" in raw and not {
            "rt_price_pool",
            "workload_pool",
            "pv_pool",
        }.issubset(raw):
            artifact = self._load_npz(self._resolve_path(raw["path"]), raw)
        else:
            artifact = FrozenForecastArtifact.from_mapping(raw)
        if artifact.operating_date != operating_date:
            raise ValueError(
                f"forecast index key {operating_date} disagrees with artifact date {artifact.operating_date}"
            )
        self._cached_date = operating_date
        self._cached_artifact = artifact
        return artifact


def _read_forecast_index(raw_index: Any, root: Path) -> Mapping[str, Any]:
    if isinstance(raw_index, Mapping) and "index_path" in raw_index:
        raw_index = raw_index["index_path"]
    index_path: Path | None = None
    if isinstance(raw_index, (str, Path)):
        path = Path(raw_index)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path.is_dir():
            path = path / "index.json"
        index_path = path
        index = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(raw_index, Mapping):
        index = dict(raw_index)
    else:
        raise TypeError("frozen forecast index must be a path or mapping")
    status = index.get("status")
    if status is not None and str(status).upper() != "COMPLETE":
        raise ValueError(f"frozen forecast index is not COMPLETE: {status}")

    def normalize_entry(value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized: Any = dict(value)
            raw_path = normalized.get("path")
        else:
            normalized = value
            raw_path = value if isinstance(value, (str, Path)) else None
        if raw_path is not None and not Path(raw_path).is_absolute():
            repository_candidate = (root / raw_path).resolve()
            sibling_candidate = (
                (index_path.parent / raw_path).resolve() if index_path is not None else None
            )
            resolved = (
                repository_candidate
                if repository_candidate.exists() or sibling_candidate is None
                else sibling_candidate
            )
            if isinstance(normalized, dict):
                normalized["path"] = str(resolved)
            else:
                normalized = str(resolved)
        return normalized

    days = index.get("days", index)
    if isinstance(days, list):
        entries: dict[str, Any] = {}
        for record in days:
            operating_date = str(record["operating_date"])
            if operating_date in entries:
                raise ValueError(f"duplicate operating date in frozen forecast index: {operating_date}")
            entries[operating_date] = normalize_entry(record)
        declared_count = index.get("day_count")
        if declared_count is not None and int(declared_count) != len(entries):
            raise ValueError(
                f"frozen forecast index day_count={declared_count} disagrees with {len(entries)} records"
            )
        return entries
    if isinstance(days, Mapping):
        entries = {str(key): normalize_entry(value) for key, value in days.items()}
        declared_count = index.get("day_count")
        if declared_count is not None and int(declared_count) != len(entries):
            raise ValueError(
                f"frozen forecast index day_count={declared_count} disagrees with {len(entries)} records"
            )
        return entries
    raise TypeError("frozen forecast index days must be a list or mapping")


def _forecast_days(payload: Mapping[str, Any], root: Path) -> FrozenForecastStore:
    raw_index = payload.get(
        "frozen_day_index",
        payload.get("forecast_day_index", payload.get("frozen_days_path")),
    )
    if raw_index is not None:
        return FrozenForecastStore(_read_forecast_index(raw_index, root), root)
    raw_days = payload.get("frozen_days", payload.get("days"))
    if raw_days is None:
        single = FrozenForecastArtifact.from_mapping(payload)
        return FrozenForecastStore({single.operating_date: single}, root)
    if isinstance(raw_days, Mapping):
        entries = {str(key): value for key, value in raw_days.items()}
    elif isinstance(raw_days, list):
        entries = {}
        for raw in raw_days:
            if isinstance(raw, FrozenForecastArtifact):
                operating_date = raw.operating_date
            elif isinstance(raw, Mapping):
                operating_date = str(raw.get("operating_date", raw.get("frozen_artifact", {}).get("operating_date")))
            else:
                raise TypeError("forecast day list entries must be mappings or FrozenForecastArtifact")
            if not operating_date or operating_date == "None":
                raise ValueError("forecast day list entry lacks operating_date")
            entries[operating_date] = raw
    else:
        raise TypeError("forecast_result days/frozen_days must be a mapping or list")
    return FrozenForecastStore(entries, root)


def _resolve_regimes(values: Iterable[Any] | None) -> tuple[DependenceRegime, ...]:
    if values is None:
        return DEFAULT_REGIMES
    known = {regime.name: regime for regime in DEFAULT_REGIMES}
    output = []
    for value in values:
        if isinstance(value, Mapping):
            regime = DependenceRegime(str(value["name"]), float(value["kappa"]))
        else:
            name = str(value)
            if name not in known:
                raise ValueError(f"unknown dependence regime {name}")
            regime = known[name]
        output.append(regime)
    return tuple(output)


def _default_cells(
    operating_dates: Iterable[str],
    *,
    seeds: Iterable[int],
    counts: Iterable[int],
    regimes: Iterable[DependenceRegime],
    phase: str,
    policy_override: Iterable[str] | None = None,
    include_h6_variant: bool = False,
) -> list[WorkCell]:
    counts = tuple(map(int, counts))
    regime_names = {regime.name for regime in regimes}
    if policy_override is not None:
        policies = tuple(map(str, policy_override))
        return [
            WorkCell(day, policy, int(seed), regime, count, "base", phase)
            for day in operating_dates
            for seed in seeds
            for regime in sorted(regime_names)
            for count in counts
            for policy in policies
            if not (policy == "B7" and regime == "independent")
        ]

    cells: list[WorkCell] = []
    for day in operating_dates:
        for seed in seeds:
            for count in counts:
                if "independent" in regime_names:
                    nominal_policies = (
                        tuple(policy for policy in POLICIES if policy != "B7")
                        if count == 20
                        else ("B6",)
                    )
                    cells.extend(WorkCell(day, policy, int(seed), "independent", count, "base", phase) for policy in nominal_policies)
                    if count == 20 and include_h6_variant:
                        cells.append(WorkCell(day, "B6", int(seed), "independent", count, "ramp_concentration", phase))
                # Dependence-regime sweeps use the preregistered 20-scenario
                # reference size.  Scenario-count sensitivity is isolated to
                # nominal B6 so the plan is not an accidental Cartesian
                # product of policies x counts x dependence regimes.
                if count == 20:
                    for regime in sorted(regime_names - {"independent"}):
                        cells.append(WorkCell(day, "B6", int(seed), regime, count, "base", phase))
                        cells.append(WorkCell(day, "B7", int(seed), regime, count, "base", phase))
    return cells


def _ofat_cells(
    operating_dates: Iterable[str],
    *,
    seed: int,
    variants: Iterable[Mapping[str, Any]],
) -> list[WorkCell]:
    """Build sparse sensitivity cells for the 96-day primary sample only."""

    return [
        WorkCell(
            operating_date=str(day),
            policy_id=str(variant["policy_id"]),
            seed=int(seed),
            regime=str(variant["regime"]),
            reduced_count=int(variant["reduced_count"]),
            constraint_variant=str(variant.get("constraint_variant", "base")),
            phase="primary_sensitivity",
            sensitivity_dimension=str(variant["dimension"]),
            variant_id=str(variant["variant_id"]),
            variant_value=variant.get("value"),
        )
        for day in operating_dates
        for variant in variants
    ]


def _fingerprint(
    cell: WorkCell,
    physical: FrozenPhysicalArtifact,
    forecast: FrozenForecastArtifact,
    solve_settings: Mapping[str, Any],
) -> str:
    return _sha256_json(
        {
            "cell": cell.as_dict(),
            "calibration_id": physical.calibration_id,
            "twin_ids": physical.twin_ids,
            "physical_hashes": dict(physical.source_hashes),
            "physical_array_hashes": {
                "idle_mw": _array_sha256(physical.idle_mw),
                "work_mw_per_unit": _array_sha256(physical.work_mw_per_unit),
                "capacity_work_units": _array_sha256(physical.capacity_work_units),
                "grid_import_limit_mw": _array_sha256(
                    physical.grid_import_limit_mw
                ),
                "battery_power_mw": _array_sha256(physical.battery_power_mw),
                "battery_energy_mwh": _array_sha256(physical.battery_energy_mwh),
                "physical_multiplier_pool": _array_sha256(physical.physical_multiplier_pool),
            },
            "forecast_id": forecast.forecast_id,
            "forecast_hashes": dict(forecast.source_hashes),
            "forecast_array_hashes": {
                "da_price": _array_sha256(forecast.da_price),
                "rt_price_pool": _array_sha256(forecast.rt_price_pool),
                "workload_pool": _array_sha256(forecast.workload_pool),
                "pv_pool": _array_sha256(forecast.pv_pool),
            },
            "settings": dict(solve_settings),
            "schema": "optimization_branch_fingerprint_v1",
            "scenario_schema": SCENARIO_SCHEMA_VERSION,
            "runner_code_sha256": _file_sha256(Path(__file__)),
            "optimization_code_sha256": _file_sha256(
                Path(__file__).with_name("optimization.py")
            ),
            "rolling_code_sha256": _file_sha256(
                Path(__file__).with_name("rolling.py")
            ),
        }
    )


def _scenario_fingerprint(
    physical: FrozenPhysicalArtifact,
    forecast: FrozenForecastArtifact,
) -> str:
    return _sha256_json(
        {
            "calibration_id": physical.calibration_id,
            "physical_source_hashes": dict(physical.source_hashes),
            "physical_multiplier_pool": _array_sha256(physical.physical_multiplier_pool),
            "forecast_id": forecast.forecast_id,
            "forecast_source_hashes": dict(forecast.source_hashes),
            "da_price": _array_sha256(forecast.da_price),
            "rt_price_pool": _array_sha256(forecast.rt_price_pool),
            "workload_pool": _array_sha256(forecast.workload_pool),
            "pv_pool": _array_sha256(forecast.pv_pool),
            "scenario_schema": SCENARIO_SCHEMA_VERSION,
        }
    )


def _load_xcel_local_subcase(
    root: Path, physical_result: Mapping[str, Any]
) -> dict[str, Any]:
    extraction = root / "data" / "raw" / "xcel" / "extraction_status.json"
    if not extraction.exists():
        return {
            "status": "XCEL_PUBLIC_DATA_BLOCKED",
            "tariff_defined_counterfactual": True,
            "event_calls": 0,
            "reason": f"missing {extraction}",
        }
    try:
        paths = build_xcel_counterfactual_outputs(root)
    except Exception as exc:
        return {
            "status": "XCEL_PUBLIC_DATA_BLOCKED",
            "tariff_defined_counterfactual": True,
            "event_calls": 0,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    manifest_path = Path(paths["manifest"])
    parquet_path = Path(paths["parquet"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("event_calls") != 0 or not manifest.get("tariff_defined_counterfactual"):
        raise ValueError("Xcel local subcase manifest violates the zero-call counterfactual boundary")
    if _file_sha256(parquet_path) != manifest["outputs"]["parquet"]["sha256"]:
        raise ValueError("Xcel counterfactual Parquet hash does not match its manifest")
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    primary = frame.loc[frame["case_role"] == "primary_overlap_era_2024"]
    if set(primary["schedule"]) != {"SG-CPP", "PG-CPP", "TG-CPP"}:
        raise ValueError("Xcel primary subcase must contain SG-CPP, PG-CPP, and TG-CPP")
    if set(primary["effective_date"].astype(str)) != {"2024-05-01"}:
        raise ValueError("Xcel primary subcase must use the 2024-05-01 overlap-era snapshot")
    outputs = physical_result.get("outputs", {})
    output_hashes = physical_result.get("output_hashes", {})
    nlr300_raw = outputs.get("nlr300_primary") if isinstance(outputs, Mapping) else None
    if not nlr300_raw:
        return {
            "status": "XCEL_LOCAL_REPLAY_BLOCKED",
            "reason": "frozen physical result does not expose nlr300_primary",
            "event_calls": 0,
            "tariff_defined_counterfactual": True,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _file_sha256(manifest_path),
            "parquet_path": str(parquet_path),
            "parquet_sha256": _file_sha256(parquet_path),
        }
    nlr300_path = Path(str(nlr300_raw))
    if not nlr300_path.is_absolute():
        nlr300_path = root / nlr300_path
    try:
        replay = build_xcel_local_tariff_replay(
            root,
            nlr300_path,
            expected_nlr300_sha256=(
                str(output_hashes.get("nlr300_primary"))
                if isinstance(output_hashes, Mapping)
                and output_hashes.get("nlr300_primary")
                else None
            ),
        )
    except Exception as exc:
        return {
            "status": "XCEL_LOCAL_REPLAY_BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "event_calls": 0,
            "tariff_defined_counterfactual": True,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _file_sha256(manifest_path),
            "parquet_path": str(parquet_path),
            "parquet_sha256": _file_sha256(parquet_path),
        }
    return {
        "status": replay["status"],
        "case_role": "separate_local_tariff_subcase_not_in_nyiso_settlement_objective",
        "primary_effective_date": "2024-05-01",
        "primary_schedules": ["SG-CPP", "PG-CPP", "TG-CPP"],
        "event_calls": 0,
        "tariff_defined_counterfactual": True,
        "training_target": False,
        "realized_response_evidence": False,
        "evidence_role": "counterfactual",
        "source_relationship": "tariff_defined_counterfactual",
        "measurement_relationship": "not_measured_by_this_study",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "parquet_path": str(parquet_path),
        "parquet_sha256": _file_sha256(parquet_path),
        "primary_rate_records": json.loads(primary.to_json(orient="records")),
        "local_replay": replay,
    }


def _write_result_table(rows: list[dict[str, Any]], path: Path) -> Path:
    machine_evidence_fields = {
        "sensitivity_setting_keys_supported",
        "sensitivity_settings_consumed",
        "sensitivity_variant",
        "sensitivity_effective_inputs",
        "sensitivity_parameters",
        "transformed_scenario_hashes",
    }
    scalar_rows = []
    for row in rows:
        serialized: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            elif key in machine_evidence_fields:
                serialized[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        scalar_rows.append(serialized)
    fields = sorted({field for row in scalar_rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)
    temporary.replace(path)
    return path


def _base_solve_settings(
    settings: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    constraint_variant: str = "base",
) -> dict[str, Any]:
    """Resolve the common solver contract identically for every branch."""

    solve_settings = dict(settings.get("solve", {}))
    for field in (
        "solver",
        "cvar_alpha",
        "risk_weight",
        "time_limit_seconds",
        "mip_relative_gap",
        "solver_threads",
        "checkpoint_granularity_work_units",
        "checkpoint_rounding_tolerance_chunks",
        "deadline_steps",
        "interval_hours",
        "validation_tolerance",
        "flexible_share",
        "migration_eligible",
        "migration_bandwidth_base_work_units_per_interval",
        "migration_bandwidth_multiplier",
        "transfer_energy_mwh_per_work_unit",
        "checkpoint_overhead_fraction",
        "migration_energy_price_usd_per_mwh",
        "pue_slope_multiplier",
        "rt_sell_back_haircut",
    ):
        if field in settings:
            solve_settings[field] = settings[field]
    workload_config = config.get("workload", {})
    workload_config = (
        workload_config if isinstance(workload_config, Mapping) else {}
    )
    solve_settings.setdefault(
        "flexible_share", float(workload_config.get("flexible_share_cap", 0.30))
    )
    solve_settings.setdefault("migration_eligible", 1.0)
    solve_settings["constraint_variant"] = str(constraint_variant)
    return solve_settings


def _scenario_decision_value_subset(
    root: Path,
    branch_root: Path,
    scenario_root: Path,
    *,
    physical: FrozenPhysicalArtifact,
    forecasts: Mapping[str, FrozenForecastArtifact],
    primary_dates: list[str],
    sensitivity_seed: int,
    result_rows: list[Mapping[str, Any]],
    settings: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate K=20 and full-100 first-stage decisions on the same 100 draws."""

    requested = max(1, int(settings.get("scenario_decision_value_subset_days", 4)))
    requested = min(requested, len(primary_dates)) if primary_dates else requested
    indices = (
        sorted(set(np.linspace(0, len(primary_dates) - 1, requested).round().astype(int)))
        if primary_dates
        else []
    )
    selected_dates = [primary_dates[index] for index in indices]
    output_root = branch_root / "scenario_decision_value"
    output_root.mkdir(parents=True, exist_ok=True)
    by_date = {
        str(row.get("operating_date")): row
        for row in result_rows
        if row.get("phase") == "primary_96_day"
        and row.get("sensitivity_dimension", "reference") == "reference"
        and row.get("policy_id") == "B6"
        and row.get("regime") == "independent"
        and int(row.get("reduced_scenario_count", -1)) == 20
        and row.get("constraint_variant", "base") == "base"
        and int(row.get("study_seed", -1)) == int(sensitivity_seed)
    }

    base_settings = _base_solve_settings(settings, config)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for operating_date in selected_dates:
        checkpoint_path = output_root / f"{operating_date}.json"
        reduced_result = by_date.get(operating_date)
        if reduced_result is None or not reduced_result.get("day_ahead_schedule_mw"):
            failures.append(
                {
                    "operating_date": operating_date,
                    "status": "PENDING_PRIMARY_REFERENCE",
                    "reason": "the matching B6 independent K=20 study-seed result is unavailable",
                }
            )
            continue
        forecast = forecasts[operating_date]
        reduced_schedule = np.asarray(
            reduced_result["day_ahead_schedule_mw"], dtype=float
        )
        if reduced_schedule.ndim == 3:
            reduced_schedule = reduced_schedule[0]
        fingerprint = _sha256_json(
            {
                "schema": "scenario_reduction_decision_value_v1",
                "scenario_schema": SCENARIO_SCHEMA_VERSION,
                "operating_date": operating_date,
                "study_seed": sensitivity_seed,
                "forecast_id": forecast.forecast_id,
                "forecast_source_hashes": dict(forecast.source_hashes),
                "physical_source_hashes": dict(physical.source_hashes),
                "reduced_schedule_sha256": _array_sha256(reduced_schedule),
                "solve_settings": base_settings,
                "optimization_code_sha256": _file_sha256(
                    Path(__file__).with_name("optimization.py")
                ),
            }
        )
        existing = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.is_file()
            else None
        )
        if isinstance(existing, Mapping) and existing.get("status") == "COMPLETE" and existing.get("fingerprint") == fingerprint:
            artifact_path = Path(str(existing.get("full_100_scenario_path", "")))
            if (
                artifact_path.is_file()
                and existing.get("full_100_scenario_sha256")
                == _file_sha256(artifact_path)
            ):
                records.append(dict(existing))
                continue
        try:
            full_bundle = build_scenario_bundle(
                physical,
                forecast,
                regime=DEFAULT_REGIMES[0],
                reduced_count=100,
                study_seed=sensitivity_seed,
                decision_value_evaluation=True,
            )
            full_path = (
                scenario_root
                / operating_date
                / f"seed={sensitivity_seed}__regime=independent__decision_value_full100.npz"
            )
            full_bundle.save(full_path)
            full_solution = solve_policy(
                full_bundle, physical, "B6", dict(base_settings), xcel_tariff=None
            )
            full_schedule = np.asarray(
                full_solution["day_ahead_schedule_mw"], dtype=float
            )
            if full_schedule.ndim == 3:
                full_schedule = full_schedule[0]
            reduced_evaluation_settings = dict(base_settings)
            reduced_evaluation_settings["fixed_day_ahead_schedule_mw"] = (
                reduced_schedule
            )
            reduced_on_full = solve_policy(
                full_bundle,
                physical,
                "B6",
                reduced_evaluation_settings,
                xcel_tariff=None,
            )
            full_evaluation_settings = dict(base_settings)
            full_evaluation_settings["fixed_day_ahead_schedule_mw"] = full_schedule
            full_on_full = solve_policy(
                full_bundle,
                physical,
                "B6",
                full_evaluation_settings,
                xcel_tariff=None,
            )
            record = {
                "schema_version": "scenario_reduction_decision_value_v1",
                "status": "COMPLETE",
                "fingerprint": fingerprint,
                "operating_date": operating_date,
                "study_seed": int(sensitivity_seed),
                "reduced_scenario_count": 20,
                "evaluation_scenario_count": 100,
                "same_full_100_evaluation_distribution": True,
                "full_population_bundle_restricted_to_decision_value_evaluation": bool(
                    full_bundle.decision_value_evaluation
                ),
                "policy_id": "B6",
                "reduced_schedule_sha256": _array_sha256(reduced_schedule),
                "full_100_schedule_sha256": _array_sha256(full_schedule),
                "reduced_schedule_on_full_expected_cost_usd": float(
                    reduced_on_full["expected_total_cost_usd"]
                ),
                "full_100_schedule_on_full_expected_cost_usd": float(
                    full_on_full["expected_total_cost_usd"]
                ),
                "expected_cost_decision_value_gap_usd": float(
                    reduced_on_full["expected_total_cost_usd"]
                    - full_on_full["expected_total_cost_usd"]
                ),
                "reduced_schedule_on_full_cvar_usd": float(
                    reduced_on_full["cvar_total_cost_usd"]
                ),
                "full_100_schedule_on_full_cvar_usd": float(
                    full_on_full["cvar_total_cost_usd"]
                ),
                "cvar_decision_value_gap_usd": float(
                    reduced_on_full["cvar_total_cost_usd"]
                    - full_on_full["cvar_total_cost_usd"]
                ),
                "reduced_schedule_on_full_objective_usd": float(
                    reduced_on_full["objective_usd"]
                ),
                "full_100_schedule_on_full_objective_usd": float(
                    full_on_full["objective_usd"]
                ),
                "objective_decision_value_gap_usd": float(
                    reduced_on_full["objective_usd"]
                    - full_on_full["objective_usd"]
                ),
                "full_100_optimized_objective_usd": float(
                    full_solution["objective_usd"]
                ),
                "fixed_schedule_validation": {
                    "reduced": reduced_on_full.get("validations", {}),
                    "full_100": full_on_full.get("validations", {}),
                },
                "solver_evidence": {
                    name: {
                        "termination_condition": result.get("termination_condition"),
                        "mip_gap": result.get("mip_gap"),
                        "mip_gap_target": result.get("mip_gap_target"),
                    }
                    for name, result in (
                        ("full_100_optimization", full_solution),
                        ("reduced_schedule_on_full", reduced_on_full),
                        ("full_100_schedule_on_full", full_on_full),
                    )
                },
                "full_100_scenario_path": str(full_path),
                "full_100_scenario_sha256": _file_sha256(full_path),
                "reduced_result_checkpoint_key": reduced_result.get(
                    "checkpoint_key"
                ),
                "reduced_result_scenario_path": reduced_result.get("scenario_path"),
                "evidence_role": "derived_counterfactual_diagnostic",
                "measurement_relationship": "not_measured_by_this_study",
                "completed_at_utc": _utc_now(),
            }
            _atomic_json(checkpoint_path, record)
            records.append(record)
        except Exception as exc:
            failure = {
                "schema_version": "scenario_reduction_decision_value_v1",
                "status": "FAILED",
                "fingerprint": fingerprint,
                "operating_date": operating_date,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "updated_at_utc": _utc_now(),
            }
            _atomic_json(checkpoint_path, failure)
            failures.append(failure)
    completed = len(records) == len(selected_dates) and not failures
    manifest = {
        "schema_version": "scenario_reduction_decision_value_subset_v1",
        "status": "COMPLETE" if completed else "PARTIAL",
        "completed": completed,
        "method": "K20_and_full100_first_stage_schedules_evaluated_on_the_same_full100_distribution",
        "subset_selection": "evenly_spaced_over_preregistered_96_primary_dates",
        "requested_day_count": requested,
        "selected_dates": selected_dates,
        "completed_day_count": len(records),
        "failed_or_pending_day_count": len(failures),
        "study_seed": int(sensitivity_seed),
        "records": records,
        "failures": failures,
        "updated_at_utc": _utc_now(),
    }
    path = _atomic_json(
        branch_root / "scenario_decision_value_subset.json", manifest
    )
    return {
        "status": manifest["status"],
        "completed": completed,
        "path": str(path),
        "sha256": _file_sha256(path),
        "completed_day_count": len(records),
        "failed_or_pending_day_count": len(failures),
    }


def _run_exact_validation_gate(
    branch_root: Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Run/reuse the registered hand-checkable exact-vs-cohort gate."""

    from .exact_validation import (
        one_site_hand_checkable_case,
        run_exact_vs_cohort,
        two_site_hand_checkable_case,
    )

    path = branch_root / "exact_validation_gate.json"
    effective = {
        "solver": str(settings.get("solver", settings.get("solve", {}).get("solver", "highs"))),
        "time_limit_seconds": float(
            settings.get(
                "exact_validation_time_limit_seconds",
                min(float(settings.get("time_limit_seconds", 30.0)), 60.0),
            )
        ),
        "mip_relative_gap": float(
            settings.get("exact_validation_mip_relative_gap", 0.0)
        ),
    }
    fingerprint = _sha256_json(
        {
            "schema": "paper_exact_validation_gate_v1",
            "settings": effective,
            "implementation_sha256": _file_sha256(
                Path(__file__).with_name("exact_validation.py")
            ),
            "optimization_sha256": _file_sha256(
                Path(__file__).with_name("optimization.py")
            ),
        }
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("passed") is True
            and existing.get("fingerprint") == fingerprint
        ):
            return {
                "status": "COMPLETE",
                "passed": True,
                "path": str(path),
                "sha256": _file_sha256(path),
                "resumed": True,
            }
    cases = [
        run_exact_vs_cohort(
            case,
            solver=effective["solver"],
            time_limit_seconds=effective["time_limit_seconds"],
            mip_relative_gap=effective["mip_relative_gap"],
        )
        for case in (one_site_hand_checkable_case(), two_site_hand_checkable_case())
    ]
    passed = bool(cases and all(case.get("passed") is True for case in cases))
    manifest = {
        "schema_version": "paper_exact_validation_gate_v1",
        "status": "COMPLETE" if passed else "FAILED",
        "passed": passed,
        "fingerprint": fingerprint,
        "settings": effective,
        "case_count": len(cases),
        "cases": cases,
        "required_before_formal_rolling_optimization": True,
        "updated_at_utc": _utc_now(),
    }
    _atomic_json(path, manifest)
    return {
        "status": manifest["status"],
        "passed": passed,
        "path": str(path),
        "sha256": _file_sha256(path),
        "resumed": False,
    }


def _rolling_scalar_result(
    rolling: Mapping[str, Any],
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    policy_id: str,
    settings: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    if rolling.get("status") != "COMPLETE":
        raise OptimizationFailure(
            f"rolling day is {rolling.get('status', 'missing')}",
            {"checkpoint_path": str(checkpoint_path), "rolling": dict(rolling)},
        )
    metrics = rolling["metrics"]
    economic = metrics["economic"]
    risk = metrics["risk"]
    service = metrics["service"]
    grid = metrics["flexibility_and_grid"]
    computation = metrics["computation"]
    if computation.get("all_interval_validations_passed") is not True:
        raise OptimizationFailure(
            "one or more rolling interval validations failed",
            {"checkpoint_path": str(checkpoint_path), "computation": computation},
        )
    solve_records = [rolling.get("day_ahead_solve", {})] + [
        item.get("solver", {}) for item in rolling.get("intervals", [])
    ]
    terminations = {
        str(record.get("termination_condition", "missing")).lower()
        for record in solve_records
    }
    if terminations - {"optimal"}:
        raise OptimizationFailure(
            "rolling solver termination is not optimal",
            {"termination_conditions": sorted(terminations)},
        )
    gaps = [
        float(record["mip_gap"])
        for record in solve_records
        if record.get("mip_gap") is not None
    ]
    total_grid_energy = 0.25 * sum(
        float(np.asarray(item["dispatch"]["grid_import_mw"], dtype=float).sum())
        for item in rolling.get("intervals", [])
    )
    day_ahead = rolling.get("day_ahead_solve", {})
    return {
        "operating_date": bundle.operating_date,
        "policy_id": policy_id,
        "regime": bundle.regime,
        "reduced_scenario_count": bundle.reduced_count,
        "scenario_seed": bundle.seed,
        "objective_usd": float(economic["simulated_total_cost_usd"]),
        "expected_total_cost_usd": float(economic["simulated_total_cost_usd"]),
        "expected_total_cost_metric_role": "realized_counterfactual_daily_cost_under_causal_15_minute_rolling_replay",
        "cvar_total_cost_usd": float(risk["empirical_cvar_95_usd"]),
        "expected_sla_shortfall_work_units": float(
            service["deadline_missed_work_units"]
        ),
        "expected_grid_energy_mwh": float(total_grid_energy),
        "rebound_peak_mw": float(grid["rebound_after_low_price_intervals_mw"]),
        "maximum_zone_concentration_share": float(
            grid["maximum_zone_concentration_share"]
        ),
        "day_ahead_schedule_mw": rolling["day_ahead_schedule_mw"],
        "scenario_costs_usd": rolling["ex_ante_risk"]["scenario_costs_usd"],
        "scenario_probabilities": rolling["ex_ante_risk"][
            "scenario_probabilities"
        ],
        "cvar_alpha": float(rolling["ex_ante_risk"]["cvar_alpha"]),
        "termination_condition": "optimal",
        "solver_status": "ok",
        "mip_gap": max(gaps, default=0.0),
        "mip_gap_target": float(
            settings.get(
                "mip_relative_gap", settings.get("mip_gap", 0.01)
            )
        ),
        "solve_time_seconds": float(computation["solve_time_seconds"]),
        "time_limit_reached": bool(computation["timeout_count"]),
        "number_of_variables": int(computation["variables_solved_total"]),
        "number_of_constraints": int(computation["constraints_solved_total"]),
        "validations": {
            "passed": True,
            "rolling_all_interval_validations_passed": True,
            "future_realized_targets_used": False,
            "maximum_validation_residual": float(
                computation["maximum_validation_residual"]
            ),
        },
        "rolling_operation": True,
        "rolling_interval_minutes": 15,
        "rolling_completed_interval_count": int(
            rolling["completed_interval_count"]
        ),
        "rolling_realized_scenario_index": int(
            rolling["realized_scenario_index"]
        ),
        "rolling_checkpoint_path": str(checkpoint_path),
        "rolling_checkpoint_sha256": _file_sha256(checkpoint_path),
        "future_realized_targets_used": False,
        "sensitivity_setting_keys_supported": day_ahead.get(
            "sensitivity_setting_keys_supported", []
        ),
        "sensitivity_settings_consumed": day_ahead.get(
            "sensitivity_settings_consumed", {}
        ),
        "sensitivity_variant": day_ahead.get("sensitivity_variant"),
        "sensitivity_effective_inputs": day_ahead.get(
            "sensitivity_effective_inputs", {}
        ),
        "counterfactual": True,
        "evidence_layer": "counterfactual",
        "calibration_id": physical.calibration_id,
        "twin_ids": list(physical.twin_ids),
        "transfer_regime": physical.transfer_regime,
        "zones": list(bundle.zones),
        "nyiso_evidence_role": "counterfactual",
        "nyiso_source_relationship": "counterfactual_using_third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
        "xcel_tariff_counterfactual": None,
    }


def _fixed_mst_physical_artifact(
    root: Path,
    physical_result: Mapping[str, Any],
    primary_physical: FrozenPhysicalArtifact,
) -> tuple[FrozenPhysicalArtifact, dict[str, Any]]:
    """Rebuild the transferred twin from the accepted fixed-MST surrogate.

    Capacity, hardware, storage, and grid assumptions remain the same frozen
    transfer assumptions.  Only the clock-dependent, independently fitted and
    accepted physical surrogate supplies idle and dynamic facility power.
    """

    outputs = physical_result.get("outputs", {})
    hashes = physical_result.get("output_hashes", {})
    if not isinstance(outputs, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("physical result lacks output/hash mappings")

    def verified_output(name: str) -> tuple[Path, str]:
        raw = outputs.get(name)
        expected = hashes.get(name)
        if not raw or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"physical {name} path or SHA-256 is missing")
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _file_sha256(path)
        if observed != expected:
            raise ValueError(f"physical {name} SHA-256 mismatch")
        return path, observed

    calibration_path, calibration_sha = verified_output("physical_manifest")
    transfer_path, transfer_sha = verified_output("transfer_table")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    fixed = calibration.get("fixed_mst_sensitivity", {})
    if not isinstance(fixed, Mapping):
        raise ValueError("physical calibration has no fixed-MST sensitivity")
    surrogate = fixed.get("surrogate", {})
    acceptance = fixed.get(
        "surrogate_acceptance",
        surrogate.get("acceptance", {}) if isinstance(surrogate, Mapping) else {},
    )
    if (
        not isinstance(surrogate, Mapping)
        or surrogate.get("frozen") is not True
        or not isinstance(acceptance, Mapping)
        or acceptance.get("accepted_for_optimization") is not True
    ):
        raise ValueError("fixed-MST surrogate is not frozen and accepted for optimization")
    interpretation = str(fixed.get("clock_interpretation", ""))
    if "fixed_mst" not in interpretation.lower():
        raise ValueError("fixed-MST calibration clock interpretation is absent")

    transfer = pd.read_parquet(transfer_path)
    required_columns = {
        "zone",
        "capacity_scale",
        "hardware_multiplier",
        "grid_import_limit_mw",
        "battery_power_mw",
        "battery_energy_mwh",
        "idle_mw",
        "work_mw_per_unit",
    }
    if not required_columns.issubset(transfer.columns):
        raise ValueError(
            f"transfer table lacks columns: {sorted(required_columns - set(transfer.columns))}"
        )
    transfer["zone"] = transfer["zone"].astype(str)
    if transfer["zone"].duplicated().any() or set(transfer["zone"]) != set(
        NYISO_ZONES
    ):
        raise ValueError("transfer table must contain one row per registered NYISO zone")
    transfer = transfer.set_index("zone").loc[list(NYISO_ZONES)]
    numeric_columns = sorted(required_columns - {"zone"})
    for column in numeric_columns:
        transfer[column] = pd.to_numeric(transfer[column], errors="raise")

    idle_base_mw = float(surrogate["idle_facility_power_kw"]) / 1000.0
    peak_base_mw = float(surrogate["peak_facility_power_kw"]) / 1000.0
    dynamic_base_mw = peak_base_mw - idle_base_mw
    if idle_base_mw < 0 or dynamic_base_mw <= 0:
        raise ValueError("fixed-MST surrogate has invalid idle/peak power")
    capacity_scale = transfer["capacity_scale"].to_numpy(dtype=float)
    hardware = transfer["hardware_multiplier"].to_numpy(dtype=float)
    fixed_idle = idle_base_mw * capacity_scale * hardware
    fixed_work = dynamic_base_mw * hardware
    fixed_capacity = capacity_scale
    fixed_grid = transfer["grid_import_limit_mw"].to_numpy(dtype=float)
    fixed_battery_power = transfer["battery_power_mw"].to_numpy(dtype=float)
    fixed_battery_energy = transfer["battery_energy_mwh"].to_numpy(dtype=float)

    # Confirm that the transfer table is the exact source of the primary
    # frozen transfer assumptions before substituting the alternative clock.
    for label, observed, expected in (
        ("primary idle", primary_physical.idle_mw, transfer["idle_mw"]),
        (
            "primary work coefficient",
            primary_physical.work_mw_per_unit,
            transfer["work_mw_per_unit"],
        ),
        ("capacity", primary_physical.capacity_work_units, fixed_capacity),
        ("grid limit", primary_physical.grid_import_limit_mw, fixed_grid),
        ("battery power", primary_physical.battery_power_mw, fixed_battery_power),
        ("battery energy", primary_physical.battery_energy_mwh, fixed_battery_energy),
    ):
        if not np.allclose(
            np.asarray(observed, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(f"{label} disagrees with the hash-verified transfer table")

    fixed_identity = _sha256_json(
        {
            "clock_interpretation": interpretation,
            "calibration_manifest_sha256": calibration_sha,
            "transfer_table_sha256": transfer_sha,
            "surrogate": dict(surrogate),
            "capacity_scale": capacity_scale.tolist(),
            "hardware_multiplier": hardware.tolist(),
        }
    )
    source_hashes = dict(primary_physical.source_hashes)
    source_hashes.update(
        {
            "fixed_mst_physical_calibration_manifest": calibration_sha,
            "fixed_transfer_table": transfer_sha,
            "fixed_mst_surrogate": _sha256_json(dict(surrogate)),
        }
    )
    fixed_physical = replace(
        primary_physical,
        calibration_id=f"fixed-mst-{fixed_identity[:20]}",
        twin_ids=tuple(f"{value}-fixed-mst" for value in primary_physical.twin_ids),
        transfer_regime="fixed_mst_clock_sensitivity_frozen_transfer",
        idle_mw=np.asarray(fixed_idle, dtype=float),
        work_mw_per_unit=np.asarray(fixed_work, dtype=float),
        capacity_work_units=np.asarray(fixed_capacity, dtype=float),
        grid_import_limit_mw=np.asarray(fixed_grid, dtype=float),
        battery_power_mw=np.asarray(fixed_battery_power, dtype=float),
        battery_energy_mwh=np.asarray(fixed_battery_energy, dtype=float),
        source_hashes=source_hashes,
    )
    evidence = {
        "method": "rebuild_transfer_from_independently_fitted_fixed_MST_accepted_surrogate",
        "clock_interpretation": interpretation,
        "surrogate_frozen": True,
        "surrogate_accepted_for_optimization": True,
        "residual_multiplier_ensemble_policy": "hold_primary_frozen_train_validation_ensemble_fixed_to_isolate_source_clock",
        "transfer_assumptions_policy": "same_hash_verified_capacity_hardware_grid_and_storage_parameters",
        "calibration_manifest_path": str(calibration_path),
        "calibration_manifest_sha256": calibration_sha,
        "transfer_table_path": str(transfer_path),
        "transfer_table_sha256": transfer_sha,
        "fixed_mst_surrogate_sha256": source_hashes["fixed_mst_surrogate"],
        "fixed_calibration_id": fixed_physical.calibration_id,
        "idle_mw": fixed_physical.idle_mw.tolist(),
        "work_mw_per_unit": fixed_physical.work_mw_per_unit.tolist(),
        "capacity_work_units": fixed_physical.capacity_work_units.tolist(),
        "grid_import_limit_mw": fixed_physical.grid_import_limit_mw.tolist(),
        "battery_power_mw": fixed_physical.battery_power_mw.tolist(),
        "battery_energy_mwh": fixed_physical.battery_energy_mwh.tolist(),
    }
    return fixed_physical, evidence


def _run_timezone_economic_sensitivity(
    root: Path,
    branch_root: Path,
    *,
    primary_physical: FrozenPhysicalArtifact,
    physical_result: Mapping[str, Any],
    primary_dates: list[str],
    seeds: tuple[int, ...],
    result_rows: list[Mapping[str, Any]],
    settings: Mapping[str, Any],
    config: Mapping[str, Any],
    allow_execution: bool,
    remaining_cell_budget: int | None,
    deadline_monotonic: float | None,
) -> dict[str, Any]:
    """Paired primary-vs-fixed-MST causal economic replay with checkpoints."""

    from .rolling import run_rolling_day

    sensitivity_root = branch_root / "timezone_sensitivity"
    checkpoint_root = sensitivity_root / "checkpoints"
    result_root = sensitivity_root / "results"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = branch_root / "timezone_sensitivity.json"
    paired_path = sensitivity_root / "paired_economic_endpoints.csv"
    fixed_artifact_path = sensitivity_root / "fixed_mst_physical_artifact.json"
    policies = ("B0", "B6")
    expected_keys = [
        (str(day), int(seed), policy)
        for day in primary_dates
        for seed in seeds
        for policy in policies
    ]
    base_settings = _base_solve_settings(settings, config)
    base_settings["continue_on_noncritical_failure"] = False
    fixed_physical, fixed_evidence = _fixed_mst_physical_artifact(
        root, physical_result, primary_physical
    )
    fixed_artifact_payload = {
        "schema_version": "fixed_mst_transferred_physical_artifact_v1",
        "frozen": True,
        "counterfactual": True,
        "calibration_id": fixed_physical.calibration_id,
        "twin_ids": list(fixed_physical.twin_ids),
        "transfer_regime": fixed_physical.transfer_regime,
        "idle_mw": fixed_physical.idle_mw.tolist(),
        "work_mw_per_unit": fixed_physical.work_mw_per_unit.tolist(),
        "capacity_work_units": fixed_physical.capacity_work_units.tolist(),
        "grid_import_limit_mw": fixed_physical.grid_import_limit_mw.tolist(),
        "battery_power_mw": fixed_physical.battery_power_mw.tolist(),
        "battery_energy_mwh": fixed_physical.battery_energy_mwh.tolist(),
        "physical_multiplier_pool_sha256": _array_sha256(
            fixed_physical.physical_multiplier_pool
        ),
        "source_hashes": dict(fixed_physical.source_hashes),
        "evidence": fixed_evidence,
        "source_relationship": "derived_from_third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
        "evidence_role": "transferred_nyiso_twin_counterfactual",
    }
    _atomic_json(fixed_artifact_path, fixed_artifact_payload)
    fixed_artifact_sha = _file_sha256(fixed_artifact_path)

    primary_rows = [
        dict(row)
        for row in result_rows
        if row.get("phase") == "primary_96_day"
        and row.get("sensitivity_dimension", "reference") == "reference"
        and row.get("regime") == "independent"
        and int(row.get("reduced_scenario_count", -1)) == 20
        and row.get("constraint_variant", "base") == "base"
        and row.get("policy_id") in policies
    ]
    primary_index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in primary_rows:
        identity = (
            str(row.get("operating_date")),
            int(row.get("study_seed", -1)),
            str(row.get("policy_id")),
        )
        if identity in primary_index:
            raise ValueError(f"duplicate primary pooled endpoint: {identity}")
        primary_index[identity] = row
    processed = 0
    cell_records: list[dict[str, Any]] = []
    fixed_results: dict[tuple[str, int, str], dict[str, Any]] = {}

    def path_for(root_path: Path, day: str, seed: int, policy: str) -> Path:
        return root_path / day / f"seed={seed}__policy={policy}.json"

    def persist_idle_record(
        checkpoint_path: Path,
        *,
        day: str,
        seed: int,
        policy: str,
        status: str,
        reason: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.is_file()
            else {}
        )
        record = {
            "schema_version": "fixed_mst_economic_sensitivity_cell_v1",
            "operating_date": day,
            "study_seed": seed,
            "policy_id": policy,
            "status": status,
            "reason": reason,
            "attempts": list(previous.get("attempts", [])),
            "updated_at_utc": _utc_now(),
            **dict(extra or {}),
        }
        _atomic_json(checkpoint_path, record)
        return record

    for day, seed, policy in expected_keys:
        key = (day, seed, policy)
        checkpoint_path = path_for(checkpoint_root, day, seed, policy)
        result_path = path_for(result_root, day, seed, policy)
        rolling_path = (
            sensitivity_root
            / "rolling_checkpoints"
            / day
            / f"seed={seed}__policy={policy}.json"
        )
        primary = primary_index.get(key)
        if primary is None:
            cell_records.append(
                persist_idle_record(
                    checkpoint_path,
                    day=day,
                    seed=seed,
                    policy=policy,
                    status="PENDING_PRIMARY_REFERENCE",
                    reason="hash-verified primary reference endpoint is not complete",
                    extra={"checkpoint_path": str(checkpoint_path)},
                )
            )
            continue
        primary_key = str(primary.get("checkpoint_key", ""))
        primary_result_path = (
            branch_root / "results" / day / f"{primary_key}.json"
        )
        scenario_path = Path(str(primary.get("scenario_path", "")))
        if not scenario_path.is_absolute():
            scenario_path = root / scenario_path
        if not primary_result_path.is_file() or not scenario_path.is_file():
            cell_records.append(
                persist_idle_record(
                    checkpoint_path,
                    day=day,
                    seed=seed,
                    policy=policy,
                    status="FAILED_PRIMARY_EVIDENCE",
                    reason="primary result or scenario artifact is missing",
                    extra={
                        "checkpoint_path": str(checkpoint_path),
                        "primary_result_path": str(primary_result_path),
                        "scenario_path": str(scenario_path),
                    },
                )
            )
            continue
        primary_result_sha = _file_sha256(primary_result_path)
        scenario_sha = _file_sha256(scenario_path)
        fingerprint = _sha256_json(
            {
                "schema": "fixed_mst_economic_sensitivity_cell_v1",
                "operating_date": day,
                "study_seed": seed,
                "policy_id": policy,
                "primary_result_sha256": primary_result_sha,
                "scenario_sha256": scenario_sha,
                "fixed_physical_artifact_sha256": fixed_artifact_sha,
                "solve_settings": base_settings,
                "rolling_code_sha256": _file_sha256(
                    Path(__file__).with_name("rolling.py")
                ),
                "optimization_code_sha256": _file_sha256(
                    Path(__file__).with_name("optimization.py")
                ),
            }
        )
        existing = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.is_file()
            else {}
        )
        reusable = bool(
            existing.get("status") == "COMPLETE"
            and existing.get("fingerprint") == fingerprint
            and result_path.is_file()
            and rolling_path.is_file()
            and existing.get("fixed_result_sha256") == _file_sha256(result_path)
            and existing.get("rolling_checkpoint_sha256") == _file_sha256(rolling_path)
            and existing.get("primary_result_sha256") == primary_result_sha
            and existing.get("scenario_sha256") == scenario_sha
        )
        if reusable:
            fixed_results[key] = json.loads(result_path.read_text(encoding="utf-8"))
            cell_records.append(dict(existing))
            continue
        matching_failed_attempts = sum(
            attempt.get("fingerprint") == fingerprint
            and attempt.get("status") == "FAILED"
            for attempt in existing.get("attempts", [])
        )
        if matching_failed_attempts >= max(
            1, int(settings.get("max_attempts_per_cell", 3))
        ):
            exhausted_record = {
                **dict(existing),
                "status": "FAILED",
                "fingerprint": fingerprint,
                "reason": "maximum attempts reached for unchanged fingerprint",
                "updated_at_utc": _utc_now(),
            }
            _atomic_json(checkpoint_path, exhausted_record)
            cell_records.append(exhausted_record)
            continue
        budget_exhausted = (
            remaining_cell_budget is not None
            and processed >= int(remaining_cell_budget)
        )
        time_exhausted = (
            deadline_monotonic is not None and time.monotonic() >= deadline_monotonic
        )
        if not allow_execution or budget_exhausted or time_exhausted:
            cell_records.append(
                persist_idle_record(
                    checkpoint_path,
                    day=day,
                    seed=seed,
                    policy=policy,
                    status="PENDING",
                    reason=(
                        "primary gate not passed"
                        if not allow_execution
                        else "invocation resource limit reached"
                    ),
                    extra={
                        "checkpoint_path": str(checkpoint_path),
                        "fingerprint": fingerprint,
                        "primary_result_path": str(primary_result_path),
                        "primary_result_sha256": primary_result_sha,
                        "scenario_path": str(scenario_path),
                        "scenario_sha256": scenario_sha,
                        "fixed_physical_artifact_path": str(fixed_artifact_path),
                        "fixed_physical_artifact_sha256": fixed_artifact_sha,
                        "rolling_checkpoint_path": str(rolling_path),
                    },
                )
            )
            continue

        attempts = list(existing.get("attempts", []))
        attempts.append(
            {
                "attempt_number": len(attempts) + 1,
                "started_at_utc": _utc_now(),
                "status": "RUNNING",
                "fingerprint": fingerprint,
            }
        )
        running_record = {
            "schema_version": "fixed_mst_economic_sensitivity_cell_v1",
            "operating_date": day,
            "study_seed": seed,
            "policy_id": policy,
            "status": "RUNNING",
            "fingerprint": fingerprint,
            "attempts": attempts,
            "checkpoint_path": str(checkpoint_path),
            "primary_result_path": str(primary_result_path),
            "primary_result_sha256": primary_result_sha,
            "scenario_path": str(scenario_path),
            "scenario_sha256": scenario_sha,
            "fixed_physical_artifact_path": str(fixed_artifact_path),
            "fixed_physical_artifact_sha256": fixed_artifact_sha,
            "rolling_checkpoint_path": str(rolling_path),
            "updated_at_utc": _utc_now(),
        }
        _atomic_json(checkpoint_path, running_record)
        processed += 1
        try:
            bundle = ScenarioBundle.load(scenario_path)
            if not (
                bundle.operating_date == day
                and bundle.regime == "independent"
                and bundle.reduced_count == 20
                and int(primary.get("scenario_seed", bundle.seed)) == bundle.seed
            ):
                raise ValueError("primary scenario bundle identity mismatch")
            if primary.get("rolling_operation") is not True:
                raise ValueError("primary endpoint is not a causal rolling result")
            realized_index = int(primary["rolling_realized_scenario_index"])
            rolling = run_rolling_day(
                bundle,
                fixed_physical,
                policy,
                base_settings,
                realized_scenario_index=realized_index,
                checkpoint_path=rolling_path,
            )
            result = _rolling_scalar_result(
                rolling,
                bundle,
                fixed_physical,
                policy,
                base_settings,
                rolling_path,
            )
            result.update(
                {
                    "run_id": str(primary.get("run_id", "")),
                    "study_seed": seed,
                    "phase": "timezone_fixed_mst_sensitivity",
                    "clock_interpretation": fixed_evidence["clock_interpretation"],
                    "primary_result_path": str(primary_result_path),
                    "primary_result_sha256": primary_result_sha,
                    "scenario_path": str(scenario_path),
                    "scenario_sha256": scenario_sha,
                    "fixed_physical_artifact_path": str(fixed_artifact_path),
                    "fixed_physical_artifact_sha256": fixed_artifact_sha,
                    "same_forecast_scenario_seed_as_primary": True,
                }
            )
            save_optimization_result(result, result_path)
            attempts[-1].update(
                {"status": "COMPLETE", "finished_at_utc": _utc_now()}
            )
            complete_record = {
                **running_record,
                "status": "COMPLETE",
                "attempts": attempts,
                "fixed_result_path": str(result_path),
                "fixed_result_sha256": _file_sha256(result_path),
                "rolling_checkpoint_sha256": _file_sha256(rolling_path),
                "updated_at_utc": _utc_now(),
            }
            _atomic_json(checkpoint_path, complete_record)
            fixed_results[key] = result
            cell_records.append(complete_record)
        except Exception as exc:
            attempts[-1].update(
                {
                    "status": "FAILED",
                    "finished_at_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            failed_record = {
                **running_record,
                "status": "FAILED",
                "attempts": attempts,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "updated_at_utc": _utc_now(),
            }
            _atomic_json(checkpoint_path, failed_record)
            cell_records.append(failed_record)

    paired_rows: list[dict[str, Any]] = []
    for day in primary_dates:
        for seed in seeds:
            primary_b0 = primary_index.get((day, seed, "B0"))
            primary_b6 = primary_index.get((day, seed, "B6"))
            fixed_b0 = fixed_results.get((day, seed, "B0"))
            fixed_b6 = fixed_results.get((day, seed, "B6"))
            if not all((primary_b0, primary_b6, fixed_b0, fixed_b6)):
                continue
            primary_scenario_paths = {
                str(primary_b0.get("scenario_path", "")),
                str(primary_b6.get("scenario_path", "")),
            }
            fixed_scenario_hashes = {
                str(fixed_b0.get("scenario_sha256", "")),
                str(fixed_b6.get("scenario_sha256", "")),
            }
            if len(primary_scenario_paths) != 1 or "" in primary_scenario_paths:
                raise ValueError(
                    f"B0/B6 primary endpoints do not share one scenario artifact: {day}/{seed}"
                )
            primary_scenario_path = Path(next(iter(primary_scenario_paths)))
            if not primary_scenario_path.is_absolute():
                primary_scenario_path = root / primary_scenario_path
            primary_scenario_sha = _file_sha256(primary_scenario_path)
            if fixed_scenario_hashes != {primary_scenario_sha}:
                raise ValueError(
                    f"fixed-MST B0/B6 endpoints do not reuse the primary scenario artifact: {day}/{seed}"
                )
            primary_value = float(primary_b0["expected_total_cost_usd"]) - float(
                primary_b6["expected_total_cost_usd"]
            )
            fixed_value = float(fixed_b0["expected_total_cost_usd"]) - float(
                fixed_b6["expected_total_cost_usd"]
            )
            primary_cvar_value = float(primary_b0["cvar_total_cost_usd"]) - float(
                primary_b6["cvar_total_cost_usd"]
            )
            fixed_cvar_value = float(fixed_b0["cvar_total_cost_usd"]) - float(
                fixed_b6["cvar_total_cost_usd"]
            )
            paired_rows.append(
                {
                    "operating_date": day,
                    "study_seed": seed,
                    "primary_b0_expected_cost_usd": float(
                        primary_b0["expected_total_cost_usd"]
                    ),
                    "primary_b6_expected_cost_usd": float(
                        primary_b6["expected_total_cost_usd"]
                    ),
                    "fixed_mst_b0_expected_cost_usd": float(
                        fixed_b0["expected_total_cost_usd"]
                    ),
                    "fixed_mst_b6_expected_cost_usd": float(
                        fixed_b6["expected_total_cost_usd"]
                    ),
                    "primary_policy_value_usd": primary_value,
                    "fixed_mst_policy_value_usd": fixed_value,
                    "clock_effect_on_policy_value_usd": fixed_value
                    - primary_value,
                    "primary_cvar_policy_value_usd": primary_cvar_value,
                    "fixed_mst_cvar_policy_value_usd": fixed_cvar_value,
                    "clock_effect_on_cvar_policy_value_usd": fixed_cvar_value
                    - primary_cvar_value,
                    "same_forecast_scenario_seed": True,
                    "scenario_artifact_path": str(primary_scenario_path),
                    "scenario_artifact_sha256": primary_scenario_sha,
                    "primary_result_evidence_json": json.dumps(
                        [
                            {
                                "path": str(
                                    branch_root
                                    / "results"
                                    / day
                                    / f"{primary_b0['checkpoint_key']}.json"
                                ),
                                "sha256": _file_sha256(
                                    branch_root
                                    / "results"
                                    / day
                                    / f"{primary_b0['checkpoint_key']}.json"
                                ),
                            },
                            {
                                "path": str(
                                    branch_root
                                    / "results"
                                    / day
                                    / f"{primary_b6['checkpoint_key']}.json"
                                ),
                                "sha256": _file_sha256(
                                    branch_root
                                    / "results"
                                    / day
                                    / f"{primary_b6['checkpoint_key']}.json"
                                ),
                            },
                        ],
                        sort_keys=True,
                    ),
                    "fixed_mst_result_evidence_json": json.dumps(
                        [
                            {
                                "path": str(path_for(result_root, day, seed, "B0")),
                                "sha256": _file_sha256(
                                    path_for(result_root, day, seed, "B0")
                                ),
                            },
                            {
                                "path": str(path_for(result_root, day, seed, "B6")),
                                "sha256": _file_sha256(
                                    path_for(result_root, day, seed, "B6")
                                ),
                            },
                        ],
                        sort_keys=True,
                    ),
                }
            )
    paired_table = _write_result_table(paired_rows, paired_path)
    paired_frame = pd.DataFrame(paired_rows)
    inference_config = settings.get("statistical_inference", {})
    inference_config = (
        inference_config if isinstance(inference_config, Mapping) else {}
    )
    block_length = int(inference_config.get("moving_block_length_days", 7))
    replicates = int(inference_config.get("bootstrap_replicates", 2000))
    confidence_level = float(inference_config.get("confidence_level", 0.95))
    alpha = float(inference_config.get("holm_alpha", 0.05))
    economic_threshold = float(
        settings.get(
            "timezone_economic_significance_fraction",
            settings.get("acceptable_expected_cost_premium_fraction", 0.05),
        )
    )
    if not paired_frame.empty:
        daily = paired_frame.groupby("operating_date", sort=True).mean(
            numeric_only=True
        )

        def endpoint_inference(
            endpoint_id: str,
            primary_column: str,
            fixed_column: str,
            effect_column: str,
        ) -> dict[str, Any]:
            values = daily[effect_column].to_numpy(dtype=float)
            lower, upper = _moving_block_interval(
                values,
                block_length=block_length,
                replicates=replicates,
                confidence_level=confidence_level,
                seed=int.from_bytes(
                    hashlib.sha256(
                        f"{branch_root.name}|fixed-mst-timezone|{endpoint_id}".encode(
                            "utf-8"
                        )
                    ).digest()[:8],
                    "big",
                ),
            )
            statistic, standard_error, p_value = _newey_west_test(
                values, max(block_length - 1, 0)
            )
            primary_mean = float(daily[primary_column].mean())
            fixed_mean = float(daily[fixed_column].mean())
            effect_mean = float(values.mean())
            effect_fraction = abs(effect_mean) / max(abs(primary_mean), 1e-9)
            direction_changed = bool(
                np.sign(primary_mean) != np.sign(fixed_mean)
                and not np.isclose(primary_mean, fixed_mean, atol=1e-9)
            )
            statistically_significant = bool(
                p_value <= alpha and (upper < 0.0 or lower > 0.0)
            )
            economically_significant = bool(
                effect_fraction >= economic_threshold
            )
            timezone_sensitive = bool(
                direction_changed
                or (statistically_significant and economically_significant)
            )
            return {
                "endpoint_id": endpoint_id,
                "method": "paired_date_seed_policy_value_then_daily_mean_circular_block_bootstrap_and_Newey_West_and_Holm",
                "difference": f"fixed_MST_minus_primary_UTC_for_{endpoint_id}",
                "paired_date_seed_count": len(paired_rows),
                "paired_day_count": len(daily),
                "primary_mean_policy_value_usd": primary_mean,
                "fixed_mst_mean_policy_value_usd": fixed_mean,
                "mean_clock_effect_usd": effect_mean,
                "block_bootstrap_lower_usd": lower,
                "block_bootstrap_upper_usd": upper,
                "newey_west_standard_error": standard_error,
                "test_statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "economic_effect_fraction_of_primary_value": effect_fraction,
                "economic_significance_threshold_fraction": economic_threshold,
                "direction_changed": direction_changed,
                "statistically_significant": statistically_significant,
                "economically_significant": economically_significant,
                "timezone_sensitive": timezone_sensitive,
                "substantive_conclusion_changed": timezone_sensitive,
            }

        expected_inference = endpoint_inference(
            "B0_minus_B6_expected_cost_policy_value",
            "primary_policy_value_usd",
            "fixed_mst_policy_value_usd",
            "clock_effect_on_policy_value_usd",
        )
        cvar_inference = endpoint_inference(
            "B0_minus_B6_CVaR_policy_value",
            "primary_cvar_policy_value_usd",
            "fixed_mst_cvar_policy_value_usd",
            "clock_effect_on_cvar_policy_value_usd",
        )
        endpoint_family = [expected_inference, cvar_inference]
        ordered = sorted(endpoint_family, key=lambda item: float(item["p_value"]))
        previous_adjusted = 0.0
        for rank, endpoint in enumerate(ordered):
            adjusted = min(
                1.0,
                max(
                    previous_adjusted,
                    (len(ordered) - rank) * float(endpoint["p_value"]),
                ),
            )
            endpoint["p_value_raw"] = float(endpoint["p_value"])
            endpoint["p_value_holm"] = adjusted
            endpoint["multiple_testing_method"] = "Holm"
            endpoint["holm_family_size"] = len(ordered)
            endpoint["statistically_significant"] = bool(
                adjusted <= alpha
                and (
                    float(endpoint["block_bootstrap_upper_usd"]) < 0.0
                    or float(endpoint["block_bootstrap_lower_usd"]) > 0.0
                )
            )
            endpoint["timezone_sensitive"] = bool(
                endpoint["direction_changed"]
                or (
                    endpoint["statistically_significant"]
                    and endpoint["economically_significant"]
                )
            )
            endpoint["substantive_conclusion_changed"] = endpoint[
                "timezone_sensitive"
            ]
            previous_adjusted = adjusted
        timezone_sensitive = bool(
            expected_inference["timezone_sensitive"]
            or cvar_inference["timezone_sensitive"]
        )
        inference = {
            **expected_inference,
            "endpoint_inferences": {
                "expected_cost_policy_value": expected_inference,
                "cvar_policy_value": cvar_inference,
            },
            "timezone_sensitive": timezone_sensitive,
            "substantive_conclusion_changed": timezone_sensitive,
        }
    else:
        inference = {
            "method": "paired_date_seed_policy_value_then_daily_mean_circular_block_bootstrap_and_Newey_West_and_Holm",
            "paired_date_seed_count": 0,
            "paired_day_count": 0,
            "timezone_sensitive": None,
            "substantive_conclusion_changed": None,
        }

    success_count = sum(record.get("status") == "COMPLETE" for record in cell_records)
    failure_count = sum(
        str(record.get("status", "")).startswith("FAILED")
        for record in cell_records
    )
    pending_count = len(expected_keys) - success_count - failure_count
    expected_pair_count = len(primary_dates) * len(seeds)
    completed = bool(
        expected_keys
        and success_count == len(expected_keys)
        and failure_count == 0
        and pending_count == 0
        and len(paired_rows) == expected_pair_count
        and int(inference.get("paired_day_count", 0)) == len(primary_dates)
    )
    manifest = {
        "schema_version": "fixed_mst_economic_sensitivity_v1",
        "status": "COMPLETE" if completed else "PARTIAL",
        "completed": completed,
        "economic_endpoint_compared": completed,
        "endpoint": "paired_counterfactual_policy_value_B0_expected_cost_minus_B6_expected_cost",
        "primary_clock": "raw_ts_assumed_UTC_canonical_UTC_Denver_local",
        "fixed_mst_clock": fixed_evidence["clock_interpretation"],
        "policy_ids": list(policies),
        "scenario_regime": "independent",
        "reduced_scenario_count": 20,
        "rolling_interval_minutes": 15,
        "same_selected_days_forecasts_scenarios_and_realized_parent_ids": True,
        "primary_day_count": len(primary_dates),
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "planned_cell_count": len(expected_keys),
        "success_cell_count": success_count,
        "failed_cell_count": failure_count,
        "pending_cell_count": pending_count,
        "paired_date_seed_count": len(paired_rows),
        "processed_this_invocation": processed,
        "fixed_physical_evidence": {
            **fixed_evidence,
            "artifact_path": str(fixed_artifact_path),
            "artifact_sha256": fixed_artifact_sha,
        },
        "paired_economic_table": {
            "path": str(paired_table),
            "sha256": _file_sha256(paired_table),
        },
        "inference": inference,
        "timezone_sensitive": inference.get("timezone_sensitive"),
        "substantive_conclusion_changed": inference.get(
            "substantive_conclusion_changed"
        ),
        "cells": cell_records,
        "updated_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "completed": completed,
        "economic_endpoint_compared": completed,
        "planned_cell_count": len(expected_keys),
        "success_cell_count": success_count,
        "failed_cell_count": failure_count,
        "pending_cell_count": pending_count,
        "paired_date_seed_count": len(paired_rows),
        "primary_day_count": len(primary_dates),
        "seed_count": len(seeds),
        "processed_this_invocation": processed,
        "timezone_sensitive": inference.get("timezone_sensitive"),
        "substantive_conclusion_changed": inference.get(
            "substantive_conclusion_changed"
        ),
        "path": str(manifest_path),
        "sha256": _file_sha256(manifest_path),
        "paired_table_path": str(paired_table),
        "paired_table_sha256": _file_sha256(paired_table),
    }


def run_optimization_branch(
    root: Path,
    config: dict,
    run_id: str,
    physical_result: dict,
    forecast_result: dict,
) -> dict[str, Any]:
    """Persistent, idempotent entrypoint for scenarios, B0--B8, and H0--H6.

    A checkpoint is written for every date x policy x seed x regime/sensitivity
    cell. Failed attempts are append-only within that cell's checkpoint.
    """

    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and hyphen")
    branch_root = root / "artifacts" / "optimization_branch" / run_id
    scenario_root = branch_root / "scenarios"
    result_root = branch_root / "results"
    checkpoint_store = CheckpointStore(branch_root / "checkpoints")
    branch_root.mkdir(parents=True, exist_ok=True)

    physical = FrozenPhysicalArtifact.from_mapping(physical_result)
    forecasts = _forecast_days(forecast_result, root)
    try:
        xcel_local_subcase = _load_xcel_local_subcase(root, physical_result)
    except Exception as exc:
        # The Xcel local replay owns a separate status and cannot abort the
        # physical/NYISO counterfactual branch.
        xcel_local_subcase = {
            "status": "XCEL_LOCAL_REPLAY_BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "event_calls": 0,
            "tariff_defined_counterfactual": True,
        }
    xcel_subcase_path = _atomic_json(
        branch_root / "xcel_local_subcase_input.json", xcel_local_subcase
    )
    optimization_section = config.get("optimization", {}) if isinstance(config.get("optimization"), Mapping) else {}
    settings = dict(optimization_section)
    settings.update(config.get("optimization_branch", {}))
    if "mip_relative_gap" not in settings and "mip_gap" in settings:
        settings["mip_relative_gap"] = settings["mip_gap"]
    if "interval_hours" not in settings and "interval_minutes" in config:
        settings["interval_hours"] = float(config["interval_minutes"]) / 60.0
    study_seed = int(config.get("seed", settings.get("study_seed", 20260809)))
    seeds = tuple(map(int, settings.get("seed_list", config.get("seed_list", [study_seed]))))
    counts = tuple(map(int, settings.get("scenario_counts", (10, 20, 40, 80))))
    if any(count not in {10, 20, 40, 80} for count in counts):
        raise ValueError("scenario_counts must be a subset of 10, 20, 40, 80")
    regimes = _resolve_regimes(settings.get("regimes"))
    target_year = int(settings.get("target_year", 2025))
    explicit_dates = settings.get("selected_dates")
    if explicit_dates is not None:
        primary_dates = [str(day) for day in explicit_dates]
        selection_records = [
            {
                "date": day,
                "selection_rule": "explicit_fixture_or_preregistered_override",
                "selection_hash": hashlib.sha256(f"{day}|{study_seed}".encode()).hexdigest(),
            }
            for day in primary_dates
        ]
    else:
        selection_records = select_primary_days(forecasts, study_seed=study_seed, target_year=target_year)
        primary_dates = [str(record["date"]) for record in selection_records]
    missing = sorted(set(primary_dates) - set(forecasts))
    if missing:
        raise ValueError(f"selected dates have no frozen forecast artifact: {missing[:5]}")
    _atomic_json(branch_root / "primary_day_selection.json", selection_records)

    reporting = config.get("reporting", {}) if isinstance(config.get("reporting"), Mapping) else {}
    full_year_extension = bool(
        settings.get(
            "full_year_extension",
            settings.get("run_full_year_extension", reporting.get("run_full_year_extension", False)),
        )
    )
    include_h6 = isinstance(settings.get("h6_constraints"), Mapping)
    branch_section = config.get("optimization_branch", {})
    # The normal paper config may list B0--B8 under ``policies``.  Treating that
    # as a Cartesian override would explode every sensitivity/regime cell.  A
    # cross-product is enabled only by this branch-specific, explicit key.
    policy_override = branch_section.get("policy_override") if isinstance(branch_section, Mapping) else None
    run_ofat = bool(
        settings.get(
            "run_ofat_sensitivities",
            policy_override is None and explicit_dates is None,
        )
    )
    exact_validation_result = (
        _run_exact_validation_gate(branch_root, settings)
        if run_ofat
        else {
            "status": "DISABLED_FIXTURE",
            "passed": False,
            "reason": "formal rolling optimization is disabled for this fixture",
        }
    )
    if run_ofat and exact_validation_result.get("passed") is not True:
        raise OptimizationFailure(
            "formal optimization is blocked by the exact-vs-cohort validation gate",
            dict(exact_validation_result),
        )
    sensitivity_seed = int(settings.get("sensitivity_seed", study_seed))
    ofat_variants = build_ofat_variants(settings, config) if run_ofat else []
    variant_by_id = {
        str(variant["variant_id"]): variant for variant in ofat_variants
    }
    if run_ofat:
        support_distances, support_distance_evidence = load_support_distances(
            root, physical_result, settings
        )
    else:
        support_distances, support_distance_evidence = None, {
            "source": "not_required_ofat_disabled"
        }
    primary_cells = _default_cells(
        primary_dates,
        seeds=seeds,
        counts=counts,
        regimes=regimes,
        phase="primary_96_day",
        policy_override=policy_override,
        include_h6_variant=include_h6,
    )
    sensitivity_cells = _ofat_cells(
        primary_dates,
        seed=sensitivity_seed,
        variants=ofat_variants,
    )
    extension_dates = (
        sorted(day for day in forecasts if day.startswith(f"{target_year:04d}-") and day not in set(primary_dates))
        if full_year_extension
        else []
    )
    expected_year_dates: list[str] = []
    cursor = date(target_year, 1, 1)
    year_end = date(target_year + 1, 1, 1)
    while cursor < year_end:
        expected_year_dates.append(cursor.isoformat())
        cursor += timedelta(days=1)
    available_year_dates = sorted(day for day in forecasts if day.startswith(f"{target_year:04d}-"))
    missing_full_year_dates = sorted(set(expected_year_dates) - set(available_year_dates))
    full_year_coverage_complete = not missing_full_year_dates
    extension_cells = _default_cells(
        extension_dates,
        seeds=seeds,
        counts=counts,
        regimes=regimes,
        phase="full_year_extension",
        policy_override=policy_override,
        include_h6_variant=include_h6,
    )
    sensitivity_manifest = {
        "schema_version": "optimization_ofat_sensitivity_plan_v1",
        "run_id": run_id,
        "enabled": run_ofat,
        "phase": "primary_sensitivity",
        "primary_day_count": len(primary_dates),
        "full_year_extension_includes_sensitivities": False,
        "seed_policy": "single_preregistered_study_seed_due_resource",
        "sensitivity_seed": sensitivity_seed,
        "reference_design_seed_count": len(seeds),
        "required_dimensions": list(REQUIRED_DIMENSIONS),
        "implemented_dimensions": sorted(
            {str(variant["dimension"]) for variant in ofat_variants}
        ),
        "variant_count_per_day": len(ofat_variants),
        "planned_cell_count": len(sensitivity_cells),
        "variants_fingerprint": variants_fingerprint(ofat_variants),
        "support_distance_evidence": support_distance_evidence,
        "variants": ofat_variants,
        "created_or_refreshed_at_utc": _utc_now(),
    }
    sensitivity_manifest_path = _atomic_json(
        branch_root / "sensitivity_plan.json", sensitivity_manifest
    )
    plan = {
        "schema_version": "optimization_cell_plan_v1",
        "run_id": run_id,
        "created_or_refreshed_at_utc": _utc_now(),
        "primary_day_count": len(primary_dates),
        "preregistered_primary_day_target": 96,
        "full_year_extension_requested": full_year_extension,
        "full_year_extension_gate": (
            "requires exactly 96 selected primary days and every reference and "
            "primary OFAT sensitivity cell SUCCESS"
        ),
        "full_year_extension_day_count": len(extension_dates),
        "full_year_available_day_count": len(available_year_dates),
        "full_year_expected_day_count": len(expected_year_dates),
        "full_year_missing_dates": missing_full_year_dates,
        "sparse_design": {
            "reference": "B0-B6 and B8 at independent/20 scenarios; B7 uses adverse mixtures only",
            "scenario_count_sensitivity": "B6 independent at 10/40/80 scenarios",
            "dependence_sensitivity": "B6 and B7 at 20 scenarios",
            "h6_constraint_variant": "B6 independent at 20 scenarios",
            "ofat_primary_only": (
                f"{len(ofat_variants)} independent variants at study seed {sensitivity_seed}"
                if run_ofat
                else "disabled by explicit fixture/branch setting"
            ),
        },
        "auxiliary_design": {
            "fixed_mst_economic_timezone_sensitivity": {
                "enabled": run_ofat,
                "phase": "timezone_fixed_mst_sensitivity",
                "primary_day_count": len(primary_dates),
                "seed_count": len(seeds),
                "policy_ids": ["B0", "B6"],
                "regime": "independent",
                "reduced_scenario_count": 20,
                "planned_cell_count": (
                    len(primary_dates) * len(seeds) * 2 if run_ofat else 0
                ),
                "full_year_extension_includes_sensitivity": False,
            }
        },
        "sensitivity_plan": str(sensitivity_manifest_path),
        "cells": [
            cell.as_dict()
            for cell in primary_cells + sensitivity_cells + extension_cells
        ],
    }
    _atomic_json(branch_root / "cell_plan.json", plan)

    max_cells = settings.get("max_cells_per_invocation")
    max_seconds = settings.get("max_runtime_seconds_per_invocation")
    started = time.monotonic()
    processed_this_invocation = 0
    scenario_cache: dict[tuple[str, int, str, int, str], ScenarioBundle] = {}
    cached_operating_date: str | None = None
    result_rows: list[dict[str, Any]] = []
    scenario_diagnostics_file = branch_root / "scenario_diagnostics.json"
    diagnostics_rows: list[dict[str, Any]] = (
        json.loads(scenario_diagnostics_file.read_text(encoding="utf-8"))
        if scenario_diagnostics_file.exists()
        else []
    )
    max_attempts_per_cell = max(1, int(settings.get("max_attempts_per_cell", 3)))
    # A stored SUCCESS is not current evidence until its fingerprint and
    # result hash have been checked during this invocation.  This matters when
    # a resumed run changes configuration and an invocation cap stops before
    # every old checkpoint has been revisited.
    effective_status: dict[str, str] = {}

    def execute_cells(cells: list[WorkCell]) -> None:
        nonlocal processed_this_invocation, cached_operating_date
        for cell in cells:
            if max_cells is not None and processed_this_invocation >= int(max_cells):
                return
            if max_seconds is not None and time.monotonic() - started >= float(max_seconds):
                return
            if cached_operating_date != cell.operating_date:
                scenario_cache.clear()
                cached_operating_date = cell.operating_date
            forecast = forecasts[cell.operating_date]
            solve_settings = _base_solve_settings(
                settings,
                config,
                constraint_variant=cell.constraint_variant,
            )
            if cell.constraint_variant == "ramp_concentration":
                constraints = dict(settings["h6_constraints"])
                if "ramp_limit_mw_per_step" not in constraints or "concentration_share_limit" not in constraints:
                    raise ValueError("h6_constraints requires ramp_limit_mw_per_step and concentration_share_limit")
                solve_settings.update(constraints)
            variant = variant_by_id.get(cell.variant_id)
            if variant is not None:
                for payload_key in (
                    "solve_overrides",
                    "scenario_transforms",
                    "physical_transforms",
                ):
                    solve_settings.update(dict(variant.get(payload_key, {})))
                solve_settings["sensitivity_variant"] = {
                    key: variant.get(key)
                    for key in (
                        "dimension",
                        "variant_id",
                        "level",
                        "value",
                        "baseline",
                    )
                }
                if cell.sensitivity_dimension == "climate_support_distance":
                    solve_settings["support_distance_evidence"] = (
                        support_distance_evidence
                    )
            fingerprint = _fingerprint(cell, physical, forecast, solve_settings)
            existing = checkpoint_store.read(cell)
            result_path = Path(existing["result_path"]) if existing and existing.get("result_path") else None
            reusable_result = bool(
                result_path
                and result_path.is_file()
                and existing.get("result_sha256")
                and _file_sha256(result_path) == existing.get("result_sha256")
            )
            if (
                reusable_result
                and run_ofat
                and cell.phase in {"primary_96_day", "full_year_extension"}
                and cell.sensitivity_dimension == "reference"
                and cell.policy_id != "B8"
            ):
                candidate = json.loads(result_path.read_text(encoding="utf-8"))
                rolling_path = Path(
                    str(candidate.get("rolling_checkpoint_path", ""))
                )
                reusable_result = bool(
                    candidate.get("rolling_operation") is True
                    and rolling_path.is_file()
                    and candidate.get("rolling_checkpoint_sha256")
                    == _file_sha256(rolling_path)
                )
            if (
                existing
                and existing.get("current_status") == "SUCCESS"
                and existing.get("fingerprint") == fingerprint
                and reusable_result
            ):
                effective_status[cell.key] = "SUCCESS"
                result_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            matching_attempts = sum(
                attempt.get("fingerprint") == fingerprint
                for attempt in (existing or {}).get("attempts", [])
            )
            if (
                existing
                and existing.get("current_status") == "FAILED"
                and matching_attempts >= max_attempts_per_cell
            ):
                effective_status[cell.key] = "FAILED"
                continue

            record = checkpoint_store.start(cell, fingerprint)
            effective_status[cell.key] = "RUNNING"
            processed_this_invocation += 1
            try:
                source_fingerprint = _scenario_fingerprint(physical, forecast)[:16]
                def load_pure(regime_name: str) -> tuple[ScenarioBundle, Path]:
                    cache_key = (cell.operating_date, cell.seed, regime_name, cell.reduced_count, "pure")
                    path = (
                        scenario_root
                        / cell.operating_date
                        / (
                            f"seed={cell.seed}__regime={regime_name}__sensitivity={cell.reduced_count}__"
                            f"source={source_fingerprint}.npz"
                        )
                    )
                    cached = scenario_cache.get(cache_key)
                    if cached is not None:
                        return cached, path
                    if path.exists():
                        loaded = ScenarioBundle.load(path)
                    else:
                        regime_spec = next(
                            item for item in tuple(regimes) + tuple(DEFAULT_REGIMES) if item.name == regime_name
                        )
                        loaded = build_scenario_bundle(
                            physical,
                            forecast,
                            regime=regime_spec,
                            reduced_count=cell.reduced_count,
                            study_seed=cell.seed,
                        )
                        loaded.save(path)
                    scenario_cache[cache_key] = loaded
                    diagnostics_rows.append(
                        scenario_diagnostics(
                            loaded,
                            support_distance_evidence=support_distance_evidence,
                        )
                    )
                    return loaded, path

                if cell.policy_id == "B7":
                    stress_mass = float(
                        solve_settings.get(
                            "stress_mass", settings.get("stress_mass", 0.20)
                        )
                    )
                    if not 0.0 <= stress_mass <= 0.40:
                        raise ValueError(
                            "paper B7 stress_mass must be within the preregistered 0-0.40 range"
                        )
                    mixture_key = (
                        cell.operating_date,
                        cell.seed,
                        cell.regime,
                        cell.reduced_count,
                        f"mixture_{stress_mass:.6f}",
                    )
                    scenario_path = (
                        scenario_root
                        / cell.operating_date
                        / (
                            f"seed={cell.seed}__regime={cell.regime}__sensitivity={cell.reduced_count}__"
                            f"mixture_stress_mass={stress_mass:.6f}__source={source_fingerprint}.npz"
                        )
                    )
                    bundle = scenario_cache.get(mixture_key)
                    if bundle is None:
                        if scenario_path.exists():
                            bundle = ScenarioBundle.load(scenario_path)
                        else:
                            nominal, _ = load_pure("independent")
                            stress, _ = load_pure(cell.regime)
                            if stress_mass == 0.0:
                                bundle = replace(
                                    nominal,
                                    regime=stress.regime,
                                    kappa=stress.kappa,
                                    mixture_stress_mass=0.0,
                                    nominal_component_count=nominal.reduced_count,
                                    stress_component_count=0,
                                )
                            else:
                                bundle = mix_nominal_and_stress(
                                    nominal,
                                    stress,
                                    stress_mass=stress_mass,
                                    study_seed=cell.seed,
                                )
                            bundle.save(scenario_path)
                        scenario_cache[mixture_key] = bundle
                        diagnostics_rows.append(
                            scenario_diagnostics(
                                bundle,
                                support_distance_evidence=support_distance_evidence,
                            )
                        )
                else:
                    bundle, scenario_path = load_pure(cell.regime)
                variant_physical = physical
                if variant is not None:
                    bundle, variant_physical, consumed_overrides = apply_ofat_variant(
                        bundle,
                        physical,
                        variant,
                        support_distance_by_zone=support_distances,
                    )
                    solve_settings.update(consumed_overrides)
                # Xcel is a separate local tariff replay.  NYISO settlement
                # cells must never receive an Xcel tariff term, even if a
                # caller accidentally places one in the optimization config.
                formal_rolling_cell = bool(
                    run_ofat
                    and cell.phase in {"primary_96_day", "full_year_extension"}
                    and cell.sensitivity_dimension == "reference"
                    and cell.policy_id != "B8"
                )
                if formal_rolling_cell:
                    from .rolling import run_rolling_day

                    parent_id = int.from_bytes(
                        hashlib.sha256(
                            f"{cell.operating_date}|{cell.seed}|realized-parent".encode(
                                "utf-8"
                            )
                        ).digest()[:8],
                        "big",
                    ) % bundle.initial_count
                    realized_scenario_index = int(
                        bundle.full_parent_mapping[parent_id]
                    )
                    rolling_checkpoint_path = (
                        branch_root / "rolling_checkpoints" / f"{cell.key}.json"
                    )
                    rolling_settings = dict(solve_settings)
                    rolling_settings["continue_on_noncritical_failure"] = False
                    rolling = run_rolling_day(
                        bundle,
                        variant_physical,
                        cell.policy_id,
                        rolling_settings,
                        realized_scenario_index=realized_scenario_index,
                        checkpoint_path=rolling_checkpoint_path,
                    )
                    result = _rolling_scalar_result(
                        rolling,
                        bundle,
                        variant_physical,
                        cell.policy_id,
                        rolling_settings,
                        rolling_checkpoint_path,
                    )
                    result["rolling_realized_parent_id"] = parent_id
                else:
                    result = solve_policy(
                        bundle,
                        variant_physical,
                        cell.policy_id,
                        solve_settings,
                        xcel_tariff=None,
                    )
                result.update(
                    {
                        "run_id": run_id,
                        "study_seed": cell.seed,
                        "phase": cell.phase,
                        "sensitivity_dimension": cell.sensitivity_dimension,
                        "variant_id": cell.variant_id,
                        "variant_value": cell.variant_value,
                        "sensitivity_parameters": (
                            {
                                "solve_overrides": dict(
                                    variant.get("solve_overrides", {})
                                ),
                                "scenario_transforms": dict(
                                    variant.get("scenario_transforms", {})
                                ),
                                "physical_transforms": dict(
                                    variant.get("physical_transforms", {})
                                ),
                                "consumed_by": list(variant.get("consumed_by", [])),
                            }
                            if variant is not None
                            else None
                        ),
                        "transformed_scenario_hashes": {
                            "da_price": _array_sha256(bundle.da_price),
                            "rt_price": _array_sha256(bundle.rt_price),
                            "workload": _array_sha256(bundle.workload),
                            "pv": _array_sha256(bundle.pv),
                            "physical_multiplier": _array_sha256(
                                bundle.physical_multiplier
                            ),
                        },
                        "checkpoint_key": cell.key,
                        "scenario_path": str(scenario_path),
                    }
                )
                result_path = result_root / cell.operating_date / f"{cell.key}.json"
                save_optimization_result(result, result_path)
                checkpoint_store.finish(cell, record, status="SUCCESS", result_path=result_path)
                effective_status[cell.key] = "SUCCESS"
                result_rows.append(result)
            except Exception as exc:
                diagnostics = {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if isinstance(exc, OptimizationFailure):
                    diagnostics["optimization_diagnostics"] = exc.diagnostics
                checkpoint_store.finish(cell, record, status="FAILED", diagnostics=diagnostics)
                effective_status[cell.key] = "FAILED"

    execute_cells(primary_cells)
    execute_cells(sensitivity_cells)

    def status_for(cells: Iterable[WorkCell]) -> dict[str, int]:
        counts_by_status = {"SUCCESS": 0, "FAILED": 0, "RUNNING": 0, "PENDING": 0}
        for cell in cells:
            status = effective_status.get(cell.key, "PENDING")
            counts_by_status[status if status in counts_by_status else "FAILED"] += 1
        return counts_by_status

    primary_reference_status = status_for(primary_cells)
    primary_sensitivity_status = status_for(sensitivity_cells)
    primary_gate_cells = primary_cells + sensitivity_cells
    primary_status = status_for(primary_gate_cells)
    primary_reference_gate_passed = bool(
        len(primary_dates) == 96
        and primary_reference_status["SUCCESS"] == len(primary_cells)
        and primary_reference_status["FAILED"] == 0
        and primary_reference_status["RUNNING"] == 0
        and primary_reference_status["PENDING"] == 0
    )
    if run_ofat:
        remaining_cell_budget = (
            None
            if max_cells is None
            else max(0, int(max_cells) - processed_this_invocation)
        )
        deadline_monotonic = (
            None
            if max_seconds is None
            else started + float(max_seconds)
        )
        try:
            timezone_sensitivity_result = _run_timezone_economic_sensitivity(
                root,
                branch_root,
                primary_physical=physical,
                physical_result=physical_result,
                primary_dates=primary_dates,
                seeds=seeds,
                result_rows=result_rows,
                settings=settings,
                config=config,
                allow_execution=primary_reference_gate_passed,
                remaining_cell_budget=remaining_cell_budget,
                deadline_monotonic=deadline_monotonic,
            )
        except Exception as exc:
            setup_failure_cells: list[dict[str, Any]] = []
            for timezone_day in primary_dates:
                for timezone_seed in seeds:
                    for timezone_policy in ("B0", "B6"):
                        setup_checkpoint = (
                            branch_root
                            / "timezone_sensitivity"
                            / "checkpoints"
                            / timezone_day
                            / f"seed={timezone_seed}__policy={timezone_policy}.json"
                        )
                        setup_record = {
                            "schema_version": "fixed_mst_economic_sensitivity_cell_v1",
                            "operating_date": timezone_day,
                            "study_seed": int(timezone_seed),
                            "policy_id": timezone_policy,
                            "status": "FAILED_SETUP",
                            "reason": "fixed-MST producer setup evidence failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "attempts": [],
                            "checkpoint_path": str(setup_checkpoint),
                            "updated_at_utc": _utc_now(),
                        }
                        _atomic_json(setup_checkpoint, setup_record)
                        setup_failure_cells.append(setup_record)
            timezone_manifest_path = _atomic_json(
                branch_root / "timezone_sensitivity.json",
                {
                    "schema_version": "fixed_mst_economic_sensitivity_v1",
                    "status": "FAILED",
                    "completed": False,
                    "economic_endpoint_compared": False,
                    "primary_day_count": len(primary_dates),
                    "seed_count": len(seeds),
                    "planned_cell_count": len(primary_dates) * len(seeds) * 2,
                    "success_cell_count": 0,
                    "failed_cell_count": len(setup_failure_cells),
                    "pending_cell_count": 0,
                    "cells": setup_failure_cells,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "updated_at_utc": _utc_now(),
                },
            )
            timezone_sensitivity_result = {
                "status": "FAILED",
                "completed": False,
                "economic_endpoint_compared": False,
                "planned_cell_count": len(primary_dates) * len(seeds) * 2,
                "success_cell_count": 0,
                "failed_cell_count": len(setup_failure_cells),
                "pending_cell_count": 0,
                "processed_this_invocation": 0,
                "path": str(timezone_manifest_path),
                "sha256": _file_sha256(timezone_manifest_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        processed_this_invocation += int(
            timezone_sensitivity_result.get("processed_this_invocation", 0) or 0
        )
    else:
        timezone_sensitivity_result = {
            "status": "DISABLED_FIXTURE",
            "completed": False,
            "economic_endpoint_compared": False,
            "planned_cell_count": 0,
            "success_cell_count": 0,
            "failed_cell_count": 0,
            "pending_cell_count": 0,
            "processed_this_invocation": 0,
            "reason": "formal fixed-MST economic sensitivity is disabled for this fixture",
        }
    extension_gate_passed = (
        len(primary_dates) == 96
        and primary_status["SUCCESS"] == len(primary_gate_cells)
        and primary_status["FAILED"] == 0
        and (
            not run_ofat
            or timezone_sensitivity_result.get("completed") is True
        )
    )
    if full_year_extension and extension_gate_passed:
        plan["full_year_extension_started_at_utc"] = _utc_now()
        _atomic_json(branch_root / "cell_plan.json", plan)
        execute_cells(extension_cells)
    elif full_year_extension:
        plan["full_year_extension_blocked_reason"] = (
            "primary gate not passed: require exactly 96 primary days and SUCCESS "
            "for every reference/OFAT cell and the paired fixed-MST economic replay"
        )
        _atomic_json(branch_root / "cell_plan.json", plan)

    all_cells = primary_cells + sensitivity_cells + extension_cells
    current_status = status_for(all_cells)
    extension_status = status_for(extension_cells)
    cell_status_records = []
    for cell in all_cells:
        record = checkpoint_store.read(cell)
        stored_status = record.get("current_status", "PENDING") if record else "PENDING"
        cell_status_records.append(
            {
                **cell.as_dict(),
                "current_status": effective_status.get(cell.key, "PENDING"),
                "stored_checkpoint_status": stored_status,
                "attempt_count": len(record.get("attempts", [])) if record else 0,
                "checkpoint_path": str(checkpoint_store.path(cell)),
                "result_path": record.get("result_path") if record else None,
            }
        )
    cell_status_path = _atomic_json(branch_root / "cell_status.json", cell_status_records)
    # Load successes not visited because a runtime cap stopped iteration.
    known_keys = {row.get("checkpoint_key") for row in result_rows}
    for cell in all_cells:
        record = checkpoint_store.read(cell)
        if (
            record
            and effective_status.get(cell.key) == "SUCCESS"
            and record.get("current_status") == "SUCCESS"
            and cell.key not in known_keys
            and record.get("result_path")
            and Path(record["result_path"]).exists()
            and record.get("result_sha256")
            and _file_sha256(Path(record["result_path"])) == record.get("result_sha256")
        ):
            result_rows.append(json.loads(Path(record["result_path"]).read_text(encoding="utf-8")))

    confirmatory_rows = [
        row
        for row in result_rows
        if row.get("phase") == "primary_96_day"
        and row.get("sensitivity_dimension", "reference") == "reference"
    ]
    if run_ofat:
        scenario_decision_value_result = _scenario_decision_value_subset(
            root,
            branch_root,
            scenario_root,
            physical=physical,
            forecasts=forecasts,
            primary_dates=primary_dates,
            sensitivity_seed=sensitivity_seed,
            result_rows=result_rows,
            settings=settings,
            config=config,
        )
    else:
        scenario_decision_value_result = {
            "status": "DISABLED_FIXTURE",
            "completed": False,
            "reason": "formal OFAT/scenario diagnostics are disabled for this fixture",
        }
    result_table_path = _write_result_table(result_rows, branch_root / "optimization_results.csv")
    sensitivity_result_rows = [
        row for row in result_rows if row.get("phase") == "primary_sensitivity"
    ]
    sensitivity_table_path = _write_result_table(
        sensitivity_result_rows,
        branch_root / "optimization_sensitivity.csv",
    )
    sensitivity_status_by_dimension: dict[str, dict[str, int]] = {}
    for record in cell_status_records:
        if record.get("phase") != "primary_sensitivity":
            continue
        dimension = str(record.get("sensitivity_dimension"))
        counts_for_dimension = sensitivity_status_by_dimension.setdefault(
            dimension,
            {"SUCCESS": 0, "FAILED": 0, "RUNNING": 0, "PENDING": 0},
        )
        status = str(record.get("current_status", "PENDING")).upper()
        counts_for_dimension[status if status in counts_for_dimension else "FAILED"] += 1
    sensitivity_complete = bool(
        run_ofat
        and set(sensitivity_status_by_dimension) == set(REQUIRED_DIMENSIONS)
        and all(
            counts["SUCCESS"] > 0
            and counts["FAILED"] == 0
            and counts["RUNNING"] == 0
            and counts["PENDING"] == 0
            for counts in sensitivity_status_by_dimension.values()
        )
    )
    sensitivity_results_path = _atomic_json(
        branch_root / "sensitivity_results.json",
        {
            "schema_version": "optimization_ofat_sensitivity_results_v1",
            "run_id": run_id,
            "status": (
                "COMPLETE"
                if sensitivity_complete
                else ("PARTIAL" if run_ofat else "DISABLED_FIXTURE")
            ),
            "completed": sensitivity_complete,
            "primary_day_count": len(primary_dates),
            "sensitivity_seed": sensitivity_seed,
            "seed_policy": "single_preregistered_study_seed_due_resource",
            "required_dimensions": list(REQUIRED_DIMENSIONS),
            "implemented_dimensions": sorted(sensitivity_status_by_dimension),
            "status_by_dimension": sensitivity_status_by_dimension,
            "planned_cell_count": len(sensitivity_cells),
            "success_result_count": len(sensitivity_result_rows),
            "full_year_extension_includes_sensitivities": False,
            "plan_path": str(sensitivity_manifest_path),
            "plan_sha256": _file_sha256(sensitivity_manifest_path),
            "result_table_path": str(sensitivity_table_path),
            "result_table_sha256": _file_sha256(sensitivity_table_path),
            "updated_at_utc": _utc_now(),
        },
    )
    diagnostics_by_key = {
        (
            row.get("operating_date"),
            row.get("regime"),
            row.get("reduced_count"),
            row.get("seed"),
            row.get("mixture_stress_mass", 0.0),
        ): row
        for row in diagnostics_rows
    }
    scenario_diagnostics_path = _atomic_json(
        scenario_diagnostics_file,
        [diagnostics_by_key[key] for key in sorted(diagnostics_by_key, key=str)],
    )
    if run_ofat:
        try:
            statistical_result = build_statistical_inference(
                root,
                branch_root,
                run_id=run_id,
                result_rows=result_rows,
                variants=ofat_variants,
                forecast_result=forecast_result,
                settings=settings,
                expected_primary_days=96,
            )
        except Exception as exc:
            statistical_path = _atomic_json(
                branch_root / "statistical_inference.json",
                {
                    "schema_version": "optimization_statistical_inference_v1",
                    "run_id": run_id,
                    "status": "FAILED",
                    "completed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "updated_at_utc": _utc_now(),
                },
            )
            statistical_result = {
                "status": "FAILED",
                "completed": False,
                "path": str(statistical_path),
                "sha256": _file_sha256(statistical_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    else:
        statistical_result = {
            "status": "DISABLED_FIXTURE",
            "completed": False,
            "reason": "OFAT sensitivities were explicitly disabled for this fixture run",
        }
    hypothesis_result = aggregate_hypotheses(
        confirmatory_rows,
        physical_metrics=physical_result.get("hypothesis_metrics"),
        energy_forecast_metrics=forecast_result.get("energy_hypothesis_metrics"),
        workload_forecast_metrics=forecast_result.get("workload_hypothesis_metrics"),
        forecast_decision_records=statistical_result.get(
            "forecast_decision_records",
            forecast_result.get("forecast_decision_records", ()),
        ),
        statistical_inference_rows=statistical_result.get("policy_rows", ()),
        expected_primary_days=96 if run_ofat else None,
        expected_seed_count=len(seeds) if run_ofat else None,
        holm_alpha=float(
            settings.get("statistical_inference", {}).get("holm_alpha", 0.05)
            if isinstance(settings.get("statistical_inference", {}), Mapping)
            else 0.05
        ),
        acceptable_expected_cost_premium_fraction=float(
            settings.get(
                "acceptable_expected_cost_premium_fraction",
                settings.get("acceptable_cost_premium_fraction", 0.05),
            )
        ),
        acceptable_sla_deterioration_work_units=float(
            settings.get("acceptable_sla_deterioration_work_units", 0.0)
        ),
    )
    hypothesis_path = _atomic_json(
        branch_root / "h0_h6_aggregation.json", hypothesis_result
    )
    completed_count = current_status["SUCCESS"]
    cell_failure_count = current_status["FAILED"]
    pending_count = current_status["PENDING"] + current_status["RUNNING"]
    actionable_pending_count = primary_status["PENDING"] + primary_status["RUNNING"]
    auxiliary_pending_count = int(
        timezone_sensitivity_result.get("pending_cell_count", 0) or 0
    )
    auxiliary_failure_count = int(
        timezone_sensitivity_result.get("failed_cell_count", 0) or 0
    )
    if run_ofat and primary_reference_gate_passed:
        actionable_pending_count += auxiliary_pending_count
    if extension_gate_passed:
        actionable_pending_count += extension_status["PENDING"] + extension_status["RUNNING"]
    gated_pending_count = (
        pending_count + auxiliary_pending_count - actionable_pending_count
    )
    extension_required_but_blocked = full_year_extension and not extension_gate_passed
    extension_required_but_incomplete = full_year_extension and not full_year_coverage_complete
    statistical_inference_complete = bool(
        not run_ofat or statistical_result.get("completed") is True
    )
    statistical_failure_count = int(
        run_ofat and pending_count == 0 and not statistical_inference_complete
    )
    scenario_decision_value_complete = bool(
        not run_ofat or scenario_decision_value_result.get("completed") is True
    )
    scenario_decision_value_failure_count = int(
        run_ofat and pending_count == 0 and not scenario_decision_value_complete
    )
    failure_count = (
        cell_failure_count
        + statistical_failure_count
        + scenario_decision_value_failure_count
    )
    completed = (
        len(all_cells) > 0
        and failure_count == 0
        and auxiliary_failure_count == 0
        and pending_count == 0
        and auxiliary_pending_count == 0
        and not extension_required_but_blocked
        and not extension_required_but_incomplete
        and statistical_inference_complete
        and scenario_decision_value_complete
        and (
            not run_ofat
            or timezone_sensitivity_result.get("completed") is True
        )
    )
    if completed:
        overall_status = "COMPLETE"
    elif (
        completed_count == 0
        and failure_count + auxiliary_failure_count > 0
        and pending_count + auxiliary_pending_count == 0
    ):
        overall_status = "FAILED"
    else:
        overall_status = "PARTIAL"
    summary = {
        "schema_version": "optimization_branch_run_v1",
        "run_id": run_id,
        "status": overall_status,
        "completed": completed,
        "completed_count": completed_count,
        "failure_count": failure_count,
        "cell_failure_count": cell_failure_count,
        "statistical_failure_count": statistical_failure_count,
        "pending_count": pending_count,
        "auxiliary_failure_count": auxiliary_failure_count,
        "auxiliary_pending_count": auxiliary_pending_count,
        "actionable_pending_count": actionable_pending_count,
        "gated_pending_count": gated_pending_count,
        "max_attempts_per_cell": max_attempts_per_cell,
        "planned_cell_count": len(all_cells),
        "processed_this_invocation": processed_this_invocation,
        "primary_day_count": len(primary_dates),
        "preregistered_primary_day_target": 96,
        "primary_cell_status": primary_status,
        "primary_reference_cell_status": primary_reference_status,
        "primary_sensitivity_cell_status": primary_sensitivity_status,
        "ofat_sensitivities_requested": run_ofat,
        "ofat_sensitivities_complete": sensitivity_complete,
        "ofat_dimension_count": len(sensitivity_status_by_dimension),
        "ofat_required_dimension_count": len(REQUIRED_DIMENSIONS),
        "ofat_variant_count_per_day": len(ofat_variants),
        "statistical_inference_required": run_ofat,
        "statistical_inference_complete": statistical_inference_complete,
        "statistical_inference_status": statistical_result.get("status"),
        "scenario_decision_value_complete": scenario_decision_value_complete,
        "scenario_decision_value_status": scenario_decision_value_result.get(
            "status"
        ),
        "scenario_decision_value_failure_count": scenario_decision_value_failure_count,
        "timezone_economic_sensitivity_required": run_ofat,
        "timezone_economic_sensitivity_complete": bool(
            not run_ofat or timezone_sensitivity_result.get("completed") is True
        ),
        "timezone_economic_sensitivity_status": timezone_sensitivity_result.get(
            "status"
        ),
        "timezone_economic_sensitivity_planned_cell_count": int(
            timezone_sensitivity_result.get("planned_cell_count", 0) or 0
        ),
        "timezone_economic_sensitivity_success_cell_count": int(
            timezone_sensitivity_result.get("success_cell_count", 0) or 0
        ),
        "timezone_economic_sensitivity_paired_date_seed_count": int(
            timezone_sensitivity_result.get("paired_date_seed_count", 0) or 0
        ),
        "timezone_sensitive": timezone_sensitivity_result.get(
            "timezone_sensitive"
        ),
        "timezone_substantive_conclusion_changed": timezone_sensitivity_result.get(
            "substantive_conclusion_changed"
        ),
        "exact_validation_required": run_ofat,
        "exact_validation_passed": bool(
            not run_ofat or exact_validation_result.get("passed") is True
        ),
        "sensitivity_seed": sensitivity_seed,
        "sensitivity_seed_policy": "single_preregistered_study_seed_due_resource",
        "full_year_extension_requested": full_year_extension,
        "full_year_extension_gate_passed": extension_gate_passed,
        "full_year_extension_started": bool(extension_cells) and extension_gate_passed,
        "full_year_coverage_complete": full_year_coverage_complete,
        "full_year_available_day_count": len(available_year_dates),
        "full_year_expected_day_count": len(expected_year_dates),
        "full_year_missing_day_count": len(missing_full_year_dates),
        "nyiso_evidence_role": "counterfactual",
        "xcel_relationship": "tariff_defined_counterfactual",
        "xcel_local_subcase_status": xcel_local_subcase["status"],
        "xcel_tariff_used_in_nyiso_settlement": False,
        "measurement_relationship": "not_measured_by_this_study",
        "paths": {
            "branch_root": str(branch_root),
            "cell_plan": str(branch_root / "cell_plan.json"),
            "cell_status": str(cell_status_path),
            "checkpoints": str(checkpoint_store.directory),
            "scenarios": str(scenario_root),
            "results": str(result_root),
            "result_table": str(result_table_path),
            "sensitivity_plan": str(sensitivity_manifest_path),
            "sensitivity_results": str(sensitivity_results_path),
            "sensitivity_table": str(sensitivity_table_path),
            "scenario_diagnostics": str(scenario_diagnostics_path),
            "scenario_decision_value_subset": scenario_decision_value_result.get(
                "path"
            ),
            "timezone_sensitivity": timezone_sensitivity_result.get("path"),
            "timezone_sensitivity_paired_table": timezone_sensitivity_result.get(
                "paired_table_path"
            ),
            "exact_validation_gate": exact_validation_result.get("path"),
            "statistical_inference": statistical_result.get("path"),
            "paired_policy_inference": statistical_result.get("outputs", {})
            .get("paired_policy_inference", {})
            .get("path"),
            "sensitivity_effects": statistical_result.get("outputs", {})
            .get("sensitivity_effects", {})
            .get("path"),
            "forecast_dm_tests": statistical_result.get("outputs", {})
            .get("forecast_dm_tests", {})
            .get("path"),
            "forecast_decision_linkage": statistical_result.get("outputs", {})
            .get("forecast_decision_linkage", {})
            .get("path"),
            "hypotheses": str(hypothesis_path),
            "xcel_local_subcase_input": str(xcel_subcase_path),
            "xcel_local_replay_manifest": xcel_local_subcase.get("local_replay", {}).get(
                "manifest"
            ),
        },
        "path_hashes": {
            "cell_plan": _file_sha256(branch_root / "cell_plan.json"),
            "cell_status": _file_sha256(cell_status_path),
            "result_table": _file_sha256(result_table_path),
            "sensitivity_plan": _file_sha256(sensitivity_manifest_path),
            "sensitivity_results": _file_sha256(sensitivity_results_path),
            "sensitivity_table": _file_sha256(sensitivity_table_path),
            "scenario_diagnostics": _file_sha256(scenario_diagnostics_path),
            "scenario_decision_value_subset": scenario_decision_value_result.get(
                "sha256"
            ),
            "timezone_sensitivity": timezone_sensitivity_result.get("sha256"),
            "timezone_sensitivity_paired_table": timezone_sensitivity_result.get(
                "paired_table_sha256"
            ),
            "exact_validation_gate": exact_validation_result.get("sha256"),
            "statistical_inference": statistical_result.get("sha256"),
            "hypotheses": _file_sha256(hypothesis_path),
        },
        "updated_at_utc": _utc_now(),
    }
    summary_path = _atomic_json(branch_root / "run_summary.json", summary)
    summary["paths"]["run_summary"] = str(summary_path)
    return summary
