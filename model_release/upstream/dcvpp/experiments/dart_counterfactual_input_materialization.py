"""Materialize candidate-independent counterfactual inputs for DART replay.

The materializer implements the preregistered dual-track settlement policy:
all selection-calendar input bundles are built independently of settlement
availability, while a separately versioned settlement ledger decides which
days may enter economic replay.  Missing settlement is never imputed and is
never represented as a solver failure.

One immutable NPZ is written per operating date.  The issue-time, replay-
forcing, and actionability indices all reference that same file, avoiding
three copies of the large 100-draw arrays.  The locked 2025 calendar is
recorded but deliberately remains closed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_real_input_preparation import (
    _REQUIRED_ACTION_SOURCE_FIELDS,
    _audit_kind,
    DESIGN_SCHEMA_VERSION,
    exact_phase_dates,
)
from dcvpp.research_branch import workload_unit_bridge
from dcvpp.research_branch.contracts import NYISO_ZONES
from dcvpp.research_branch.workload_unit_bridge import (
    apply_workload_normalization,
    derive_workload_normalization_contract,
)


SCHEMA_VERSION = "dcvpp.dart_counterfactual_input_materialization.v1"
DAILY_SCHEMA_VERSION = "dcvpp.dart_counterfactual_daily_bundle.v1"
ROOT_SEED = 20260809
DRAW_COUNT = 100
SEASONAL_POOL_LIMIT = 56
INTERVALS_PER_DAY = 96
INTERVAL_MINUTES = 15


class CounterfactualMaterializationError(RuntimeError):
    """Raised when a causal or frozen-input contract cannot be satisfied."""


def finalize_selection_input_block(
    *,
    repository_root: Path,
    output_root: str | Path = "artifacts/dart_value_select/counterfactual_inputs/v1",
) -> dict[str, Any]:
    """Audit the three completed indices and persist the forecast-result block.

    This is deliberately separate from materialization so it can be rerun
    without touching a daily bundle.
    """

    root = Path(repository_root).resolve()
    destination = _resolve(root, output_root)
    manifest_path = destination / "materialization_manifest.json"
    ledger_path = (
        root
        / "artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1/"
        "selection_conditional_settlement_ledger.json"
    )
    official_index = (
        root
        / "artifacts/dart_value_select/real_input_preparation/v1/official_rt_recovery/"
        "selection_nyiso_replay_by_date_index.json"
    )
    for required in (manifest_path, ledger_path, official_index):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    dates = tuple(map(str, ledger.get("valid_operating_dates", ())))
    if manifest.get("status") != "SELECTION_INPUTS_COMPLETE_LOCKED_TEST_CLOSED":
        raise CounterfactualMaterializationError("materialization manifest is not COMPLETE")
    if len(dates) != int(manifest["selection"]["materialized_day_count"]):
        raise CounterfactualMaterializationError("valid ledger count differs from materialization")
    official = json.loads(official_index.read_text(encoding="utf-8"))
    if official.get("status") != "COMPLETE" or int(official.get("day_count", -1)) != len(dates):
        raise CounterfactualMaterializationError("official replay index differs from valid panel")

    index_paths = manifest["selection"]["three_index_paths"]
    reports: dict[str, Any] = {}
    for kind, key in (
        ("issue_time", "issue_time_ensembles_by_date"),
        ("forcing", "replay_forcing_by_date"),
        ("actionability", "actionability_by_date"),
    ):
        report, _ = _audit_kind(
            kind,
            locator=index_paths[key],
            expected_dates=dates,
            root=root,
        )
        reports[kind] = report
    if any(report["status"] != "COMPLETE" for report in reports.values()):
        raise CounterfactualMaterializationError("one or more selection input indices failed audit")

    block = {
        "schema_version": SCHEMA_VERSION,
        "status": "SELECTION_READY_LOCKED_TEST_CLOSED",
        "phase": "selection",
        "selection_valid_date_count": len(dates),
        "full_calendar_ledger": {
            "path": manifest["selection"]["calendar_ledger_path"],
            "sha256": manifest["selection"]["calendar_ledger_sha256"],
        },
        "conditional_settlement_ledger": {
            "path": _relative(root, ledger_path),
            "sha256": sha256_file(ledger_path),
        },
        "dart_real_evidence_inputs": {
            "issue_time_ensembles_by_date": {
                "index_path": index_paths["issue_time_ensembles_by_date"]
            },
            "replay_forcing_by_date": {
                "index_path": index_paths["replay_forcing_by_date"]
            },
            "actionability_by_date": {
                "index_path": index_paths["actionability_by_date"]
            },
            "nyiso_replay_by_date": {"index_path": _relative(root, official_index)},
            "referenced_nyiso_data_root": "data/processed/referenced/nyiso",
        },
        "injection": (
            "forecast_result['dart_real_evidence_inputs'] = "
            "payload['dart_real_evidence_inputs']"
        ),
        "locked_test_opened": False,
    }
    block_path = destination / "selection_config_ready_input_block.json"
    atomic_json(block_path, block)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "validated_at_utc": utc_now(),
        "selection_valid_date_count": len(dates),
        "three_index_reports": reports,
        "official_nyiso_replay_index": {
            "path": _relative(root, official_index),
            "sha256": sha256_file(official_index),
            "day_count": int(official["day_count"]),
        },
        "materialization_manifest": {
            "path": _relative(root, manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "config_ready_input_block": {
            "path": _relative(root, block_path),
            "sha256": sha256_file(block_path),
        },
        "locked_test_opened": False,
    }
    validation["validation_fingerprint"] = sha256_json(validation)
    validation_path = destination / "selection_input_validation.json"
    atomic_json(validation_path, validation)
    return {
        **validation,
        "validation_path": _relative(root, validation_path),
        "validation_sha256": sha256_file(validation_path),
    }


@dataclass(frozen=True)
class DailyPath:
    operating_date: str
    values: np.ndarray
    max_timestamp_utc: pd.Timestamp
    max_available_at_utc: pd.Timestamp | None = None


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(root.resolve()).as_posix()
        if resolved.is_relative_to(root.resolve())
        else resolved.as_posix()
    )


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _stable_seed(day: str, component: str, *, root_seed: int = ROOT_SEED) -> int:
    payload = f"{SCHEMA_VERSION}|{root_seed}|{day}|{component}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def _circular_day_distance(left: str, right: str) -> int:
    left_day = pd.Timestamp(left).dayofyear
    right_day = pd.Timestamp(right).dayofyear
    raw = abs(int(left_day) - int(right_day))
    return min(raw, 366 - raw)


def _seasonal_pool(
    histories: Mapping[str, DailyPath],
    *,
    target_day: str,
    issue_time: pd.Timestamp,
    component: str,
    same_weekday: bool,
) -> list[DailyPath]:
    eligible = [
        path
        for path in histories.values()
        if path.max_timestamp_utc < issue_time
        and (
            path.max_available_at_utc is None
            or path.max_available_at_utc <= issue_time
        )
    ]
    if same_weekday:
        weekday = pd.Timestamp(target_day).weekday()
        matched = [
            path
            for path in eligible
            if pd.Timestamp(path.operating_date).weekday() == weekday
        ]
        if matched:
            eligible = matched
    eligible.sort(
        key=lambda path: (
            _circular_day_distance(path.operating_date, target_day),
            -pd.Timestamp(path.operating_date).toordinal(),
        )
    )
    if not eligible:
        raise CounterfactualMaterializationError(
            f"no causal {component} history exists before {issue_time.isoformat()}"
        )
    return eligible[: min(SEASONAL_POOL_LIMIT, len(eligible))]


def _draw_paths(
    histories: Mapping[str, DailyPath],
    *,
    target_day: str,
    issue_time: pd.Timestamp,
    component: str,
    same_weekday: bool,
    draw_count: int,
) -> tuple[np.ndarray, tuple[str, ...], pd.Timestamp, pd.Timestamp | None]:
    pool = _seasonal_pool(
        histories,
        target_day=target_day,
        issue_time=issue_time,
        component=component,
        same_weekday=same_weekday,
    )
    rng = np.random.default_rng(_stable_seed(target_day, component))
    selected = [pool[int(index)] for index in rng.integers(0, len(pool), size=draw_count)]
    values = np.stack([path.values for path in selected]).astype(np.float32, copy=False)
    max_timestamp = max(path.max_timestamp_utc for path in selected)
    availabilities = [
        path.max_available_at_utc
        for path in selected
        if path.max_available_at_utc is not None
    ]
    max_available = max(availabilities) if availabilities else None
    return (
        values,
        tuple(path.operating_date for path in selected),
        max_timestamp,
        max_available,
    )


def _complete_workload_days(
    frame: pd.DataFrame, *, bridge_contract: Mapping[str, Any]
) -> dict[str, DailyPath]:
    rows = frame[["timestamp_utc", "submitted_nodes_requested"]].copy()
    rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
    rows = rows.sort_values("timestamp_utc", kind="stable")
    if rows.duplicated("timestamp_utc").any():
        raise CounterfactualMaterializationError("NLR workload contains duplicate timestamps")
    rows["operating_date"] = rows["timestamp_utc"].dt.date.astype(str)
    output: dict[str, DailyPath] = {}
    for day, group in rows.groupby("operating_date", sort=True):
        timestamps = pd.date_range(day, periods=INTERVALS_PER_DAY, freq="15min", tz="UTC")
        indexed = group.set_index("timestamp_utc")["submitted_nodes_requested"].reindex(
            timestamps
        )
        raw = indexed.to_numpy(dtype=float)
        if not np.isfinite(raw).all() or (raw < 0).any():
            continue
        normalized = apply_workload_normalization(raw[None, :], bridge_contract)[0]
        output[day] = DailyPath(
            operating_date=day,
            values=normalized.astype(np.float32),
            max_timestamp_utc=timestamps[-1],
        )
    if not output:
        raise CounterfactualMaterializationError("NLR source has no complete 96-interval days")
    return output


def _complete_pv_days(frame: pd.DataFrame) -> dict[str, DailyPath]:
    required = {
        "timestamp_utc",
        "zone",
        "deployable_available_at_utc",
        "pv_dc_availability_mw_per_mwdc",
    }
    if missing := required.difference(frame.columns):
        raise CounterfactualMaterializationError(
            f"NASA-derived PV panel lacks {sorted(missing)}"
        )
    rows = frame[list(required)].copy()
    rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
    rows["deployable_available_at_utc"] = pd.to_datetime(
        rows["deployable_available_at_utc"], utc=True
    )
    rows["operating_date"] = rows["timestamp_utc"].dt.date.astype(str)
    output: dict[str, DailyPath] = {}
    for day, group in rows.groupby("operating_date", sort=True):
        timestamps = pd.date_range(day, periods=INTERVALS_PER_DAY, freq="15min", tz="UTC")
        if group.duplicated(["timestamp_utc", "zone"]).any():
            continue
        pivot = group.pivot(
            index="zone",
            columns="timestamp_utc",
            values="pv_dc_availability_mw_per_mwdc",
        ).reindex(index=NYISO_ZONES, columns=timestamps)
        values = pivot.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            continue
        output[day] = DailyPath(
            operating_date=day,
            values=values.astype(np.float32),
            max_timestamp_utc=timestamps[-1],
            max_available_at_utc=group["deployable_available_at_utc"].max(),
        )
    if not output:
        raise CounterfactualMaterializationError("PV source has no complete four-zone days")
    return output


def _selection_oos_price_days(
    repository_root: Path,
    *,
    stage_root: str | Path = (
        "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos"
    ),
) -> tuple[dict[str, DailyPath], dict[str, str]]:
    """Load candidate-independent seasonal-naive OOS price distributions.

    These are frozen predictions available at the common DA gate, not target-
    day DA/RT settlement.  A1 holds their DA median fixed; A2 replaces both
    price blocks with candidate-specific forecast distributions downstream.
    """

    root = repository_root.resolve()
    stage = _resolve(root, stage_root)
    output: dict[str, DailyPath] = {}
    hashes: dict[str, str] = {}
    origins = (
        ("through_2021_to_2022_oos", 2022),
        ("through_2022_to_2023_oos", 2023),
        ("through_2023_to_2024_oos", 2024),
    )
    for origin, year in origins:
        path = (
            stage
            / "dart"
            / origin
            / "seasonal_naive"
            / "deterministic_seed_0"
            / "dart_draws.npz"
        )
        manifest_path = path.with_suffix(".json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_sha = sha256_file(path)
        if manifest.get("status") != "COMPLETE" or manifest.get("artifact_sha256") != artifact_sha:
            raise CounterfactualMaterializationError(
                f"seasonal-naive OOS price artifact is not frozen COMPLETE: {origin}"
            )
        hashes[f"{origin}_seasonal_naive_dart_draws"] = artifact_sha
        hashes[f"{origin}_seasonal_naive_dart_manifest"] = sha256_file(manifest_path)
        with np.load(path, allow_pickle=False) as archive:
            event_keys = np.asarray(archive["event_keys"]).astype(str)
            da = np.asarray(archive["da_lbmp_draws"], dtype=float)
            rt = np.asarray(archive["rt_lbmp_draws"], dtype=float)
        expected = 365 + (1 if year == 2024 else 0)
        if da.shape != rt.shape or da.shape != (expected * 4 * 96, DRAW_COUNT):
            raise CounterfactualMaterializationError(
                f"seasonal-naive OOS price shape differs for {origin}: {da.shape}"
            )
        parsed_days = np.asarray([value.split("T", 1)[0] for value in event_keys])
        for day in sorted(set(parsed_days.tolist())):
            indices = np.flatnonzero(parsed_days == day)
            if len(indices) != 4 * 96:
                raise CounterfactualMaterializationError(
                    f"OOS price day has wrong event count: {day}"
                )
            # Event order is frozen zone-major, time-minor in the OOS artifact.
            day_da = da[indices].reshape(4, 96, DRAW_COUNT).transpose(2, 0, 1)
            day_rt = rt[indices].reshape(4, 96, DRAW_COUNT).transpose(2, 0, 1)
            output[day] = DailyPath(
                operating_date=day,
                values=np.stack([day_da, day_rt], axis=1).astype(np.float32),
                max_timestamp_utc=pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=1),
            )
    if len(output) != 1096:
        raise CounterfactualMaterializationError(
            f"seasonal-naive OOS prices cover {len(output)} days, expected 1096"
        )
    return output, hashes


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        # Most 100-draw arrays are high entropy; uncompressed NPZ is materially
        # faster here and avoids wasting CPU during the one-time materialization.
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_daily_arrays(
    day: str,
    *,
    workload_days: Mapping[str, DailyPath],
    pv_days: Mapping[str, DailyPath],
    price_days: Mapping[str, DailyPath],
    workload_forcing_days: Sequence[DailyPath],
    physical: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    draw_count: int = DRAW_COUNT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamps = pd.date_range(day, periods=INTERVALS_PER_DAY, freq="15min", tz="UTC")
    issue_time = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=1)
    workload_pool, workload_sources, workload_max, _ = _draw_paths(
        workload_days,
        target_day=day,
        issue_time=issue_time,
        component="workload",
        same_weekday=True,
        draw_count=draw_count,
    )
    pv_pool, pv_sources, pv_max, pv_available_max = _draw_paths(
        pv_days,
        target_day=day,
        issue_time=issue_time,
        component="pv",
        same_weekday=False,
        draw_count=draw_count,
    )
    price_day = price_days.get(day)
    if price_day is None:
        raise CounterfactualMaterializationError(
            f"frozen seasonal-naive OOS price distribution is absent: {day}"
        )
    price_pool = np.asarray(price_day.values, dtype=np.float32)
    if price_pool.shape != (draw_count, 2, len(NYISO_ZONES), INTERVALS_PER_DAY):
        raise CounterfactualMaterializationError(
            f"frozen OOS price shape is invalid for {day}: {price_pool.shape}"
        )
    da_price = np.median(price_pool[:, 0], axis=0).astype(np.float32)
    rt_price_pool = price_pool[:, 1].astype(np.float32, copy=False)
    price_sources = tuple([day] * draw_count)
    price_max = price_day.max_timestamp_utc

    target_pv = pv_days.get(day)
    if target_pv is None:
        raise CounterfactualMaterializationError(f"target-day reconstructed PV is absent: {day}")
    forcing_position = (
        pd.Timestamp(day).toordinal() - pd.Timestamp("2022-01-01").toordinal()
    ) % len(workload_forcing_days)
    forcing_workload = workload_forcing_days[forcing_position]

    battery_power = np.asarray(physical["battery_power_mw"], dtype=float)
    battery_energy = np.asarray(physical["battery_energy_mwh"], dtype=float)
    physical_pool = np.asarray(physical["physical_multiplier_pool"], dtype=float)
    if (
        battery_power.shape != (len(NYISO_ZONES),)
        or battery_energy.shape != battery_power.shape
        or physical_pool.ndim != 2
        or physical_pool.shape[1] != len(NYISO_ZONES)
    ):
        raise CounterfactualMaterializationError("frozen physical array shapes are invalid")
    initial_soc = 0.5 * battery_energy
    flexible = np.minimum.reduce(
        [battery_power * 0.25, initial_soc, battery_energy - initial_soc]
    )
    weight = np.divide(
        flexible,
        np.maximum(battery_power * 0.25, np.finfo(float).eps),
    )
    feasible_by_event = np.repeat(flexible[:, None], INTERVALS_PER_DAY, axis=1)
    weight_by_event = np.repeat(np.clip(weight, 0.0, 1.0)[:, None], INTERVALS_PER_DAY, axis=1)
    no_action = np.full((len(NYISO_ZONES), INTERVALS_PER_DAY), 1.0, dtype=np.float32)
    physical_multiplier = np.median(physical_pool, axis=0).astype(np.float32)

    source_hash_text = json.dumps(dict(sorted(source_hashes.items())), sort_keys=True)
    arrays: dict[str, Any] = {
        "daily_schema_version": np.asarray(DAILY_SCHEMA_VERSION),
        "issue_time_utc": np.asarray(issue_time.isoformat()),
        "operating_date": np.asarray(day),
        "forecast_id": np.asarray(f"dart_counterfactual_common_{day}"),
        "timestamps_utc": timestamps.to_numpy(dtype="datetime64[ns]"),
        "da_price": da_price,
        "rt_price_pool": rt_price_pool,
        "workload_pool": workload_pool,
        "pv_pool": pv_pool,
        "zones": np.asarray(NYISO_ZONES),
        "source_hashes_json": np.asarray(source_hash_text),
        "locked_test": np.asarray(False),
        "workload": np.asarray(forcing_workload.values, dtype=np.float32),
        "pv": np.asarray(target_pv.values, dtype=np.float32),
        "physical_multiplier": physical_multiplier,
        "candidate_independent": np.asarray(True),
        "selected_scenario_as_realized": np.asarray(False),
        "evidence_roles_json": np.asarray(
            json.dumps(
                {
                    "workload": "replayed",
                    "pv": "reconstructed",
                    "physical_multiplier": "counterfactual",
                },
                sort_keys=True,
            )
        ),
        "feasible_flexible_energy_mwh_by_event": feasible_by_event.astype(np.float32),
        "normalized_feasibility_weight_by_event": weight_by_event.astype(np.float32),
        "lower_no_action_bound": -no_action,
        "upper_no_action_bound": no_action,
        "available_at_gate": np.ones(
            (len(NYISO_ZONES), INTERVALS_PER_DAY), dtype=bool
        ),
        "source_fields_json": np.asarray(
            json.dumps(sorted(_REQUIRED_ACTION_SOURCE_FIELDS))
        ),
        "issue_workload_source_dates": np.asarray(workload_sources),
        "issue_pv_source_dates": np.asarray(pv_sources),
        "issue_price_source_dates": np.asarray(price_sources),
        "forcing_workload_source_date": np.asarray(forcing_workload.operating_date),
        "forcing_pv_source_date": np.asarray(day),
        "migration_feasibility_weight_by_event": np.zeros(
            (len(NYISO_ZONES), INTERVALS_PER_DAY), dtype=np.float32
        ),
        "initial_soc_mwh_by_zone": initial_soc.astype(np.float32),
        "actionability_policy_json": np.asarray(
            json.dumps(
                {
                    "migration_actions_enabled": False,
                    "migration_eligibility_observed": False,
                    "migration_absence_policy": "known_zero_not_imputed_observation",
                    "battery_only": True,
                    "initial_soc_fraction": 0.5,
                    "soc_runtime_update": (
                        "recursive_from_initial_soc_and_causally_revealed_actions; "
                        "no future action path is precomputed"
                    ),
                    "no_action_band_usd_per_mwh_equivalent": [-1.0, 1.0],
                    "battery_degradation_usd_per_mwh": 1.0,
                    "transaction_migration_sla_minimum_dispatch_settlement_adders": 0.0,
                },
                sort_keys=True,
            )
        ),
        "forcing_reveal_policy": np.asarray(
            "interval_sequential_after_day_ahead_first_stage_freeze"
        ),
    }
    diagnostics = {
        "issue_time_utc": issue_time.isoformat(),
        "workload_history_max_utc": workload_max.isoformat(),
        "pv_history_max_utc": pv_max.isoformat(),
        "pv_deployable_available_max_utc": (
            pv_available_max.isoformat() if pv_available_max is not None else None
        ),
        "price_history_max_utc": price_max.isoformat(),
        "causal_workload": bool(workload_max < issue_time),
        "causal_pv": bool(
            pv_max < issue_time
            and pv_available_max is not None
            and pv_available_max <= issue_time
        ),
        "causal_candidate_independent_price": bool(price_max <= issue_time),
        "workload_draw_source_dates_sha256": sha256_json(list(workload_sources)),
        "pv_draw_source_dates_sha256": sha256_json(list(pv_sources)),
        "price_draw_source_dates_sha256": sha256_json(list(price_sources)),
        "forcing_workload_source_date": forcing_workload.operating_date,
        "forcing_pv_source_date": day,
        "migration_actions_enabled": False,
        "battery_only_actionability": True,
    }
    if not all(
        diagnostics[name]
        for name in (
            "causal_workload",
            "causal_pv",
            "causal_candidate_independent_price",
        )
    ):
        raise CounterfactualMaterializationError(f"causal gate failed for {day}")
    return arrays, diagnostics


def materialize_selection_counterfactual_inputs(
    *,
    repository_root: Path,
    output_root: str | Path = "artifacts/dart_value_select/counterfactual_inputs/v1",
    workload_path: str | Path = (
        "artifacts/real_forecasting_recovery/run_id=paper-forecast-v1-recovery-v1/"
        "data/nlr295_15m_workload.parquet"
    ),
    pv_path: str | Path = "data/processed/nyiso_zone_pv_15m.parquet",
    physical_path: str | Path = (
        "artifacts/research/physical/run_id=paper-physical-v1/"
        "frozen_physical_artifact.json"
    ),
    twin_path: str | Path = (
        "data/processed/research/physical/run_id=paper-physical-v1/"
        "nyiso_zone_twins.parquet"
    ),
    coverage_path: str | Path = (
        "artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1/"
        "selection_conditional_settlement_ledger.json"
    ),
    selection_dates: Sequence[str] | None = None,
    draw_count: int = DRAW_COUNT,
    workers: int = 4,
    require_exact_calendar: bool = True,
) -> dict[str, Any]:
    """Materialize full-calendar selection inputs without opening locked 2025."""

    if draw_count != DRAW_COUNT:
        raise CounterfactualMaterializationError("formal materialization requires 100 draws")
    root = Path(repository_root).resolve()
    destination = _resolve(root, output_root)
    destination.mkdir(parents=True, exist_ok=True)
    workload_source = _resolve(root, workload_path)
    pv_source = _resolve(root, pv_path)
    physical_source = _resolve(root, physical_path)
    twin_source = _resolve(root, twin_path)
    coverage_source = _resolve(root, coverage_path)
    for required in (
        workload_source,
        pv_source,
        physical_source,
        twin_source,
        coverage_source,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    coverage = json.loads(coverage_source.read_text(encoding="utf-8"))
    calendar_dates = exact_phase_dates("selection")
    valid_dates = tuple(map(str, coverage.get("valid_operating_dates", ())))
    if not valid_dates:
        valid_dates = tuple(
            str(record["operating_date"])
            for record in coverage.get("records", [])
            if record.get("settlement_evaluable") is True
        )
    dates = tuple(selection_dates or valid_dates)
    if (
        len(dates) != len(set(dates))
        or (require_exact_calendar and dates != valid_dates)
    ):
        raise CounterfactualMaterializationError(
            "selection dates must equal the frozen conditional valid-date panel"
        )
    if any(pd.Timestamp(day).year not in (2022, 2023, 2024) for day in dates):
        raise CounterfactualMaterializationError("selection dates must be within 2022-2024")

    workload_sha = sha256_file(workload_source)
    pv_sha = sha256_file(pv_source)
    physical_sha = sha256_file(physical_source)
    twin_sha = sha256_file(twin_source)
    coverage_sha = sha256_file(coverage_source)
    physical = json.loads(physical_source.read_text(encoding="utf-8"))
    if physical.get("frozen") is not True or tuple(physical.get("zones", ())) != tuple(
        NYISO_ZONES
    ):
        raise CounterfactualMaterializationError("physical artifact is not frozen four-zone")

    workload_frame = pq.ParquetFile(workload_source).read(
        columns=["timestamp_utc", "submitted_nodes_requested"]
    ).to_pandas()
    workload_frame["timestamp_utc"] = pd.to_datetime(
        workload_frame["timestamp_utc"], utc=True
    )
    workload_frame = workload_frame.sort_values("timestamp_utc", kind="stable").reset_index(
        drop=True
    )
    bridge_contract = derive_workload_normalization_contract(
        workload_frame,
        physical,
        panel_sha256=workload_sha,
        physical_artifact_sha256=physical_sha,
    )
    bridge_contract["implementation_sha256"] = sha256_file(
        Path(workload_unit_bridge.__file__).resolve()
    )
    bridge_sha = sha256_json(bridge_contract)

    pv_frame = pq.ParquetFile(pv_source).read().to_pandas()
    workload_days = _complete_workload_days(
        workload_frame, bridge_contract=bridge_contract
    )
    pv_days = _complete_pv_days(pv_frame)
    price_days, oos_price_hashes = _selection_oos_price_days(root)

    selection_start = pd.Timestamp("2022-01-01T00:00:00Z")
    workload_forcing_days = tuple(
        path
        for path in sorted(workload_days.values(), key=lambda value: value.operating_date)
        if path.max_timestamp_utc < selection_start
    )
    if not workload_forcing_days:
        raise CounterfactualMaterializationError(
            "no pre-selection NLR days exist for fixed forcing mapping"
        )

    coverage_records = {
        str(record["operating_date"]): dict(record)
        for record in coverage.get("records", [])
    }
    if require_exact_calendar and set(coverage_records) != set(calendar_dates):
        raise CounterfactualMaterializationError(
            "settlement ledger and selection calendar differ"
        )
    materialization_contract = {
        "schema_version": SCHEMA_VERSION,
        "design_schema_version": DESIGN_SCHEMA_VERSION,
        "authorization": "user_approved_conditional_settlement_dual_track_2026_08_12",
        "selection_calendar_day_count": len(calendar_dates),
        "selection_conditional_valid_day_count": len(dates),
        "draw_count": draw_count,
        "root_seed": ROOT_SEED,
        "seasonal_pool_limit": SEASONAL_POOL_LIMIT,
        "issue_time": "previous_operating_day_00_00_UTC",
        "workload_issue_method": "causal_same_weekday_circular_doy_block_bootstrap",
        "pv_issue_method": "causal_deployable_circular_doy_block_bootstrap",
        "common_price_method": (
            "frozen_selection_OOS_seasonal_naive_DA_RT_draws_at_common_DA_gate"
        ),
        "workload_forcing_method": (
            "fixed_circular_mapping_over_complete_pre_2022_NLR_days"
        ),
        "pv_forcing_method": "target_day_NASA_reconstruction_interval_reveal",
        "actionability": "battery_only_migration_disabled",
        "no_action_band_usd_per_mwh_equivalent": [-1.0, 1.0],
        "candidate_independent": True,
        "settlement_independent_materialization": True,
        "economic_replay_mask": "frozen_post_recovery_conditional_settlement_ledger",
        "locked_2025_opened": False,
        "workload_unit_bridge_contract_sha256": bridge_sha,
    }
    materialization_contract_sha = sha256_json(materialization_contract)
    source_hashes = {
        "nlr295_workload": workload_sha,
        "nasa_derived_pv": pv_sha,
        "frozen_physical_artifact": physical_sha,
        "frozen_nyiso_twins": twin_sha,
        "workload_unit_bridge_contract": bridge_sha,
        "materialization_contract": materialization_contract_sha,
        "settlement_coverage_snapshot": coverage_sha,
        **oos_price_hashes,
    }
    for label, digest in source_hashes.items():
        if len(digest) != 64:
            raise CounterfactualMaterializationError(f"invalid source hash: {label}")

    bundles_root = destination / "selection" / "daily_bundles"
    progress_path = destination / "selection" / "materialization_progress.json"

    def write_day(day: str) -> dict[str, Any]:
        path = bundles_root / f"{day}.npz"
        record_path = bundles_root / f"{day}.record.json"
        if path.is_file() and record_path.is_file():
            cached = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                cached.get("materialization_contract_sha256")
                == materialization_contract_sha
                and cached.get("sha256") == sha256_file(path)
                and cached.get("operating_date") == day
            ):
                return dict(cached["record"])
        arrays, diagnostics = _build_daily_arrays(
            day,
            workload_days=workload_days,
            pv_days=pv_days,
            price_days=price_days,
            workload_forcing_days=workload_forcing_days,
            physical=physical,
            source_hashes=source_hashes,
            draw_count=draw_count,
        )
        _atomic_npz(path, arrays)
        digest = sha256_file(path)
        settlement = coverage_records.get(day, {})
        record = {
            "operating_date": day,
            "path": _relative(root, path),
            "sha256": digest,
            "forecast_id": str(arrays["forecast_id"].item()),
            "issue_time_utc": diagnostics["issue_time_utc"],
            "timestamp_panel_sha256": sha256_json(
                [value.isoformat() for value in pd.date_range(day, periods=96, freq="15min", tz="UTC")]
            ),
            "causal_gates": {
                "workload": diagnostics["causal_workload"],
                "pv": diagnostics["causal_pv"],
                "candidate_independent_price": diagnostics[
                    "causal_candidate_independent_price"
                ],
            },
            "draw_source_dates_sha256": {
                "workload": diagnostics["workload_draw_source_dates_sha256"],
                "pv": diagnostics["pv_draw_source_dates_sha256"],
                "candidate_independent_price": diagnostics[
                    "price_draw_source_dates_sha256"
                ],
            },
            "forcing_workload_source_date": diagnostics[
                "forcing_workload_source_date"
            ],
            "settlement_status_snapshot": settlement.get(
                "decision_replay_cell_status", "UNASSESSED"
            ),
            "solver_launched": False,
        }
        atomic_json(
            record_path,
            {
                "schema_version": SCHEMA_VERSION,
                "materialization_contract_sha256": materialization_contract_sha,
                "operating_date": day,
                "sha256": digest,
                "record": record,
            },
        )
        return record

    records: list[dict[str, Any]] = []
    atomic_json(
        progress_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "RUNNING",
            "expected_day_count": len(dates),
            "completed_day_count": 0,
            "pending_day_count": len(dates),
            "materialization_contract_sha256": materialization_contract_sha,
            "started_at_utc": utc_now(),
        },
    )
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(write_day, day): day for day in dates}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed_count % 16 == 0 or completed_count == len(dates):
                atomic_json(
                    progress_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": (
                            "MATERIALIZED_AWAITING_FINAL_INDEX"
                            if completed_count == len(dates)
                            else "RUNNING"
                        ),
                        "expected_day_count": len(dates),
                        "completed_day_count": completed_count,
                        "pending_day_count": len(dates) - completed_count,
                        "materialization_contract_sha256": materialization_contract_sha,
                        "updated_at_utc": utc_now(),
                    },
                )
    records.sort(key=lambda record: record["operating_date"])

    index_paths: dict[str, str] = {}
    for kind in (
        "issue_time_ensembles_by_date",
        "replay_forcing_by_date",
        "actionability_by_date",
    ):
        path = destination / "selection" / f"{kind}_index.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "phase": "selection",
            "kind": kind,
            "day_count": len(records),
            "shared_immutable_daily_bundles": True,
            "days": records,
            "records_sha256": sha256_json(records),
        }
        atomic_json(path, payload)
        index_paths[kind] = _relative(root, path)

    calendar_ledger = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_CALENDAR_WITH_DYNAMIC_SETTLEMENT_MASK",
        "phase": "selection",
        "calendar_day_count": len(calendar_dates),
        "input_materialized_day_count": len(records),
        "date_drop_count": 0,
        "missing_settlement_imputed": False,
        "settlement_snapshot_path": _relative(root, coverage_source),
        "settlement_snapshot_sha256": coverage_sha,
        "current_snapshot_evaluable_day_count": sum(
            bool(coverage_records.get(day, {}).get("settlement_evaluable", False))
            for day in calendar_dates
        ),
        "current_snapshot_invalid_day_count": sum(
            not bool(coverage_records.get(day, {}).get("settlement_evaluable", False))
            for day in calendar_dates
        ),
        "conditional_valid_operating_dates_sha256": sha256_json(list(dates)),
        "conditional_invalid_operating_dates": [
            day
            for day in calendar_dates
            if not bool(coverage_records.get(day, {}).get("settlement_evaluable", False))
        ],
        "final_mask_binding": (
            "resolve_latest_post_recovery_conditional_settlement_ledger_before_solver_launch"
        ),
        "invalid_day_policy": "INVALID_MISSING_SETTLEMENT_NO_MILP_NO_COST",
        "records": [
            {
                "operating_date": day,
                "input_path": (
                    next(
                        (
                            record["path"]
                            for record in records
                            if record["operating_date"] == day
                        ),
                        None,
                    )
                ),
                "input_sha256": (
                    next(
                        (
                            record["sha256"]
                            for record in records
                            if record["operating_date"] == day
                        ),
                        None,
                    )
                ),
                "settlement_status_snapshot": coverage_records.get(day, {}).get(
                    "decision_replay_cell_status", "UNASSESSED"
                ),
                "solver_launched": False,
            }
            for day in calendar_dates
        ],
    }
    calendar_ledger["records_sha256"] = sha256_json(calendar_ledger["records"])
    calendar_path = destination / "selection" / "full_calendar_input_ledger.json"
    atomic_json(calendar_path, calendar_ledger)

    locked_indices: dict[str, str] = {}
    for kind in (
        "issue_time_ensembles_by_date",
        "replay_forcing_by_date",
        "actionability_by_date",
    ):
        path = destination / "locked_test" / f"{kind}_index.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "CLOSED_NOT_OPENED",
            "phase": "locked_test",
            "kind": kind,
            "expected_calendar_day_count": len(exact_phase_dates("locked_test")),
            "materialized_day_count": 0,
            "days": [],
            "reason": (
                "2025 remains sealed until selection replay and the static selection rule are frozen"
            ),
            "locked_predictions_or_targets_opened": False,
        }
        atomic_json(path, payload)
        locked_indices[kind] = _relative(root, path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "SELECTION_INPUTS_COMPLETE_LOCKED_TEST_CLOSED",
        "created_at_utc": utc_now(),
        "selection": {
            "calendar_day_count": len(calendar_dates),
            "materialized_day_count": len(records),
            "candidate_independent": True,
            "shared_daily_bundle_count": len(records),
            "three_index_paths": index_paths,
            "calendar_ledger_path": _relative(root, calendar_path),
            "calendar_ledger_sha256": sha256_file(calendar_path),
            "solver_runs_launched": 0,
            "model_training_runs_launched": 0,
            "model_prediction_runs_launched": 0,
        },
        "locked_test": {
            "status": "CLOSED_NOT_OPENED",
            "calendar_day_count": len(exact_phase_dates("locked_test")),
            "materialized_day_count": 0,
            "three_index_paths": locked_indices,
            "locked_predictions_or_targets_opened": False,
        },
        "materialization_contract": materialization_contract,
        "materialization_contract_sha256": materialization_contract_sha,
        "workload_unit_bridge_contract": bridge_contract,
        "workload_unit_bridge_contract_sha256": bridge_sha,
        "source_hashes": source_hashes,
        "source_paths": {
            "nlr295_workload": _relative(root, workload_source),
            "nasa_derived_pv": _relative(root, pv_source),
            "frozen_physical_artifact": _relative(root, physical_source),
            "frozen_nyiso_twins": _relative(root, twin_source),
            "settlement_coverage_snapshot": _relative(root, coverage_source),
        },
        "causal_audit": {
            "selection_day_count": len(records),
            "all_issue_workload_sources_strictly_before_issue": all(
                record["causal_gates"]["workload"] for record in records
            ),
            "all_issue_pv_sources_deployable_by_issue": all(
                record["causal_gates"]["pv"] for record in records
            ),
            "all_common_price_sources_strictly_before_issue": all(
                record["causal_gates"]["candidate_independent_price"]
                for record in records
            ),
            "target_day_pv_only_in_replay_forcing": True,
            "future_realized_workload_or_pv_used_at_issue": False,
            "locked_2025_opened": False,
        },
        "claim_boundaries": {
            "counterfactual": True,
            "nlr_nyiso_joint_observation_claimed": False,
            "nlr_forcing_role": "fixed_cross_domain_replay_from_pre_selection_days",
            "nasa_pv_role": "third_party_reconstruction_not_site_measurement",
            "migration_state_observed": False,
            "migration_actions_enabled": False,
            "settlement_completeness_claimed": False,
        },
        "replay_admission": {
            "selection_inputs_ready": True,
            "economic_solver_launch_allowed_by_this_manifest": False,
            "required_before_solver": (
                "bind final post-recovery conditional settlement valid-date ledger"
            ),
            "invalid_dates_are_solver_failures": False,
        },
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    manifest_path = destination / "materialization_manifest.json"
    atomic_json(manifest_path, manifest)
    config_ready = {
        "schema_version": SCHEMA_VERSION,
        "status": "SELECTION_READY_LOCKED_TEST_CLOSED",
        "phase": "selection",
        "selection_valid_date_count": len(records),
        "full_calendar_ledger": {
            "path": _relative(root, calendar_path),
            "sha256": sha256_file(calendar_path),
        },
        "conditional_settlement_ledger": {
            "path": _relative(root, coverage_source),
            "sha256": coverage_sha,
        },
        "dart_real_evidence_inputs": {
            "issue_time_ensembles_by_date": {
                "index_path": index_paths["issue_time_ensembles_by_date"]
            },
            "replay_forcing_by_date": {
                "index_path": index_paths["replay_forcing_by_date"]
            },
            "actionability_by_date": {
                "index_path": index_paths["actionability_by_date"]
            },
            "nyiso_replay_by_date": {
                "index_path": (
                    "artifacts/dart_value_select/real_input_preparation/v1/"
                    "official_rt_recovery/selection_nyiso_replay_by_date_index.json"
                )
            },
            "referenced_nyiso_data_root": "data/processed/referenced/nyiso",
        },
        "injection": (
            "forecast_result['dart_real_evidence_inputs'] = "
            "payload['dart_real_evidence_inputs']"
        ),
        "locked_test_opened": False,
    }
    config_path = destination / "selection_config_ready_input_block.json"
    atomic_json(config_path, config_ready)
    atomic_json(
        progress_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "expected_day_count": len(dates),
            "completed_day_count": len(records),
            "pending_day_count": 0,
            "materialization_contract_sha256": materialization_contract_sha,
            "manifest_path": _relative(root, manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "config_ready_input_block_path": _relative(root, config_path),
            "config_ready_input_block_sha256": sha256_file(config_path),
            "completed_at_utc": utc_now(),
        },
    )
    return {
        **manifest,
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
