"""Delayed sealed-corpus factory for the one-shot R-ECMoE final evaluation.

Importing this module performs no filesystem discovery or data loading.  The
public factory is called only after :mod:`r_ecmoe_sealed_2025` has consumed the
authorized one-shot sentinel.  At that point it checks exact configured daily
paths (never directory scans), loads immutable train-only statistics, and
constructs final packets with the existing WorldMoE packet machinery.

The development corpus remains unchanged and continues to reject final-year
dates.  Small final-only adapters below relax only the date gate after the
one-shot authorization; they retain pre-final NASA and migration evidence and
never refit normalization, proper-score thresholds, or model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from dcvpp.e2e_moe_r4.constrained_runtime import load_physical_limits
from dcvpp.e2e_moe_r4.physical_twin import load_frozen_physical_twin_core
from dcvpp.e2e_moe_r4.training_data import (
    EMPIRICAL_SCENARIOS,
    _RawDay,
    _allocate_total_workload,
    _day_to_tensors,
    _finite,
    _resolve_under,
)

from . import data as world_data
from .config import WorldMoEConfig
from .lineage import validate_train_only_normalization
from .migration import (
    MigrationEvidenceError,
    MigrationEvidenceStore,
    MigrationSnapshot,
)
from .nasa_power_weather import NasaPowerLagPersistenceStore
from .r_ecmoe_sealed_2025 import (
    FINAL_SPLIT,
    FINAL_YEAR,
    FROZEN_MARKER_SCHEMA,
    SEALED_CORPUS_SCHEMA,
)


class RECMoESealedCorpusError(RuntimeError):
    """The frozen final corpus cannot be constructed without data adaptation."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RECMoESealedCorpusError(f"{label} must be a mapping")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise RECMoESealedCorpusError(f"{label} path is required")
    return _resolve_under(root, value, label)


def _date_range(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _locked_sibling(path: Path) -> Path:
    """Derive the configured locked-test sibling without scanning directories."""

    parts = list(path.resolve().parts)
    indices = [index for index, value in enumerate(parts) if value.lower() == "selection"]
    if not indices:
        raise RECMoESealedCorpusError(
            f"configured development root has no selection component: {path}"
        )
    parts[indices[-1]] = "locked_test"
    return Path(*parts)


def _protocol_dates(
    config: Mapping[str, Any], marker: Mapping[str, Any]
) -> tuple[date, date, int]:
    if marker.get("schema") != FROZEN_MARKER_SCHEMA:
        raise RECMoESealedCorpusError("factory requires the validated frozen marker")
    if (
        marker.get("final_year") != FINAL_YEAR
        or marker.get("model_selection_frozen") is not True
        or marker.get("thresholds_frozen") is not True
        or marker.get("data_statistics_frozen") is not True
        or marker.get("locked_final_opened") is not False
    ):
        raise RECMoESealedCorpusError("factory marker does not authorize frozen final data")
    protocol = _mapping(config.get("protocol"), "protocol")
    splits = _mapping(protocol.get("splits"), "protocol.splits")
    final = _mapping(splits.get("final_test"), "protocol.splits.final_test")
    try:
        start = date.fromisoformat(str(final.get("start")))
        end = date.fromisoformat(str(final.get("end")))
    except ValueError as exc:
        raise RECMoESealedCorpusError("final split dates must use ISO format") from exc
    configured_days = int(final.get("days", -1))
    marker_days = int(marker.get("expected_evaluable_days", -1))
    calendar_days = int(final.get("calendar_days", (end - start).days + 1))
    if (
        start.year != FINAL_YEAR
        or end.year != FINAL_YEAR
        or start > end
        or calendar_days != (end - start).days + 1
        or configured_days != marker_days
        or final.get("sealed_until_all_methods_frozen") is not True
        or final.get("execute_once") is not True
    ):
        raise RECMoESealedCorpusError("configured final split differs from frozen marker")
    return start, end, marker_days


def _complete_exact_days(
    bundle_root: Path,
    settlement_root: Path,
    dates: Sequence[date],
    *,
    required_count: int | None,
) -> tuple[date, ...]:
    """Check exact expected filenames; directory enumeration is prohibited."""

    available: list[date] = []
    for operating in dates:
        day = operating.isoformat()
        paths = (
            bundle_root / f"{day}.npz",
            bundle_root / f"{day}.record.json",
            settlement_root / f"{day}.npz",
        )
        exists = tuple(path.is_file() for path in paths)
        if any(exists) and not all(exists):
            raise RECMoESealedCorpusError(
                f"daily final modalities are partially available for {day}"
            )
        if all(exists):
            available.append(operating)
    if required_count is not None and len(available) != int(required_count):
        raise RECMoESealedCorpusError(
            f"final corpus expected {required_count} complete days, found {len(available)}"
        )
    return tuple(available)


def _load_frozen_world_statistics(
    path: Path,
) -> tuple[world_data.WorldNormalizationStatistics, world_data.WorldTargetStatistics]:
    if not path.is_file():
        raise RECMoESealedCorpusError(f"frozen normalization artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RECMoESealedCorpusError("frozen normalization artifact is invalid") from exc
    if not isinstance(payload, Mapping):
        raise RECMoESealedCorpusError("frozen normalization root must be a mapping")
    validate_train_only_normalization(payload)
    if (
        payload.get("fit_split") != "train_2022_2023_only"
        or payload.get("used_validation_statistics") is not False
        or payload.get("opened_final_test_2025") is not False
    ):
        raise RECMoESealedCorpusError("normalization artifact is not train-only frozen")
    fit_dates = tuple(str(value) for value in payload.get("fit_dates", ()))

    def tensor(name: str, *, fallback: str | None = None) -> Tensor:
        value = payload.get(name)
        if value is None and fallback is not None:
            value = payload.get(fallback)
        if value is None:
            raise RECMoESealedCorpusError(f"normalization omitted {name}")
        return torch.as_tensor(value, dtype=torch.float32)

    caps = _mapping(payload.get("forecast_residual_caps"), "forecast_residual_caps")
    normalization = world_data.WorldNormalizationStatistics(
        fast_scale=tensor("fast_slot_feature_scale", fallback="fast_feature_scale"),
        medium_scale=tensor(
            "medium_slot_feature_scale", fallback="medium_feature_scale"
        ),
        slow_scale=tensor("slow_slot_feature_scale", fallback="slow_feature_scale"),
        fast_center=tensor("fast_slot_feature_center"),
        medium_center=tensor("medium_slot_feature_center"),
        slow_center=tensor("slow_slot_feature_center"),
        pv_capacity=tensor("pv_capacity_train_max_by_zone"),
        pue_min=tensor("pue_train_min_by_zone"),
        pue_max=tensor("pue_train_max_by_zone"),
        pue_issue_center=tensor("pue_issue_center_train_median_96x4"),
        da_residual_cap=torch.as_tensor(caps.get("da"), dtype=torch.float32),
        fit_dates=fit_dates,
    ).validate()
    target_payload = _mapping(
        payload.get("target_score_statistics"), "target_score_statistics"
    )
    if (
        target_payload.get("fit_split") != "train_2022_2023_only"
        or target_payload.get("opened_final_test_2025") is not False
    ):
        raise RECMoESealedCorpusError("target thresholds are not train-only frozen")
    target_dates = tuple(str(value) for value in target_payload.get("fit_dates", ()))
    state_baselines = _mapping(
        target_payload.get("state_training_baselines"), "state_training_baselines"
    )
    state_baseline_metadata = _mapping(
        target_payload.get("state_training_baseline_metadata"),
        "state_training_baseline_metadata",
    )
    gate_baselines = _mapping(
        target_payload.get("state_gate_baselines"), "state_gate_baselines"
    )
    target = world_data.WorldTargetStatistics(
        center=torch.as_tensor(target_payload.get("center"), dtype=torch.float32),
        scale=torch.as_tensor(target_payload.get("scale"), dtype=torch.float32),
        tail_thresholds=torch.as_tensor(
            target_payload.get("tail_thresholds"), dtype=torch.float32
        ),
        state_center=torch.as_tensor(
            target_payload.get("state_center"), dtype=torch.float32
        ),
        state_scale=torch.as_tensor(
            target_payload.get("state_scale"), dtype=torch.float32
        ),
        queue_event_prevalence=torch.tensor(
            float(target_payload.get("train_aggregate_queue_event_prevalence")),
            dtype=torch.float32,
        ),
        baseline_daily_cost=torch.tensor(
            float(target_payload.get("train_frozen_baseline_daily_cost")),
            dtype=torch.float32,
        ),
        fit_dates=target_dates,
        state_training_baseline_soc=torch.tensor(
            float(state_baselines.get("soc")), dtype=torch.float32
        ),
        state_training_baseline_queue_occurrence=torch.tensor(
            float(state_baselines.get("queue_occurrence")), dtype=torch.float32
        ),
        state_training_baseline_queue_severity=torch.tensor(
            float(state_baselines.get("queue_severity")), dtype=torch.float32
        ),
        state_gate_baseline_soc_persistence=torch.tensor(
            float(gate_baselines.get("soc_persistence")), dtype=torch.float32
        ),
        state_training_baseline_soc_reference=str(
            state_baseline_metadata.get("soc", "frozen_train_reference")
        ),
    ).validate()
    if target.fit_dates != normalization.fit_dates:
        raise RECMoESealedCorpusError(
            "feature and target statistics use different frozen train dates"
        )
    return normalization, target


@dataclass(frozen=True)
class _IdentityLegacyStatistics:
    """Lossless carrier for raw issue distributions, never a fitted statistic."""

    market_scale: np.ndarray
    workload_scale: np.ndarray
    pv_scale: np.ndarray
    forecast_da_residual_cap: np.ndarray
    forecast_pv_residual_cap: np.ndarray
    forecast_pue_residual_cap: np.ndarray

    @classmethod
    def from_world(
        cls, statistics: world_data.WorldNormalizationStatistics
    ) -> "_IdentityLegacyStatistics":
        da = statistics.da_residual_cap
        if da is None:
            raise RECMoESealedCorpusError("frozen world statistics omitted DA cap")
        return cls(
            market_scale=np.ones(16, dtype=np.float32),
            workload_scale=np.ones(5, dtype=np.float32),
            pv_scale=np.ones(8, dtype=np.float32),
            forecast_da_residual_cap=da.cpu().numpy().astype(np.float32),
            forecast_pv_residual_cap=np.ones(4, dtype=np.float32),
            forecast_pue_residual_cap=np.ones(4, dtype=np.float32),
        )

    def forecast_residual_cap_sha256(self) -> str:
        return "NOT_COMPUTED_FINAL_INFERENCE_ONLY"


def _load_sealed_raw_day(
    bundle_path: Path,
    settlement_path: Path,
    capacity_work_units: np.ndarray,
    *,
    expected_locked: bool,
) -> _RawDay:
    """Read one explicitly authorized day using the registered bundle schema."""

    day_text = bundle_path.stem
    operating = date.fromisoformat(day_text)
    if expected_locked != (operating.year == FINAL_YEAR):
        raise RECMoESealedCorpusError("raw-day authorization does not match its date")
    record_path = bundle_path.with_suffix(".record.json")
    if not record_path.is_file():
        raise RECMoESealedCorpusError(f"causal-gate record is missing: {record_path}")
    record_payload = json.loads(record_path.read_text(encoding="utf-8"))
    record = record_payload.get("record", {}) if isinstance(record_payload, Mapping) else {}
    gates = record.get("causal_gates", {}) if isinstance(record, Mapping) else {}
    required_gates = ("candidate_independent_price", "pv", "workload")
    if not all(gates.get(name) is True for name in required_gates):
        raise RECMoESealedCorpusError(f"causal gates failed for {day_text}")
    with np.load(bundle_path, allow_pickle=False) as source:
        if str(source["operating_date"].item())[:10] != day_text:
            raise RECMoESealedCorpusError("bundle operating date mismatch")
        if bool(source["locked_test"].item()) is not bool(expected_locked):
            raise RECMoESealedCorpusError("bundle locked-test flag mismatches authorization")
        da = _finite(source["da_price"], (4, world_data.FAST_STEPS), "da_price")
        rt_pool = _finite(
            source["rt_price_pool"],
            (EMPIRICAL_SCENARIOS, 4, world_data.FAST_STEPS),
            "rt_price_pool",
        )
        workload_pool = _finite(
            source["workload_pool"],
            (EMPIRICAL_SCENARIOS, world_data.FAST_STEPS),
            "workload_pool",
        )
        pv_pool = _finite(
            source["pv_pool"],
            (EMPIRICAL_SCENARIOS, 4, world_data.FAST_STEPS),
            "pv_pool",
        )
        realized_workload = _finite(
            source["workload"], (world_data.FAST_STEPS,), "workload"
        )
        realized_pv = _finite(
            source["pv"], (4, world_data.FAST_STEPS), "pv"
        )
        multiplier = _finite(source["physical_multiplier"], (4,), "physical_multiplier")
        initial_soc = _finite(source["initial_soc_mwh_by_zone"], (4,), "initial_soc")
        issue_time = str(source["issue_time_utc"].item())
    with np.load(settlement_path, allow_pickle=False) as settlement:
        realized_da = _finite(
            settlement["da_lbmp"], (4, world_data.FAST_STEPS), "realized_da"
        )
        realized_rt = _finite(
            settlement["rt_lbmp"], (4, world_data.FAST_STEPS), "realized_rt"
        )
    da_time = da.T
    rt_mean = rt_pool.mean(axis=0).T
    rt_std = rt_pool.std(axis=0).T
    market = np.concatenate((da_time, rt_mean, da_time - rt_mean, rt_std), axis=-1)
    quantiles = np.quantile(workload_pool, (0.1, 0.5, 0.9), axis=0)
    workload = np.stack(
        (workload_pool.mean(0), workload_pool.std(0), *quantiles), axis=-1
    ).astype(np.float32)
    issue_pv = pv_pool.mean(0).T.astype(np.float32)
    issue_utilization = _allocate_total_workload(
        workload_pool.mean(0), capacity_work_units
    ) / np.asarray(capacity_work_units, dtype=np.float32)[None]
    physical_dynamic = np.concatenate(
        (issue_pv, pv_pool.std(0).T, issue_utilization), axis=-1
    ).astype(np.float32)
    return _RawDay(
        operating_date=day_text,
        market=market.astype(np.float32),
        workload=workload,
        physical_static=multiplier,
        physical_dynamic=physical_dynamic,
        initial_soc=initial_soc,
        realized_da=realized_da.T,
        realized_rt=realized_rt.T,
        realized_workload=realized_workload,
        realized_pv=realized_pv.T,
        issue_workload=workload_pool.mean(0).astype(np.float32),
        issue_pv=issue_pv,
        provenance={
            "bridge": "sealed_final_counterfactual_input_record_v1",
            "record_path": str(record_path.resolve()),
            "bundle_path": str(bundle_path.resolve()),
            "settlement_path": str(settlement_path.resolve()),
            "hash_verification": "NOT_COMPUTED_FINAL_INFERENCE_ONLY",
            "issue_time_utc": issue_time,
            "causal_gates": {name: True for name in required_gates},
            "opened_final_test_2025": bool(expected_locked),
        },
    )


class SealedFinalNasaPowerLagPersistenceStore(NasaPowerLagPersistenceStore):
    """Use only the parent's pre-final source table for final lag persistence."""

    @staticmethod
    def _timestamps(
        decision_origin: datetime | pd.Timestamp | str,
        valid_times: Sequence[datetime | pd.Timestamp | str],
    ) -> tuple[pd.Timestamp, tuple[pd.Timestamp, ...], date]:
        origin = pd.Timestamp(decision_origin)
        if origin.tzinfo is None:
            raise RECMoESealedCorpusError("weather origin must be timezone-aware")
        origin = origin.tz_convert("UTC")
        targets = tuple(pd.Timestamp(value) for value in valid_times)
        if not targets or any(value.tzinfo is None for value in targets):
            raise RECMoESealedCorpusError("weather targets must be timezone-aware")
        targets = tuple(value.tz_convert("UTC") for value in targets)
        operating_days = {value.date() for value in targets}
        if (
            origin.year != FINAL_YEAR
            or len(operating_days) != 1
            or next(iter(operating_days)).year != FINAL_YEAR
        ):
            raise RECMoESealedCorpusError("sealed weather query left the final year")
        return origin, targets, next(iter(operating_days))


class SealedFinalMigrationEvidenceStore(MigrationEvidenceStore):
    """Permit final cutoffs while retaining the pre-final evidence table."""

    def snapshot(self, cutoff: datetime) -> MigrationSnapshot:
        if cutoff.tzinfo is None:
            raise MigrationEvidenceError("migration cutoff must be timezone-aware")
        instant = cutoff.astimezone(timezone.utc)
        if instant.year != FINAL_YEAR:
            raise MigrationEvidenceError("sealed migration cutoff left the final year")
        cutoff_ns = int(pd.Timestamp(instant).value)
        snapshot = MigrationSnapshot.unavailable()
        capacity = snapshot.capacity.clone()
        allowed = snapshot.allowed.clone()
        features = snapshot.edge_features.clone()
        ages = snapshot.age_minutes.clone()
        observed = snapshot.observed_timestamp.clone()
        for (source, destination), values in self._routes.items():
            index = int(
                np.searchsorted(values["timestamp_ns"], cutoff_ns, side="right") - 1
            )
            if index < 0:
                continue
            timestamp_ns = int(values["timestamp_ns"][index])
            age = (cutoff_ns - timestamp_ns) / 60.0e9
            median = float(values["median_rtt_ms"][index])
            p95 = float(values["p95_rtt_ms"][index])
            loss = float(values["packet_loss_rate"][index])
            sent = int(values["sent_packets"][index])
            available = (
                0 <= age <= self.max_age_minutes
                and np.isfinite(median)
                and median >= 0
                and np.isfinite(p95)
                and p95 >= median
                and np.isfinite(loss)
                and 0 <= loss <= self.max_packet_loss_rate
                and sent >= self.min_sent_packets
            )
            ages[source, destination] = float(age)
            observed[source, destination] = timestamp_ns // 1_000_000_000
            if not available:
                continue
            capacity[source, destination] = self.capacity_work_units
            allowed[source, destination] = True
            features[source, destination] = torch.tensor(
                (
                    self.capacity_work_units,
                    median / 100.0,
                    p95 / 100.0,
                    loss,
                    age / self.max_age_minutes,
                    1.0,
                ),
                dtype=torch.float32,
            )
        return MigrationSnapshot(capacity, allowed, features, ages, observed)

    def causal_day_series(
        self, operating_day: date, *, origin: int, steps: int = 96
    ) -> tuple[Tensor, Tensor, MigrationSnapshot]:
        # WorldMoECorpus receives a non-final alias solely to pass its immutable
        # development date guard.  Restore the authorized year here.
        if operating_day.year != FINAL_YEAR:
            operating_day = operating_day.replace(year=FINAL_YEAR)
        if not 0 <= int(origin) < int(steps):
            raise MigrationEvidenceError("rolling origin is outside the day")
        day_start = datetime.combine(operating_day, time.min, tzinfo=timezone.utc)
        features: list[Tensor] = []
        masks: list[Tensor] = []
        origin_snapshot: MigrationSnapshot | None = None
        for step in range(int(steps)):
            effective = min(step, int(origin))
            snapshot = self.snapshot(day_start + timedelta(minutes=15 * effective))
            features.append(snapshot.edge_features)
            masks.append(snapshot.allowed)
            if step == int(origin):
                origin_snapshot = snapshot
        if origin_snapshot is None:
            origin_snapshot = self.snapshot(
                day_start + timedelta(minutes=15 * int(origin))
            )
        return torch.stack(features), torch.stack(masks), origin_snapshot


def _alias_development_date(value: date) -> date:
    return value if value.year < 2025 else value.replace(year=2024)


def _calendar_correct_context(
    values: Tensor, mask: Tensor, operating: date, *, holiday: float, dst: float
) -> tuple[Tensor, Tensor]:
    values = values.clone()
    values[:, 39, 0] = operating.weekday() / 6.0
    values[:, 39, 1] = (operating.month - 1) / 11.0
    values[:, 39, 2] = holiday
    values[:, 39, 3] = dst
    values[:, 39, 7] = torch.arange(world_data.FAST_STEPS) / world_data.FAST_STEPS
    mask = mask.clone()
    mask[:, 39] = True
    return values, mask


class SealedFinalWorldMoECorpus(world_data.WorldMoECorpus):
    """Final-only view reusing frozen WorldMoE packet and batch construction."""

    def __init__(
        self,
        config: Mapping[str, Any],
        repository_root: Path,
        marker: Mapping[str, Any],
        *,
        final_bundle_root: Path,
        final_settlement_root: Path,
        history_bundle_root: Path,
        history_settlement_root: Path,
        final_dates: Sequence[date],
        final_start: date,
        statistics: world_data.WorldNormalizationStatistics,
        target_statistics: world_data.WorldTargetStatistics,
    ) -> None:
        # Deliberately do not call WorldMoECorpus.__init__: it fits statistics
        # and rejects final-year records, both correct for development only.
        self.config = dict(config)
        self.repository_root = repository_root.resolve()
        self.model_config = WorldMoEConfig.from_mapping(
            _mapping(config.get("model"), "model")
        )
        data_config = _mapping(config.get("data"), "data")
        self.deadline_intervals = int(data_config.get("deadline_intervals", 4))
        if not 1 <= self.deadline_intervals <= world_data.FAST_STEPS:
            raise RECMoESealedCorpusError("deadline_intervals must lie in [1,96]")
        self.statistics = statistics.validate()
        self.target_statistics = target_statistics.validate()
        identity = _IdentityLegacyStatistics.from_world(self.statistics)
        self.base = SimpleNamespace(statistics=identity)

        physical_path = _resolve(
            self.repository_root, data_config.get("frozen_physical_twin"), "physical twin"
        )
        surrogate_path = _resolve(
            self.repository_root,
            data_config.get("frozen_physical_surrogate"),
            "physical surrogate",
        )
        calibration_path = _resolve(
            self.repository_root,
            data_config.get("physical_calibration_manifest"),
            "physical calibration manifest",
        )
        self.physical = load_physical_limits(physical_path)
        physical_core = load_frozen_physical_twin_core(
            artifact_path=physical_path,
            surrogate_path=surrogate_path,
            calibration_manifest_path=calibration_path,
        )

        weather_config = _mapping(data_config.get("weather", {}), "data.weather")
        self.weather_source = str(
            weather_config.get("primary_source", world_data.NASA_POWER_LAG_SOURCE)
        )
        self.weather_store: SealedFinalNasaPowerLagPersistenceStore | None = None
        if weather_config.get("enabled") is True:
            weather_path = _resolve(
                self.repository_root, weather_config.get("source_path"), "NASA weather"
            )
            self.weather_store = SealedFinalNasaPowerLagPersistenceStore(weather_path)

        migration_config = _mapping(data_config.get("migration"), "data.migration")
        self.migration: SealedFinalMigrationEvidenceStore | None = None
        if migration_config.get("enabled") is True:
            availability = _mapping(
                migration_config.get("availability"), "migration availability"
            )
            evidence_path = _resolve(
                self.repository_root,
                migration_config.get("evidence_path"),
                "migration evidence",
            )
            manifest_path = _resolve(
                self.repository_root,
                migration_config.get("manifest_path"),
                "migration manifest",
            )
            self.migration = SealedFinalMigrationEvidenceStore.from_artifacts(
                evidence_path,
                manifest_path,
                max_age_minutes=float(availability.get("max_age_minutes")),
                max_packet_loss_rate=float(availability.get("max_packet_loss_rate")),
                min_sent_packets=int(availability.get("min_sent_packets")),
            )
        self.graph = world_data._build_graph_contract(
            tuple(self.migration.supported_routes) if self.migration else ()
        )

        history_dates = tuple(
            _date_range(final_start - timedelta(days=world_data.SLOW_STEPS), final_start - timedelta(days=1))
        )
        available_history = _complete_exact_days(
            history_bundle_root,
            history_settlement_root,
            history_dates,
            required_count=None,
        )
        capacity = np.asarray(self.physical.capacity_work_units, dtype=np.float32)
        raw_history = [
            _load_sealed_raw_day(
                history_bundle_root / f"{value.isoformat()}.npz",
                history_settlement_root / f"{value.isoformat()}.npz",
                capacity,
                expected_locked=False,
            )
            for value in available_history
        ]
        raw_final = [
            _load_sealed_raw_day(
                final_bundle_root / f"{value.isoformat()}.npz",
                final_settlement_root / f"{value.isoformat()}.npz",
                capacity,
                expected_locked=True,
            )
            for value in final_dates
        ]
        base_history: list[Mapping[str, Any]] = []
        raw_history_chain: list[_RawDay] = []
        for raw in raw_history:
            record = _day_to_tensors(
                raw, identity, self.physical, physical_core, raw_history_chain
            )
            base_history.append(record)
            raw_history_chain.append(raw)
        final_base_list: list[Mapping[str, Any]] = []
        for raw in raw_final:
            record = _day_to_tensors(
                raw, identity, self.physical, physical_core, raw_history_chain
            )
            final_base_list.append(record)
            raw_history_chain.append(raw)
        final_base = tuple(final_base_list)
        # Each final anchor sees all prior history but no later final outcome.
        # This path applies immutable statistics and performs no fit.

        completed_cache: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        completed: list[Mapping[str, Any]] = list(base_history)
        for record in (*base_history, *final_base):
            operating = date.fromisoformat(str(record["operating_date"]))
            alias = dict(record)
            alias["operating_date"] = _alias_development_date(operating).isoformat()
            values, mask = world_data._completed_day_context(
                alias,
                physical=self.physical,
                pv_capacity=self.statistics.pv_capacity,
                pue_min=self.statistics.pue_min,
                pue_max=self.statistics.pue_max,
            )
            values, mask = _calendar_correct_context(
                values,
                mask,
                operating,
                holiday=float(record.get("holiday", 0)),
                dst=float(record.get("dst", 0)),
            )
            completed_cache[str(record["operating_date"])] = (
                world_data._aggregate_completed_day(values, mask)
            )

        final_raw_records: list[dict[str, Any]] = []
        for record in final_base:
            issue = world_data._issue_time(record)
            eligible = [
                item
                for item in completed
                if datetime.combine(
                    date.fromisoformat(str(item["operating_date"])) + timedelta(days=1),
                    time.min,
                    tzinfo=timezone.utc,
                )
                <= issue
            ]
            da_anchor, da_metadata = world_data._causal_da_anchor(record, eligible)
            operating = date.fromisoformat(str(record["operating_date"]))
            history = world_data._history_context(
                eligible,
                completed_cache,
                cutoff_day=issue.date() - timedelta(days=1),
            )
            alias = dict(record)
            alias["operating_date"] = _alias_development_date(operating).isoformat()
            fast = world_data._raw_fast_context(
                alias,
                physical=self.physical,
                legacy_statistics=identity,
                deadline_intervals=self.deadline_intervals,
                pv_capacity=self.statistics.pv_capacity,
                pue_min=self.statistics.pue_min,
                pue_max=self.statistics.pue_max,
                pue_issue_center=self.statistics.pue_issue_center,
            )
            fast_values, fast_mask = _calendar_correct_context(
                fast["values"],
                fast["mask"],
                operating,
                holiday=float(record.get("holiday", 0)),
                dst=float(record.get("dst", 0)),
            )
            final_raw_records.append(
                {
                    "operating_date": str(record["operating_date"]),
                    "split": FINAL_SPLIT,
                    "fast_values": fast_values,
                    "fast_mask": fast_mask,
                    "source_ids": fast["source"],
                    "quality_ids": fast["quality"],
                    "forecast_anchors": {"da": da_anchor},
                    "forecast_anchor_metadata": {
                        **da_metadata,
                        "opened_final_test_2025": True,
                    },
                    **history,
                    "base_record": record,
                    "issue_time_utc": issue.isoformat(),
                }
            )
            completed.append(record)
        normalized = tuple(
            world_data._normalize_record(record, self.statistics)
            for record in final_raw_records
        )
        self._records = {FINAL_SPLIT: normalized}
        self._next_day = {
            str(record["operating_date"]): None for record in normalized
        }
        self.sealed_final_lineage = {
            "schema": SEALED_CORPUS_SCHEMA,
            "split": FINAL_SPLIT,
            "year": FINAL_YEAR,
            "opened_final_test": True,
            "normalization_fit": "train_2022_2023_only",
            "proper_score_thresholds_frozen": True,
            "fit_on_final_test": False,
            "selection_feedback_allowed": False,
            "evaluable_days": len(normalized),
            "one_shot_id": marker.get("one_shot_id"),
            "daily_inventory": "exact_configured_paths_no_directory_scan",
            "weather": "pre_final_NASA_lag_persistence_only",
            "migration": "pre_final_RIPE_asof_with_true_final_age",
            "hash_computed": False,
        }

    def records(
        self, split: str, *, require_h96: bool = False
    ) -> Sequence[Mapping[str, Any]]:
        if str(split).lower() != FINAL_SPLIT or require_h96:
            raise RECMoESealedCorpusError(
                "sealed corpus exposes only locked_test without h96 supervision"
            )
        return self._records[FINAL_SPLIT]

    def _weather_context(
        self, record: Mapping[str, Any], *, origin: int
    ) -> tuple[Tensor, Tensor]:
        values = torch.zeros(world_data.FAST_STEPS, world_data.INPUT_DIM)
        mask = torch.zeros(world_data.FAST_STEPS, dtype=torch.bool)
        if self.weather_store is None or not 0 <= origin < world_data.FAST_STEPS:
            return values, mask
        operating = date.fromisoformat(str(record.get("sealed_operating_date")))
        operating_start = datetime.combine(operating, time.min, tzinfo=timezone.utc)
        decision_origin = operating_start + timedelta(minutes=15 * int(origin))
        valid_times = tuple(
            operating_start + timedelta(minutes=15 * step)
            for step in range(origin, world_data.FAST_STEPS)
        )
        dense = self.weather_store.query_causal_array_as_of(
            decision_origin,
            valid_times=valid_times,
            source=self.weather_source,
            zones=tuple(world_data.NYISO_WEATHER_SITES),
            features=world_data.WEATHER_FEATURES,
        )
        if dense.values.shape != (
            len(valid_times),
            world_data.NODES,
            len(world_data.WEATHER_FEATURES),
        ):
            return values, mask
        raw = torch.from_numpy(dense.values)
        feature_mask = torch.from_numpy(dense.mask)
        temperature = raw[..., 0]
        shortwave = raw[..., 1]
        plausible = feature_mask.clone()
        plausible[..., 0] &= temperature.ge(180.0) & temperature.le(340.0)
        plausible[..., 1] &= shortwave.ge(-5.0) & shortwave.le(1600.0)
        complete = plausible.all(dim=(-1, -2))
        transformed = torch.cat(
            (
                (temperature - world_data.WEATHER_TEMPERATURE_CENTER_K)
                / world_data.WEATHER_TEMPERATURE_SCALE_K,
                shortwave.clamp_min(0.0) / world_data.WEATHER_SHORTWAVE_SCALE_WM2,
            ),
            dim=-1,
        ).to(torch.float32)
        transformed = torch.where(
            complete[:, None], transformed, torch.zeros_like(transformed)
        )
        values[origin:] = transformed
        mask[origin:] = complete
        return values, mask

    def _packet(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        origin: int,
        device: torch.device,
    ) -> Any:
        proxied: list[dict[str, Any]] = []
        for record in records:
            actual = date.fromisoformat(str(record["operating_date"]))
            base = dict(_mapping(record.get("base_record"), "base_record"))
            base["operating_date"] = _alias_development_date(actual).isoformat()
            item = dict(record)
            item["base_record"] = base
            item["sealed_operating_date"] = actual.isoformat()
            proxied.append(item)
        return super()._packet(proxied, origin=origin, device=device)

    def batches(self, split: str, batch_size: int, **kwargs: Any) -> Iterator[dict[str, Any]]:
        if str(split).lower() != FINAL_SPLIT:
            raise RECMoESealedCorpusError("sealed corpus accepts only locked_test")
        forbidden = {
            "shuffle_seed": kwargs.get("shuffle_seed"),
            "max_days": kwargs.get("max_days"),
            "require_h96": kwargs.get("require_h96", False),
            "include_next_day": kwargs.get("include_next_day", False),
            "shard_index": kwargs.get("shard_index", 0),
            "shard_count": kwargs.get("shard_count", 1),
            "start_batch": kwargs.get("start_batch", 0),
            "batch_ordinals": kwargs.get("batch_ordinals"),
        }
        if forbidden != {
            "shuffle_seed": None,
            "max_days": None,
            "require_h96": False,
            "include_next_day": False,
            "shard_index": 0,
            "shard_count": 1,
            "start_batch": 0,
            "batch_ordinals": None,
        }:
            raise RECMoESealedCorpusError(
                "one-shot final batches must consume the complete ordered split"
            )
        for batch in super().batches(split, batch_size, **kwargs):
            result = dict(batch)
            for name in (
                "forecast_anchor_metadata",
                "forecast_residual_cap_metadata",
                "state_training_baseline_metadata",
                "target_metadata",
            ):
                metadata = result.get(name)
                if isinstance(metadata, Mapping):
                    result[name] = {
                        **metadata,
                        "opened_final_test_2025": True,
                        "fit_statistics_on_final_test": False,
                        "updated_thresholds_on_final_test": False,
                    }
            yield result


def build_sealed_final_corpus(
    *,
    config: Mapping[str, Any],
    repository_root: str | Path,
    frozen_evaluation_marker: Mapping[str, Any],
) -> SealedFinalWorldMoECorpus:
    """Construct the complete final corpus after one-shot authorization."""

    root = Path(repository_root).expanduser().resolve()
    start, end, expected_days = _protocol_dates(config, frozen_evaluation_marker)
    data_config = _mapping(config.get("data"), "data")
    history_bundle_root = _resolve(
        root, data_config.get("selection_bundle_root"), "selection bundle root"
    )
    history_settlement_root = _resolve(
        root, data_config.get("official_settlement_root"), "selection settlement root"
    )
    final_bundle_root = _locked_sibling(history_bundle_root)
    final_settlement_root = _locked_sibling(history_settlement_root)
    final_dates = _complete_exact_days(
        final_bundle_root,
        final_settlement_root,
        tuple(_date_range(start, end)),
        required_count=expected_days,
    )
    normalization_config = _mapping(
        data_config.get("normalization"), "data.normalization"
    )
    if (
        normalization_config.get("fit_split") != "train_2022_2023_only"
        or normalization_config.get("validation_statistics_allowed") is not False
        or normalization_config.get("final_test_statistics_allowed") is not False
    ):
        raise RECMoESealedCorpusError("configuration permits final statistic fitting")
    normalization_path = _resolve(
        root, normalization_config.get("artifact_path"), "frozen normalization"
    )
    statistics, target_statistics = _load_frozen_world_statistics(normalization_path)
    return SealedFinalWorldMoECorpus(
        config,
        root,
        frozen_evaluation_marker,
        final_bundle_root=final_bundle_root,
        final_settlement_root=final_settlement_root,
        history_bundle_root=history_bundle_root,
        history_settlement_root=history_settlement_root,
        final_dates=final_dates,
        final_start=start,
        statistics=statistics,
        target_statistics=target_statistics,
    )


__all__ = [
    "RECMoESealedCorpusError",
    "SealedFinalMigrationEvidenceStore",
    "SealedFinalNasaPowerLagPersistenceStore",
    "SealedFinalWorldMoECorpus",
    "build_sealed_final_corpus",
]
