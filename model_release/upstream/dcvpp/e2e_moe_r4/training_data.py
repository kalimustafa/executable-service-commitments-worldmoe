"""Executable, split-locked training data path for formal R4.

The loader opens only exact 2022--2023 daily paths named by the governing
configuration.  It never scans or opens a 2024/2025 payload.  Legacy daily
bundles are accepted only through their adjacent causal-gate record; this is a
documented provenance bridge, not permission to infer missing availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import Tensor

from .constrained_runtime import PhysicalLimits, load_physical_limits
from .data import INTERVALS, LOCKED_TEST_YEAR, R4DataError, temporal_split
from .hashing import sha256_file, sha256_json
from .physical_twin import (
    ArtifactBackedFrozenPhysicalTwinCore,
    load_frozen_physical_twin_core,
)


TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 6, 30)
VALIDATION_START = date(2023, 7, 1)
VALIDATION_END = date(2023, 12, 31)
EMPIRICAL_SCENARIOS = 100
FORECAST_RESIDUAL_CAP_SCHEMA = "train_only_absolute_residual_p95_raw_units_v1"


def _dates(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _resolve_under(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    # Daily payload paths are constructed only after a date has passed the
    # immutable split check.  Root names themselves need not encode a year.
    return path


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise R4DataError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise R4DataError(f"{label} contains non-finite values")
    return array


def _positive_cap(value: Any, label: str) -> np.ndarray:
    cap = _finite(value, (4,), label)
    if not bool((cap > 0.0).all()):
        raise R4DataError(f"{label} must be strictly positive")
    return cap


@dataclass(frozen=True)
class R4InputStatistics:
    market_scale: np.ndarray
    workload_scale: np.ndarray
    pv_scale: np.ndarray
    forecast_da_residual_cap: np.ndarray
    forecast_pv_residual_cap: np.ndarray
    forecast_pue_residual_cap: np.ndarray

    def forecast_residual_cap_payload(self) -> dict[str, list[float]]:
        return {
            "da": self.forecast_da_residual_cap.tolist(),
            "pv": self.forecast_pv_residual_cap.tolist(),
            "pue": self.forecast_pue_residual_cap.tolist(),
        }

    def forecast_residual_cap_sha256(self) -> str:
        # Use the repository's isolated hashing process.  Direct OpenSSL
        # hashlib calls have caused native access violations in this Windows
        # Unicode workspace and must not run in the training process.
        return sha256_json(self.forecast_residual_cap_payload())

    def as_json(self) -> dict[str, Any]:
        return {
            "fit_split": "train",
            "market_transform": "asinh_train_median_absolute_scale",
            "workload_transform": "log1p_train_median_absolute_scale",
            "pv_transform": "log1p_train_median_absolute_scale",
            "market_scale": self.market_scale.tolist(),
            "workload_scale": self.workload_scale.tolist(),
            "pv_scale": self.pv_scale.tolist(),
            "forecast_residual_cap_schema": FORECAST_RESIDUAL_CAP_SCHEMA,
            "forecast_residual_cap_fit_split": "train",
            "forecast_residual_cap_quantile": 0.95,
            "forecast_residual_cap_units": "raw_target_units_by_zone",
            "forecast_residual_caps": self.forecast_residual_cap_payload(),
            "forecast_residual_cap_sha256": self.forecast_residual_cap_sha256(),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "R4InputStatistics":
        if value.get("fit_split") != "train":
            raise R4DataError("normalization artifact was not fitted on train")
        if value.get("forecast_residual_cap_schema") != FORECAST_RESIDUAL_CAP_SCHEMA:
            raise R4DataError("normalization artifact lacks train-only residual caps")
        caps = value.get("forecast_residual_caps")
        if not isinstance(caps, Mapping):
            raise R4DataError("normalization artifact residual caps must be a mapping")
        result = cls(
            market_scale=_finite(value["market_scale"], (16,), "market_scale"),
            workload_scale=_finite(value["workload_scale"], (5,), "workload_scale"),
            pv_scale=_finite(value["pv_scale"], (8,), "pv_scale"),
            forecast_da_residual_cap=_positive_cap(caps["da"], "DA residual cap"),
            forecast_pv_residual_cap=_positive_cap(caps["pv"], "PV residual cap"),
            forecast_pue_residual_cap=_positive_cap(caps["pue"], "PUE residual cap"),
        )
        if value.get("forecast_residual_cap_sha256") != (
            result.forecast_residual_cap_sha256()
        ):
            raise R4DataError("normalization artifact residual-cap hash mismatch")
        return result


def validate_forecast_anchors(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Validate the shared unbatched/batched DA and PV anchor interface."""

    value = batch.get("forecast_anchors")
    if not isinstance(value, Mapping):
        raise R4DataError("forecast_anchors must be an explicit mapping")
    output: dict[str, Tensor] = {}
    prefix: tuple[int, ...] | None = None
    for name in ("da", "pv"):
        if value.get(name) is None:
            raise R4DataError(f"forecast_anchors.{name} is required")
        tensor = torch.as_tensor(value[name])
        if tensor.ndim not in (2, 3) or tensor.shape[-2:] != (INTERVALS, 4):
            raise R4DataError(
                f"forecast_anchors.{name} must be [96,4] or [batch,96,4]"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise R4DataError(f"forecast_anchors.{name} contains non-finite values")
        current_prefix = tuple(tensor.shape[:-2])
        if prefix is not None and current_prefix != prefix:
            raise R4DataError("DA and PV forecast anchors have different batch shapes")
        prefix = current_prefix
        output[name] = tensor
    return output


def attach_forecast_anchors(
    batch: Mapping[str, Any],
    *,
    da_anchor: Any,
    pv_anchor: Any | None = None,
    da_source: str = "latest_fully_completed_operating_day_before_issue",
    pv_source: str = "issue_time_ensemble_mean",
) -> dict[str, Any]:
    """Attach one deployment-compatible anchor contract without normalization."""

    result = dict(batch)
    if pv_anchor is None:
        dynamic = result.get("physical_dynamic")
        if dynamic is None:
            raise R4DataError("pv_anchor or raw physical_dynamic is required")
        pv_anchor = torch.as_tensor(dynamic)[..., :4]
    result["forecast_anchors"] = {
        "da": torch.as_tensor(da_anchor),
        "pv": torch.as_tensor(pv_anchor),
    }
    validate_forecast_anchors(result)
    result["forecast_anchor_metadata"] = {
        "schema": "r4_causal_forecast_anchors_v1",
        "da_source": str(da_source),
        "pv_source": str(pv_source),
        "normalization": "none_raw_target_units",
    }
    return result


def normalize_runtime_batch(
    batch: Mapping[str, Any], statistics: R4InputStatistics
) -> dict[str, Any]:
    """Apply the exact frozen train-only transforms to raw deployment tensors."""

    result = dict(batch)
    market = torch.as_tensor(result["market"])
    workload = torch.as_tensor(result["workload"])
    dynamic = torch.as_tensor(result["physical_dynamic"]).clone()
    market_scale = torch.as_tensor(
        statistics.market_scale, device=market.device, dtype=market.dtype
    )
    workload_scale = torch.as_tensor(
        statistics.workload_scale, device=workload.device, dtype=workload.dtype
    )
    pv_scale = torch.as_tensor(
        statistics.pv_scale, device=dynamic.device, dtype=dynamic.dtype
    )
    result["market_raw"] = market.clone()
    result["issue_workload_raw"] = workload[..., 0].clone()
    result["issue_pv_raw"] = dynamic[..., :4].clone()
    result["market"] = torch.asinh(market / market_scale)
    result["workload"] = torch.log1p(workload.clamp_min(0.0)) / workload_scale
    dynamic[..., :8] = torch.log1p(dynamic[..., :8].clamp_min(0.0)) / pv_scale
    # The final four channels are issue-time utilization in [0,1], not a
    # repeated static multiplier.  They are already on their physical scale
    # and are consumed unchanged by the frozen artifact-backed twin.
    result["physical_dynamic"] = dynamic
    cap = lambda value: torch.as_tensor(
        value, device=market.device, dtype=market.dtype
    )
    result["forecast_residual_caps"] = {
        "da": cap(statistics.forecast_da_residual_cap),
        "pv": cap(statistics.forecast_pv_residual_cap),
        "pue": cap(statistics.forecast_pue_residual_cap),
    }
    result["forecast_residual_cap_metadata"] = {
        "schema": FORECAST_RESIDUAL_CAP_SCHEMA,
        "fit_split": "train",
        "quantile": 0.95,
        "units": "raw_target_units_by_zone",
        "sha256": statistics.forecast_residual_cap_sha256(),
    }
    if "forecast_anchors" in result:
        validate_forecast_anchors(result)
    return result


def load_normalization_artifact(path: str | Path) -> R4InputStatistics:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise R4DataError("normalization artifact root must be a mapping")
    return R4InputStatistics.from_json(payload)


@dataclass
class _RawDay:
    operating_date: str
    market: np.ndarray
    workload: np.ndarray
    physical_static: np.ndarray
    physical_dynamic: np.ndarray
    initial_soc: np.ndarray
    realized_da: np.ndarray
    realized_rt: np.ndarray
    realized_workload: np.ndarray
    realized_pv: np.ndarray
    issue_workload: np.ndarray
    issue_pv: np.ndarray
    provenance: Mapping[str, Any]


def _allocate_total_workload(
    total_workload: np.ndarray, capacity_work_units: np.ndarray
) -> np.ndarray:
    """Deterministically distribute one VPP-wide workload across four sites."""

    total = np.asarray(total_workload, dtype=np.float32)
    capacity = np.asarray(capacity_work_units, dtype=np.float32)
    if total.ndim != 1 or capacity.shape != (4,) or (capacity <= 0).any():
        raise R4DataError("workload allocation requires [time] totals and four capacities")
    served = np.minimum(np.maximum(total, 0.0), float(capacity.sum()))
    allocation = served[:, None] * capacity[None] / float(capacity.sum())
    return np.minimum(allocation, capacity[None]).astype(np.float32)


def _load_raw_day(
    bundle_path: Path,
    settlement_path: Path,
    capacity_work_units: np.ndarray,
    *,
    allowed_window: tuple[date, date] | None = None,
    verify_hashes: bool = True,
) -> _RawDay:
    day_text = bundle_path.stem
    day = date.fromisoformat(day_text)
    if allowed_window is None:
        # Preserve the formal R4 default exactly: callers which do not opt in
        # remain restricted to its original 543/179 train-validation lineage.
        if temporal_split(day) not in {"train", "validation"}:
            raise R4DataError(f"training loader may open only 2022--2023: {bundle_path}")
    else:
        allowed_start, allowed_end = allowed_window
        if day.year >= LOCKED_TEST_YEAR:
            # This guard runs before record JSON or either NPZ is opened.
            raise R4DataError(f"locked {LOCKED_TEST_YEAR} data may not be opened")
        if allowed_start > allowed_end or not allowed_start <= day <= allowed_end:
            raise R4DataError(
                f"development loader date lies outside its approved window: {day_text}"
            )
    record_path = bundle_path.with_suffix(".record.json")
    if not record_path.is_file():
        raise R4DataError(f"missing causal-gate record: {record_path}")
    record_payload = json.loads(record_path.read_text(encoding="utf-8"))
    record = record_payload.get("record", {})
    gates = record.get("causal_gates", {}) if isinstance(record, Mapping) else {}
    required_gates = ("candidate_independent_price", "pv", "workload")
    if not all(gates.get(name) is True for name in required_gates):
        raise R4DataError(f"legacy provenance bridge failed causal gates for {day_text}")
    expected_hash = str(record.get("sha256", ""))
    if not expected_hash:
        raise R4DataError(f"causal-gate record omitted bundle hash: {bundle_path}")
    if bool(verify_hashes) and expected_hash != sha256_file(bundle_path):
        raise R4DataError(f"bundle hash does not match causal-gate record: {bundle_path}")

    with np.load(bundle_path, allow_pickle=False) as source:
        if str(source["operating_date"].item())[:10] != day_text:
            raise R4DataError(f"bundle operating date mismatch: {bundle_path}")
        if bool(source["locked_test"].item()):
            raise R4DataError("locked_test=true is forbidden in R4 training")
        da = _finite(source["da_price"], (4, INTERVALS), "da_price")
        rt_pool = _finite(
            source["rt_price_pool"], (EMPIRICAL_SCENARIOS, 4, INTERVALS),
            "rt_price_pool",
        )
        workload_pool = _finite(
            source["workload_pool"], (EMPIRICAL_SCENARIOS, INTERVALS),
            "workload_pool",
        )
        pv_pool = _finite(
            source["pv_pool"], (EMPIRICAL_SCENARIOS, 4, INTERVALS), "pv_pool"
        )
        realized_workload = _finite(source["workload"], (INTERVALS,), "workload")
        realized_pv = _finite(source["pv"], (4, INTERVALS), "pv")
        multiplier = _finite(source["physical_multiplier"], (4,), "physical_multiplier")
        initial_soc = _finite(source["initial_soc_mwh_by_zone"], (4,), "initial_soc")
        issue_time = str(source["issue_time_utc"].item())

    with np.load(settlement_path, allow_pickle=False) as settlement:
        realized_da = _finite(settlement["da_lbmp"], (4, INTERVALS), "realized_da")
        realized_rt = _finite(settlement["rt_lbmp"], (4, INTERVALS), "realized_rt")

    da_tz = da.T
    rt_mean = rt_pool.mean(axis=0).T
    rt_std = rt_pool.std(axis=0).T
    market = np.concatenate((da_tz, rt_mean, da_tz - rt_mean, rt_std), axis=-1)
    quantiles = np.quantile(workload_pool, (0.1, 0.5, 0.9), axis=0)
    workload = np.stack(
        (workload_pool.mean(0), workload_pool.std(0), *quantiles), axis=-1
    ).astype(np.float32)
    issue_pv = pv_pool.mean(0).T.astype(np.float32)
    issue_utilization = _allocate_total_workload(
        workload_pool.mean(0), capacity_work_units
    ) / np.asarray(capacity_work_units, dtype=np.float32)[None]
    physical_dynamic = np.concatenate(
        (issue_pv, pv_pool.std(0).T, issue_utilization),
        axis=-1,
    ).astype(np.float32)
    provenance = {
        "bridge": "adjacent_counterfactual_input_materialization_record_v1",
        "record_path": str(record_path.resolve()),
        "record_sha256": (
            sha256_file(record_path) if verify_hashes else "NOT_RECOMPUTED"
        ),
        "bundle_sha256": expected_hash,
        "settlement_path": str(settlement_path.resolve()),
        "settlement_sha256": (
            sha256_file(settlement_path) if verify_hashes else "NOT_RECOMPUTED"
        ),
        "hash_verification": (
            "verified_on_load" if verify_hashes else "trusted_registered_record"
        ),
        "issue_time_utc": issue_time,
        "causal_gates": {name: True for name in required_gates},
    }
    return _RawDay(
        day_text,
        market.astype(np.float32),
        workload,
        multiplier,
        physical_dynamic,
        initial_soc,
        realized_da.T,
        realized_rt.T,
        realized_workload,
        realized_pv.T,
        workload_pool.mean(0).astype(np.float32),
        issue_pv,
        provenance,
    )


def _safe_median_scale(values: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    scale = np.median(np.abs(values), axis=axis)
    return np.maximum(scale.astype(np.float32), 1.0e-3)


def _fit_statistics(
    days: Sequence[_RawDay],
    physical: PhysicalLimits,
    physical_core: ArtifactBackedFrozenPhysicalTwinCore,
) -> R4InputStatistics:
    if not days:
        raise R4DataError("cannot fit normalization without training days")
    market = np.stack([item.market for item in days])
    workload = np.log1p(np.stack([item.workload for item in days]))
    pv = np.log1p(np.stack([item.physical_dynamic[..., :8] for item in days]))
    da_residuals: list[np.ndarray] = []
    history: list[_RawDay] = []
    for item in days:
        anchor, _ = _causal_da_anchor(item, history)
        da_residuals.append(item.realized_da - anchor)
        history.append(item)
    pv_residuals = np.stack(
        [item.realized_pv - item.issue_pv for item in days]
    )
    capacity = np.asarray(physical.capacity_work_units, dtype=np.float32)
    issue_utilization = np.stack(
        [item.physical_dynamic[..., 8:12] for item in days]
    )
    realized_utilization = np.stack(
        [
            _allocate_total_workload(item.realized_workload, capacity)
            / capacity[None]
            for item in days
        ]
    )
    multiplier = np.stack([item.physical_static for item in days])
    with torch.inference_mode():
        issue_pue = physical_core(
            torch.as_tensor(multiplier, dtype=torch.float64),
            torch.as_tensor(issue_utilization, dtype=torch.float64),
        )["pue"].cpu().numpy()
        target_pue = physical_core(
            torch.as_tensor(multiplier, dtype=torch.float64),
            torch.as_tensor(realized_utilization, dtype=torch.float64),
        )["pue"].cpu().numpy()

    def raw_p95(residual: np.ndarray) -> np.ndarray:
        cap = np.quantile(np.abs(residual), 0.95, axis=(0, 1))
        return np.maximum(cap.astype(np.float32), 1.0e-6)

    return R4InputStatistics(
        market_scale=_safe_median_scale(market, (0, 1)),
        workload_scale=_safe_median_scale(workload, (0, 1)),
        pv_scale=_safe_median_scale(pv, (0, 1)),
        forecast_da_residual_cap=raw_p95(np.stack(da_residuals)),
        forecast_pv_residual_cap=raw_p95(pv_residuals),
        forecast_pue_residual_cap=raw_p95(target_pue - issue_pue),
    )


def _state_sequence(raw: _RawDay, physical: PhysicalLimits) -> np.ndarray:
    """Causal passive-EDF state trace used as teacher context, never future plan data."""

    state = np.zeros((INTERVALS, 49), dtype=np.float32)
    state[:, 0:4] = raw.initial_soc / np.maximum(physical.battery_energy_mwh, 1.0e-6)
    capacity = np.asarray(physical.capacity_work_units, dtype=np.float32)
    queue_age = np.zeros(5, dtype=np.float32)
    issue_rt = raw.market[:, 4:8]
    issue_workload = raw.issue_workload
    for step in range(1, INTERVALS):
        previous = step - 1
        queue_age[-1] += float(raw.realized_workload[previous])
        served = _allocate_total_workload(
            np.asarray([min(float(queue_age.sum()), float(capacity.sum()))]),
            capacity,
        )[0]
        service_left = float(served.sum())
        for bucket in range(queue_age.size):
            consumed = min(service_left, float(queue_age[bucket]))
            queue_age[bucket] -= consumed
            service_left -= consumed
        # Expired work is accounted as SLA shortfall in the runtime.  This
        # teacher context carries only the remaining causal backlog.
        queue_age[0] = 0.0
        queue_age[:-1] = queue_age[1:]
        queue_age[-1] = 0.0
        backlog_total = float(queue_age.sum())
        load = raw.physical_static * (
            physical.idle_mw + physical.work_mw_per_unit * served
        )
        grid = np.maximum(load - raw.realized_pv[previous], 0.0)
        state[step, 4:8] = backlog_total / 4.0
        deadline_fraction = (
            float(queue_age[0] / backlog_total) if backlog_total > 0 else 0.0
        )
        state[step, 8:12] = deadline_fraction
        state[step, 12:16] = grid
        state[step, 16:20] = 0.0
        state[step, 20:24] = served
        state[step, 24:40] = 0.0
        state[step, 40:44] = raw.realized_rt[previous] - issue_rt[previous]
        state[step, 44] = raw.realized_workload[previous] - issue_workload[previous]
        state[step, 45:49] = raw.realized_pv[previous] - raw.issue_pv[previous]
    return state


def _calendar(day_text: str) -> tuple[int, int, int, int]:
    operating = date.fromisoformat(day_text)
    local = datetime(
        operating.year, operating.month, operating.day, 12,
        tzinfo=ZoneInfo("America/New_York"),
    )
    return operating.weekday(), operating.month - 1, 0, int(bool(local.dst()))


def _causal_da_anchor(raw: _RawDay, history: Sequence[_RawDay]) -> tuple[np.ndarray, bool]:
    """Return the latest fully completed DA trajectory available at issue time."""

    issue_text = str(raw.provenance.get("issue_time_utc", "")).replace("Z", "+00:00")
    issue = datetime.fromisoformat(issue_text)
    if issue.tzinfo is None:
        raise R4DataError("issue_time_utc must be timezone-aware for the DA anchor")
    issue = issue.astimezone(timezone.utc)
    for source in reversed(history):
        operating = date.fromisoformat(source.operating_date)
        completed = datetime.combine(
            operating + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        if completed <= issue:
            return source.realized_da.copy(), True
    # The first formal train date has no completed in-window predecessor.  Its
    # issue-time DA path is causal and is used only as a documented warm-up
    # fallback; validation always has a completed persistence anchor.
    return raw.market[:, :4].copy(), False


def _day_to_tensors(
    raw: _RawDay,
    statistics: R4InputStatistics,
    physical: PhysicalLimits,
    physical_core: ArtifactBackedFrozenPhysicalTwinCore,
    history: Sequence[_RawDay],
) -> dict[str, Any]:
    market = np.arcsinh(raw.market / statistics.market_scale)
    workload = np.log1p(raw.workload) / statistics.workload_scale
    dynamic = raw.physical_dynamic.copy()
    dynamic[..., :8] = np.log1p(dynamic[..., :8]) / statistics.pv_scale
    historical = np.zeros((7, 16), dtype=np.float32)
    available = np.zeros(7, dtype=bool)
    selected = list(history[-7:])
    offset = 7 - len(selected)
    for index, item in enumerate(selected, start=offset):
        historical[index] = np.arcsinh(
            item.market / statistics.market_scale
        ).mean(axis=0)
        available[index] = True

    realized_work = _allocate_total_workload(
        raw.realized_workload, np.asarray(physical.capacity_work_units)
    )
    realized_utilization = realized_work / np.asarray(
        physical.capacity_work_units, dtype=np.float32
    )[None]
    with torch.inference_mode():
        physical_target = physical_core(
            torch.as_tensor(raw.physical_static[None], dtype=torch.float64),
            torch.as_tensor(realized_utilization[None], dtype=torch.float64),
        )
    pue = physical_target["pue"][0].cpu().numpy().astype(np.float32)
    dart = raw.realized_da - raw.realized_rt
    scenario_target = np.concatenate(
        (
            raw.realized_da,
            raw.realized_rt,
            dart,
            raw.realized_workload[:, None],
            raw.realized_pv,
            pue,
        ),
        axis=-1,
    ).astype(np.float32)
    issue_work = _allocate_total_workload(
        raw.issue_workload, np.asarray(physical.capacity_work_units)
    )
    issue_load = raw.physical_static[None] * (
        np.asarray(physical.idle_mw)[None]
        + np.asarray(physical.work_mw_per_unit)[None] * issue_work
    )
    issue_grid = np.maximum(issue_load - raw.issue_pv, 0.0)
    issue_da = np.minimum(
        issue_grid, np.asarray(physical.grid_import_limit_mw)[None]
    ).T.astype(np.float32)
    realized_load = raw.physical_static[None] * (
        np.asarray(physical.idle_mw)[None]
        + np.asarray(physical.work_mw_per_unit)[None] * realized_work
    )
    realized_grid = np.maximum(realized_load - raw.realized_pv, 0.0)
    baseline_cost = 0.25 * (
        issue_da.T * raw.realized_da
        + (realized_grid - issue_da.T) * raw.realized_rt
    ).sum()
    dow, month, holiday, dst = _calendar(raw.operating_date)
    causal_da, causal_da_available = _causal_da_anchor(raw, history)
    tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32)
    return {
        "market": tensor(market),
        "market_raw": tensor(raw.market),
        "issue_workload_raw": tensor(raw.issue_workload),
        "issue_pv_raw": tensor(raw.issue_pv),
        "forecast_anchors": {
            "da": tensor(causal_da),
            "pv": tensor(raw.issue_pv),
        },
        "forecast_anchor_metadata": {
            "schema": "r4_causal_forecast_anchors_v1",
            "da_source": (
                "latest_fully_completed_operating_day_before_issue"
                if causal_da_available
                else "first_train_day_issue_time_da_warmup"
            ),
            "pv_source": "issue_time_ensemble_mean",
            "normalization": "none_raw_target_units",
        },
        "forecast_residual_caps": {
            "da": tensor(statistics.forecast_da_residual_cap),
            "pv": tensor(statistics.forecast_pv_residual_cap),
            "pue": tensor(statistics.forecast_pue_residual_cap),
        },
        "forecast_residual_cap_metadata": {
            "schema": FORECAST_RESIDUAL_CAP_SCHEMA,
            "fit_split": "train",
            "quantile": 0.95,
            "units": "raw_target_units_by_zone",
            "sha256": statistics.forecast_residual_cap_sha256(),
        },
        "causal_da_anchor_available": torch.tensor(
            causal_da_available, dtype=torch.bool
        ),
        "historical_market": tensor(historical),
        "historical_market_available": torch.as_tensor(available, dtype=torch.bool),
        "workload": tensor(workload),
        "physical_static": tensor(raw.physical_static),
        "physical_dynamic": tensor(dynamic),
        "state_sequence": tensor(_state_sequence(raw, physical)),
        "origin": torch.tensor(0, dtype=torch.long),
        "day_of_week": torch.tensor(dow, dtype=torch.long),
        "month": torch.tensor(month, dtype=torch.long),
        "holiday": torch.tensor(holiday, dtype=torch.long),
        "dst": torch.tensor(dst, dtype=torch.long),
        "targets": {
            "da": tensor(raw.realized_da),
            "rt": tensor(raw.realized_rt),
            "dart": tensor(dart),
            "workload": tensor(raw.realized_workload[:, None]),
            "pv": tensor(raw.realized_pv),
            "pue": tensor(pue),
        },
        "scenario_target_zonal": tensor(scenario_target),
        "rollout_targets": {
            "da_price": tensor(raw.realized_da.T),
            "rt_price": tensor(raw.realized_rt.T),
            "workload": tensor(raw.realized_workload),
            "pv": tensor(raw.realized_pv.T),
            "pue": tensor(pue.T),
            "initial_soc": tensor(raw.initial_soc),
            "capacity_work_units": tensor(physical.capacity_work_units),
            "idle_mw": tensor(physical.idle_mw),
            "work_mw_per_unit": tensor(physical.work_mw_per_unit),
            "physical_multiplier": tensor(raw.physical_static),
            "grid_limit_mw": tensor(physical.grid_import_limit_mw),
            "battery_power_mw": tensor(physical.battery_power_mw),
            "battery_energy_mwh": tensor(physical.battery_energy_mwh),
            "baseline_actions": {
                "day_ahead": tensor(issue_da),
                "battery": torch.zeros((4, INTERVALS), dtype=torch.float32),
                "compute": tensor(realized_work.T),
            },
            "baseline_realized_cost": tensor(np.asarray(baseline_cost)),
        },
        "operating_date": raw.operating_date,
        "provenance": dict(raw.provenance),
    }


def _collate(items: Sequence[Any]) -> Any:
    first = items[0]
    if isinstance(first, Tensor):
        return torch.stack(list(items))
    if isinstance(first, Mapping):
        return {key: _collate([item[key] for item in items]) for key in first}
    return list(items)


def move_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: move_batch(value, device) for key, value in batch.items()}
    return batch


class PreloadedR4Corpus:
    """One host-memory corpus with train-fitted immutable normalization."""

    def __init__(self, config: Mapping[str, Any], repository_root: str | Path) -> None:
        root = Path(repository_root).resolve()
        data = config.get("data", {})
        if not isinstance(data, Mapping):
            raise R4DataError("R4 config.data must be a mapping")
        bundle_root = _resolve_under(root, data.get("selection_bundle_root"), "bundle root")
        settlement_root = _resolve_under(
            root, data.get("official_settlement_root"), "settlement root"
        )
        physical_path = _resolve_under(
            root, data.get("frozen_physical_twin"), "frozen physical twin"
        )
        self.physical = load_physical_limits(physical_path)
        self.physical_sha256 = sha256_file(physical_path)
        surrogate_path = _resolve_under(
            root, data.get("frozen_physical_surrogate"), "frozen physical surrogate"
        )
        manifest_path = _resolve_under(
            root, data.get("physical_calibration_manifest"), "physical calibration manifest"
        )
        self.physical_core = load_frozen_physical_twin_core(
            artifact_path=physical_path,
            surrogate_path=surrogate_path,
            calibration_manifest_path=manifest_path,
        )
        self.physical_lineage = dict(self.physical_core.artifact_metadata)
        raw: dict[str, _RawDay] = {}
        for operating in _dates(TRAIN_START, VALIDATION_END):
            day_text = operating.isoformat()
            bundle = bundle_root / f"{day_text}.npz"
            settlement = settlement_root / f"{day_text}.npz"
            if bundle.is_file() != settlement.is_file():
                raise R4DataError(f"bundle/settlement availability mismatch for {day_text}")
            if not bundle.is_file():
                continue
            raw[day_text] = _load_raw_day(
                bundle, settlement, np.asarray(self.physical.capacity_work_units)
            )
        train_raw = [raw[key] for key in sorted(raw) if key <= TRAIN_END.isoformat()]
        validation_raw = [
            raw[key] for key in sorted(raw) if key >= VALIDATION_START.isoformat()
        ]
        if len(train_raw) != 543 or len(validation_raw) != 179:
            raise R4DataError(
                f"formal corpus requires 543 train and 179 validation days, got "
                f"{len(train_raw)} and {len(validation_raw)}"
            )
        self.statistics = _fit_statistics(
            train_raw, self.physical, self.physical_core
        )
        ordered = [raw[key] for key in sorted(raw)]
        self._records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        history: list[_RawDay] = []
        for item in ordered:
            split = temporal_split(item.operating_date)
            self._records[split].append(
                _day_to_tensors(
                    item,
                    self.statistics,
                    self.physical,
                    self.physical_core,
                    history,
                )
            )
            history.append(item)
        if torch.cuda.is_available():
            self._records = _pin(self._records)
        baseline = torch.stack(
            [item["rollout_targets"]["baseline_realized_cost"] for item in self._records["train"]]
        )
        self.fixed_training_cost_scale = float(
            baseline.abs().median().clamp_min(1.0).item()
        )
        for split_records in self._records.values():
            for item in split_records:
                item["provenance"]["physical_sha256"] = self.physical_sha256
                item["provenance"]["physical_twin_lineage"] = dict(
                    self.physical_lineage
                )

    def records(self, split: str) -> Sequence[Mapping[str, Any]]:
        if split not in self._records:
            raise R4DataError(f"unknown R4 corpus split: {split}")
        return self._records[split]

    def batches(
        self,
        split: str,
        batch_size: int,
        *,
        device: torch.device,
        shuffle_seed: int | None = None,
        max_days: int | None = None,
        origin: int = 0,
    ) -> Iterator[dict[str, Any]]:
        records = list(self.records(split))
        if shuffle_seed is not None:
            generator = torch.Generator().manual_seed(int(shuffle_seed))
            order = torch.randperm(len(records), generator=generator).tolist()
            records = [records[index] for index in order]
        if max_days is not None:
            records = records[: int(max_days)]
        for start in range(0, len(records), int(batch_size)):
            batch = _collate(records[start : start + int(batch_size)])
            batch["num_days"] = len(batch["operating_date"])
            batch["operating_dates"] = tuple(batch["operating_date"])
            batch["origin"] = torch.full(
                (len(batch["operating_date"]),), int(origin), dtype=torch.long
            )
            yield move_batch(batch, device)


def _pin(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.pin_memory()
    if isinstance(value, Mapping):
        return {key: _pin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pin(item) for item in value]
    return value


__all__ = [
    "FORECAST_RESIDUAL_CAP_SCHEMA",
    "PreloadedR4Corpus",
    "R4InputStatistics",
    "attach_forecast_anchors",
    "load_normalization_artifact",
    "move_batch",
    "normalize_runtime_batch",
    "validate_forecast_anchors",
]
