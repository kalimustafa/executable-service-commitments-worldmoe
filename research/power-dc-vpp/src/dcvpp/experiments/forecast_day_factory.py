from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dcvpp.core import atomic_json, sha256_file
from dcvpp.research_branch.contracts import FrozenForecastArtifact


ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**32 - 1)


def _daily_energy(path: Path) -> dict[str, dict[str, Any]]:
    frame = pq.ParquetFile(path).read().to_pandas()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    result: dict[str, dict[str, Any]] = {}
    for (split, day), group in frame.groupby(["split", "operating_date"], sort=True):
        ordered = group.sort_values(["zone", "timestamp_utc"], kind="stable")
        if len(ordered) != 4 * 96:
            continue
        timestamps = np.sort(ordered["timestamp_utc"].unique())
        if len(timestamps) != 96:
            continue
        arrays: dict[str, Any] = {"timestamps": timestamps}
        for column in ("observed_target", "q50"):
            pivot = ordered.pivot(index="zone", columns="timestamp_utc", values=column).reindex(
                index=ZONES, columns=timestamps
            )
            arrays[column] = pivot.to_numpy(dtype="float64")
        finite_target = np.isfinite(arrays["observed_target"])
        inferred_complete = bool(finite_target.all())
        declared_complete = (
            bool(ordered["day_realized_target_complete"].all())
            if "day_realized_target_complete" in ordered
            else inferred_complete
        )
        if declared_complete != inferred_complete:
            raise ValueError(
                f"realized-target completeness flag disagrees with values for {split}:{day}"
            )
        arrays["target_quality"] = {
            "realized_target_complete": inferred_complete,
            "eligible_for_locked_test_scoring": bool(
                ordered["eligible_for_locked_test_scoring"].all()
                if "eligible_for_locked_test_scoring" in ordered
                else split == "locked_test" and inferred_complete
            ),
            "missing_realized_target_interval_count": int(
                (~finite_target).any(axis=0).sum()
            ),
            "missing_realized_target_value_count": int((~finite_target).sum()),
            "missing_realized_target_intervals_by_zone": {
                zone: int((~finite_target[zone_index]).sum())
                for zone_index, zone in enumerate(ZONES)
            },
            "missing_realized_targets_imputed": False,
        }
        result[f"{split}:{day}"] = arrays
    return result


def _daily_workload(path: Path) -> dict[str, dict[str, np.ndarray]]:
    frame = pq.ParquetFile(path).read().to_pandas()
    result: dict[str, dict[str, np.ndarray]] = {}
    for (split, day), group in frame.groupby(["split", "operating_date"], sort=True):
        ordered = group.sort_values("timestamp_utc", kind="stable")
        if len(ordered) != 96:
            continue
        result[f"{split}:{day}"] = {
            "observed": ordered["observed_nodes_requested"].to_numpy(dtype="float64"),
            "predicted": ordered["predicted_nodes_requested"].to_numpy(dtype="float64"),
        }
    return result


def _pv_validation_days(path: Path) -> list[np.ndarray]:
    frame = pq.ParquetFile(path).read(
        columns=["timestamp_utc", "zone", "pv_dc_availability_mw_per_mwdc"]
    ).to_pandas()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.loc[frame["timestamp_utc"].dt.year.eq(2024)].copy()
    frame["operating_date"] = frame["timestamp_utc"].dt.date.astype(str)
    days = []
    for _, group in frame.groupby("operating_date", sort=True):
        timestamps = np.sort(group["timestamp_utc"].unique())
        if len(timestamps) != 96:
            continue
        pivot = group.pivot(
            index="zone", columns="timestamp_utc", values="pv_dc_availability_mw_per_mwdc"
        ).reindex(index=ZONES, columns=timestamps)
        values = pivot.to_numpy(dtype="float64")
        if np.isfinite(values).all():
            days.append(values)
    if not days:
        raise ValueError("PV panel has no complete 2024 validation days")
    return days


def _sample_rows(values: list[np.ndarray], *, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=count, replace=len(values) < count)
    return np.stack([values[int(index)] for index in indices])


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def build_frozen_forecast_day_index(
    *,
    repository_root: Path,
    output_root: Path,
    da_prediction_path: Path,
    rt_prediction_path: Path,
    workload_prediction_path: Path,
    pv_panel_path: Path,
    draw_count: int = 100,
    physical_workload_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if draw_count < 100:
        raise ValueError("frozen forecast artifacts require at least 100 marginal draws")
    da = _daily_energy(da_prediction_path)
    rt = _daily_energy(rt_prediction_path)
    workload = _daily_workload(workload_prediction_path)
    pv_days = _pv_validation_days(pv_panel_path)
    rt_validation_residuals = [
        value["observed_target"] - value["q50"]
        for key, value in rt.items()
        if key.startswith("validation:")
    ]
    workload_validation_residuals = [
        value["observed"] - value["predicted"]
        for key, value in workload.items()
        if key.startswith("validation:")
    ]
    workload_test = [
        value["predicted"] for key, value in workload.items() if key.startswith("locked_test:")
    ]
    deployment_days = sorted(
        key.split(":", 1)[1]
        for key in da
        if key.startswith("deployment:")
        and f"deployment:{key.split(':', 1)[1]}" in rt
    )
    locked_test_days = sorted(
        key.split(":", 1)[1]
        for key in da
        if key.startswith("locked_test:") and f"locked_test:{key.split(':', 1)[1]}" in rt
    )
    day_role = "deployment" if deployment_days else "locked_test"
    forecast_days = deployment_days if deployment_days else locked_test_days
    if not rt_validation_residuals or not workload_validation_residuals or not workload_test:
        raise ValueError("forecast-day factory lacks validation residuals or workload replay days")
    source_hashes = {
        "da_predictions": sha256_file(da_prediction_path),
        "rt_predictions": sha256_file(rt_prediction_path),
        "workload_predictions": sha256_file(workload_prediction_path),
        "pv_panel": sha256_file(pv_panel_path),
    }
    if physical_workload_contract is not None:
        contract_sha256 = physical_workload_contract.get("contract_sha256")
        event_sha256 = physical_workload_contract.get("sha256")
        if not contract_sha256 or not event_sha256:
            raise ValueError("physical workload contract lacks its contract or event hash")
        source_hashes["physical_workload_contract"] = str(contract_sha256)
        source_hashes["physical_workload_events"] = str(event_sha256)
    records = []
    for position, operating_date in enumerate(forecast_days):
        key = f"{day_role}:{operating_date}"
        da_values = da[key]
        rt_values = rt[key]
        da_quality = dict(da_values["target_quality"])
        rt_quality = dict(rt_values["target_quality"])
        realized_targets_complete = bool(
            da_quality["realized_target_complete"]
            and rt_quality["realized_target_complete"]
        )
        eligible_for_locked_test_scoring = bool(
            realized_targets_complete
            and (
                operating_date in locked_test_days
                or (
                    da_quality["eligible_for_locked_test_scoring"]
                    and rt_quality["eligible_for_locked_test_scoring"]
                )
            )
        )
        missing_reasons = []
        if not da_quality["realized_target_complete"]:
            missing_reasons.append(
                "DA historical realized target has incomplete zone-quarter-hour values"
            )
        if not rt_quality["realized_target_complete"]:
            missing_reasons.append(
                "RT historical realized target lacks one or more complete nominal three-record 15-minute blocks"
            )
        seed = _stable_seed(operating_date)
        rt_residual_pool = _sample_rows(rt_validation_residuals, count=draw_count, seed=seed)
        rt_pool = rt_values["q50"][None, :, :] + rt_residual_pool
        workload_base = workload_test[position % len(workload_test)]
        workload_residual_pool = _sample_rows(
            workload_validation_residuals, count=draw_count, seed=seed + 1
        )
        workload_pool = np.maximum(workload_base[None, :] + workload_residual_pool, 0.0)
        pv_pool = _sample_rows(pv_days, count=draw_count, seed=seed + 2)
        timestamps = pd.to_datetime(da_values["timestamps"], utc=True).to_numpy(dtype="datetime64[ns]")
        forecast_id = f"real_forecast_{operating_date}_{seed}"
        issue_time = (pd.Timestamp(operating_date, tz="UTC") - pd.Timedelta(days=1)).isoformat()
        artifact_payload = {
            "issue_time_utc": issue_time,
            "operating_date": operating_date,
            "forecast_id": forecast_id,
            "timestamps_utc": timestamps,
            "da_price": da_values["q50"],
            "rt_price_pool": rt_pool,
            "workload_pool": workload_pool,
            "pv_pool": pv_pool,
            "zones": ZONES,
            "source_hashes": source_hashes,
            "locked_test": True,
        }
        FrozenForecastArtifact.from_mapping(artifact_payload)
        day_path = output_root / "forecast_days" / f"{operating_date}.npz"
        _atomic_npz(
            day_path,
            issue_time_utc=np.asarray(issue_time),
            operating_date=np.asarray(operating_date),
            forecast_id=np.asarray(forecast_id),
            timestamps_utc=timestamps,
            da_price=da_values["q50"],
            rt_price_pool=rt_pool,
            workload_pool=workload_pool,
            pv_pool=pv_pool,
            zones=np.asarray(ZONES),
            source_hashes_json=np.asarray(json.dumps(source_hashes, sort_keys=True)),
            locked_test=np.asarray(True),
            forecast_role=np.asarray("deployment_counterfactual" if day_role == "deployment" else "locked_test_evaluation"),
            realized_targets_complete=np.asarray(realized_targets_complete),
            eligible_for_locked_test_scoring=np.asarray(eligible_for_locked_test_scoring),
            realized_target_quality_json=np.asarray(
                json.dumps(
                    {"da": da_quality, "rt": rt_quality},
                    sort_keys=True,
                )
            ),
        )
        records.append(
            {
                "operating_date": operating_date,
                "issue_time_utc": issue_time,
                "forecast_id": forecast_id,
                "path": day_path.resolve().relative_to(repository_root.resolve()).as_posix(),
                "sha256": sha256_file(day_path),
                "draw_count": draw_count,
                "horizon_intervals": 96,
                "zones": list(ZONES),
                "forecast_role": (
                    "deployment_counterfactual"
                    if day_role == "deployment"
                    else "locked_test_evaluation"
                ),
                "realized_targets_complete": realized_targets_complete,
                "eligible_for_locked_test_scoring": eligible_for_locked_test_scoring,
                "missing_realized_target_reasons": missing_reasons,
                "realized_target_quality": {"da": da_quality, "rt": rt_quality},
                "realized_targets_used_to_generate_forecast": False,
                "missing_realized_targets_imputed": False,
                "workload_replay_mapping": (
                    "cyclic mapping to separately held-out NLR day; no joint NYISO-NLR observation claimed"
                ),
                "pv_pool_source": "bootstrap of 2024 NASA-derived counterfactual PV day shapes",
            }
        )
    index_path = output_root / "forecast_days/index.json"
    scorable_days = [
        item["operating_date"] for item in records
        if item["eligible_for_locked_test_scoring"]
    ]
    deployment_only_missing_target_days = [
        {
            "operating_date": item["operating_date"],
            "missing_realized_target_reasons": item["missing_realized_target_reasons"],
            "realized_target_quality": item["realized_target_quality"],
        }
        for item in records
        if not item["realized_targets_complete"]
    ]
    expected_deployment_dates = [
        (pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=index))
        .date()
        .isoformat()
        for index in range(365)
    ]
    missing_deployment_dates = (
        sorted(set(expected_deployment_dates) - set(forecast_days))
        if day_role == "deployment"
        else []
    )
    deployment_coverage_complete = bool(
        day_role == "deployment" and not missing_deployment_dates and len(records) == 365
    )
    index = {
        "schema_version": 3,
        "status": (
            "COMPLETE"
            if records and (day_role != "deployment" or deployment_coverage_complete)
            else "PARTIAL"
            if records
            else "FAILED"
        ),
        "factory": "dcvpp.experiments.forecast_day_factory.load_frozen_forecast_day",
        "access_mode": "lazy_hash_verified_per_day_npz",
        "eager_array_materialization": False,
        "draw_count": draw_count,
        "source_hashes": source_hashes,
        "day_count": len(records),
        "days": records,
        "source_relationship": "counterfactual_coupling_of_separately_referenced_domains",
        "measurement_relationship": "not_measured_by_this_study",
        "forecast_role": (
            "deployment_counterfactual_full_year"
            if day_role == "deployment"
            else "locked_test_evaluation_only"
        ),
        "deployment_day_count": len(records) if day_role == "deployment" else 0,
        "deployment_target_year": 2025 if day_role == "deployment" else None,
        "deployment_coverage_complete": deployment_coverage_complete,
        "missing_deployment_dates": missing_deployment_dates,
        "available_locked_test_day_count": len(locked_test_days),
        "scorable_locked_test_day_count": len(scorable_days),
        "scorable_locked_test_dates": scorable_days,
        "deployment_only_missing_target_day_count": len(
            deployment_only_missing_target_days
        ),
        "deployment_only_missing_target_days": deployment_only_missing_target_days,
        "deployment_predictions_use_realized_targets": False,
        "missing_realized_targets_imputed": False,
        "price_support_policy": "negative DA/RT prices and forecast draws are preserved without clipping",
        "date_policy": (
            "all generated 2025 deployment origins; realized-target completeness gates scoring only, not forecast generation"
            if day_role == "deployment"
            else "all complete DA/RT locked-test dates; downstream primary selection remains deterministic"
        ),
    }
    atomic_json(index_path, index)
    return {
        "status": index["status"],
        "day_count": len(records),
        "draw_count": draw_count,
        "forecast_role": index["forecast_role"],
        "deployment_day_count": index["deployment_day_count"],
        "deployment_coverage_complete": index["deployment_coverage_complete"],
        "missing_deployment_dates": index["missing_deployment_dates"],
        "scorable_locked_test_day_count": index["scorable_locked_test_day_count"],
        "deployment_only_missing_target_day_count": index[
            "deployment_only_missing_target_day_count"
        ],
        "deployment_only_missing_target_days": index[
            "deployment_only_missing_target_days"
        ],
        "index_path": index_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "factory": index["factory"],
    }


def load_frozen_forecast_day(index_path: Path, operating_date: str) -> dict[str, Any]:
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record = next(
        (item for item in index["days"] if item["operating_date"] == operating_date), None
    )
    if record is None:
        raise KeyError(f"operating date is not in frozen forecast index: {operating_date}")
    sibling = index_path.parent / f"{operating_date}.npz"
    if sibling.is_file():
        path = sibling
    else:
        repository_root = index_path
        while repository_root.name != "power-dc-vpp" and repository_root.parent != repository_root:
            repository_root = repository_root.parent
        path = repository_root / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"frozen forecast day hash mismatch: {operating_date}")
    with np.load(path, allow_pickle=False) as data:
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
            "source_hashes": json.loads(str(data["source_hashes_json"].item())),
            "locked_test": bool(data["locked_test"].item()),
            "forecast_role": (
                str(data["forecast_role"].item())
                if "forecast_role" in data.files
                else "locked_test_evaluation"
            ),
            "realized_targets_complete": (
                bool(data["realized_targets_complete"].item())
                if "realized_targets_complete" in data.files
                else True
            ),
            "eligible_for_locked_test_scoring": (
                bool(data["eligible_for_locked_test_scoring"].item())
                if "eligible_for_locked_test_scoring" in data.files
                else True
            ),
            "realized_target_quality": (
                json.loads(str(data["realized_target_quality_json"].item()))
                if "realized_target_quality_json" in data.files
                else {}
            ),
        }
    FrozenForecastArtifact.from_mapping(payload)
    return {"frozen_artifact": payload}
