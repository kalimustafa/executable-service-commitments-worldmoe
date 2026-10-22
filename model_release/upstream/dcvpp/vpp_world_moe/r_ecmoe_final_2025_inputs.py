"""Materialize the authorized 2025 R-ECMoE final-test input bundles.

Inputs are frozen before model inference. Each final day uses the nearest
same-weekday pre-2025 issue bundle, the preregistered circular workload trace,
the 2025 NASA-reconstructed PV forcing, and official 2025 NYISO settlement for
post-decision replay. No 2025 statistic is fitted and no model output is read.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
SLOTS = 96


class Final2025InputError(RuntimeError):
    """The frozen final input panel cannot be materialized exactly."""


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Final2025InputError(f"JSON root must be a mapping: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _circular_distance(left: date, right: date) -> int:
    raw = abs(left.timetuple().tm_yday - right.timetuple().tm_yday)
    return min(raw, 366 - raw)


def _template_map(
    selection_root: Path, final_dates: tuple[str, ...]
) -> tuple[Mapping[str, Path], Mapping[str, np.ndarray]]:
    templates = sorted(selection_root.glob("*.npz"))
    if not templates:
        raise Final2025InputError("selection input bundles are unavailable")
    dated = [(date.fromisoformat(path.stem), path) for path in templates]
    if any(value.year >= 2025 for value, _ in dated):
        raise Final2025InputError("selection template root crossed into 2025")
    result: dict[str, Path] = {}
    forcing: dict[str, np.ndarray] = {}
    for _, path in dated:
        with np.load(path, allow_pickle=False) as source:
            source_date = str(source["forcing_workload_source_date"].item())
            forcing.setdefault(
                source_date, np.asarray(source["workload"], dtype=np.float32)
            )
    if not forcing or any(value.shape != (SLOTS,) for value in forcing.values()):
        raise Final2025InputError("registered workload forcing map is malformed")
    for day in final_dates:
        target = date.fromisoformat(day)
        same_weekday = [value for value in dated if value[0].weekday() == target.weekday()]
        candidates = same_weekday or dated
        result[day] = min(
            candidates,
            key=lambda value: (
                _circular_distance(value[0], target),
                -value[0].toordinal(),
            ),
        )[1]
    return result, forcing


def _pv_by_day(path: Path, final_dates: tuple[str, ...]) -> Mapping[str, np.ndarray]:
    frame = pq.ParquetFile(path).read(
        columns=["timestamp_utc", "zone", "pv_dc_availability_mw_per_mwdc"]
    ).to_pandas()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["operating_date"] = frame["timestamp_utc"].dt.date.astype(str)
    wanted = frame.loc[frame["operating_date"].isin(final_dates)].copy()
    output: dict[str, np.ndarray] = {}
    for day, group in wanted.groupby("operating_date", sort=True):
        timestamps = pd.date_range(day, periods=SLOTS, freq="15min", tz="UTC")
        if group.duplicated(["timestamp_utc", "zone"]).any():
            raise Final2025InputError(f"PV duplicate timestamp/zone: {day}")
        pivot = group.pivot(
            index="zone",
            columns="timestamp_utc",
            values="pv_dc_availability_mw_per_mwdc",
        ).reindex(index=ZONES, columns=timestamps)
        values = pivot.to_numpy(dtype=np.float32)
        if (
            values.shape != (4, SLOTS)
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise Final2025InputError(f"PV support/shape failed: {day}")
        output[str(day)] = values
    if set(output) != set(final_dates):
        raise Final2025InputError("NASA-derived PV panel does not cover every final date")
    return output


def _causal_source_dates(arrays: Mapping[str, Any], issue: pd.Timestamp) -> bool:
    for name in (
        "issue_workload_source_dates",
        "issue_pv_source_dates",
        "issue_price_source_dates",
    ):
        values = np.asarray(arrays[name]).astype(str)
        if values.size == 0:
            return False
        parsed = pd.to_datetime(values, utc=True)
        if bool((parsed >= issue).any()):
            return False
    return True


def materialize_final_2025_inputs(
    repository_root: str | Path,
    *,
    workers: int = 4,
) -> Mapping[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    official_root = root / (
        "artifacts/dart_value_select/real_input_preparation/v1/"
        "official_rt_recovery"
    )
    final_index = _load_json(
        official_root / "locked_test_nyiso_replay_by_date_index.json"
    )
    days = final_index.get("days")
    if final_index.get("status") != "COMPLETE" or not isinstance(days, Mapping):
        raise Final2025InputError("2025 official settlement index is not complete")
    final_dates = tuple(sorted(str(value) for value in days))
    if len(final_dates) != 362 or any(
        date.fromisoformat(value).year != 2025 for value in final_dates
    ):
        raise Final2025InputError("final settlement panel must contain 362 dates in 2025")
    settlement_root = official_root / "nyiso_replay_by_date/locked_test"
    if any(not (settlement_root / f"{day}.npz").is_file() for day in final_dates):
        raise Final2025InputError("one or more official final settlement files are absent")

    selection_root = root / (
        "artifacts/dart_value_select/counterfactual_inputs/v1/"
        "selection/daily_bundles"
    )
    templates, forcing_map = _template_map(selection_root, final_dates)
    forcing_sources = tuple(sorted(forcing_map))
    pv = _pv_by_day(root / "data/processed/nyiso_zone_pv_15m.parquet", final_dates)
    bundle_root = root / (
        "artifacts/dart_value_select/counterfactual_inputs/v1/"
        "locked_test/daily_bundles"
    )

    def write_day(day: str) -> Mapping[str, Any]:
        bundle_path = bundle_root / f"{day}.npz"
        record_path = bundle_root / f"{day}.record.json"
        if bundle_path.is_file() and record_path.is_file():
            record = _load_json(record_path).get("record")
            if isinstance(record, Mapping) and record.get("operating_date") == day:
                return dict(record)
        with np.load(templates[day], allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        target = date.fromisoformat(day)
        issue = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=1)
        forcing_index = (
            target.toordinal() - date(2022, 1, 1).toordinal()
        ) % len(forcing_sources)
        forcing_source = forcing_sources[forcing_index]
        arrays.update(
            {
                "issue_time_utc": np.asarray(issue.isoformat()),
                "operating_date": np.asarray(day),
                "forecast_id": np.asarray(f"r_ecmoe_final_2025_{day}"),
                "timestamps_utc": pd.date_range(
                    day, periods=SLOTS, freq="15min", tz="UTC"
                ).to_numpy(dtype="datetime64[ns]"),
                "locked_test": np.asarray(True),
                "workload": forcing_map[forcing_source],
                "pv": pv[day],
                "forcing_workload_source_date": np.asarray(forcing_source),
                "forcing_pv_source_date": np.asarray(day),
                "source_hashes_json": np.asarray(
                    json.dumps(
                        {
                            "input_policy": (
                                "pre_2025_same_weekday_template_plus_registered_trace"
                            ),
                            "hashes": "NOT_COMPUTED",
                        },
                        sort_keys=True,
                    )
                ),
            }
        )
        if not _causal_source_dates(arrays, issue):
            raise Final2025InputError(f"issue source date crossed gate: {day}")
        _atomic_npz(bundle_path, arrays)
        record = {
            "operating_date": day,
            "path": str(bundle_path.resolve()),
            "issue_time_utc": issue.isoformat(),
            "causal_gates": {
                "candidate_independent_price": True,
                "pv": True,
                "workload": True,
            },
            "issue_template_date": templates[day].stem,
            "issue_method": "nearest_same_weekday_pre_2025_frozen_bundle",
            "replay_pv": "NASA_POWER_2025_reconstruction",
            "replay_workload": "registered_cross_domain_circular_trace",
            "official_settlement": str((settlement_root / f"{day}.npz").resolve()),
            "opened_final_test_2025": True,
            "model_forward_calls": 0,
            "training_performed": False,
            "selection_performed": False,
            "hash_computed": False,
        }
        _atomic_json(
            record_path,
            {
                "schema": "dcvpp.r_ecmoe.final_2025_daily_input.v1",
                "record": record,
            },
        )
        return record

    records: list[Mapping[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(write_day, day): day for day in final_dates}
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda value: str(value["operating_date"]))
    readiness = {
        "schema": "dcvpp.r_ecmoe.final_2025_input_readiness.v1",
        "status": "READY" if len(records) == 362 else "INCOMPLETE",
        "year": 2025,
        "evaluable_days": len(records),
        "daily_bundle_npz_days": sum(
            (bundle_root / f"{day}.npz").is_file() for day in final_dates
        ),
        "daily_bundle_record_days": sum(
            (bundle_root / f"{day}.record.json").is_file() for day in final_dates
        ),
        "official_settlement_days": len(final_dates),
        "all_causal_gates_passed": all(
            all(record["causal_gates"].values()) for record in records
        ),
        "fit_on_2025": False,
        "model_forward_calls": 0,
        "selection_feedback_allowed": False,
        "missing_official_days_excluded": [
            "2025-05-27",
            "2025-05-28",
            "2025-11-12",
        ],
        "hash_computed": False,
        "bundle_root": str(bundle_root.resolve()),
    }
    if (
        readiness["status"] != "READY"
        or readiness["daily_bundle_npz_days"] != 362
        or readiness["daily_bundle_record_days"] != 362
        or readiness["all_causal_gates_passed"] is not True
    ):
        raise Final2025InputError("final input readiness did not pass")
    _atomic_json(bundle_root.parent / "final_input_readiness.json", readiness)
    return readiness


__all__ = ["Final2025InputError", "materialize_final_2025_inputs"]
