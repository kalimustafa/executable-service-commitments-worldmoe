"""Resumable, append-only orchestration for the 364-day C1 closed loop."""

from __future__ import annotations

import inspect
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch


C1_DAILY_RECORD_SCHEMA = "dcvpp.vpp-world-moe.c1-daily-record.v1"
C1_PROGRESS_SCHEMA = "dcvpp.vpp-world-moe.c1-progress.v1"
C1_SCHEDULE_SCHEMA = "dcvpp.vpp-world-moe.c1-schedule.v1"
C1_INTRADAY_INTERVALS = 96


class C1EvaluationError(RuntimeError):
    """C1 cannot preserve its immutable checkpoint or daily evidence contract."""


@dataclass(frozen=True)
class C1EvaluationResult:
    records: tuple[Mapping[str, Any], ...]
    completed_days: int
    appended_days: int
    record_path: str
    progress_path: str
    checkpoint_id: str


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _call_supported(callback: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(callback)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = (
        kwargs
        if accepts_kwargs
        else {key: value for key, value in kwargs.items() if key in signature.parameters}
    )
    return callback(**selected)


def _finite_scalar(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise C1EvaluationError(f"C1 daily field {label} must be numeric") from exc
    if not math.isfinite(result):
        raise C1EvaluationError(f"C1 daily field {label} must be finite")
    return result


def _mapping_scalars(
    record: Mapping[str, Any], name: str, keys: Sequence[str]
) -> None:
    value = record.get(name)
    if not isinstance(value, Mapping) or set(map(str, value)) != set(keys):
        raise C1EvaluationError(f"C1 daily field {name} must contain {list(keys)}")
    normalized = {str(key): item for key, item in value.items()}
    for key in keys:
        _finite_scalar(normalized[key], label=f"{name}.{key}")


def validate_daily_record(
    raw: Mapping[str, Any],
    *,
    expected_date: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Normalize one complete day and fail before append on incomplete evidence."""

    record = dict(raw)
    day = str(record.get("date", expected_date))[:10]
    if day != expected_date:
        raise C1EvaluationError(
            f"daily evaluator returned date {day}, expected {expected_date}"
        )
    try:
        parsed = date.fromisoformat(day)
    except ValueError as exc:
        raise C1EvaluationError("C1 date must be ISO YYYY-MM-DD") from exc
    if parsed.year != 2024:
        raise C1EvaluationError("C1 may evaluate development year 2024 only")
    supplied_checkpoint = str(
        record.get(
            "checkpoint_id",
            record.get("checkpoint_sha256", record.get("checkpoint", checkpoint_id)),
        )
    ).strip()
    if supplied_checkpoint != checkpoint_id:
        raise C1EvaluationError("daily record checkpoint differs from immutable J1 SHA")
    if record.get("locked_2025_opened") is not False:
        raise C1EvaluationError(
            "daily record must explicitly prove locked_2025_opened=false"
        )
    intervals = int(record.get("intraday_intervals_evaluated", -1))
    execution = str(record.get("intraday_execution", ""))
    if intervals != C1_INTRADAY_INTERVALS or execution != "sequential_0_to_95":
        raise C1EvaluationError("each C1 day must execute 96 intervals sequentially")

    _mapping_scalars(
        record,
        "model_reconstruction_losses",
        ("public_21", "queue", "soc"),
    )
    _mapping_scalars(
        record,
        "baseline_reconstruction_losses",
        ("public_21", "queue", "soc"),
    )
    _mapping_scalars(
        record,
        "model_dynamics_losses",
        ("1", "4", "16", "96"),
    )
    _mapping_scalars(
        record,
        "baseline_dynamics_losses",
        ("1", "4", "16", "96"),
    )
    required_scalars = (
        "model_nll",
        "baseline_nll",
        "model_crps",
        "baseline_crps",
        "model_energy",
        "baseline_energy",
        "model_variogram",
        "baseline_variogram",
        "model_tail",
        "baseline_tail",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "scenario_diversity",
        "scenario_duplicate_rate",
        "model_closed_loop_cost",
        "baseline_closed_loop_cost",
        "model_closed_loop_cvar",
        "baseline_closed_loop_cvar",
        "constraint_violation_max_abs",
        "model_sla_violation_rate",
        "baseline_sla_violation_rate",
        "dart_identity_max_abs",
        "power_balance_residual_max_abs",
        "soc_balance_residual_max_abs",
        "queue_balance_residual_max_abs",
        "action_bound_violation_max_abs",
        "shield_residual_max_abs",
    )
    for name in required_scalars:
        _finite_scalar(record.get(name), label=name)
    record.update(
        schema=C1_DAILY_RECORD_SCHEMA,
        date=day,
        checkpoint_id=checkpoint_id,
        locked_2025_opened=False,
        intraday_intervals_evaluated=C1_INTRADAY_INTERVALS,
        intraday_execution="sequential_0_to_95",
    )
    record.pop("checkpoint_sha256", None)
    record.pop("checkpoint", None)
    return record


def validate_schedule(dates: Sequence[str], *, expected_days: int = 364) -> tuple[str, ...]:
    normalized = tuple(str(value)[:10] for value in dates)
    if len(normalized) != int(expected_days) or len(set(normalized)) != len(normalized):
        raise C1EvaluationError(
            f"C1 schedule requires exactly {expected_days} unique dates"
        )
    try:
        parsed = tuple(date.fromisoformat(value) for value in normalized)
    except ValueError as exc:
        raise C1EvaluationError("C1 schedule contains a non-ISO date") from exc
    if any(value.year != 2024 for value in parsed):
        raise C1EvaluationError("C1 schedule must remain entirely within 2024")
    if tuple(sorted(normalized)) != normalized:
        raise C1EvaluationError("C1 schedule must be chronological")
    return normalized


class C1DailyRecordStore:
    """One append-only JSONL source plus an immutable schedule sidecar."""

    def __init__(
        self,
        path: str | Path,
        *,
        checkpoint_id: str,
        dates: Sequence[str],
        resume: bool,
    ) -> None:
        self.path = Path(path).resolve()
        self.checkpoint_id = str(checkpoint_id)
        self.dates = validate_schedule(dates)
        self.schedule_path = self.path.with_name("schedule.json")
        self._seen_dates: set[str] | None = None
        if self.path.is_file() and self.path.stat().st_size and not resume:
            raise C1EvaluationError("C1 daily evidence exists; explicit resume is required")
        schedule = {
            "schema": C1_SCHEDULE_SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "dates": list(self.dates),
            "intraday_intervals": C1_INTRADAY_INTERVALS,
            "locked_2025_opened": False,
        }
        if self.schedule_path.is_file():
            existing = json.loads(self.schedule_path.read_text(encoding="utf-8"))
            if existing != schedule:
                raise C1EvaluationError("C1 resume schedule/checkpoint changed")
        else:
            _atomic_json(self.schedule_path, schedule)

    def records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.is_file():
            return ()
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        allowed = set(self.dates)
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise C1EvaluationError(
                        f"C1 JSONL has an incomplete line at {line_number}"
                    )
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise C1EvaluationError(
                        f"C1 JSONL line {line_number} is malformed"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise C1EvaluationError("C1 JSONL rows must be mappings")
                day = str(raw.get("date", ""))[:10]
                if day not in allowed or day in seen:
                    raise C1EvaluationError("C1 JSONL has an unknown or duplicate date")
                rows.append(
                    validate_daily_record(
                        raw, expected_date=day, checkpoint_id=self.checkpoint_id
                    )
                )
                seen.add(day)
        rows.sort(key=lambda item: self.dates.index(str(item["date"])))
        self._seen_dates = {str(item["date"]) for item in rows}
        return tuple(rows)

    def append(self, record: Mapping[str, Any]) -> None:
        day = str(record.get("date", ""))[:10]
        normalized = validate_daily_record(
            record, expected_date=day, checkpoint_id=self.checkpoint_id
        )
        if day not in self.dates:
            raise C1EvaluationError("cannot append a date outside the frozen schedule")
        if self._seen_dates is None:
            self.records()
        if day in (self._seen_dates or set()):
            raise C1EvaluationError(f"C1 date {day} is already append-only evidence")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if self._seen_dates is None:
            self._seen_dates = set()
        self._seen_dates.add(day)


def run_daily_closed_loop(
    *,
    dates: Sequence[str],
    checkpoint_id: str,
    record_path: str | Path,
    progress_path: str | Path,
    evaluate_day: Callable[..., Mapping[str, Any]],
    model: Any,
    corpus: Any,
    resume: bool,
    day_batch_size: int = 1,
    max_workers: int = 1,
    parallel_safe: bool = False,
) -> C1EvaluationResult:
    """Evaluate independent days in batches; every callback receives 0..95 order."""

    if int(day_batch_size) < 1 or int(max_workers) < 1:
        raise C1EvaluationError("C1 day batch size and worker count must be positive")
    if int(max_workers) > 1 and not parallel_safe:
        raise C1EvaluationError("parallel C1 days require an explicit parallel-safe evaluator")
    schedule = validate_schedule(dates)
    store = C1DailyRecordStore(
        record_path,
        checkpoint_id=checkpoint_id,
        dates=schedule,
        resume=resume,
    )
    existing = store.records()
    completed = {str(record["date"]) for record in existing}
    missing = [value for value in schedule if value not in completed]
    progress = Path(progress_path).resolve()

    def one_day(day: str) -> dict[str, Any]:
        with torch.inference_mode():
            raw = _call_supported(
                evaluate_day,
                date=day,
                model=model,
                corpus=corpus,
                checkpoint_id=checkpoint_id,
                interval_indices=tuple(range(C1_INTRADAY_INTERVALS)),
                intraday_intervals=C1_INTRADAY_INTERVALS,
                sequential_intraday=True,
                split="validation",
            )
        if not isinstance(raw, Mapping):
            raise C1EvaluationError("daily evaluator must return one mapping")
        return validate_daily_record(
            raw, expected_date=day, checkpoint_id=checkpoint_id
        )

    appended = 0
    for start in range(0, len(missing), int(day_batch_size)):
        batch = missing[start : start + int(day_batch_size)]
        if int(max_workers) == 1:
            rows = [one_day(day) for day in batch]
        else:
            with ThreadPoolExecutor(max_workers=min(int(max_workers), len(batch))) as pool:
                rows = list(pool.map(one_day, batch))
        rows.sort(key=lambda item: schedule.index(str(item["date"])))
        for row in rows:
            store.append(row)
            appended += 1
            _atomic_json(
                progress,
                {
                    "schema": C1_PROGRESS_SCHEMA,
                    "checkpoint_id": checkpoint_id,
                    "completed_days": len(completed) + appended,
                    "expected_days": len(schedule),
                    "last_completed_date": row["date"],
                    "record_path": str(store.path),
                    "locked_2025_opened": False,
                },
            )
    records = store.records()
    return C1EvaluationResult(
        records=records,
        completed_days=len(records),
        appended_days=appended,
        record_path=str(store.path),
        progress_path=str(progress),
        checkpoint_id=checkpoint_id,
    )


__all__ = [
    "C1_DAILY_RECORD_SCHEMA",
    "C1_INTRADAY_INTERVALS",
    "C1DailyRecordStore",
    "C1EvaluationError",
    "C1EvaluationResult",
    "run_daily_closed_loop",
    "validate_daily_record",
    "validate_schedule",
]
