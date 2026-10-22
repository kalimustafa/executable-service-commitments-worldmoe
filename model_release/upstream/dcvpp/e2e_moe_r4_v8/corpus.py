"""R4-V8 development corpus and residual-anchor lineage.

The heavy daily-tensor construction is shared with the already verified V7
development loader, but V8 writes a distinct anchor schema and artifact path so
no V7 capability result can be inherited accidentally.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dcvpp.e2e_moe_r4_v7.corpus import (
    DEVELOPMENT_TRAIN_DAYS,
    DEVELOPMENT_TRAINVAL_DAYS,
    DEVELOPMENT_VALIDATION_DAYS,
    PreloadedV7DevelopmentCorpus,
    V7DevelopmentCorpusError,
    development_date_inventory,
    materialize_dynamic_residual_anchor,
)


V8_ANCHOR_SCHEMA = "dcvpp.r4.v8.trifusion-development-residual-anchor.v1"
REPRESENTATION_ANCHOR_SCHEMA = "dcvpp.r4.v8.representation-causal-anchors.v1"
REPRESENTATION_REPAIR_STATISTICS_SCHEMA = (
    "dcvpp.r4.v8.representation-repair-statistics.v1"
)
REPRESENTATION_REPAIR_CAP_QUANTILE = 0.95
WORKLOAD_ZERO_TARGET_THRESHOLD = 1.0e-6


class V8CorpusError(V7DevelopmentCorpusError):
    """Raised when V8 data would violate the registered temporal lineage."""


@dataclass(frozen=True)
class V8CandidateRepairStatistics:
    """Train-only caps for one fully specified causal anchor candidate."""

    name: str
    eligible_train_dates: tuple[str, ...]
    price_level_cap: Tensor
    price_spread_cap: Tensor
    workload_cap: Tensor

    def validate(self) -> "V8CandidateRepairStatistics":
        if self.name not in {"persistence", "seasonal"}:
            raise V8CorpusError("repair candidate must be persistence or seasonal")
        dates = self.eligible_train_dates
        if (
            not dates
            or dates != tuple(sorted(dates))
            or len(dates) != len(set(dates))
            or any(not day.startswith(("2022-", "2023-")) for day in dates)
        ):
            raise V8CorpusError(
                f"{self.name} repair caps require unique ordered 2022--2023 dates"
            )
        for label, value, shape in (
            ("price_level_cap", self.price_level_cap, (4,)),
            ("price_spread_cap", self.price_spread_cap, (4,)),
            ("workload_cap", self.workload_cap, (1,)),
        ):
            tensor = torch.as_tensor(value)
            if (
                tensor.shape != shape
                or not bool(torch.isfinite(tensor).all())
                or bool((tensor <= 0).any())
            ):
                raise V8CorpusError(
                    f"{self.name} {label} must be finite positive {shape}"
                )
        return self

    def to(
        self, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> "V8CandidateRepairStatistics":
        return V8CandidateRepairStatistics(
            name=self.name,
            eligible_train_dates=self.eligible_train_dates,
            price_level_cap=self.price_level_cap.to(device=device, dtype=dtype),
            price_spread_cap=self.price_spread_cap.to(device=device, dtype=dtype),
            workload_cap=self.workload_cap.to(device=device, dtype=dtype),
        ).validate()

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "eligible_train_dates": self.eligible_train_dates,
            "eligible_train_days": len(self.eligible_train_dates),
            "price_level_cap": self.price_level_cap.detach().cpu().tolist(),
            "price_spread_cap": self.price_spread_cap.detach().cpu().tolist(),
            "workload_cap": self.workload_cap.detach().cpu().tolist(),
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, expected_name: str
    ) -> "V8CandidateRepairStatistics":
        result = cls(
            name=str(value.get("name", "")),
            eligible_train_dates=tuple(
                str(day) for day in value.get("eligible_train_dates", ())
            ),
            price_level_cap=torch.as_tensor(
                value.get("price_level_cap"), dtype=torch.float32
            ),
            price_spread_cap=torch.as_tensor(
                value.get("price_spread_cap"), dtype=torch.float32
            ),
            workload_cap=torch.as_tensor(
                value.get("workload_cap"), dtype=torch.float32
            ),
        ).validate()
        if result.name != expected_name or int(
            value.get("eligible_train_days", len(result.eligible_train_dates))
        ) != len(result.eligible_train_dates):
            raise V8CorpusError(f"malformed {expected_name} repair statistics")
        return result


@dataclass(frozen=True)
class V8RepresentationRepairStatistics:
    """Candidate caps and intermittent-workload facts fitted on train only."""

    persistence: V8CandidateRepairStatistics
    seasonal: V8CandidateRepairStatistics
    workload_zero_target_threshold: float
    workload_zero_day_count: int
    workload_near_zero_day_count: int
    workload_active_day_count: int
    workload_nonzero_day_count: int
    workload_zero_pos_weight: float
    workload_positive_scale_floor: float
    workload_near_zero_target_threshold: float
    workload_active_median: float
    cap_quantile: float = REPRESENTATION_REPAIR_CAP_QUANTILE
    locked_2025_opened: bool = False

    def validate(self) -> "V8RepresentationRepairStatistics":
        self.persistence.validate()
        self.seasonal.validate()
        if abs(float(self.cap_quantile) - REPRESENTATION_REPAIR_CAP_QUANTILE) > 1e-12:
            raise V8CorpusError("Representation repair caps must use train-only q95")
        if (
            abs(
                float(self.workload_zero_target_threshold)
                - WORKLOAD_ZERO_TARGET_THRESHOLD
            )
            > 1e-12
        ):
            raise V8CorpusError("unexpected workload zero-target threshold")
        if (
            int(self.workload_zero_day_count) < 0
            or int(self.workload_near_zero_day_count) < 0
            or int(self.workload_active_day_count) < 0
            or int(self.workload_nonzero_day_count) < 0
            or int(self.workload_zero_day_count) + int(self.workload_nonzero_day_count)
            != len(self.persistence.eligible_train_dates)
            or int(self.workload_near_zero_day_count)
            + int(self.workload_active_day_count)
            != int(self.workload_nonzero_day_count)
        ):
            raise V8CorpusError(
                "workload hurdle counts do not match eligible train days"
            )
        if (
            not math.isfinite(float(self.workload_zero_pos_weight))
            or float(self.workload_zero_pos_weight) <= 0.0
            or not math.isfinite(float(self.workload_positive_scale_floor))
            or float(self.workload_positive_scale_floor) <= 0.0
            or not math.isfinite(float(self.workload_near_zero_target_threshold))
            or float(self.workload_near_zero_target_threshold)
            <= float(self.workload_zero_target_threshold)
            or not math.isfinite(float(self.workload_active_median))
            or float(self.workload_active_median)
            < float(self.workload_near_zero_target_threshold)
        ):
            raise V8CorpusError("workload hurdle train statistics are invalid")
        if self.locked_2025_opened is not False:
            raise V8CorpusError("2025 cannot enter Representation repair statistics")
        return self

    def to(
        self, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> "V8RepresentationRepairStatistics":
        return V8RepresentationRepairStatistics(
            persistence=self.persistence.to(device, dtype),
            seasonal=self.seasonal.to(device, dtype),
            workload_zero_target_threshold=self.workload_zero_target_threshold,
            workload_zero_day_count=self.workload_zero_day_count,
            workload_near_zero_day_count=self.workload_near_zero_day_count,
            workload_active_day_count=self.workload_active_day_count,
            workload_nonzero_day_count=self.workload_nonzero_day_count,
            workload_zero_pos_weight=self.workload_zero_pos_weight,
            workload_positive_scale_floor=self.workload_positive_scale_floor,
            workload_near_zero_target_threshold=(
                self.workload_near_zero_target_threshold
            ),
            workload_active_median=self.workload_active_median,
            cap_quantile=self.cap_quantile,
            locked_2025_opened=False,
        ).validate()

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": REPRESENTATION_REPAIR_STATISTICS_SCHEMA,
            "fit_split": "2022-01-01_to_2023-12-31_only",
            "eligibility": (
                "source operating day fully completed no later than issue_time_utc"
            ),
            "cap_quantile": float(self.cap_quantile),
            "persistence": self.persistence.as_dict(),
            "seasonal": self.seasonal.as_dict(),
            "workload_zero_target_threshold": float(
                self.workload_zero_target_threshold
            ),
            "workload_zero_day_count": int(self.workload_zero_day_count),
            "workload_near_zero_day_count": int(self.workload_near_zero_day_count),
            "workload_active_day_count": int(self.workload_active_day_count),
            "workload_nonzero_day_count": int(self.workload_nonzero_day_count),
            "workload_zero_pos_weight": float(self.workload_zero_pos_weight),
            "workload_positive_scale_floor": float(self.workload_positive_scale_floor),
            "workload_near_zero_target_threshold": float(
                self.workload_near_zero_target_threshold
            ),
            "workload_active_median": float(self.workload_active_median),
            "locked_2025_opened": False,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "V8RepresentationRepairStatistics":
        if value.get("schema") != REPRESENTATION_REPAIR_STATISTICS_SCHEMA:
            raise V8CorpusError("unexpected Representation repair statistics schema")
        persistence = value.get("persistence")
        seasonal = value.get("seasonal")
        if not isinstance(persistence, Mapping) or not isinstance(seasonal, Mapping):
            raise V8CorpusError(
                "Representation repair candidate statistics are missing"
            )
        return cls(
            persistence=V8CandidateRepairStatistics.from_mapping(
                persistence, expected_name="persistence"
            ),
            seasonal=V8CandidateRepairStatistics.from_mapping(
                seasonal, expected_name="seasonal"
            ),
            workload_zero_target_threshold=float(
                value.get("workload_zero_target_threshold", float("nan"))
            ),
            workload_zero_day_count=int(value.get("workload_zero_day_count", -1)),
            workload_near_zero_day_count=int(
                value.get("workload_near_zero_day_count", -1)
            ),
            workload_active_day_count=int(value.get("workload_active_day_count", -1)),
            workload_nonzero_day_count=int(value.get("workload_nonzero_day_count", -1)),
            workload_zero_pos_weight=float(
                value.get("workload_zero_pos_weight", float("nan"))
            ),
            workload_positive_scale_floor=float(
                value.get("workload_positive_scale_floor", float("nan"))
            ),
            workload_near_zero_target_threshold=float(
                value.get("workload_near_zero_target_threshold", float("nan"))
            ),
            workload_active_median=float(
                value.get("workload_active_median", float("nan"))
            ),
            cap_quantile=float(value.get("cap_quantile", float("nan"))),
            locked_2025_opened=value.get("locked_2025_opened") is not False,
        ).validate()

    @property
    def price_level_cap(self) -> Tensor:
        return self.persistence.price_level_cap

    @property
    def price_spread_cap(self) -> Tensor:
        return self.persistence.price_spread_cap

    @property
    def workload_cap(self) -> Tensor:
        return self.persistence.workload_cap

    def as_batch_mapping(self) -> dict[str, Tensor]:
        """Return the selected persistence statistics with stable tensor names."""

        self.validate()
        reference = self.persistence.price_level_cap

        def scalar(value: float) -> Tensor:
            return torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype
            )

        return {
            "price_level_cap": self.price_level_cap,
            "price_spread_cap": self.price_spread_cap,
            "workload_cap": self.workload_cap,
            "zero_target_epsilon": scalar(self.workload_zero_target_threshold),
            "near_zero_threshold": scalar(self.workload_near_zero_target_threshold),
            "active_workload_median": scalar(self.workload_active_median),
            "positive_scale_floor": scalar(self.workload_positive_scale_floor),
            "workload_zero_pos_weight": scalar(self.workload_zero_pos_weight),
        }


class PreloadedV8Corpus(PreloadedV7DevelopmentCorpus):
    """The 722/364 V8 development corpus with a 1,086-day refit view."""

    def __init__(self, config: Mapping[str, Any], repository_root: str | Path) -> None:
        super().__init__(config, repository_root)
        self.representation_repair_statistics = attach_v8_representation_anchors(
            self._records["train"], self._records["validation"]
        )

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
        for batch in super().batches(
            split,
            batch_size,
            device=device,
            shuffle_seed=shuffle_seed,
            max_days=max_days,
            origin=origin,
        ):
            statistics = self.representation_repair_statistics.to(device)
            batch["representation_repair_statistics"] = statistics.as_batch_mapping()
            yield batch


@dataclass(frozen=True)
class V8ResidualAnchor:
    residuals: Tensor
    channel_scale: Tensor
    pue_center: Tensor
    train_dates: tuple[str, ...]
    source_path: Path
    metadata: Mapping[str, Any]
    representation_repair_statistics: V8RepresentationRepairStatistics | None = None

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "V8ResidualAnchor":
        return V8ResidualAnchor(
            residuals=self.residuals.to(device=device, dtype=dtype),
            channel_scale=self.channel_scale.to(device=device, dtype=dtype),
            pue_center=self.pue_center.to(device=device, dtype=dtype),
            train_dates=self.train_dates,
            source_path=self.source_path,
            metadata=dict(self.metadata),
            representation_repair_statistics=(
                None
                if self.representation_repair_statistics is None
                else self.representation_repair_statistics.to(device, dtype)
            ),
        )

    def deterministic_indices(
        self,
        sample_count: int,
        seed: int,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        count = int(sample_count)
        day_count = int(self.residuals.shape[0])
        if not 1 <= count <= day_count:
            raise V8CorpusError(f"sample_count must lie in [1,{day_count}]")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        result = torch.randperm(day_count, generator=generator)[:count]
        return result if device is None else result.to(device)


def _record_date(record: Mapping[str, Any]) -> date:
    try:
        value = date.fromisoformat(str(record.get("operating_date", ""))[:10])
    except ValueError as exc:
        raise V8CorpusError(
            "Representation anchor record has no ISO operating date"
        ) from exc
    if value.year >= 2025:
        raise V8CorpusError("locked 2025 is rejected before Representation anchors")
    return value


def _record_issue_time(record: Mapping[str, Any]) -> datetime:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise V8CorpusError("Representation anchor record omitted provenance")
    raw = str(provenance.get("issue_time_utc", "")).replace("Z", "+00:00")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise V8CorpusError("Representation anchor issue_time_utc is invalid") from exc
    if value.tzinfo is None:
        raise V8CorpusError("Representation anchor issue_time_utc must be aware")
    return value.astimezone(timezone.utc)


def _completed_before_issue(
    source: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    completed = datetime.combine(
        _record_date(source) + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    return completed <= _record_issue_time(current)


def _public_target(record: Mapping[str, Any]) -> Tensor:
    value = record.get("scenario_target_zonal")
    if not isinstance(value, Tensor):
        raise V8CorpusError(
            "Representation anchor source omitted scenario_target_zonal"
        )
    tensor = value.detach().cpu().float()
    if tensor.shape != (96, 21) or not bool(torch.isfinite(tensor).all()):
        raise V8CorpusError("Representation public target must be finite [96,21]")
    return tensor


def _issue_center_public(record: Mapping[str, Any], pue_center: Tensor) -> Tensor:
    market = record.get("market_raw")
    workload = record.get("issue_workload_raw")
    pv = record.get("issue_pv_raw")
    if not all(isinstance(value, Tensor) for value in (market, workload, pv)):
        raise V8CorpusError("Representation issue-center fields are missing")
    assert isinstance(market, Tensor)
    assert isinstance(workload, Tensor)
    assert isinstance(pv, Tensor)
    market = market.detach().cpu().float()
    workload = workload.detach().cpu().float()
    pv = pv.detach().cpu().float()
    if (
        market.shape != (96, 16)
        or workload.shape != (96,)
        or pv.shape != (96, 4)
        or pue_center.shape != (96, 4)
    ):
        raise V8CorpusError("Representation issue-center tensors have invalid shapes")
    da, rt = market[:, 0:4], market[:, 4:8]
    result = torch.cat((da, rt, da - rt, workload[:, None], pv, pue_center), dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise V8CorpusError("Representation issue center contains non-finite values")
    return result


def _candidate_statistics(
    train_records: Sequence[MutableMapping[str, Any]], candidate: str
) -> V8CandidateRepairStatistics:
    if candidate not in {"persistence", "seasonal"}:
        raise V8CorpusError("unknown Representation anchor candidate")
    candidate_index = 0 if candidate == "persistence" else 1
    level_residuals: list[Tensor] = []
    spread_residuals: list[Tensor] = []
    workload_residuals: list[Tensor] = []
    dates: list[str] = []
    for record in sorted(train_records, key=_record_date):
        anchors = record.get("representation_anchors")
        if not isinstance(anchors, Mapping):
            raise V8CorpusError("record omitted attached Representation anchors")
        availability = anchors.get("availability")
        if (
            not isinstance(availability, Tensor)
            or availability.shape != (2,)
            or availability.dtype is not torch.bool
        ):
            raise V8CorpusError("Representation anchor availability must be bool [2]")
        if not bool(availability[candidate_index]):
            continue
        raw_anchor = anchors.get(f"{candidate}_public")
        if not isinstance(raw_anchor, Tensor):
            raise V8CorpusError("Representation candidate anchor must be a tensor")
        anchor = raw_anchor.detach().cpu().float()
        target = _public_target(record)
        if anchor.shape != (96, 21) or not bool(torch.isfinite(anchor).all()):
            raise V8CorpusError("Representation candidate anchor must be [96,21]")
        target_level = 0.5 * (target[:, 0:4] + target[:, 4:8])
        anchor_level = 0.5 * (anchor[:, 0:4] + anchor[:, 4:8])
        target_spread = target[:, 0:4] - target[:, 4:8]
        anchor_spread = anchor[:, 0:4] - anchor[:, 4:8]
        level_residuals.append(target_level - anchor_level)
        spread_residuals.append(target_spread - anchor_spread)
        workload_residuals.append(target[:, 12:13] - anchor[:, 12:13])
        dates.append(_record_date(record).isoformat())
    if not dates:
        raise V8CorpusError(
            f"no eligible train dates exist for {candidate} repair caps"
        )

    def cap(values: Sequence[Tensor]) -> Tensor:
        flattened = torch.stack(tuple(values)).abs().flatten(0, 1)
        return torch.quantile(
            flattened, REPRESENTATION_REPAIR_CAP_QUANTILE, dim=0
        ).clamp_min(WORKLOAD_ZERO_TARGET_THRESHOLD)

    return V8CandidateRepairStatistics(
        name=candidate,
        eligible_train_dates=tuple(dates),
        price_level_cap=cap(level_residuals),
        price_spread_cap=cap(spread_residuals),
        workload_cap=cap(workload_residuals),
    ).validate()


def _repair_statistics(
    train_records: Sequence[MutableMapping[str, Any]],
) -> V8RepresentationRepairStatistics:
    persistence = _candidate_statistics(train_records, "persistence")
    seasonal = _candidate_statistics(train_records, "seasonal")
    by_date = {_record_date(record).isoformat(): record for record in train_records}
    workload_scales = torch.stack(
        [
            _public_target(by_date[day])[:, 12:13].abs().mean()
            for day in persistence.eligible_train_dates
        ]
    )
    zero = workload_scales <= WORKLOAD_ZERO_TARGET_THRESHOLD
    positive = workload_scales[~zero]
    positive_floor = (
        float(torch.quantile(positive, 0.01).clamp_min(WORKLOAD_ZERO_TARGET_THRESHOLD))
        if positive.numel()
        else WORKLOAD_ZERO_TARGET_THRESHOLD
    )
    near_threshold = (
        float(
            torch.quantile(positive, 0.05).clamp_min(
                2.0 * WORKLOAD_ZERO_TARGET_THRESHOLD
            )
        )
        if positive.numel()
        else 2.0 * WORKLOAD_ZERO_TARGET_THRESHOLD
    )
    near = (~zero) & (workload_scales < near_threshold)
    active = (~zero) & ~near
    active_values = workload_scales[active]
    active_median = (
        float(torch.median(active_values)) if active_values.numel() else near_threshold
    )
    zero_days = int(zero.sum().item())
    nonzero_days = int((~zero).sum().item())
    return V8RepresentationRepairStatistics(
        persistence=persistence,
        seasonal=seasonal,
        workload_zero_target_threshold=WORKLOAD_ZERO_TARGET_THRESHOLD,
        workload_zero_day_count=zero_days,
        workload_near_zero_day_count=int(near.sum().item()),
        workload_active_day_count=int(active.sum().item()),
        workload_nonzero_day_count=nonzero_days,
        workload_zero_pos_weight=math.sqrt(max(nonzero_days, 1) / max(zero_days, 1)),
        workload_positive_scale_floor=positive_floor,
        workload_near_zero_target_threshold=near_threshold,
        workload_active_median=active_median,
        cap_quantile=REPRESENTATION_REPAIR_CAP_QUANTILE,
        locked_2025_opened=False,
    ).validate()


def attach_v8_representation_anchors(
    train_records: Sequence[MutableMapping[str, Any]],
    validation_records: Sequence[MutableMapping[str, Any]],
) -> V8RepresentationRepairStatistics:
    """Attach causal candidates and fit repair caps without opening 2025.

    Candidate values are copied only from an earlier operating day that had
    fully completed by the current record's registered issue time. Unavailable
    candidates contain zeros plus a false availability mask.
    """

    ordered = sorted([*train_records, *validation_records], key=_record_date)
    if len({_record_date(record) for record in ordered}) != len(ordered):
        raise V8CorpusError("Representation anchor records must have unique dates")
    if any(_record_date(record).year not in {2022, 2023} for record in train_records):
        raise V8CorpusError("Representation repair fitting may use only 2022--2023")
    if any(_record_date(record).year != 2024 for record in validation_records):
        raise V8CorpusError("Representation validation anchors may use only 2024")
    pue_center = (
        torch.stack([_public_target(record)[:, 17:21] for record in train_records])
        .median(dim=0)
        .values
    )

    history: list[MutableMapping[str, Any]] = []
    for record in ordered:
        target = _public_target(record)
        safe = [source for source in history if _completed_before_issue(source, record)]
        persistence = safe[-1] if safe else None
        weekday = _record_date(record).weekday()
        same_weekday = [
            source for source in safe if _record_date(source).weekday() == weekday
        ]
        seasonal = same_weekday[-1] if same_weekday else None

        def candidate_value(source: Mapping[str, Any] | None) -> Tensor:
            return (
                torch.zeros_like(target)
                if source is None
                else _public_target(source).clone()
            )

        record["representation_anchors"] = {
            "issue_center_public": _issue_center_public(record, pue_center),
            "persistence_public": candidate_value(persistence),
            "seasonal_public": candidate_value(seasonal),
            "availability": torch.tensor(
                (persistence is not None, seasonal is not None), dtype=torch.bool
            ),
        }
        record["representation_anchor_metadata"] = {
            "schema": REPRESENTATION_ANCHOR_SCHEMA,
            "availability_order": ("persistence", "seasonal"),
            "eligibility": (
                "source operating day fully completed no later than issue_time_utc"
            ),
            "persistence_source_date": (
                None if persistence is None else _record_date(persistence).isoformat()
            ),
            "seasonal_source_date": (
                None if seasonal is None else _record_date(seasonal).isoformat()
            ),
            "unavailable_tensor_semantics": "zeros_ignored_by_boolean_mask",
            "locked_2025_opened": False,
        }
        history.append(record)
    return _repair_statistics(train_records)


def materialize_v8_residual_anchor(
    corpus: PreloadedV8Corpus,
    output_path: str | Path,
) -> V8ResidualAnchor:
    """Fit one train-only 722-day anchor and save it atomically."""

    base = materialize_dynamic_residual_anchor(corpus)
    destination = Path(output_path).resolve()
    payload = {
        "schema": V8_ANCHOR_SCHEMA,
        "architecture_lineage": "r4-v8-trifusion",
        "fit_split": "2022-01-01_to_2023-12-31_only",
        "train_day_count": len(base.train_dates),
        "train_dates": base.train_dates,
        "split_lineage": dict(base.split_lineage),
        "trajectory_sampling": "whole_day_joint_residual_without_replacement",
        "normalization": "train_residual_channel_median_absolute_scale",
        "pue_center": base.pue_center,
        "residuals": base.residuals,
        "scale": base.channel_scale,
        "representation_repair_statistics": (
            corpus.representation_repair_statistics.as_dict()
        ),
        "final_test_2025_opened": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return load_v8_residual_anchor(destination)


def load_v8_residual_anchor(path: str | Path) -> V8ResidualAnchor:
    source = Path(path).resolve()
    if not source.is_file():
        raise V8CorpusError(f"V8 residual anchor is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != V8_ANCHOR_SCHEMA:
        raise V8CorpusError("unexpected V8 residual-anchor schema")
    residuals = torch.as_tensor(payload.get("residuals"), dtype=torch.float32)
    scale = torch.as_tensor(payload.get("scale"), dtype=torch.float32)
    pue_center = torch.as_tensor(payload.get("pue_center"), dtype=torch.float32)
    dates = tuple(str(item) for item in payload.get("train_dates", ()))
    if residuals.shape != (DEVELOPMENT_TRAIN_DAYS, 96, 21):
        raise V8CorpusError("V8 residuals must be [722,96,21]")
    if (
        scale.shape != (21,)
        or bool((scale <= 0).any())
        or not bool(torch.isfinite(scale).all())
    ):
        raise V8CorpusError("V8 channel scale must contain 21 positive values")
    if pue_center.shape != (96, 4) or not bool(torch.isfinite(pue_center).all()):
        raise V8CorpusError("V8 PUE centre must be [96,4]")
    if (
        len(dates) != DEVELOPMENT_TRAIN_DAYS
        or len(set(dates)) != DEVELOPMENT_TRAIN_DAYS
        or dates != tuple(sorted(dates))
        or not all("2022-01-01" <= day <= "2023-12-31" for day in dates)
    ):
        raise V8CorpusError("V8 anchor dates do not match the 722-day train split")
    if payload.get("final_test_2025_opened") is not False:
        raise V8CorpusError("V8 anchor does not prove that 2025 remained sealed")
    repair_raw = payload.get("representation_repair_statistics")
    if repair_raw is not None and not isinstance(repair_raw, Mapping):
        raise V8CorpusError("V8 repair statistics payload must be a mapping")
    repair_statistics = (
        None
        if repair_raw is None
        else V8RepresentationRepairStatistics.from_mapping(repair_raw)
    )
    if repair_statistics is not None:
        train_date_set = set(dates)
        for candidate in (
            repair_statistics.persistence,
            repair_statistics.seasonal,
        ):
            if not set(candidate.eligible_train_dates).issubset(train_date_set):
                raise V8CorpusError(
                    "V8 repair statistics contain dates outside the residual anchor"
                )
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"residuals", "scale", "pue_center"}
    }
    return V8ResidualAnchor(
        residuals=residuals,
        channel_scale=scale,
        pue_center=pue_center,
        train_dates=dates,
        source_path=source,
        metadata=metadata,
        representation_repair_statistics=repair_statistics,
    )


__all__ = [
    "DEVELOPMENT_TRAIN_DAYS",
    "DEVELOPMENT_TRAINVAL_DAYS",
    "DEVELOPMENT_VALIDATION_DAYS",
    "PreloadedV8Corpus",
    "REPRESENTATION_ANCHOR_SCHEMA",
    "REPRESENTATION_REPAIR_CAP_QUANTILE",
    "REPRESENTATION_REPAIR_STATISTICS_SCHEMA",
    "V8_ANCHOR_SCHEMA",
    "V8CandidateRepairStatistics",
    "V8CorpusError",
    "V8RepresentationRepairStatistics",
    "V8ResidualAnchor",
    "WORKLOAD_ZERO_TARGET_THRESHOLD",
    "attach_v8_representation_anchors",
    "development_date_inventory",
    "load_v8_residual_anchor",
    "materialize_v8_residual_anchor",
]
