"""Development-only 2022--2024 corpus and dynamic residual anchor.

This module is intentionally separate from the formal R4 loader.  The legacy
``PreloadedR4Corpus`` retains its 543/179 split, while this V7 development view
uses every complete 2022--2023 day for fitting and every complete 2024 day for
validation.  No helper in this module will construct or inspect a 2025 daily
payload path.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from dcvpp.e2e_moe_r4.constrained_runtime import load_physical_limits
from dcvpp.e2e_moe_r4.data import INTERVALS, LOCKED_TEST_YEAR, R4DataError
from dcvpp.e2e_moe_r4.hashing import sha256_file
from dcvpp.e2e_moe_r4.physical_twin import load_frozen_physical_twin_core
from dcvpp.e2e_moe_r4.training_data import (
    _collate,
    _day_to_tensors,
    _fit_statistics,
    _load_raw_day,
    _pin,
    _resolve_under,
    move_batch,
)


CHANNELS = 21
DEVELOPMENT_TRAIN_DAYS = 722
DEVELOPMENT_VALIDATION_DAYS = 364
DEVELOPMENT_TRAINVAL_DAYS = 1086
DYNAMIC_ANCHOR_SCHEMA = "dcvpp.r4.v7.development-residual-anchor.v1"


class V7DevelopmentCorpusError(R4DataError):
    """Raised before an invalid development split or daily path can be opened."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V7DevelopmentCorpusError(f"{label} must be a mapping")
    return value


def _date(value: date | str, label: str) -> date:
    try:
        result = value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError as exc:
        raise V7DevelopmentCorpusError(f"{label} must be an ISO operating date") from exc
    if result.year >= LOCKED_TEST_YEAR:
        raise V7DevelopmentCorpusError(
            f"locked {LOCKED_TEST_YEAR} operating dates are rejected before path access"
        )
    return result


def _date_range(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@dataclass(frozen=True)
class DevelopmentSplitContract:
    """Config-derived V7 development split, disjoint from formal R4 defaults."""

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    source_lineage: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DevelopmentSplitContract":
        protocol = _mapping(config.get("protocol"), "protocol")
        splits = _mapping(protocol.get("splits"), "protocol.splits")
        configured_train = _mapping(splits.get("train"), "protocol.splits.train")
        configured_validation = _mapping(
            splits.get("validation"), "protocol.splits.validation"
        )
        configured_refit = _mapping(
            splits.get("trainval_refit"), "protocol.splits.trainval_refit"
        )
        result = cls(
            train_start=_date(configured_train.get("start"), "train.start"),
            train_end=_date(configured_train.get("end"), "train.end"),
            validation_start=_date(
                configured_validation.get("start"), "validation.start"
            ),
            validation_end=_date(
                configured_validation.get("end"), "validation.end"
            ),
            source_lineage={
                "configured_train": dict(configured_train),
                "configured_validation": dict(configured_validation),
                "configured_trainval_refit": dict(configured_refit),
                "development_mapping": {
                    "train": "configured_train",
                    "validation": "configured_validation",
                    "trainval": "development_train_union_development_validation",
                },
            },
        )
        result.validate()
        if (
            int(configured_train.get("days", -1)) != DEVELOPMENT_TRAIN_DAYS
            or int(configured_validation.get("days", -1))
            != DEVELOPMENT_VALIDATION_DAYS
            or configured_refit.get("start") != configured_train.get("start")
            or configured_refit.get("end") != configured_validation.get("end")
            or int(configured_refit.get("days", -1)) != DEVELOPMENT_TRAINVAL_DAYS
        ):
            raise V7DevelopmentCorpusError(
                "configured development split counts must be 722/364/1086"
            )
        return result

    def validate(self) -> None:
        expected = (
            date(2022, 1, 1),
            date(2023, 12, 31),
            date(2024, 1, 1),
            date(2024, 12, 31),
        )
        actual = (
            self.train_start,
            self.train_end,
            self.validation_start,
            self.validation_end,
        )
        if actual != expected:
            raise V7DevelopmentCorpusError(
                "V7 development corpus must be train=2022--2023 and validation=2024"
            )
        if self.train_end >= self.validation_start:
            raise V7DevelopmentCorpusError("development train/validation dates overlap")

    def classify(self, operating_date: date | str) -> str:
        value = _date(operating_date, "operating_date")
        if self.train_start <= value <= self.train_end:
            return "train"
        if self.validation_start <= value <= self.validation_end:
            return "validation"
        raise V7DevelopmentCorpusError(
            f"operating date is outside the 2022--2024 development corpus: {value}"
        )

    def dates(self, split: str) -> tuple[date, ...]:
        name = str(split).lower()
        if name == "train":
            return tuple(_date_range(self.train_start, self.train_end))
        if name == "validation":
            return tuple(_date_range(self.validation_start, self.validation_end))
        if name == "trainval":
            return self.dates("train") + self.dates("validation")
        raise V7DevelopmentCorpusError("split must be train, validation, or trainval")

    def as_lineage(self) -> dict[str, Any]:
        return {
            "schema": "dcvpp.r4.v7-development-split.v1",
            "train": {
                "start": self.train_start.isoformat(),
                "end": self.train_end.isoformat(),
                "calendar_days": len(self.dates("train")),
                "required_complete_days": DEVELOPMENT_TRAIN_DAYS,
            },
            "validation": {
                "start": self.validation_start.isoformat(),
                "end": self.validation_end.isoformat(),
                "calendar_days": len(self.dates("validation")),
                "required_complete_days": DEVELOPMENT_VALIDATION_DAYS,
            },
            "trainval_required_complete_days": DEVELOPMENT_TRAINVAL_DAYS,
            "source_lineage": dict(self.source_lineage),
            "locked_2025_opened": False,
        }


@dataclass(frozen=True)
class DevelopmentDateInventory:
    contract: DevelopmentSplitContract
    available: Mapping[str, tuple[str, ...]]
    missing: Mapping[str, tuple[str, ...]]

    def validate_counts(self) -> None:
        counts = {name: len(self.available[name]) for name in ("train", "validation")}
        if counts != {
            "train": DEVELOPMENT_TRAIN_DAYS,
            "validation": DEVELOPMENT_VALIDATION_DAYS,
        }:
            raise V7DevelopmentCorpusError(
                "development corpus requires 722 train and 364 validation days, "
                f"got {counts['train']} and {counts['validation']}"
            )

    @property
    def trainval(self) -> tuple[str, ...]:
        return self.available["train"] + self.available["validation"]


def development_date_inventory(
    config: Mapping[str, Any], repository_root: str | Path
) -> DevelopmentDateInventory:
    """Inspect only expected 2022--2024 filenames and adjacent record metadata."""

    root = Path(repository_root).resolve()
    data = _mapping(config.get("data"), "data")
    bundle_root = _resolve_under(root, data.get("selection_bundle_root"), "bundle root")
    settlement_root = _resolve_under(
        root, data.get("official_settlement_root"), "settlement root"
    )
    contract = DevelopmentSplitContract.from_config(config)
    available: dict[str, list[str]] = {"train": [], "validation": []}
    missing: dict[str, list[str]] = {"train": [], "validation": []}
    for split in ("train", "validation"):
        for operating in contract.dates(split):
            # classify rejects 2025 before these paths are constructed.
            contract.classify(operating)
            day = operating.isoformat()
            paths = (
                bundle_root / f"{day}.npz",
                bundle_root / f"{day}.record.json",
                settlement_root / f"{day}.npz",
            )
            exists = tuple(path.is_file() for path in paths)
            if any(exists) and not all(exists):
                raise V7DevelopmentCorpusError(
                    f"development modalities are only partially available for {day}"
                )
            (available if all(exists) else missing)[split].append(day)
    result = DevelopmentDateInventory(
        contract=contract,
        available={key: tuple(value) for key, value in available.items()},
        missing={key: tuple(value) for key, value in missing.items()},
    )
    result.validate_counts()
    return result


class PreloadedV7DevelopmentCorpus:
    """Host-memory 722/364 corpus with normalization fitted on 2022--2023."""

    def __init__(self, config: Mapping[str, Any], repository_root: str | Path) -> None:
        root = Path(repository_root).resolve()
        data = _mapping(config.get("data"), "data")
        verify_daily_hashes = bool(data.get("verify_daily_hashes", True))
        inventory = development_date_inventory(config, root)
        self.split_contract = inventory.contract
        self.split_lineage = self.split_contract.as_lineage() | {
            "available_train_days": len(inventory.available["train"]),
            "available_validation_days": len(inventory.available["validation"]),
            "missing_train_dates": inventory.missing["train"],
            "missing_validation_dates": inventory.missing["validation"],
        }
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
            root,
            data.get("physical_calibration_manifest"),
            "physical calibration manifest",
        )
        self.physical_core = load_frozen_physical_twin_core(
            artifact_path=physical_path,
            surrogate_path=surrogate_path,
            calibration_manifest_path=manifest_path,
        )
        self.physical_lineage = dict(self.physical_core.artifact_metadata)

        raw: dict[str, Any] = {}
        capacity = np.asarray(self.physical.capacity_work_units)
        allowed_window = (
            self.split_contract.train_start,
            self.split_contract.validation_end,
        )
        for day in inventory.trainval:
            # The inventory has already rejected every date outside 2022--2024.
            raw[day] = _load_raw_day(
                bundle_root / f"{day}.npz",
                settlement_root / f"{day}.npz",
                capacity,
                allowed_window=allowed_window,
                verify_hashes=verify_daily_hashes,
            )
        train_raw = [raw[day] for day in inventory.available["train"]]
        self.statistics = _fit_statistics(
            train_raw, self.physical, self.physical_core
        )

        records: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
        }
        history: list[Any] = []
        for day in inventory.trainval:
            item = raw[day]
            split = self.split_contract.classify(day)
            record = _day_to_tensors(
                item,
                self.statistics,
                self.physical,
                self.physical_core,
                history,
            )
            record["provenance"]["physical_sha256"] = self.physical_sha256
            record["provenance"]["physical_twin_lineage"] = dict(
                self.physical_lineage
            )
            record["provenance"]["development_split"] = split
            record["provenance"]["development_split_lineage"] = self.split_lineage
            records[split].append(record)
            history.append(item)
        if torch.cuda.is_available():
            records = _pin(records)
        self._records: dict[str, list[dict[str, Any]]] = {
            "train": records["train"],
            "validation": records["validation"],
            "trainval": [*records["train"], *records["validation"]],
        }
        baseline = torch.stack(
            [
                item["rollout_targets"]["baseline_realized_cost"]
                for item in self._records["train"]
            ]
        )
        self.fixed_training_cost_scale = float(
            baseline.abs().median().clamp_min(1.0).item()
        )

    def records(self, split: str) -> Sequence[Mapping[str, Any]]:
        name = str(split).lower()
        if name not in self._records:
            raise V7DevelopmentCorpusError(
                "development split must be train, validation, or trainval"
            )
        return self._records[name]

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
        size = int(batch_size)
        if size <= 0:
            raise V7DevelopmentCorpusError("batch_size must be positive")
        for start in range(0, len(records), size):
            batch = _collate(records[start : start + size])
            batch["num_days"] = len(batch["operating_date"])
            batch["operating_dates"] = tuple(batch["operating_date"])
            batch["origin"] = torch.full(
                (len(batch["operating_date"]),), int(origin), dtype=torch.long
            )
            yield move_batch(batch, device)


@dataclass(frozen=True)
class DynamicResidualAnchor:
    """Train-derived 722-day issue-center and residual distribution."""

    issue_centers: Tensor
    residuals: Tensor
    channel_scale: Tensor
    pue_center: Tensor
    train_dates: tuple[str, ...]
    split_lineage: Mapping[str, Any]
    source_path: Path | None = None

    def deterministic_indices(
        self,
        sample_count: int,
        seed: int,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        count = int(sample_count)
        if not 1 <= count <= len(self.train_dates):
            raise V7DevelopmentCorpusError(
                f"sample_count must lie in [1,{len(self.train_dates)}]"
            )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        indices = torch.randperm(len(self.train_dates), generator=generator)[:count]
        return indices if device is None else indices.to(device)

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema": DYNAMIC_ANCHOR_SCHEMA,
            "name": "v7_development_training_residual_sampling",
            "fit_split": "development_train_2022-01-01_to_2023-12-31_only",
            "train_day_count": len(self.train_dates),
            "train_dates": self.train_dates,
            "split_lineage": dict(self.split_lineage),
            "issue_center_contract": "DA+RT+(DA-RT)+workload+PV+median_train_PUE",
            "pue_center_method": "per_interval_per_zone_train_median",
            "trajectory_sampling": "whole_day_joint_residual_without_replacement",
            "normalization": "train_residual_channel_median_absolute_scale",
            "locked_2025_opened": False,
            "pue_center": self.pue_center,
            "issue_centers": self.issue_centers,
            "residuals": self.residuals,
            "scale": self.channel_scale,
        }


def _issue_center(record: Mapping[str, Any], pue_center: Tensor) -> Tensor:
    market = torch.as_tensor(record["market_raw"], dtype=torch.float32).cpu()
    workload = torch.as_tensor(
        record["issue_workload_raw"], dtype=torch.float32
    ).cpu()
    pv = torch.as_tensor(record["issue_pv_raw"], dtype=torch.float32).cpu()
    if market.shape != (INTERVALS, 16):
        raise V7DevelopmentCorpusError("market_raw must be [96,16]")
    if workload.shape != (INTERVALS,) or pv.shape != (INTERVALS, 4):
        raise V7DevelopmentCorpusError("issue workload/PV shapes are invalid")
    da = market[:, 0:4]
    rt = market[:, 4:8]
    return torch.cat((da, rt, da - rt, workload[:, None], pv, pue_center), dim=-1)


def materialize_dynamic_residual_anchor(
    corpus: Any,
    output_path: str | Path | None = None,
) -> DynamicResidualAnchor:
    """Build the 722-day anchor; optionally save one atomic lineage artifact."""

    train = tuple(corpus.records("train"))
    if len(train) != DEVELOPMENT_TRAIN_DAYS:
        raise V7DevelopmentCorpusError(
            f"dynamic residual anchor requires {DEVELOPMENT_TRAIN_DAYS} train records"
        )
    dates = tuple(str(record.get("operating_date", "")) for record in train)
    if len(set(dates)) != DEVELOPMENT_TRAIN_DAYS or any(
        not day.startswith(("2022-", "2023-")) for day in dates
    ):
        raise V7DevelopmentCorpusError(
            "dynamic residual anchor requires 722 unique 2022--2023 dates"
        )
    targets = torch.stack(
        [
            torch.as_tensor(record["scenario_target_zonal"], dtype=torch.float32).cpu()
            for record in train
        ]
    )
    if targets.shape != (DEVELOPMENT_TRAIN_DAYS, INTERVALS, CHANNELS):
        raise V7DevelopmentCorpusError("train targets must stack to [722,96,21]")
    if not bool(torch.isfinite(targets).all()):
        raise V7DevelopmentCorpusError("train targets contain non-finite values")
    pue_center = targets[..., 17:21].median(dim=0).values
    centers = torch.stack([_issue_center(record, pue_center) for record in train])
    residuals = targets - centers
    scale = residuals.abs().flatten(0, 1).median(dim=0).values
    scale = torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))
    lineage = getattr(corpus, "split_lineage", None)
    if not isinstance(lineage, Mapping):
        raise V7DevelopmentCorpusError("corpus must expose development split_lineage")
    anchor = DynamicResidualAnchor(
        issue_centers=centers,
        residuals=residuals,
        channel_scale=scale,
        pue_center=pue_center,
        train_dates=dates,
        split_lineage=dict(lineage),
        source_path=(None if output_path is None else Path(output_path).resolve()),
    )
    if output_path is not None:
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(anchor.as_payload(), temporary)
        os.replace(temporary, destination)
    return anchor


def load_dynamic_residual_anchor(path: str | Path) -> DynamicResidualAnchor:
    """Load one saved 722-day development anchor without accepting 2025 lineage."""

    source = Path(path).resolve()
    if not source.is_file():
        raise V7DevelopmentCorpusError(f"dynamic residual anchor is missing: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != DYNAMIC_ANCHOR_SCHEMA:
        raise V7DevelopmentCorpusError("unexpected dynamic residual anchor schema")
    dates = tuple(str(value) for value in payload.get("train_dates", ()))
    centers = torch.as_tensor(payload.get("issue_centers"), dtype=torch.float32)
    residuals = torch.as_tensor(payload.get("residuals"), dtype=torch.float32)
    scale = torch.as_tensor(payload.get("scale"), dtype=torch.float32)
    pue = torch.as_tensor(payload.get("pue_center"), dtype=torch.float32)
    lineage = payload.get("split_lineage")
    if not isinstance(lineage, Mapping):
        raise V7DevelopmentCorpusError("dynamic anchor has no split lineage")
    if payload.get("locked_2025_opened") is not False or lineage.get(
        "locked_2025_opened"
    ) is not False:
        raise V7DevelopmentCorpusError("dynamic anchor does not prove 2025 stayed closed")
    if (
        centers.shape != (DEVELOPMENT_TRAIN_DAYS, INTERVALS, CHANNELS)
        or residuals.shape != (DEVELOPMENT_TRAIN_DAYS, INTERVALS, CHANNELS)
        or scale.shape != (CHANNELS,)
        or pue.shape != (INTERVALS, 4)
    ):
        raise V7DevelopmentCorpusError("dynamic residual anchor tensor shapes are invalid")
    if len(dates) != DEVELOPMENT_TRAIN_DAYS or len(set(dates)) != len(dates):
        raise V7DevelopmentCorpusError("dynamic anchor must contain 722 unique dates")
    if any(not value.startswith(("2022-", "2023-")) for value in dates):
        raise V7DevelopmentCorpusError("dynamic anchor contains a non-development date")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (centers, residuals, scale, pue)
    ) or bool((scale <= 0).any()):
        raise V7DevelopmentCorpusError("dynamic residual anchor contains invalid values")
    return DynamicResidualAnchor(
        issue_centers=centers,
        residuals=residuals,
        channel_scale=scale,
        pue_center=pue,
        train_dates=dates,
        split_lineage=dict(lineage),
        source_path=source,
    )


__all__ = [
    "DEVELOPMENT_TRAIN_DAYS",
    "DEVELOPMENT_TRAINVAL_DAYS",
    "DEVELOPMENT_VALIDATION_DAYS",
    "DYNAMIC_ANCHOR_SCHEMA",
    "DevelopmentDateInventory",
    "DevelopmentSplitContract",
    "DynamicResidualAnchor",
    "PreloadedV7DevelopmentCorpus",
    "V7DevelopmentCorpusError",
    "development_date_inventory",
    "load_dynamic_residual_anchor",
    "materialize_dynamic_residual_anchor",
]
