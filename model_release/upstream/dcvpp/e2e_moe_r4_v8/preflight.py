"""Fail-closed development-only preflight for the R4-V8 lineage."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from dcvpp.e2e_moe_r4.hashing import sha256_file

from .config import ARCHITECTURE_VERSION, SPLITS, load_v8_config
from .corpus import (
    PreloadedV8Corpus,
    V8ResidualAnchor,
    development_date_inventory,
    load_v8_residual_anchor,
    materialize_v8_residual_anchor,
)


PREFLIGHT_SCHEMA = "dcvpp.r4.v8.development-preflight.v1"
SPLIT_SUMMARY_SCHEMA = "dcvpp.r4.v8.development-splits.v1"
LINEAGE_SUMMARY_SCHEMA = "dcvpp.r4.v8.initial-lineage.v1"
DEVELOPMENT_YEARS = (2022, 2023, 2024)


class V8PreflightError(RuntimeError):
    """V8 cannot prepare without weakening its temporal or lineage contract."""


@dataclass(frozen=True)
class V8PreflightResult:
    config: dict[str, Any]
    repository_root: Path
    resolved_paths: dict[str, Path]
    residual_anchor: V8ResidualAnchor
    split_summary: Mapping[str, Any]
    lineage_summary: Mapping[str, Any]
    summary_paths: Mapping[str, Path]
    anchor_materialized: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def assert_development_dates(values: Sequence[Any], *, label: str) -> tuple[str, ...]:
    """Accept only unique ISO dates from 2022 through 2024."""

    output: list[str] = []
    for raw in values:
        text = str(raw)[:10]
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise V8PreflightError(f"{label} contains an invalid date: {raw!r}") from exc
        if parsed.year not in DEVELOPMENT_YEARS:
            raise V8PreflightError(
                f"{label} attempted to access a non-development date: {text}"
            )
        output.append(text)
    if len(output) != len(set(output)):
        raise V8PreflightError(f"{label} contains duplicate operating dates")
    return tuple(output)


def _inventory_sections(inventory: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(inventory, Mapping):
        available = inventory.get("available")
        missing = inventory.get("missing", {})
    else:
        available = getattr(inventory, "available", None)
        missing = getattr(inventory, "missing", {})
    if not isinstance(available, Mapping) or not isinstance(missing, Mapping):
        raise V8PreflightError("development inventory is malformed")
    return available, missing


def _configured_development_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = config["protocol"]["splits"]
    output: dict[str, Any] = {}
    for name in ("train", "validation", "trainval_refit"):
        start, end, days, role = SPLITS[name]
        item = configured[name]
        if int(str(start)[:4]) not in DEVELOPMENT_YEARS or int(str(end)[:4]) not in DEVELOPMENT_YEARS:
            raise V8PreflightError(f"{name} extends outside 2022--2024")
        output[name] = {
            "start": start,
            "end": end,
            "days": days,
            "role": role,
            "configured": dict(item),
        }
    return output


def _split_summary(config: Mapping[str, Any], inventory: Any) -> dict[str, Any]:
    available, missing = _inventory_sections(inventory)
    train = assert_development_dates(tuple(available.get("train", ())), label="train inventory")
    validation = assert_development_dates(
        tuple(available.get("validation", ())), label="validation inventory"
    )
    train_missing = assert_development_dates(
        tuple(missing.get("train", ())), label="missing train inventory"
    )
    validation_missing = assert_development_dates(
        tuple(missing.get("validation", ())), label="missing validation inventory"
    )
    expected_train = int(config["protocol"]["splits"]["train"]["days"])
    expected_validation = int(config["protocol"]["splits"]["validation"]["days"])
    if len(train) != expected_train or len(validation) != expected_validation:
        raise V8PreflightError(
            "development inventory count mismatch: "
            f"train={len(train)}/{expected_train}, "
            f"validation={len(validation)}/{expected_validation}"
        )
    if set(train).intersection(validation):
        raise V8PreflightError("development train and validation inventories overlap")
    if set(train).intersection(train_missing) or set(validation).intersection(
        validation_missing
    ):
        raise V8PreflightError("available and unavailable development dates overlap")
    split_config = config["protocol"]["splits"]
    expected_missing = {
        name: (
            date.fromisoformat(str(split_config[name]["end"]))
            - date.fromisoformat(str(split_config[name]["start"]))
        ).days
        + 1
        - int(split_config[name]["days"])
        for name in ("train", "validation")
    }
    if len(train_missing) != expected_missing["train"] or len(
        validation_missing
    ) != expected_missing["validation"]:
        raise V8PreflightError(
            "development unavailable-date inventory does not close the calendar"
        )
    return {
        "schema": SPLIT_SUMMARY_SCHEMA,
        "architecture_version": ARCHITECTURE_VERSION,
        "method_revision": str(config["project"].get("method_revision", "v8_base")),
        "allowed_years": list(DEVELOPMENT_YEARS),
        "configured_splits": _configured_development_splits(config),
        "available": {
            "train": {"days": len(train), "start": train[0], "end": train[-1]},
            "validation": {
                "days": len(validation),
                "start": validation[0],
                "end": validation[-1],
            },
            "trainval": {"days": len(train) + len(validation)},
        },
        "unavailable": {
            "train": {"days": len(train_missing), "dates": list(train_missing)},
            "validation": {
                "days": len(validation_missing),
                "dates": list(validation_missing),
            },
        },
        "training_split": "train",
        "validation_split": "validation",
        "final_test_data_accessed": False,
        "locked_2025_opened": False,
    }


def _lineage_summary(
    *,
    config_path: Path,
    resolved: Mapping[str, Path],
    anchor: V8ResidualAnchor,
    split_summary: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_dates = assert_development_dates(anchor.train_dates, label="V8 residual anchor")
    if not anchor_dates or anchor_dates[0] < "2022-01-01" or anchor_dates[-1] > "2023-12-31":
        raise V8PreflightError("V8 residual anchor is not train-only 2022--2023")
    sources = {
        "config": config_path.resolve(),
        "representation_checkpoint": resolved["representation_checkpoint"],
        "source_residual_anchor": resolved["source_residual_anchor"],
        "residual_anchor": anchor.source_path.resolve(),
    }
    for name in ("repair_parent_checkpoint", "repair_parent_gate"):
        if name in resolved:
            sources[name] = resolved[name]
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise V8PreflightError(f"lineage source(s) are missing: {missing}")
    return {
        "schema": LINEAGE_SUMMARY_SCHEMA,
        "architecture_version": ARCHITECTURE_VERSION,
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "residual_anchor": {
            "shape": list(anchor.residuals.shape),
            "channel_scale_shape": list(anchor.channel_scale.shape),
            "fit_start": anchor_dates[0],
            "fit_end": anchor_dates[-1],
            "fit_days": len(anchor_dates),
            "metadata": dict(anchor.metadata),
        },
        "development_split_schema": split_summary["schema"],
        "optimizer_or_scheduler_inherited": False,
        "locked_2025_opened": False,
    }


def prepare_v8(
    config_path: str | Path,
    *,
    materialize_anchor: bool = True,
    write_summaries: bool = True,
    corpus: PreloadedV8Corpus | None = None,
    corpus_factory: Callable[[Mapping[str, Any], Path], PreloadedV8Corpus] = PreloadedV8Corpus,
    inventory_fn: Callable[[Mapping[str, Any], Path], Any] = development_date_inventory,
    anchor_materializer: Callable[[PreloadedV8Corpus, str | Path], V8ResidualAnchor] = materialize_v8_residual_anchor,
) -> V8PreflightResult:
    """Load strict YAML, prepare the train-only anchor and attest 2022--2024.

    The function never constructs, enumerates or opens a final-test data path.
    """

    source = Path(config_path).resolve()
    config, root, resolved = load_v8_config(source)
    output_root = resolved["output_root"]
    anchor_path = resolved["residual_anchor"]
    created = False
    if anchor_path.is_file():
        anchor = load_v8_residual_anchor(anchor_path)
    else:
        if not materialize_anchor:
            raise V8PreflightError(f"V8 residual anchor is missing: {anchor_path}")
        prepared_corpus = corpus or corpus_factory(config, root)
        anchor = anchor_materializer(prepared_corpus, anchor_path)
        created = True
    inventory = inventory_fn(config, root)
    split_summary = _split_summary(config, inventory)
    lineage_summary = _lineage_summary(
        config_path=source,
        resolved=resolved,
        anchor=anchor,
        split_summary=split_summary,
    )
    paths = {
        "split_summary": output_root / "preflight" / "split_summary.json",
        "lineage_summary": output_root / "preflight" / "lineage_summary.json",
        "preflight": output_root / "preflight" / "preflight.json",
    }
    if write_summaries:
        _atomic_json(paths["split_summary"], split_summary)
        _atomic_json(paths["lineage_summary"], lineage_summary)
        _atomic_json(
            paths["preflight"],
            {
                "schema": PREFLIGHT_SCHEMA,
                "status": "READY",
                "architecture_version": ARCHITECTURE_VERSION,
                "method_revision": str(
                    config["project"].get("method_revision", "v8_base")
                ),
                "created_at_utc": _utc_now(),
                "anchor_materialized": created,
                "residual_anchor": str(anchor.source_path),
                "split_summary": str(paths["split_summary"]),
                "lineage_summary": str(paths["lineage_summary"]),
                "training_started": False,
                "final_test_data_accessed": False,
                "locked_2025_opened": False,
            },
        )
    return V8PreflightResult(
        config=dict(config),
        repository_root=root,
        resolved_paths=dict(resolved),
        residual_anchor=anchor,
        split_summary=split_summary,
        lineage_summary=lineage_summary,
        summary_paths=paths,
        anchor_materialized=created,
    )


__all__ = [
    "DEVELOPMENT_YEARS",
    "LINEAGE_SUMMARY_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "SPLIT_SUMMARY_SCHEMA",
    "V8PreflightError",
    "V8PreflightResult",
    "assert_development_dates",
    "prepare_v8",
]
