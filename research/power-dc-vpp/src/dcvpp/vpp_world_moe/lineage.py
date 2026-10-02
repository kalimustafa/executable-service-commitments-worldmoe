"""Small, append-safe lineage artifacts for the WorldMoE data adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


NORMALIZATION_SCHEMA = "dcvpp.vpp_world_moe.train-normalization.v1"
DATA_LINEAGE_SCHEMA = "dcvpp.vpp_world_moe.data-lineage.v1"


class WorldDataLineageError(ValueError):
    """Raised when a data artifact would overstate its evidence boundary."""


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a deterministic JSON artifact without exposing a partial file."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def validate_train_only_normalization(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != NORMALIZATION_SCHEMA:
        raise WorldDataLineageError("unexpected WorldMoE normalization schema")
    if (
        payload.get("fit_split") != "train_2022_2023_only"
        or int(payload.get("fit_days", -1)) != 722
        or payload.get("used_validation_statistics") is not False
        or payload.get("opened_final_test_2025") is not False
    ):
        raise WorldDataLineageError(
            "normalization must use only the 722 complete 2022--2023 days"
        )


def validate_data_lineage(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != DATA_LINEAGE_SCHEMA:
        raise WorldDataLineageError("unexpected WorldMoE data-lineage schema")
    counts = payload.get("complete_days")
    if not isinstance(counts, Mapping) or {
        str(key): int(value) for key, value in counts.items()
    } != {"train": 722, "validation": 364, "trainval_refit": 1086}:
        raise WorldDataLineageError("data lineage does not preserve 722/364/1086")
    final = payload.get("final_test_2025")
    if not isinstance(final, Mapping) or (
        final.get("opened") is not False or final.get("status") != "SEALED_ONE_SHOT"
    ):
        raise WorldDataLineageError("2025 must remain sealed in development lineage")
    migration = payload.get("migration")
    if not isinstance(migration, Mapping) or (
        migration.get("available") is not True
        or migration.get("enabled_routes")
        != [["WEST", "N.Y.C."], ["N.Y.C.", "WEST"]]
        or migration.get("capacity_source")
        != "ub_nysernet_published_100gbps_upper_bound"
        or migration.get("latency_source")
        != "ripe_atlas_direct_anchor_mesh_ipv4"
        or migration.get("action_mask") != "ripe_asof_origin_gated"
        or migration.get("capacity_evidence_role")
        != "operational_proxy_sensitivity_not_execution_log"
        or migration.get("future_results_allowed") is not False
    ):
        raise WorldDataLineageError("migration evidence lineage is incompatible")


def materialize_lineage_pair(
    *,
    normalization_path: str | Path,
    normalization: Mapping[str, Any],
    lineage_path: str | Path,
    lineage: Mapping[str, Any],
) -> tuple[Path, Path]:
    validate_train_only_normalization(normalization)
    validate_data_lineage(lineage)
    return (
        atomic_write_json(normalization_path, normalization),
        atomic_write_json(lineage_path, lineage),
    )


__all__ = [
    "DATA_LINEAGE_SCHEMA",
    "NORMALIZATION_SCHEMA",
    "WorldDataLineageError",
    "atomic_write_json",
    "materialize_lineage_pair",
    "validate_data_lineage",
    "validate_train_only_normalization",
]
