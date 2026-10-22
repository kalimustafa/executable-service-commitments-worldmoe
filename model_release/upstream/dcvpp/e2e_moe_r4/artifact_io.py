"""Atomic and compact artifact I/O for the independent formal R4 lineage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import os
import tempfile

import pandas as pd
import torch
import yaml

from .hashing import sha256_file


TERMINAL_STATUSES = frozenset({"COMPLETE", "PARTIAL", "BLOCKED", "FAILED"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        Path(path),
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8"),
    )


def atomic_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        Path(path),
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True).encode("utf-8"),
    )


def atomic_text(path: str | Path, text: str) -> None:
    _atomic_bytes(Path(path), text.encode("utf-8"))


def atomic_parquet(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        pd.DataFrame.from_records(records).to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def validate_state(state: Mapping[str, Any]) -> None:
    required = {
        "experiment_id",
        "architecture_version",
        "training_semantics_revision",
        "method",
        "seed",
        "phase",
        "split",
        "predecessor_checkpoint",
        "capability_gate_status",
        "status",
        "resume_command",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"state.json is missing required fields: {missing}")
    if state["status"] not in TERMINAL_STATUSES:
        raise ValueError(f"invalid R4 status: {state['status']!r}")
    if state["status"] == "COMPLETE" and state["capability_gate_status"] != "PASSED":
        raise ValueError("R4 phase cannot be COMPLETE before its capability gate passes")
    if not str(state["resume_command"]).strip():
        raise ValueError("state.json must contain an exact resume command")


def write_state(path: str | Path, state: Mapping[str, Any]) -> None:
    validate_state(state)
    payload = dict(state)
    payload["updated_at_utc"] = utc_now()
    atomic_json(path, payload)


def build_compact_provenance(
    *,
    governing_config: str | Path,
    split_manifest: str | Path,
    canonical_data_index: str | Path,
    active_source_files: Sequence[str | Path],
    selected_checkpoints: Sequence[str | Path] = (),
    final_metrics: Sequence[str | Path] = (),
    final_reports: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Hash exactly the registered formal artifacts, never every daily bundle."""

    groups = {
        "governing_config": [governing_config],
        "split_manifest": [split_manifest],
        "canonical_data_index": [canonical_data_index],
        "active_r4_source_manifest": list(active_source_files),
        "selected_best_checkpoints": list(selected_checkpoints),
        "final_metrics": list(final_metrics),
        "final_reports": list(final_reports),
    }
    return {
        "schema_version": "dcvpp.e2e_moe_hybrid_r4.provenance.v1",
        "created_at_utc": utc_now(),
        "hash_algorithm": "SHA-256",
        "daily_bundles_hashed": False,
        "groups": {
            name: [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(path),
                    "bytes": Path(path).stat().st_size,
                }
                for path in paths
            ]
            for name, paths in groups.items()
        },
    }


def _atomic_bytes(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "TERMINAL_STATUSES",
    "atomic_json",
    "atomic_parquet",
    "atomic_text",
    "atomic_torch_save",
    "atomic_yaml",
    "build_compact_provenance",
    "read_json",
    "sha256_file",
    "utc_now",
    "validate_state",
    "write_state",
]
