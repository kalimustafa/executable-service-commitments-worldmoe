"""Fail-closed, lineage-preserving V7 to V8 checkpoint upgrade.

Only shape-identical shared causal encoders are copied.  Every V8-only tensor
must belong to the explicit new-key registry, and every retired V7 tensor must
belong to the explicit retired-prefix registry.  Optimizer, scheduler, scaler
and RNG training state are deliberately never migrated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from dcvpp.e2e_moe_r4.hashing import sha256_file
from dcvpp.e2e_moe_r4_v7.config import ARCHITECTURE_VERSION as V7_ARCHITECTURE_VERSION

from .model import ARCHITECTURE_VERSION as V8_ARCHITECTURE_VERSION


UPGRADE_SCHEMA = "dcvpp.r4.v8.checkpoint-upgrade.v1"
REGISTERED_NEW_PREFIXES = (
    "modality_projections.",
    "trifusion.",
    "representation.",
    "distribution.",
    "decision_bridge.",
    "policy.",
)
REGISTERED_RETIRED_V7_PREFIXES = (
    "cross_fusion.",
    "moe.",
    "forecast_decoder.",
    "distribution.",
    "policy.",
)
TRAINING_STATE_KEYS = (
    "optimizer",
    "optimizer_state",
    "optimizers",
    "scheduler",
    "scheduler_state",
    "scaler",
    "scaler_state",
    "grad_scaler",
    "rng_state",
)


class CheckpointUpgradeError(RuntimeError):
    """Raised when a checkpoint cannot be upgraded without guessing."""


@dataclass(frozen=True)
class CheckpointUpgradeReport:
    schema: str
    source_architecture: str
    target_architecture: str
    copied_keys: tuple[str, ...]
    initialized_new_keys: tuple[str, ...]
    retired_source_keys: tuple[str, ...]
    discarded_training_state_keys: tuple[str, ...]
    optimizer_state_migrated: bool
    source_checkpoint: str | None = None
    source_checkpoint_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_mapping(payload: Mapping[str, Any]) -> Mapping[str, Tensor]:
    candidate = payload.get("model_state", payload.get("state_dict"))
    if not isinstance(candidate, Mapping):
        raise CheckpointUpgradeError("source checkpoint lacks model_state/state_dict")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in candidate.items()):
        raise CheckpointUpgradeError("source model state must map string keys to tensors")
    return candidate


def _has_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _source_architecture(payload: Mapping[str, Any]) -> str:
    value = payload.get("architecture_version")
    if value is None and isinstance(payload.get("model"), Mapping):
        value = payload["model"].get("architecture_version")
    if value is None:
        value = payload.get("representation_architecture_version")
    return str(value or "")


def upgrade_checkpoint_payload(
    source_payload: Mapping[str, Any],
    target_model: nn.Module,
    *,
    source_checkpoint: str | Path | None = None,
    source_checkpoint_sha256: str | None = None,
    registered_new_prefixes: Sequence[str] = REGISTERED_NEW_PREFIXES,
    registered_retired_prefixes: Sequence[str] = REGISTERED_RETIRED_V7_PREFIXES,
) -> tuple[dict[str, Any], CheckpointUpgradeReport]:
    """Create a strict V8 initialization payload without optimizer migration."""

    source_architecture = _source_architecture(source_payload)
    if source_architecture != V7_ARCHITECTURE_VERSION:
        raise CheckpointUpgradeError(
            "source architecture is not the registered V7 lineage: "
            f"{source_architecture!r}"
        )
    target_architecture = str(getattr(target_model, "architecture_version", ""))
    if target_architecture != V8_ARCHITECTURE_VERSION:
        raise CheckpointUpgradeError("target model is not the registered V8 architecture")

    source_state = _state_mapping(source_payload)
    target_state = target_model.state_dict()
    copied: list[str] = []
    initialized: list[str] = []
    retired: list[str] = []
    unregistered_missing: list[str] = []
    unregistered_source: list[str] = []
    upgraded: dict[str, Tensor] = {}

    for name, target_value in target_state.items():
        source_value = source_state.get(name)
        is_new = _has_prefix(name, registered_new_prefixes)
        if source_value is None:
            if not is_new:
                unregistered_missing.append(name)
                continue
            upgraded[name] = target_value.detach().clone()
            initialized.append(name)
            continue
        if is_new:
            # A coincidental name match in a retired V7 branch must not silently
            # initialize a semantically different V8 tensor.
            upgraded[name] = target_value.detach().clone()
            initialized.append(name)
            continue
        if source_value.shape != target_value.shape or source_value.dtype != target_value.dtype:
            raise CheckpointUpgradeError(
                f"shared checkpoint tensor {name!r} changed shape or dtype: "
                f"{tuple(source_value.shape)}/{source_value.dtype} -> "
                f"{tuple(target_value.shape)}/{target_value.dtype}"
            )
        upgraded[name] = source_value.detach().clone()
        copied.append(name)

    for name in source_state:
        if name in copied:
            continue
        if _has_prefix(name, registered_retired_prefixes):
            retired.append(name)
        elif name in target_state and _has_prefix(name, registered_new_prefixes):
            retired.append(name)
        else:
            unregistered_source.append(name)

    if unregistered_missing:
        raise CheckpointUpgradeError(
            "target contains unregistered new keys: " + ", ".join(sorted(unregistered_missing))
        )
    if unregistered_source:
        raise CheckpointUpgradeError(
            "source contains unregistered retired/extra keys: "
            + ", ".join(sorted(unregistered_source))
        )
    if set(upgraded) != set(target_state):
        raise CheckpointUpgradeError("upgraded state does not cover the complete V8 model")

    discarded = tuple(
        key for key in TRAINING_STATE_KEYS if source_payload.get(key) is not None
    )
    report = CheckpointUpgradeReport(
        schema=UPGRADE_SCHEMA,
        source_architecture=source_architecture,
        target_architecture=target_architecture,
        copied_keys=tuple(sorted(copied)),
        initialized_new_keys=tuple(sorted(initialized)),
        retired_source_keys=tuple(sorted(retired)),
        discarded_training_state_keys=discarded,
        optimizer_state_migrated=False,
        source_checkpoint=(
            None if source_checkpoint is None else str(Path(source_checkpoint).resolve())
        ),
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    payload = {
        "schema": UPGRADE_SCHEMA,
        "architecture_version": V8_ARCHITECTURE_VERSION,
        "parent_architecture_version": source_architecture,
        "phase": "v8_initialization",
        "model_state": upgraded,
        "upgrade_report": report.as_dict(),
        "optimizer_state_migrated": False,
        "parent_checkpoint": report.source_checkpoint,
        "parent_checkpoint_sha256": report.source_checkpoint_sha256,
        "locked_2025_opened": False,
    }
    if any(key in payload for key in TRAINING_STATE_KEYS):
        raise AssertionError("V8 upgrade payload unexpectedly contains training state")
    return payload, report


def upgrade_v7_checkpoint(
    source: str | Path,
    destination: str | Path,
    target_model: nn.Module,
    *,
    registered_new_prefixes: Sequence[str] = REGISTERED_NEW_PREFIXES,
    registered_retired_prefixes: Sequence[str] = REGISTERED_RETIRED_V7_PREFIXES,
) -> CheckpointUpgradeReport:
    """Write a new V8 checkpoint; never overwrite either source or destination."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file():
        raise CheckpointUpgradeError(f"source checkpoint is missing: {source_path}")
    if source_path == destination_path:
        raise CheckpointUpgradeError("checkpoint upgrade cannot overwrite its source")
    if destination_path.exists():
        raise CheckpointUpgradeError(f"destination already exists: {destination_path}")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointUpgradeError("source checkpoint payload must be a mapping")
    upgraded, report = upgrade_checkpoint_payload(
        payload,
        target_model,
        source_checkpoint=source_path,
        source_checkpoint_sha256=sha256_file(source_path),
        registered_new_prefixes=registered_new_prefixes,
        registered_retired_prefixes=registered_retired_prefixes,
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    try:
        torch.save(upgraded, temporary)
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report


__all__ = [
    "CheckpointUpgradeError",
    "CheckpointUpgradeReport",
    "REGISTERED_NEW_PREFIXES",
    "REGISTERED_RETIRED_V7_PREFIXES",
    "TRAINING_STATE_KEYS",
    "UPGRADE_SCHEMA",
    "upgrade_checkpoint_payload",
    "upgrade_v7_checkpoint",
]
