"""Zero-copy v4 overlay recovery for formal DART neural checkpoints.

The immutable ``paper-dart-forecast-v3`` predecessor stopped before any OOS
unit was materialized because an arbitrary non-anchor CUDA LSTM epoch could not
be reconstructed bitwise.  The v12 amendment restricts final checkpoint
candidates to exact persisted five-epoch anchors.

This module does not mutate v3.  It validates an externally frozen manifest of
all positive v3 anchors, imports only each seed's highest anchor into the v4
resume namespace, and exposes a narrow compatibility context:

* provisional and final selection use exact common five-epoch anchors;
* final selection uses the union of hash-verified v3 anchors and new v4 anchors;
* selected weights are copied directly from the chosen source/local anchor;
* no train/replay step occurs during selected-checkpoint materialization.

The frozen trainer bytes remain unchanged, so v3 fingerprints and failure
evidence stay reproducible.  The v4 import changes only the new resume
fingerprint and records the exact scientific-contract digest linking the runs.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


FROZEN_DART_CHECKPOINT_TRAINING_SHA256 = (
    "8e43fc4f3c61be2aca7f8ec67ffc821a909aef50b23626955ca9caa2ae2d10a7"
)
COMPATIBILITY_SCHEMA_VERSION = "dcvpp.dart_anchor_only_overlay.v1"
EXPECTED_V3_POSITIVE_ANCHOR_COUNT = 1266
EXPECTED_V3_POSITIVE_ANCHOR_BYTES = 6_726_342_704
EXPECTED_V3_ANCHOR_RECORD_CORE_SHA256 = (
    "43030db511b7aa9591216753e3b76d2c73e7937f99b7d0183fd88c94c5d76a36"
)


class DARTCheckpointReplayCompatibilityError(RuntimeError):
    """Raised when the immutable predecessor or overlay contract differs."""


_CONTEXT_LOCK = threading.RLock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def archive_v4_completed_epoch(
    *,
    latest_checkpoint_path: Path,
    epoch_archive_root: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Persist one completed epoch checkpoint as immutable full-state evidence."""

    latest = Path(latest_checkpoint_path).resolve()
    if not latest.is_file():
        raise DARTCheckpointReplayCompatibilityError(
            f"v4 latest checkpoint is absent: {latest}"
        )
    import torch

    payload = torch.load(latest, map_location="cpu", weights_only=False)
    epoch = int(payload.get("epoch", -1))
    required = {
        "model_state",
        "optimizer_state",
        "gradient_scaler_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "dataloader_generator_state",
        "dataloader_sampler_state",
        "epoch_ledger",
        "epoch_ledger_sha256",
        "training_health",
        "training_health_sha256",
        "resume_fingerprint",
    }
    if epoch < 1 or not required.issubset(payload):
        raise DARTCheckpointReplayCompatibilityError(
            "v4 completed epoch lacks immutable full-state fields"
        )
    if len(payload["epoch_ledger"]) != epoch:
        raise DARTCheckpointReplayCompatibilityError(
            "v4 completed epoch and validation ledger length differ"
        )
    if _sha256_json(payload["epoch_ledger"]) != payload["epoch_ledger_sha256"]:
        raise DARTCheckpointReplayCompatibilityError(
            "v4 completed epoch validation ledger hash changed"
        )
    if _sha256_json(payload["training_health"]) != payload["training_health_sha256"]:
        raise DARTCheckpointReplayCompatibilityError(
            "v4 completed epoch health hash changed"
        )
    archive = Path(epoch_archive_root).resolve()
    archive_path = archive / f"epoch_{epoch:04d}.pt"
    source_sha = _sha256_file(latest)
    if archive_path.is_file():
        if _sha256_file(archive_path) != source_sha:
            raise DARTCheckpointReplayCompatibilityError(
                "immutable v4 epoch archive already exists with different bytes"
            )
    else:
        _atomic_copy(latest, archive_path)
    if _sha256_file(archive_path) != source_sha:
        raise DARTCheckpointReplayCompatibilityError(
            "v4 epoch archive copy hash mismatch"
        )
    evidence_path = archive_path
    if repository_root is not None:
        repo = Path(repository_root).resolve()
        if repo not in archive_path.parents:
            raise DARTCheckpointReplayCompatibilityError(
                "v4 epoch archive escapes repository"
            )
        evidence_path = archive_path.relative_to(repo)
    return {
        "epoch": epoch,
        "path": evidence_path.as_posix(),
        "sha256": source_sha,
        "byte_count": archive_path.stat().st_size,
        "resume_fingerprint": payload["resume_fingerprint"],
        "epoch_ledger_sha256": payload["epoch_ledger_sha256"],
        "training_health_sha256": payload["training_health_sha256"],
        "model_state_digest": _state_digest(payload["model_state"]),
        "optimizer_state_digest": _state_digest(payload["optimizer_state"]),
        "scaler_state_digest": _state_digest(payload["gradient_scaler_state"]),
        "rng_and_dataloader_state_digest": _state_digest(
            {
                key: payload[key]
                for key in (
                    "python_rng_state",
                    "numpy_rng_state",
                    "torch_rng_state",
                    "cuda_rng_state",
                    "dataloader_generator_state",
                    "dataloader_sampler_state",
                )
            }
        ),
        "formal_selection_candidate": epoch % 5 == 0,
        "non_five_epoch_role": (
            None if epoch % 5 == 0 else "AUDIT_AND_RESUME_ONLY"
        ),
    }


def update_v4_epoch_archive_manifest(
    *,
    manifest_path: Path,
    track_identity: Mapping[str, Any],
    archive_record: Mapping[str, Any],
    predecessor_epoch: int,
) -> dict[str, Any]:
    """Append one unique immutable epoch record and self-hash the track ledger."""

    path = Path(manifest_path).resolve()
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        core = {key: value for key, value in existing.items() if key != "manifest_sha256"}
        if existing.get("manifest_sha256") != _sha256_json(core):
            raise DARTCheckpointReplayCompatibilityError(
                "v4 epoch archive manifest self-hash changed"
            )
        if existing.get("track_identity") != dict(track_identity):
            raise DARTCheckpointReplayCompatibilityError(
                "v4 epoch archive track identity changed"
            )
        if int(existing.get("predecessor_imported_epoch", -1)) != int(
            predecessor_epoch
        ):
            raise DARTCheckpointReplayCompatibilityError(
                "v4 epoch archive predecessor epoch changed"
            )
        records = list(existing.get("records", ()))
    else:
        records = []
    record = dict(archive_record)
    duplicates = [item for item in records if int(item["epoch"]) == int(record["epoch"])]
    if duplicates:
        if duplicates != [record]:
            raise DARTCheckpointReplayCompatibilityError(
                "v4 epoch archive manifest epoch differs"
            )
    else:
        records.append(record)
    records.sort(key=lambda value: int(value["epoch"]))
    epochs = [int(value["epoch"]) for value in records]
    if (
        epochs[0] != int(predecessor_epoch) + 1
        or epochs != list(range(epochs[0], epochs[-1] + 1))
    ):
        raise DARTCheckpointReplayCompatibilityError(
            "v4 epoch archive manifest is not contiguous"
        )
    core = {
        "schema_version": "dcvpp.dart_v4_full_epoch_archive.v1",
        "status": "COMPLETE_THROUGH_CURRENT_EPOCH",
        "track_identity": dict(track_identity),
        "predecessor_imported_epoch": int(predecessor_epoch),
        "records": records,
        "record_count": len(records),
        "first_epoch": epochs[0],
        "latest_epoch": epochs[-1],
        "every_completed_epoch_full_state_persisted": True,
        "formal_selection_candidates_divisible_by_five_only": True,
    }
    manifest = {**core, "manifest_sha256": _sha256_json(core)}
    _atomic_json(path, manifest)
    return manifest


def initialize_v4_epoch_archive_manifest(
    *,
    manifest_path: Path,
    track_identity: Mapping[str, Any],
    predecessor_epoch: int,
) -> dict[str, Any]:
    """Create the zero-new-epoch ledger for one imported v4 seed track."""

    path = Path(manifest_path).resolve()
    core = {
        "schema_version": "dcvpp.dart_v4_full_epoch_archive.v1",
        "status": "COMPLETE_THROUGH_CURRENT_EPOCH",
        "track_identity": dict(track_identity),
        "predecessor_imported_epoch": int(predecessor_epoch),
        "records": [],
        "record_count": 0,
        "first_epoch": None,
        "latest_epoch": None,
        "every_completed_epoch_full_state_persisted": True,
        "formal_selection_candidates_divisible_by_five_only": True,
    }
    manifest = {**core, "manifest_sha256": _sha256_json(core)}
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise DARTCheckpointReplayCompatibilityError(
                "existing empty v4 epoch archive manifest differs"
            )
    else:
        _atomic_json(path, manifest)
    return manifest


def install_v4_full_epoch_archive(
    module: Any,
    *,
    imported_epoch_by_latest_path: Mapping[str, int],
    repository_root: Path | None = None,
    v4_stage_root: Path | None = None,
) -> Any:
    """Return a wrapper that archives every epoch atomically after core writes latest."""

    original = module._atomic_torch
    stage = Path(v4_stage_root).resolve() if v4_stage_root is not None else None
    imported = {
        Path(key).resolve().as_posix(): int(value)
        for key, value in imported_epoch_by_latest_path.items()
    }

    def atomic(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
        destination = Path(path).resolve()
        new_v4_epoch = (
            destination.name == "latest.pt"
            and int(payload.get("epoch", 0)) > 0
            and payload.get("namespace") == "formal_dart_checkpoint_training"
            and "overlay_import_provenance" not in payload
            and (stage is None or stage in destination.parents)
        )
        persisted = dict(payload)
        if new_v4_epoch:
            persisted["dataloader_sampler_state"] = {
                "sampler_class": "RandomSampler",
                "checkpoint_boundary": "completed_epoch",
                "persistent_in_epoch_cursor": None,
                "generator_state_digest": _state_digest(
                    payload["dataloader_generator_state"]
                ),
            }
        result = original(persisted, destination)
        if new_v4_epoch:
            predecessor_epoch = imported.get(destination.as_posix(), 0)
            if int(payload["epoch"]) <= predecessor_epoch:
                raise DARTCheckpointReplayCompatibilityError(
                    "v4 archive attempted to persist an imported predecessor epoch"
                )
            archive_record = archive_v4_completed_epoch(
                latest_checkpoint_path=destination,
                epoch_archive_root=destination.parent / "epoch_archive",
                repository_root=repository_root,
            )
            track_identity = {
                "seed_directory": destination.parent.parent.name,
                "batch_directory": destination.parent.name,
                "task": payload["task"],
                "model_name": payload["model_name"],
                "seed": int(payload["seed"]),
                "resume_fingerprint": payload["resume_fingerprint"],
            }
            update_v4_epoch_archive_manifest(
                manifest_path=destination.parent
                / "epoch_archive/epoch_archive_manifest.json",
                track_identity=track_identity,
                archive_record=archive_record,
                predecessor_epoch=predecessor_epoch,
            )
        return result

    atomic.__dart_v4_full_epoch_archive__ = COMPATIBILITY_SCHEMA_VERSION
    return atomic


def validate_v4_epoch_archives(
    *,
    v4_stage_root: Path,
    expected_track_count: int = 108,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate every per-track immutable full-epoch archive manifest."""

    root = Path(v4_stage_root).resolve()
    repo = Path(repository_root).resolve() if repository_root is not None else None
    manifests = sorted(
        (root / "checkpoint_groups").rglob(
            "epoch_archive/epoch_archive_manifest.json"
        )
    )
    if len(manifests) != int(expected_track_count):
        raise DARTCheckpointReplayCompatibilityError(
            f"v4 full-epoch archive has {len(manifests)} tracks; "
            f"expected {expected_track_count}"
        )
    records: list[dict[str, Any]] = []
    for path in manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        core = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if (
            value.get("schema_version") != "dcvpp.dart_v4_full_epoch_archive.v1"
            or value.get("status") != "COMPLETE_THROUGH_CURRENT_EPOCH"
            or value.get("manifest_sha256") != _sha256_json(core)
            or value.get("every_completed_epoch_full_state_persisted") is not True
            or value.get("formal_selection_candidates_divisible_by_five_only")
            is not True
        ):
            raise DARTCheckpointReplayCompatibilityError(
                f"v4 epoch archive manifest is invalid: {path}"
            )
        epochs = [int(record["epoch"]) for record in value.get("records", ())]
        predecessor_epoch = int(value.get("predecessor_imported_epoch", -1))
        if epochs and (
            epochs[0] != predecessor_epoch + 1
            or epochs != list(range(epochs[0], epochs[-1] + 1))
        ):
            raise DARTCheckpointReplayCompatibilityError(
                f"v4 epoch archive is not contiguous: {path}"
            )
        for record in value["records"]:
            raw_checkpoint = Path(str(record["path"]))
            checkpoint = (
                raw_checkpoint.resolve()
                if raw_checkpoint.is_absolute()
                else ((repo or Path.cwd()) / raw_checkpoint).resolve()
            )
            if repo is not None and repo not in checkpoint.parents:
                raise DARTCheckpointReplayCompatibilityError(
                    "v4 archived epoch escapes repository"
                )
            if (
                not checkpoint.is_file()
                or _sha256_file(checkpoint) != record["sha256"]
                or checkpoint.stat().st_size != int(record["byte_count"])
                or bool(record["formal_selection_candidate"])
                != (int(record["epoch"]) % 5 == 0)
            ):
                raise DARTCheckpointReplayCompatibilityError(
                    f"v4 archived epoch bytes/role changed: {checkpoint}"
                )
        records.append(
            {
                "path": (
                    path.relative_to(repo).as_posix()
                    if repo is not None and repo in path.parents
                    else path.as_posix()
                ),
                "sha256": _sha256_file(path),
                "track_identity": value["track_identity"],
                "first_epoch": value["first_epoch"],
                "latest_epoch": value["latest_epoch"],
                "record_count": value["record_count"],
                "predecessor_imported_epoch": predecessor_epoch,
            }
        )
    core = {
        "schema_version": "dcvpp.dart_v4_full_epoch_archive_inventory.v1",
        "status": "COMPLETE",
        "track_count": len(records),
        "track_manifests": records,
        "track_manifests_sha256": _sha256_json(records),
        "every_completed_epoch_full_state_persisted": True,
        "formal_selection_candidates_divisible_by_five_only": True,
    }
    inventory = {**core, "manifest_sha256": _sha256_json(core)}
    path = root / "recovery_overlay/v4_epoch_archive_inventory.json"
    _atomic_json(path, inventory)
    return {
        **inventory,
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
    }


def _manifest_core_sha(records: Sequence[Mapping[str, Any]]) -> str:
    """Stable digest independent of display-only record ordering."""

    core = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "verified_at_utc",
                "recorded_at_utc",
                "written_at_utc",
            }
        }
        for record in records
    ]
    return _sha256_json(sorted(core, key=lambda value: _sha256_json(value)))


def _minimal_anchor_record_core_sha(
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Digest pinned before deep state fields were added to the snapshot."""

    core = [
        {
            "path": str(record.get("source_path") or record.get("path")),
            "sha256": str(record["sha256"]),
            "byte_count": int(record["byte_count"]),
            "epoch": int(record["epoch"]),
            "seed": int(record["seed"]),
        }
        for record in records
    ]
    return _sha256_json(core)


def _state_digest(value: Any) -> str:
    """Content digest for nested tensor/array/scalar checkpoint state."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if hasattr(item, "detach") and hasattr(item, "cpu"):
            array = np.ascontiguousarray(item.detach().cpu().numpy())
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda value: str(value)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for value in item:
                update(value)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(
                json.dumps(item, sort_keys=True, allow_nan=False).encode("utf-8")
            )
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def build_v3_anchor_snapshot(
    module: Any,
    *,
    repository_root: Path,
    source_stage_root: Path,
    destination_manifest_path: Path,
    source_readiness_manifest_path: Path,
    source_policy_freeze_path: Path,
    expected_source_hashes: Mapping[str, str],
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Atomically build the immutable positive-anchor source manifest.

    Every anchor is opened and its embedded hashes/state digests are verified.
    Mutable aliases, epoch zero, selected checkpoints, and group/scheduler
    manifests are excluded by construction.  Evidence paths are stored
    repository-relative and must resolve inside the repository.
    """

    repo = Path(repository_root).resolve()
    stage = Path(source_stage_root).resolve()
    destination = Path(destination_manifest_path).resolve()
    if repo not in stage.parents or repo not in destination.parents:
        raise DARTCheckpointReplayCompatibilityError(
            "snapshot source/destination must remain inside repository"
        )
    selection_manifest = stage / "selection_stage_manifest.json"
    scheduler_state = stage / "checkpoint_scheduler/hybrid_scheduler_state.json"
    evidence = {
        "selection_stage_manifest": selection_manifest,
        "scheduler_state": scheduler_state,
        "source_readiness_manifest": Path(source_readiness_manifest_path).resolve(),
        "source_policy_freeze": Path(source_policy_freeze_path).resolve(),
        "source_training_module": Path(module.__file__).resolve(),
    }
    for key, path in evidence.items():
        if repo not in path.parents or not path.is_file():
            raise DARTCheckpointReplayCompatibilityError(
                f"snapshot evidence is absent or outside repository: {key}"
            )
        expected = str(expected_source_hashes.get(key, "")).lower()
        if expected and _sha256_file(path) != expected:
            raise DARTCheckpointReplayCompatibilityError(
                f"snapshot predecessor evidence hash changed: {key}"
            )
    stage_manifest = json.loads(selection_manifest.read_text(encoding="utf-8"))
    policy_freeze = json.loads(
        evidence["source_policy_freeze"].read_text(encoding="utf-8")
    )
    expected_old_config_sha = str(expected_source_hashes.get("source_config", ""))
    if (
        stage_manifest.get("status") != "PARTIAL"
        or int(stage_manifest.get("completed_unit_count", -1)) != 0
        or any((stage / "units").rglob("quantile_predictions.npz"))
    ):
        raise DARTCheckpointReplayCompatibilityError(
            "v3 predecessor is not the fail-closed pre-OOS state"
        )
    if (
        len(expected_old_config_sha) != 64
        or stage_manifest.get("config_sha256") != expected_old_config_sha
        or stage_manifest.get("input_fingerprints", {}).get(
            "dart_value_select_config"
        )
        != expected_old_config_sha
        or policy_freeze.get("contract", {}).get("config_sha256")
        != expected_old_config_sha
    ):
        raise DARTCheckpointReplayCompatibilityError(
            "v3 config lineage claim is not cross-verified by stage and v7 policy"
        )

    records: list[dict[str, Any]] = []
    track_count: dict[tuple[str, str, str, int], int] = {}
    for path in sorted((stage / "checkpoint_groups").rglob("anchors/epoch_*.pt")):
        epoch = int(path.stem.rsplit("_", 1)[-1])
        if epoch <= 0:
            continue
        if epoch % 5 != 0 or path.name == "epoch_0000.pt":
            raise DARTCheckpointReplayCompatibilityError(
                f"positive source anchor is off the five-epoch grid: {path}"
            )
        relative = path.relative_to(stage / "checkpoint_groups")
        parts = relative.parts
        if len(parts) != 7 or parts[-2] != "anchors":
            raise DARTCheckpointReplayCompatibilityError(
                f"source anchor layout changed: {path}"
            )
        origin, target, model = parts[:3]
        seed = int(parts[3].removeprefix("seed_"))
        batch_size = int(parts[4].removeprefix("batch_"))
        payload = module.torch.load(path, map_location="cpu", weights_only=False)
        module._validate_checkpoint(
            payload,
            fingerprint=str(payload["resume_fingerprint"]),
            model_name=model,
            task=target,
            seed=seed,
            max_epochs=int(payload["max_epochs"]),
            batch_size=batch_size,
        )
        architecture_sha = _state_digest(payload["architecture"])
        scientific_contract = {
            "origin_name": origin,
            "target_name": target,
            "model_name": model,
            "seed": seed,
            "batch_size": batch_size,
            "architecture_sha256": architecture_sha,
            "max_epochs": int(payload["max_epochs"]),
            "loss_id": "uniform_pinball_plus_0.1_crossing",
            "anchor_interval_epochs": 5,
            # Resume/epoch lineage is recorded separately; it is deliberately
            # excluded so this scientific digest is stable across anchors in
            # one seed track and across the v3 -> v4 namespace transition.
        }
        fit_year = int(origin.split("through_")[-1].split("_to_")[0])
        forecast_year = int(origin.split("_to_")[-1].split("_oos")[0])
        record = {
            "source_path": path.relative_to(repo).as_posix(),
            "sha256": _sha256_file(path),
            "byte_count": path.stat().st_size,
            "origin_name": origin,
            "fit_through_year": fit_year,
            "forecast_year": forecast_year,
            "target_name": target,
            "model_name": model,
            "seed": seed,
            "batch_size": batch_size,
            "epoch": epoch,
            "embedded_resume_fingerprint": payload["resume_fingerprint"],
            "epoch_ledger_sha256": payload["epoch_ledger_sha256"],
            "training_health_sha256": payload["training_health_sha256"],
            "architecture_sha256": architecture_sha,
            "model_state_digest": _state_digest(payload["model_state"]),
            "optimizer_state_digest": _state_digest(payload["optimizer_state"]),
            "scaler_state_digest": _state_digest(
                payload.get("gradient_scaler_state", {})
            ),
            "rng_state_digest": _state_digest(
                {
                    key: payload[key]
                    for key in (
                        "python_rng_state",
                        "numpy_rng_state",
                        "torch_rng_state",
                        "cuda_rng_state",
                        "dataloader_generator_state",
                    )
                }
            ),
            "scientific_contract_digest": _sha256_json(scientific_contract),
        }
        records.append(record)
        track_count[(origin, target, model, seed)] = (
            track_count.get((origin, target, model, seed), 0) + 1
        )
    total_bytes = sum(int(record["byte_count"]) for record in records)
    counts = dict(expected_counts or {})
    if (
        len(track_count) != int(counts.get("track_count", 108))
        or len(records) != int(counts.get("record_count", len(records)))
        or total_bytes != int(counts.get("total_bytes", total_bytes))
    ):
        raise DARTCheckpointReplayCompatibilityError(
            "v3 anchor snapshot track/count/byte contract changed"
        )
    manifest = {
        "schema_version": "dcvpp.dart_v3_pre_oos_anchor_snapshot.v1",
        "status": "COMPLETE_HASH_PINNED_READ_ONLY_SOURCE",
        "source_run_id": "paper-dart-forecast-v3",
        "destination_run_id": "paper-dart-forecast-v4",
        "source_stage_root": stage.relative_to(repo).as_posix(),
        "source_training_module_sha256": _sha256_file(
            evidence["source_training_module"]
        ),
        "source_config_sha256": expected_old_config_sha,
        "source_config_file_archived": False,
        "source_config_hash_cross_verified_from_stage_and_v7_policy": True,
        "source_readiness_manifest_sha256": _sha256_file(
            evidence["source_readiness_manifest"]
        ),
        "source_selection_stage_manifest": {
            "path": selection_manifest.relative_to(repo).as_posix(),
            "sha256": _sha256_file(selection_manifest),
        },
        "source_scheduler_state": {
            "path": scheduler_state.relative_to(repo).as_posix(),
            "sha256": _sha256_file(scheduler_state),
        },
        "record_count": len(records),
        "total_bytes": total_bytes,
        "records_sha256": _minimal_anchor_record_core_sha(records),
        "minimal_record_core_sha256": _minimal_anchor_record_core_sha(records),
        "expanded_records_sha256": _manifest_core_sha(records),
        "track_count": len(track_count),
        "completed_oos_unit_count": 0,
        "source_files_mutable_aliases_forbidden": ["latest.pt"],
        "excluded_source_classes": [
            "epoch_0000.pt",
            "selected_epoch_or_selected_anchor_pt",
            "checkpoint_group_manifest",
            "group_coordinator_state",
            "scheduler_state_as_checkpoint_source",
        ],
        "copy_hardlink_or_source_rewrite_performed": False,
        "records": records,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != manifest:
            raise DARTCheckpointReplayCompatibilityError(
                "existing v3 anchor snapshot differs"
            )
    else:
        _atomic_json(destination, manifest)
    return manifest


def _records_from_snapshot(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    for key in ("positive_anchors", "anchor_records", "anchors", "records"):
        values = snapshot.get(key)
        if isinstance(values, list):
            return tuple(dict(value) for value in values)
    raise DARTCheckpointReplayCompatibilityError(
        "v3 anchor snapshot has no positive-anchor record list"
    )


def validate_v3_anchor_snapshot(
    snapshot_manifest_path: Path,
    *,
    repository_root: Path | None = None,
    strict_paper_counts: bool = True,
) -> dict[str, Any]:
    """Validate the immutable v3 positive-anchor inventory without opening PT."""

    path = Path(snapshot_manifest_path).resolve()
    if not path.is_file():
        raise DARTCheckpointReplayCompatibilityError(
            f"v3 anchor snapshot manifest is absent: {path}"
        )
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") not in (
        None,
        "dcvpp.dart_v3_pre_oos_anchor_snapshot.v1",
    ) or snapshot.get("status") not in (
        None,
        "COMPLETE_HASH_PINNED_READ_ONLY_SOURCE",
    ):
        raise DARTCheckpointReplayCompatibilityError(
            "v3 anchor snapshot schema/status changed"
        )
    claimed_manifest_sha = snapshot.get("manifest_sha256")
    if claimed_manifest_sha:
        core = {
            key: value
            for key, value in snapshot.items()
            if key != "manifest_sha256"
        }
        if claimed_manifest_sha != _sha256_json(core):
            raise DARTCheckpointReplayCompatibilityError(
                "v3 anchor snapshot self-hash changed"
            )
    records = _records_from_snapshot(snapshot)
    repo = Path(repository_root).resolve() if repository_root is not None else None
    if repo is None and strict_paper_counts:
        candidates = tuple(path.parents)
        repo = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "configs/dart_value_select.yaml").is_file()
            ),
            None,
        )
        if repo is None:
            raise DARTCheckpointReplayCompatibilityError(
                "cannot derive repository root for v3 snapshot"
            )
    count = len(records)
    total_bytes = 0
    identities: set[tuple[str, str, str, int, int]] = set()
    for record in records:
        raw_path = Path(
            str(
                record.get("source_path")
                or record.get("path")
                or record.get("anchor_path")
                or ""
            )
        )
        if raw_path.is_absolute():
            if strict_paper_counts:
                raise DARTCheckpointReplayCompatibilityError(
                    "formal snapshot anchor paths must be repository-relative"
                )
            anchor_path = raw_path.resolve()
        else:
            anchor_path = ((repo or Path.cwd()) / raw_path).resolve()
        if repo is not None and repo not in anchor_path.parents:
            raise DARTCheckpointReplayCompatibilityError(
                "v3 snapshot anchor escapes repository"
            )
        claimed_sha = str(
            record.get("sha256")
            or record.get("source_sha256")
            or record.get("anchor_sha256")
            or ""
        )
        claimed_bytes = int(
            record.get("byte_count")
            or record.get("source_byte_count")
            or record.get("anchor_byte_count")
            or -1
        )
        epoch = int(record.get("epoch", -1))
        if (
            epoch <= 0
            or epoch % 5 != 0
            or not anchor_path.is_file()
            or anchor_path.name in {"latest.pt", "epoch_0000.pt"}
            or "selected_" in anchor_path.name
            or len(claimed_sha) != 64
            or _sha256_file(anchor_path) != claimed_sha
            or anchor_path.stat().st_size != claimed_bytes
        ):
            raise DARTCheckpointReplayCompatibilityError(
                f"v3 positive-anchor snapshot record is invalid: {anchor_path}"
            )
        origin = str(record.get("origin") or record.get("origin_name") or "")
        target = str(record.get("target") or record.get("target_name") or "")
        model = str(record.get("model") or record.get("model_name") or "")
        seed = int(record.get("seed", -1))
        identity = (origin, target, model, seed, epoch)
        if not all(identity[:3]) or seed <= 0 or identity in identities:
            raise DARTCheckpointReplayCompatibilityError(
                "v3 anchor snapshot identity is incomplete or duplicated"
            )
        identities.add(identity)
        total_bytes += claimed_bytes
        record["_resolved_source_path"] = anchor_path.as_posix()
    computed_record_core_sha = _minimal_anchor_record_core_sha(records)
    record_core_sha = str(
        snapshot.get("records_sha256")
        or snapshot.get("minimal_record_core_sha256")
        or snapshot.get("sorted_record_core_sha256")
        or snapshot.get("anchor_record_core_sha256")
        or computed_record_core_sha
    )
    if record_core_sha != computed_record_core_sha:
        raise DARTCheckpointReplayCompatibilityError(
            "v3 anchor snapshot minimal record-core digest changed"
        )
    if strict_paper_counts and (
        count != EXPECTED_V3_POSITIVE_ANCHOR_COUNT
        or total_bytes != EXPECTED_V3_POSITIVE_ANCHOR_BYTES
        or record_core_sha != EXPECTED_V3_ANCHOR_RECORD_CORE_SHA256
    ):
        raise DARTCheckpointReplayCompatibilityError(
            "v3 paper anchor snapshot count/bytes/core digest changed"
        )
    return {
        "manifest_path": path.as_posix(),
        "manifest_sha256": _sha256_file(path),
        "record_count": count,
        "total_bytes": total_bytes,
        "record_core_sha256": record_core_sha,
        "records": records,
    }


def _identity(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record.get("origin") or record.get("origin_name") or ""),
        str(record.get("target") or record.get("target_name") or ""),
        str(record.get("model") or record.get("model_name") or ""),
        int(record.get("seed", -1)),
    )


def _expected_v4_fingerprint(
    module: Any,
    *,
    source: Mapping[str, Any],
    v4_parent_fingerprint: str,
) -> str:
    return module._seed_fingerprint(
        parent_fingerprint=v4_parent_fingerprint,
        model_name=str(source["model_name"]),
        task=str(source["task"]),
        seed=int(source["seed"]),
        architecture=dict(source["architecture"]),
        max_epochs=int(source["max_epochs"]),
        batch_size=int(source["batch_size"]),
        learning_rate=float(source["learning_rate"]),
        gradient_accumulation_steps=int(source["gradient_accumulation_steps"]),
        weight_decay=float(source["weight_decay"]),
        clip_norm=float(source["clip_norm"]),
        anchor_interval_epochs=int(source["anchor_interval_epochs"]),
        selection_config=source["selection_config"],
        tail_threshold_contract=source["tail_threshold_contract"],
    )


def import_highest_v3_anchor(
    module: Any,
    *,
    source_record: Mapping[str, Any],
    destination_latest_path: Path,
    v4_parent_fingerprint: str,
    scientific_contract_digest: str,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Import one highest v3 anchor as a v4 latest checkpoint.

    Only the resume fingerprint changes.  Model, optimizer, scaler, RNG,
    complete epoch ledger, training health, and all scientific state remain
    byte-for-byte equal after deserialization.
    """

    source_path = Path(
        str(
            source_record.get("_resolved_source_path")
            or source_record.get("source_path")
            or source_record.get("path")
            or source_record.get("anchor_path")
        )
    ).resolve()
    repo = Path(repository_root).resolve() if repository_root is not None else None
    if repo is not None and repo not in source_path.parents:
        raise DARTCheckpointReplayCompatibilityError(
            "source anchor import escapes repository"
        )
    source_sha = _sha256_file(source_path)
    claimed_sha = str(
        source_record.get("sha256")
        or source_record.get("source_sha256")
        or source_record.get("anchor_sha256")
    )
    if source_sha != claimed_sha:
        raise DARTCheckpointReplayCompatibilityError(
            "source anchor changed before v4 import"
        )
    source = module.torch.load(source_path, map_location="cpu", weights_only=False)
    expected_identity = _identity(source_record)
    payload_identity = (
        str(source_record.get("origin") or source_record.get("origin_name") or ""),
        str(source["task"]),
        str(source["model_name"]),
        int(source["seed"]),
    )
    if expected_identity != payload_identity:
        raise DARTCheckpointReplayCompatibilityError(
            "source snapshot identity differs from checkpoint payload"
        )
    mutable = copy.deepcopy(source)
    old_fingerprint = str(mutable["resume_fingerprint"])
    source_evidence = (
        source_path.relative_to(repo).as_posix()
        if repo is not None
        else source_path.as_posix()
    )
    new_fingerprint = str(source_record.get("v4_resume_fingerprint") or "")
    if not new_fingerprint:
        required = {
            "learning_rate",
            "gradient_accumulation_steps",
            "weight_decay",
            "clip_norm",
            "anchor_interval_epochs",
            "selection_config",
            "tail_threshold_contract",
        }
        if not required.issubset(source_record):
            raise DARTCheckpointReplayCompatibilityError(
                "v4 fingerprint inputs are absent from source import record"
            )
        new_fingerprint = _expected_v4_fingerprint(
            module,
            source={**source_record, **mutable},
            v4_parent_fingerprint=v4_parent_fingerprint,
        )
    mutable["resume_fingerprint"] = new_fingerprint
    mutable["overlay_import_provenance"] = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "source_v3_anchor_path": source_evidence,
        "source_v3_anchor_sha256": source_sha,
        "source_v3_resume_fingerprint": old_fingerprint,
        "destination_v4_resume_fingerprint": new_fingerprint,
        "scientific_contract_digest": scientific_contract_digest,
        "only_resume_namespace_changed": True,
        "training_or_replay_steps_during_import": 0,
    }
    destination = Path(destination_latest_path).resolve()
    output = module._atomic_torch(mutable, destination)
    if _sha256_file(source_path) != source_sha:
        raise DARTCheckpointReplayCompatibilityError(
            "v3 source anchor bytes changed during v4 import"
        )
    destination_evidence = (
        destination.relative_to(repo).as_posix()
        if repo is not None and repo in destination.parents
        else destination.as_posix()
    )
    return {
        **output,
        "path": destination_evidence,
        "source_path": source_evidence,
        "source_sha256": source_sha,
        "source_epoch": int(source["epoch"]),
        "source_resume_fingerprint": old_fingerprint,
        "destination_resume_fingerprint": new_fingerprint,
        "training_or_replay_steps_during_import": 0,
    }


def prepare_v4_overlay_import(
    module: Any,
    *,
    snapshot_manifest_path: Path,
    v4_stage_root: Path,
    group_import_contracts: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    scientific_contract: Mapping[str, Any],
    repository_root: Path | None = None,
    strict_paper_counts: bool = True,
) -> dict[str, Any]:
    """Import the highest source anchor for each of 108 seed groups into v4."""

    snapshot = validate_v3_anchor_snapshot(
        snapshot_manifest_path,
        repository_root=repository_root,
        strict_paper_counts=strict_paper_counts,
    )
    ledger_path = (
        Path(v4_stage_root).resolve()
        / "recovery_overlay/v3_anchor_import_ledger.json"
    )
    if ledger_path.is_file():
        existing = json.loads(ledger_path.read_text(encoding="utf-8"))
        core = {
            key: value
            for key, value in existing.items()
            if key != "manifest_sha256"
        }
        if (
            existing.get("status")
            != "V4_OVERLAY_IMPORTED_NO_TRAINING_LAUNCHED"
            or existing.get("manifest_sha256") != _sha256_json(core)
            or existing.get("v3_snapshot_manifest_sha256")
            != snapshot["manifest_sha256"]
        ):
            raise DARTCheckpointReplayCompatibilityError(
                "existing v4 overlay import ledger changed"
            )
        for record in existing.get("imports", ()):
            base = (
                Path(repository_root).resolve()
                if repository_root is not None
                else Path.cwd()
            )
            raw_source = Path(str(record.get("source_path", "")))
            source = (
                raw_source.resolve()
                if raw_source.is_absolute()
                else (base / raw_source).resolve()
            )
            if (
                not source.is_file()
                or _sha256_file(source) != record.get("source_sha256")
            ):
                raise DARTCheckpointReplayCompatibilityError(
                    "existing v4 overlay import source changed"
                )
        return {
            **existing,
            "ledger_path": ledger_path.as_posix(),
            "ledger_sha256": _sha256_file(ledger_path),
        }
    scientific_digest = _sha256_json(dict(scientific_contract))
    by_identity: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for record in snapshot["records"]:
        by_identity.setdefault(_identity(record), []).append(dict(record))
    if set(by_identity) != set(group_import_contracts):
        raise DARTCheckpointReplayCompatibilityError(
            "v3 snapshot identities differ from v4 import contracts"
        )
    imports: list[dict[str, Any]] = []
    for identity in sorted(by_identity):
        source = max(by_identity[identity], key=lambda value: int(value["epoch"]))
        contract = dict(group_import_contracts[identity])
        group_relative = Path(
            f"checkpoint_groups/{identity[0]}/{identity[1]}/{identity[2]}"
        )
        batch_size = int(contract["batch_size"])
        destination = (
            Path(v4_stage_root).resolve()
            / group_relative
            / f"seed_{identity[3]}"
            / f"batch_{batch_size}"
            / "latest.pt"
        )
        source_record = {**source, **contract}
        import_record = import_highest_v3_anchor(
            module,
            source_record=source_record,
            destination_latest_path=destination,
            v4_parent_fingerprint=str(contract["v4_parent_fingerprint"]),
            scientific_contract_digest=scientific_digest,
            repository_root=repository_root,
        )
        initialize_v4_epoch_archive_manifest(
            manifest_path=destination.parent
            / "epoch_archive/epoch_archive_manifest.json",
            track_identity={
                "seed_directory": destination.parent.parent.name,
                "batch_directory": destination.parent.name,
                "task": identity[1],
                "model_name": identity[2],
                "seed": int(identity[3]),
                "resume_fingerprint": import_record[
                    "destination_resume_fingerprint"
                ],
            },
            predecessor_epoch=int(import_record["source_epoch"]),
        )
        imports.append({"identity": list(identity), **import_record})
    ledger = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "status": "V4_OVERLAY_IMPORTED_NO_TRAINING_LAUNCHED",
        "v3_snapshot_manifest_path": snapshot["manifest_path"],
        "v3_snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "v3_positive_anchor_count": snapshot["record_count"],
        "v3_positive_anchor_bytes": snapshot["total_bytes"],
        "v3_anchor_record_core_sha256": snapshot["record_core_sha256"],
        "scientific_contract": dict(scientific_contract),
        "scientific_contract_digest": scientific_digest,
        "imported_highest_seed_state_count": len(imports),
        "imports": imports,
        "v3_files_moved_modified_or_deleted": 0,
        "source_latest_epoch0_selected_and_group_manifests_imported": False,
        "training_launched": False,
    }
    ledger["manifest_sha256"] = _sha256_json(ledger)
    _atomic_json(ledger_path, ledger)
    return {**ledger, "ledger_path": ledger_path.as_posix(), "ledger_sha256": _sha256_file(ledger_path)}


def initialize_v4_scheduler_state(
    module: Any,
    *,
    jobs: Sequence[Mapping[str, Any]],
    scheduler_root: Path,
    import_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a core-valid INITIALIZED state before imported latest files exist.

    The frozen scheduler intentionally reconstructs active/completed sets and
    40/50/60 assessments from returned ledger prefixes on every invocation; it
    consumes only the prior event list.  Imported max-epoch states therefore
    remain untouched.  Any post-stop source surplus is disclosed by the import
    ledger and naturally excluded when the frozen stop rule rematerializes the
    group at the correct global boundary.
    """

    ordered_jobs = tuple(dict(job) for job in jobs)
    contexts = tuple(module._context_from_job(job) for job in ordered_jobs)
    fingerprints = tuple(str(value["group_fingerprint"]) for value in contexts)
    policies = {
        (
            tuple(context["seeds"]),
            int(job["max_epochs"]),
            int(job.get("minimum_epochs", 40)),
            int(job.get("evaluation_interval_epochs", 5)),
            int(job.get("stop_commit_interval_epochs", 10)),
            int(job.get("stability_evaluations", 3)),
            float(job.get("early_stopping_min_delta", 1.0e-8)),
        )
        for job, context in zip(ordered_jobs, contexts, strict=True)
    }
    if len(policies) != 1:
        raise DARTCheckpointReplayCompatibilityError(
            "v4 scheduler jobs do not share one frozen policy"
        )
    (
        seeds,
        max_epochs,
        minimum_epochs,
        interval,
        global_round_interval,
        stability,
        min_delta,
    ) = next(iter(policies))
    scheduler_fingerprint = module.sha256_json(
        {
            "schema_version": module.SCHEMA_VERSION,
            "order": "warmup_to_40_then_10_epoch_active_group_rounds",
            "group_fingerprints_in_order": list(fingerprints),
            "study_seeds": list(seeds),
            "max_epochs": max_epochs,
            "minimum_epochs": minimum_epochs,
            "evaluation_interval_epochs": interval,
            "global_round_interval_epochs": global_round_interval,
            "stability_evaluations": stability,
            "early_stopping_min_delta": min_delta,
            "module_sha256": module.sha256_file(Path(module.__file__)),
        }
    )
    state_path = Path(scheduler_root).resolve() / "hybrid_scheduler_state.json"
    if state_path.exists():
        raise DARTCheckpointReplayCompatibilityError(
            "v4 scheduler state already exists before overlay initialization"
        )
    state_core = {
        "schema_version": module.SCHEMA_VERSION,
        "status": "INITIALIZED",
        "scheduler_fingerprint": scheduler_fingerprint,
        "declared_order": "warmup_to_40_then_10_epoch_active_group_rounds",
        "group_fingerprints_in_order": list(fingerprints),
        "execution_events": [],
        "individual_seed_early_stopping": False,
        "market_replay_or_locked_test_consumed": False,
        "v4_overlay_import_ledger_sha256": str(import_ledger["ledger_sha256"]),
        "frozen_core_reconstructs_assessments_from_imported_ledger_prefixes": True,
        "reconstructed_global_boundaries": [40, 50, 60],
        "imported_post_stop_surplus_role": "AUDIT_ONLY_UNTOUCHED",
        "written_at_utc": module.utc_now(),
    }
    state = {**state_core, "state_sha256": module.sha256_json(state_core)}
    module.atomic_json(state_path, state)
    return state


def build_v4_overlay_contracts(
    runner: Any,
    units: Sequence[Any],
    *,
    plan_fingerprint: str,
    quantiles: Sequence[float],
) -> tuple[
    dict[tuple[str, str, str, int], dict[str, Any]],
    dict[str, tuple[str, str, str]],
    dict[str, Any],
    tuple[dict[str, Any], ...],
]:
    """Build exact v4 import fingerprints using the production runner contract.

    This performs the same read-only data/contract construction as neural
    pretraining but does not call the trainer.  It returns 108 seed import
    contracts, a v4 group-root identity map for materialization, and one
    scientific-contract object whose digest is pinned in every import.
    """

    from dcvpp.experiments.dart_forecast_oos import OOSForecastRequest, STUDY_SEEDS

    unique: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    for unit in units:
        if unit.candidate.family != "neural":
            continue
        identity = (unit.origin.name, unit.target.name, unit.candidate.name)
        if identity not in seen:
            seen.add(identity)
            unique.append(unit)
    if len(unique) != 36:
        raise DARTCheckpointReplayCompatibilityError(
            "v4 overlay requires all 36 origin/target/neural groups"
        )
    device = runner._neural_device()
    contracts: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    roots: dict[str, tuple[str, str, str]] = {}
    scientific_groups: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for unit in unique:
        request = OOSForecastRequest(
            unit_id=f"overlay__{unit.unit_id}",
            request_fingerprint=_sha256_json(
                {
                    "role": "v4_anchor_overlay_import",
                    "plan_fingerprint": plan_fingerprint,
                    "origin": unit.origin.name,
                    "target": unit.target.name,
                    "candidate": unit.candidate.name,
                }
            ),
            plan_fingerprint=plan_fingerprint,
            origin_name=unit.origin.name,
            fit_through_year=unit.origin.fit_through_year,
            forecast_year=unit.origin.forecast_year,
            target_name=unit.target.name,
            target_field=unit.target.target_field,
            candidate_name=unit.candidate.name,
            candidate_family=unit.candidate.family,
            seed_slot_name=unit.seed_slot.name,
            seed=STUDY_SEEDS[0],
            quantiles=tuple(map(float, quantiles)),
            training_start_utc=unit.origin.training_start_utc,
            training_cutoff_exclusive_utc=unit.origin.training_cutoff_exclusive_utc,
            forecast_start_utc=unit.origin.forecast_start_utc,
            forecast_end_exclusive_utc=unit.origin.forecast_end_exclusive_utc,
            checkpoint_directory=Path(runner.output_root) / "recovery_overlay/request_views",
            resume_checkpoint=None,
            expected_information_set_fingerprint=None,
        )
        runner._validate_request(request)
        data = runner._data(request)
        kwargs = runner._neural_group_kwargs(request, data, device=device)
        jobs.append(dict(kwargs))
        group_root = Path(kwargs["output_root"]).resolve()
        group_identity = (
            unit.origin.name,
            unit.target.name,
            unit.candidate.name,
        )
        roots[group_root.as_posix()] = group_identity
        scientific_group = {
            "origin_name": unit.origin.name,
            "fit_through_year": int(unit.origin.fit_through_year),
            "forecast_year": int(unit.origin.forecast_year),
            "target_name": unit.target.name,
            "model_name": unit.candidate.name,
            "group_parent_fingerprint": str(kwargs["resume_fingerprint"]),
            "training_data_fingerprint": data.training_data_fingerprint,
            "tail_threshold_contract_sha256": data.tail_thresholds_by_target[
                unit.target.name
            ]["threshold_contract_sha256"],
            "max_epochs": int(kwargs["max_epochs"]),
            "learning_rate": float(kwargs["learning_rate"]),
            "gradient_accumulation_steps": int(
                kwargs["gradient_accumulation_steps"]
            ),
            "anchor_interval_epochs": int(kwargs["anchor_interval_epochs"]),
            "selection_config_fingerprint": kwargs[
                "selection_config"
            ].fingerprint,
        }
        scientific_groups.append(scientific_group)
        for seed in STUDY_SEEDS:
            contracts[(*group_identity, int(seed))] = {
                "batch_size": int(kwargs["batch_sizes"][0]),
                "v4_parent_fingerprint": str(kwargs["resume_fingerprint"]),
                "learning_rate": float(kwargs["learning_rate"]),
                "gradient_accumulation_steps": int(
                    kwargs["gradient_accumulation_steps"]
                ),
                "weight_decay": float(kwargs.get("weight_decay", 1.0e-4)),
                "clip_norm": float(kwargs.get("clip_norm", 1.0)),
                "anchor_interval_epochs": int(kwargs["anchor_interval_epochs"]),
                "selection_config": kwargs["selection_config"],
                "tail_threshold_contract": kwargs["tail_threshold_contract"],
            }
    scientific_contract = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "candidate_epoch_policy": "v3_union_v4_exact_5_epoch_anchor_only",
        "frozen_training_loss": "uniform_pinball_plus_0.1_crossing",
        "study_seeds": list(map(int, STUDY_SEEDS)),
        "group_count": len(unique),
        "groups": sorted(scientific_groups, key=_sha256_json),
        "provisional_candidate_grid_changed_to_exact_five_epoch_anchors": True,
        "oos_locked_test_market_replay_or_action_metric_consumed": False,
    }
    return contracts, roots, scientific_contract, tuple(jobs)


def prepare_v4_overlay_recovery(
    module: Any,
    *,
    runner: Any,
    units: Sequence[Any],
    plan_fingerprint: str,
    quantiles: Sequence[float],
    snapshot_manifest_path: Path,
    v4_stage_root: Path,
    repository_root: Path | None = None,
    strict_paper_counts: bool = True,
) -> dict[str, Any]:
    """Single orchestration entrypoint: import v3 maxima and initialize v4."""

    contracts, roots, scientific_contract, jobs = build_v4_overlay_contracts(
        runner,
        units,
        plan_fingerprint=plan_fingerprint,
        quantiles=quantiles,
    )
    # The frozen seed checkpoint fingerprint is parented by the prepared
    # three-seed group fingerprint, not by the adapter's group-input hash.
    for job in jobs:
        context = module._context_from_job(job)
        root_key = Path(context["group_root"]).resolve().as_posix()
        origin, target, model = roots[root_key]
        for seed in context["seeds"]:
            contracts[(origin, target, model, int(seed))][
                "v4_parent_fingerprint"
            ] = str(context["group_fingerprint"])
    import_ledger = prepare_v4_overlay_import(
        module,
        snapshot_manifest_path=snapshot_manifest_path,
        v4_stage_root=v4_stage_root,
        group_import_contracts=contracts,
        scientific_contract=scientific_contract,
        repository_root=repository_root,
        strict_paper_counts=strict_paper_counts,
    )
    scheduler_path = (
        Path(v4_stage_root).resolve()
        / "checkpoint_scheduler/hybrid_scheduler_state.json"
    )
    if scheduler_path.is_file():
        scheduler_state = json.loads(scheduler_path.read_text(encoding="utf-8"))
    else:
        scheduler_state = initialize_v4_scheduler_state(
            module,
            jobs=jobs,
            scheduler_root=Path(v4_stage_root) / "checkpoint_scheduler",
            import_ledger=import_ledger,
        )
    return {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "status": "V4_OVERLAY_PREPARED_NO_TRAINING_LAUNCHED",
        "import_ledger": import_ledger,
        "scheduler_state": scheduler_state,
        "jobs": jobs,
        "group_identity_by_root": roots,
        "scientific_contract": scientific_contract,
        "snapshot_manifest_path": Path(snapshot_manifest_path).resolve(),
        "v4_stage_root": Path(v4_stage_root).resolve(),
        "repository_root": (
            Path(repository_root).resolve()
            if repository_root is not None
            else None
        ),
    }


def _select_from_anchor_grid(
    module: Any,
    records: Sequence[Any],
    *,
    config: Any,
    selector: Any,
    provisional: bool = False,
) -> dict[str, Any]:
    anchor_records = tuple(record for record in records if int(record.epoch) % 5 == 0)
    if not anchor_records:
        raise module.DARTCheckpointTrainingError(
            "anchor-only selector has no eligible epoch"
        )
    result = dict(
        selector(anchor_records, config=config, provisional=bool(provisional))
    )
    result.update(
        {
            "candidate_epoch_policy": "hash_verified_v3_union_v4_5_epoch_anchors",
            "candidate_anchor_epochs": [record.epoch for record in anchor_records],
            "arbitrary_non_anchor_epoch_replay_forbidden": True,
            "provisional_and_final_candidate_grid_anchor_only": True,
            "recovery_schema_version": COMPATIBILITY_SCHEMA_VERSION,
        }
    )
    if result.get("status") == "SELECTED":
        result["selection_manifest_sha256"] = module.sha256_json(
            {key: value for key, value in result.items() if key != "selection_manifest_sha256"}
        )
    return result


def _anchor_index(
    snapshot_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    return {
        (*_identity(record), int(record["epoch"])): dict(record)
        for record in snapshot_records
    }


def _make_selected_anchor_materializer(
    module: Any,
    *,
    source_index: Mapping[tuple[str, str, str, int, int], Mapping[str, Any]],
    group_identity_by_root: Mapping[str, tuple[str, str, str]],
    repository_root: Path | None = None,
) -> Any:
    def materialize(
        *,
        seed_result: Mapping[str, Any],
        selection_result: Mapping[str, Any],
        aggregation_manifest: Mapping[str, Any],
        model_name: str,
        task: str,
        seed: int,
        group_root: Path,
        **_: Any,
    ) -> dict[str, Any]:
        selected_epoch = int(selection_result["selected_epoch"])
        root_key = Path(group_root).resolve().as_posix()
        local_anchor: Path | None = None
        try:
            candidate = module._anchor_path_for_epoch(seed_result, selected_epoch)
            if (
                candidate.is_file()
                and int(candidate.stem.rsplit("_", 1)[-1]) == selected_epoch
            ):
                local_anchor = candidate
        except (FileNotFoundError, module.DARTCheckpointTrainingError):
            local_anchor = None
        if local_anchor is not None:
            source_path = local_anchor
            source_kind = "v4_local_anchor"
            source_claimed_sha = module.sha256_file(source_path)
        elif root_key in group_identity_by_root:
            origin, target, model = group_identity_by_root[root_key]
            identity = (origin, target, model, int(seed), selected_epoch)
            if identity not in source_index:
                raise module.DARTCheckpointTrainingError(
                    "selected epoch has no hash-verified v3 or v4 anchor"
                )
            record = source_index[identity]
            raw_source = Path(
                str(record.get("_resolved_source_path") or record.get("source_path") or record.get("path"))
            )
            source_path = (
                raw_source.resolve()
                if raw_source.is_absolute()
                else ((Path(repository_root).resolve() if repository_root is not None else Path.cwd()) / raw_source).resolve()
            )
            source_kind = "v3_source_anchor"
            source_claimed_sha = str(record.get("sha256") or record.get("anchor_sha256"))
        else:
            raise module.DARTCheckpointTrainingError(
                "v4 group has neither an exact local anchor nor imported identity"
            )
        if module.sha256_file(source_path) != source_claimed_sha:
            raise module.DARTCheckpointTrainingError("selected source anchor hash changed")
        anchor = module.torch.load(source_path, map_location="cpu", weights_only=False)
        if int(anchor["epoch"]) != selected_epoch:
            raise module.DARTCheckpointTrainingError("selected source anchor epoch changed")
        ledger = tuple(seed_result["ledger"])
        selected_record = next(record for record in ledger if record.epoch == selected_epoch)
        expected_record_sha = module.sha256_json(selected_record.as_dict())
        if module.sha256_json(anchor["epoch_ledger"][selected_epoch - 1]) != expected_record_sha:
            raise module.DARTCheckpointTrainingError("selected source anchor ledger changed")
        payload = {
            "checkpoint_schema_version": module.SELECTED_CHECKPOINT_SCHEMA_VERSION,
            "namespace": "formal_dart_selected_common_anchor_epoch_v4_overlay",
            "task": task,
            "model_name": model_name,
            "seed": int(seed),
            "selected_common_epoch": selected_epoch,
            "model_state": anchor["model_state"],
            "optimizer_state": anchor["optimizer_state"],
            "gradient_scaler_state": anchor.get("gradient_scaler_state", {}),
            "dataloader_generator_state": anchor["dataloader_generator_state"],
            "architecture": dict(seed_result["architecture"]),
            "batch_size": int(seed_result["batch_size"]),
            "resume_fingerprint": seed_result["fingerprint"],
            "epoch_ledger_sha256": seed_result["ledger_sha256"],
            "selected_epoch_record": selected_record.as_dict(),
            "selected_epoch_record_sha256": expected_record_sha,
            "selection_manifest_sha256": selection_result["selection_manifest_sha256"],
            "seed_aggregation_manifest_sha256": aggregation_manifest["manifest_sha256"],
            "source_anchor": {
                "kind": source_kind,
                "path": source_path.as_posix(),
                "sha256": source_claimed_sha,
                "epoch": selected_epoch,
            },
            "selected_model_state_source": "exact_hash_verified_anchor_no_replay",
            "training_or_replay_steps_during_materialization": 0,
            "market_replay_or_action_metric_consumed": False,
            "oos_or_locked_target_consumed": False,
            "python_rng_state": anchor["python_rng_state"],
            "numpy_rng_state": anchor["numpy_rng_state"],
            "torch_rng_state": anchor["torch_rng_state"],
            "cuda_rng_state": anchor["cuda_rng_state"],
            "written_at_utc": module.utc_now(),
        }
        selected_path = Path(group_root) / f"seed_{seed}/selected_anchor_epoch_{selected_epoch:04d}.pt"
        output = module._atomic_torch(payload, selected_path)
        return {
            **output,
            "seed": int(seed),
            "batch_size": int(seed_result["batch_size"]),
            "architecture": dict(seed_result["architecture"]),
            "resume_fingerprint": seed_result["fingerprint"],
            "selected_epoch_record_sha256": expected_record_sha,
            "selected_model_state_source": "exact_hash_verified_anchor_no_replay",
            "training_or_replay_steps_during_materialization": 0,
        }

    return materialize


@contextmanager
def selected_epoch_replay_compatibility(
    *,
    snapshot_manifest_path: Path,
    group_identity_by_root: Mapping[str, tuple[str, str, str]],
    ledger_path: Path | None = None,
    repository_root: Path | None = None,
    strict_paper_counts: bool = True,
    v4_stage_root: Path | None = None,
    import_ledger: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Apply exact-anchor selection/materialization and full v4 epoch archives."""

    from dcvpp.experiments import dart_checkpoint_training as frozen

    core_path = Path(frozen.__file__).resolve()
    core_sha = _sha256_file(core_path)
    if core_sha != FROZEN_DART_CHECKPOINT_TRAINING_SHA256:
        raise DARTCheckpointReplayCompatibilityError(
            "frozen trainer SHA-256 changed: " + core_sha
        )
    snapshot = validate_v3_anchor_snapshot(
        snapshot_manifest_path,
        repository_root=repository_root,
        strict_paper_counts=strict_paper_counts,
    )
    record = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "scope": "v4_provisional_final_anchor_grid_materialization_and_epoch_archive",
        "frozen_trainer_sha256": core_sha,
        "v3_snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "v3_anchor_record_core_sha256": snapshot["record_core_sha256"],
        "provisional_candidate_grid_changed_to_exact_five_epoch_anchors": True,
        "training_loss_optimizer_model_or_data_changed": False,
        "v3_files_moved_modified_or_deleted": 0,
        "arbitrary_non_anchor_replay_forbidden": True,
        "oos_locked_test_market_replay_or_action_metric_consumed": False,
    }
    if ledger_path is not None:
        destination = Path(ledger_path).resolve()
        if destination.is_file():
            if json.loads(destination.read_text(encoding="utf-8")) != record:
                raise DARTCheckpointReplayCompatibilityError(
                    "existing v4 compatibility ledger differs"
                )
        else:
            _atomic_json(destination, record)
    original_selector = frozen.select_checkpoint_epoch
    original_materializer = frozen._materialize_selected_checkpoint
    original_atomic_torch = frozen._atomic_torch

    def selector(records: Sequence[Any], *, config: Any = None, provisional: bool = False) -> dict[str, Any]:
        return _select_from_anchor_grid(
            frozen,
            records,
            config=config,
            selector=original_selector,
            provisional=provisional,
        )

    materializer = _make_selected_anchor_materializer(
        frozen,
        source_index=_anchor_index(snapshot["records"]),
        group_identity_by_root={
            Path(key).resolve().as_posix(): tuple(value)
            for key, value in group_identity_by_root.items()
        },
        repository_root=repository_root,
    )
    imports = list((import_ledger or {}).get("imports", ()))
    repo = Path(repository_root).resolve() if repository_root is not None else Path.cwd()
    imported_epochs = {}
    for value in imports:
        raw = Path(str(value["path"]))
        latest = raw.resolve() if raw.is_absolute() else (repo / raw).resolve()
        imported_epochs[latest.as_posix()] = int(value["source_epoch"])
    archive_atomic = install_v4_full_epoch_archive(
        frozen,
        imported_epoch_by_latest_path=imported_epochs,
        repository_root=repository_root,
        v4_stage_root=v4_stage_root,
    )
    with _CONTEXT_LOCK:
        frozen.select_checkpoint_epoch = selector
        frozen._materialize_selected_checkpoint = materializer
        frozen._atomic_torch = archive_atomic
        try:
            yield record
        finally:
            if (
                frozen.select_checkpoint_epoch is not selector
                or frozen._materialize_selected_checkpoint is not materializer
                or frozen._atomic_torch is not archive_atomic
            ):
                raise DARTCheckpointReplayCompatibilityError(
                    "v4 compatibility functions changed inside context"
                )
            frozen.select_checkpoint_epoch = original_selector
            frozen._materialize_selected_checkpoint = original_materializer
            frozen._atomic_torch = original_atomic_torch


__all__ = [
    "COMPATIBILITY_SCHEMA_VERSION",
    "DARTCheckpointReplayCompatibilityError",
    "FROZEN_DART_CHECKPOINT_TRAINING_SHA256",
    "archive_v4_completed_epoch",
    "build_v3_anchor_snapshot",
    "build_v4_overlay_contracts",
    "import_highest_v3_anchor",
    "initialize_v4_scheduler_state",
    "install_v4_full_epoch_archive",
    "prepare_v4_overlay_recovery",
    "prepare_v4_overlay_import",
    "selected_epoch_replay_compatibility",
    "update_v4_epoch_archive_manifest",
    "validate_v3_anchor_snapshot",
    "validate_v4_epoch_archives",
]
