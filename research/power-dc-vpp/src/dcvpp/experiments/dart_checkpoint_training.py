"""DART-only neural training with validation-only common-epoch selection.

This module intentionally does not modify or reuse the resume namespace of the
frozen ``paper-forecast-v1`` trainer.  A formal DART neural group consists of
the exact three study seeds for one annual-origin/candidate/target cell.  Every
seed is trained through the full epoch grid; daily validation evidence is then
averaged within date across seeds and one common epoch is selected.

The chronological evidence boundary is supplied by :mod:`dart_training`:

``core training -> checkpoint calibration fit -> checkpoint selection ->
final deployment calibration -> annual OOS forecast``.

Only the first three panels enter this module.  Market replay, action metrics,
the annual OOS target, and the locked 2025 outcome are not accepted inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_checkpoint_selection import (
    CheckpointSelectionConfig,
    DARTCheckpointSelectionError,
    EpochMetricRecord,
    aggregate_seed_epoch_records,
    build_epoch_metric_record,
    select_checkpoint_epoch,
)
from dcvpp.experiments.paper_training import (
    QUANTILES,
    _energy_factory,
    _energy_predict,
    _pinball_loss,
    _seed_everything,
)


SCHEMA_VERSION = "dcvpp.dart_checkpoint_training.v1"
SELECTED_CHECKPOINT_SCHEMA_VERSION = "dcvpp.dart_selected_neural_checkpoint.v1"
STUDY_SEEDS = (20260809, 20260819, 20260829)


class DARTCheckpointTrainingError(RuntimeError):
    """Raised when a DART-only checkpoint group cannot be reproduced."""


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_torch(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".pt", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "dataloader_generator_state": generator.get_state(),
    }


def _restore_rng(payload: Mapping[str, Any], generator: torch.Generator) -> None:
    required = (
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "dataloader_generator_state",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise DARTCheckpointTrainingError(
            f"DART checkpoint lacks resumable RNG state: {missing}"
        )
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    generator.set_state(payload["dataloader_generator_state"])


def _architecture(
    model: nn.Module,
    *,
    model_name: str,
    history: int,
    horizon: int,
) -> dict[str, Any]:
    if hasattr(model, "architecture_contract"):
        return dict(model.architecture_contract)
    return {"model": model_name, "history": int(history), "horizon": int(horizon)}


def _make_model(
    model_name: str,
    *,
    history: int,
    horizon: int,
    site_count: int,
    device: torch.device,
) -> nn.Module:
    return _energy_factory(
        model_name,
        history=history,
        horizon=horizon,
        sites=site_count,
        quantiles=len(QUANTILES),
    ).to(device)


def _inverse(
    prediction: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(prediction, dtype="float64")
        * np.asarray(scale, dtype="float64")[None, :, None, None]
        + np.asarray(center, dtype="float64")[None, :, None, None]
    )


def _loader(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    batch_size: int,
    generator: torch.Generator,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
    )


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    *,
    device: torch.device,
    gradient_accumulation_steps: int,
    clip_norm: float,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    maximum_gradient_norm = 0.0
    nonfinite_gradient_steps = 0
    gradient_clipped_steps = 0
    for batch_index, (x_batch, y_batch) in enumerate(loader, start=1):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(x_batch.to(device, non_blocking=True))
            raw_loss = _pinball_loss(
                prediction, y_batch.to(device, non_blocking=True)
            )
            if not bool(torch.isfinite(raw_loss).detach().cpu()):
                raise DARTCheckpointTrainingError("DART training loss is non-finite")
            loss = raw_loss / int(gradient_accumulation_steps)
        losses.append(float(raw_loss.detach().float().cpu()))
        scaler.scale(loss).backward()
        if (
            batch_index % int(gradient_accumulation_steps) == 0
            or batch_index == len(loader)
        ):
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_norm)
            )
            gradient_value = float(gradient_norm.detach().float().cpu())
            if not math.isfinite(gradient_value):
                nonfinite_gradient_steps += 1
                raise DARTCheckpointTrainingError("DART gradient norm is non-finite")
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_value)
            if gradient_value > float(clip_norm):
                gradient_clipped_steps += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    if not losses:
        raise DARTCheckpointTrainingError("DART training loader produced no batches")
    return {
        "training_loss": float(np.mean(losses)),
        "maximum_gradient_norm": float(maximum_gradient_norm),
        "nonfinite_gradient_steps": int(nonfinite_gradient_steps),
        "gradient_clipped_steps": int(gradient_clipped_steps),
    }


def _epoch_record(
    *,
    model: nn.Module,
    epoch: int,
    epoch_health: Mapping[str, Any],
    x_checkpoint_calibration: np.ndarray,
    y_checkpoint_calibration: np.ndarray,
    checkpoint_calibration_dates: Sequence[str],
    x_checkpoint_selection: np.ndarray,
    y_checkpoint_selection: np.ndarray,
    checkpoint_selection_dates: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    tail_threshold_contract: Mapping[str, Any],
    selection_config: CheckpointSelectionConfig,
    device: torch.device,
    batch_size: int,
) -> EpochMetricRecord:
    calibration_prediction = _inverse(
        _energy_predict(
            model,
            x_checkpoint_calibration,
            device=device,
            batch_size=batch_size,
        ),
        center=center,
        scale=scale,
    )
    selection_prediction = _inverse(
        _energy_predict(
            model,
            x_checkpoint_selection,
            device=device,
            batch_size=batch_size,
        ),
        center=center,
        scale=scale,
    )
    return build_epoch_metric_record(
        epoch=int(epoch),
        calibration_actual=np.asarray(y_checkpoint_calibration, dtype="float64"),
        calibration_prediction=calibration_prediction,
        calibration_dates=tuple(map(str, checkpoint_calibration_dates)),
        selection_actual=np.asarray(y_checkpoint_selection, dtype="float64"),
        selection_prediction=selection_prediction,
        selection_dates=tuple(map(str, checkpoint_selection_dates)),
        tail_threshold_contract=tail_threshold_contract,
        training_loss=float(epoch_health["training_loss"]),
        maximum_gradient_norm=float(epoch_health["maximum_gradient_norm"]),
        nonfinite_gradient_steps=int(epoch_health["nonfinite_gradient_steps"]),
        gradient_clipped_steps=int(epoch_health["gradient_clipped_steps"]),
        minimum_calibration_days=selection_config.minimum_calibration_days,
        minimum_calibration_samples=selection_config.minimum_calibration_samples,
    )


def _selection_signature(selection: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return only the stable terminal selection state used for group stopping."""

    if selection.get("status") != "SELECTED":
        return None
    selected = int(selection["selected_epoch"])
    global_set = tuple(
        sorted(
            map(
                int,
                selection.get("global_crps_confidence_set", {}).get(
                    "included_epochs", ()
                ),
            )
        )
    )
    tail_set = tuple(
        sorted(
            map(
                int,
                selection.get("proper_tail_twcrps_confidence_set", {}).get(
                    "included_epochs", ()
                ),
            )
        )
    )
    health_pass = set(map(int, selection.get("health_pass_epochs", ())))
    calibration_pass = set(
        map(int, selection.get("calibration_pass_epochs", ()))
    )
    if not global_set or not tail_set:
        return None
    return {
        "selected_epoch": selected,
        "global_crps_included_epochs": list(global_set),
        "proper_tail_twcrps_included_epochs": list(tail_set),
        "selected_epoch_health_pass": selected in health_pass,
        "selected_epoch_calibration_pass": selected in calibration_pass,
    }


def _best_mean_loss(
    selection: Mapping[str, Any], confidence_set_name: str
) -> float | None:
    values = selection.get(confidence_set_name, {}).get("mean_losses", {})
    if not values:
        return None
    numeric = np.asarray(list(values.values()), dtype="float64")
    if numeric.size == 0 or not np.isfinite(numeric).all():
        return None
    return float(np.min(numeric))


def should_stop_group(
    assessments: Sequence[Mapping[str, Any]],
    *,
    minimum_epochs: int = 40,
    evaluation_interval_epochs: int = 5,
    consecutive_evaluations: int = 3,
    min_delta: float = 1.0e-8,
) -> dict[str, Any]:
    """Fail-closed synchronized early-stop decision for an aligned seed group.

    The signature deliberately excludes the complete health/calibration pass
    lists: adding a new credible epoch must not by itself reset stability.  It
    includes only the provisional winner, both final confidence sets, and the
    winner's own health/calibration state.
    """

    if (
        int(minimum_epochs) < 1
        or int(evaluation_interval_epochs) < 1
        or int(consecutive_evaluations) < 1
        or not math.isfinite(float(min_delta))
        or float(min_delta) < 0.0
    ):
        raise ValueError("invalid synchronized DART early-stopping policy")
    rows = tuple(dict(value) for value in assessments)
    epochs = [int(row.get("assessment_epoch", -1)) for row in rows]
    if epochs != sorted(set(epochs)):
        raise DARTCheckpointTrainingError(
            "group early-stopping assessments are not unique and increasing"
        )
    eligible = tuple(
        row
        for row in rows
        if int(row.get("assessment_epoch", -1)) >= int(minimum_epochs)
    )
    base = {
        "stop": False,
        "minimum_epochs": int(minimum_epochs),
        "evaluation_interval_epochs": int(evaluation_interval_epochs),
        "consecutive_evaluations": int(consecutive_evaluations),
        "min_delta": float(min_delta),
        "assessment_count": len(rows),
        "eligible_assessment_count": len(eligible),
    }
    if len(eligible) < int(consecutive_evaluations):
        return {**base, "reason": "INSUFFICIENT_STABILITY_ASSESSMENTS"}
    recent = eligible[-int(consecutive_evaluations) :]
    recent_epochs = tuple(int(row["assessment_epoch"]) for row in recent)
    if any(not bool(row.get("aligned_exact_three_seed_ledgers")) for row in recent):
        return {**base, "reason": "SEED_LEDGERS_NOT_ALIGNED"}
    if any(tuple(map(int, row.get("study_seeds", ()))) != STUDY_SEEDS for row in recent):
        return {**base, "reason": "EXACT_THREE_SEED_CONTRACT_FAILED"}
    if any(
        later - earlier != int(evaluation_interval_epochs)
        for earlier, later in zip(recent_epochs, recent_epochs[1:])
    ):
        return {**base, "reason": "ASSESSMENT_INTERVAL_CHANGED"}
    signatures = tuple(
        _selection_signature(row.get("selection_result", {})) for row in recent
    )
    if any(signature is None for signature in signatures):
        return {**base, "reason": "PROVISIONAL_SELECTION_NOT_AVAILABLE"}
    signature_hashes = tuple(sha256_json(signature) for signature in signatures)
    if len(set(signature_hashes)) != 1:
        return {
            **base,
            "reason": "SELECTION_SIGNATURE_CHANGED",
            "recent_assessment_epochs": list(recent_epochs),
            "signature_sha256_by_assessment": list(signature_hashes),
        }

    global_losses = tuple(
        _best_mean_loss(row["selection_result"], "global_crps_confidence_set")
        for row in recent
    )
    tail_losses = tuple(
        _best_mean_loss(
            row["selection_result"], "proper_tail_twcrps_confidence_set"
        )
        for row in recent
    )
    if any(value is None for value in global_losses + tail_losses):
        return {**base, "reason": "BEST_MEAN_SCORE_NOT_AVAILABLE"}

    def largest_improvement(values: Sequence[float | None]) -> float:
        numeric = tuple(float(value) for value in values if value is not None)
        return max(
            (numeric[earlier] - numeric[later])
            for earlier in range(len(numeric))
            for later in range(earlier + 1, len(numeric))
        ) if len(numeric) > 1 else 0.0

    global_improvement = largest_improvement(global_losses)
    tail_improvement = largest_improvement(tail_losses)
    epoch_budget = int(consecutive_evaluations) * int(evaluation_interval_epochs)
    if epoch_budget < 15:
        return {**base, "reason": "STABILITY_WINDOW_BELOW_15_EPOCHS"}
    if global_improvement > float(min_delta) or tail_improvement > float(min_delta):
        return {
            **base,
            "reason": "BEST_MEAN_SCORE_STILL_IMPROVING",
            "recent_assessment_epochs": list(recent_epochs),
            "largest_global_crps_improvement": global_improvement,
            "largest_proper_tail_twcrps_improvement": tail_improvement,
        }
    return {
        **base,
        "stop": True,
        "reason": "SYNCHRONIZED_GROUP_STABILITY_REACHED",
        "stop_epoch": recent_epochs[-1],
        "recent_assessment_epochs": list(recent_epochs),
        "stability_epoch_budget": epoch_budget,
        "stable_signature": signatures[-1],
        "stable_signature_sha256": signature_hashes[-1],
        "best_mean_global_crps_by_assessment": list(map(float, global_losses)),
        "best_mean_proper_tail_twcrps_by_assessment": list(map(float, tail_losses)),
        "largest_global_crps_improvement": global_improvement,
        "largest_proper_tail_twcrps_improvement": tail_improvement,
    }


def _checkpoint_payload(
    *,
    model_name: str,
    task: str,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    epoch: int,
    max_epochs: int,
    batch_size: int,
    architecture: Mapping[str, Any],
    fingerprint: str,
    ledger: Sequence[EpochMetricRecord],
    training_runtime_seconds: float,
    peak_cuda_memory_allocated_bytes: int,
) -> dict[str, Any]:
    ledger_rows = [record.as_dict() for record in ledger]
    health = {
        "training_runtime_seconds": float(training_runtime_seconds),
        "maximum_gradient_norm_before_clipping": float(
            max((record.maximum_gradient_norm for record in ledger), default=0.0)
        ),
        "nonfinite_gradient_steps": int(
            sum(record.nonfinite_gradient_steps for record in ledger)
        ),
        "gradient_clipped_steps": int(
            sum(record.gradient_clipped_steps for record in ledger)
        ),
        "peak_cuda_memory_allocated_bytes": int(
            peak_cuda_memory_allocated_bytes
        ),
    }
    return {
        "checkpoint_schema_version": SCHEMA_VERSION,
        "namespace": "formal_dart_checkpoint_training",
        "task": task,
        "model_name": model_name,
        "seed": int(seed),
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "max_epochs": int(max_epochs),
        "completed": int(epoch) == int(max_epochs),
        "batch_size": int(batch_size),
        "architecture": dict(architecture),
        "resume_fingerprint": fingerprint,
        "epoch_ledger": ledger_rows,
        "epoch_ledger_sha256": sha256_json(ledger_rows),
        "training_health": health,
        "training_health_sha256": sha256_json(health),
        "market_replay_or_action_metric_consumed": False,
        "oos_or_locked_target_consumed": False,
        **_capture_rng(generator),
        "written_at_utc": utc_now(),
    }


def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    fingerprint: str,
    model_name: str,
    task: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
) -> tuple[EpochMetricRecord, ...]:
    if payload.get("checkpoint_schema_version") != SCHEMA_VERSION:
        raise DARTCheckpointTrainingError("DART checkpoint schema changed")
    expected = {
        "resume_fingerprint": fingerprint,
        "model_name": model_name,
        "task": task,
        "seed": int(seed),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "market_replay_or_action_metric_consumed": False,
        "oos_or_locked_target_consumed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DARTCheckpointTrainingError(
                f"DART checkpoint fingerprint field changed: {key}"
            )
    ledger = tuple(
        EpochMetricRecord.from_dict(row) for row in payload.get("epoch_ledger", ())
    )
    if [record.epoch for record in ledger] != list(range(1, len(ledger) + 1)):
        raise DARTCheckpointTrainingError("DART epoch ledger is not contiguous")
    if sha256_json([record.as_dict() for record in ledger]) != payload.get(
        "epoch_ledger_sha256"
    ):
        raise DARTCheckpointTrainingError("DART epoch ledger hash mismatch")
    if int(payload.get("epoch", -1)) != len(ledger):
        raise DARTCheckpointTrainingError("DART checkpoint epoch and ledger differ")
    if payload.get("training_health_sha256") != sha256_json(
        payload.get("training_health", {})
    ):
        raise DARTCheckpointTrainingError("DART training-health hash mismatch")
    return ledger


def _seed_fingerprint(
    *,
    parent_fingerprint: str,
    model_name: str,
    task: str,
    seed: int,
    architecture: Mapping[str, Any],
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    weight_decay: float,
    clip_norm: float,
    anchor_interval_epochs: int,
    selection_config: CheckpointSelectionConfig,
    tail_threshold_contract: Mapping[str, Any],
) -> str:
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "parent": parent_fingerprint,
            "module_sha256": sha256_file(Path(__file__)),
            "frozen_training_loss": "uniform_pinball_plus_0.1_crossing",
            "model_name": model_name,
            "task": task,
            "seed": int(seed),
            "architecture": dict(architecture),
            "max_epochs": int(max_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "weight_decay": float(weight_decay),
            "clip_norm": float(clip_norm),
            "anchor_interval_epochs": int(anchor_interval_epochs),
            "selection_config_fingerprint": selection_config.fingerprint,
            "tail_threshold_contract_sha256": tail_threshold_contract.get(
                "threshold_contract_sha256"
            ),
        }
    )


def _fit_seed_ledger(
    *,
    model_name: str,
    task: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_checkpoint_calibration: np.ndarray,
    y_checkpoint_calibration: np.ndarray,
    checkpoint_calibration_dates: Sequence[str],
    x_checkpoint_selection: np.ndarray,
    y_checkpoint_selection: np.ndarray,
    checkpoint_selection_dates: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    tail_threshold_contract: Mapping[str, Any],
    history: int,
    horizon: int,
    max_epochs: int,
    train_through_epoch: int,
    batch_sizes: Sequence[int],
    learning_rate: float,
    gradient_accumulation_steps: int,
    weight_decay: float,
    clip_norm: float,
    anchor_interval_epochs: int,
    selection_config: CheckpointSelectionConfig,
    device: torch.device,
    output_root: Path,
    parent_fingerprint: str,
) -> dict[str, Any]:
    if not 1 <= int(train_through_epoch) <= int(max_epochs):
        raise ValueError("train_through_epoch must lie inside the formal epoch grid")
    site_count = int(y_train.shape[1])
    last_oom: Exception | None = None
    for batch_size in tuple(dict.fromkeys(int(v) for v in batch_sizes if int(v) > 0)):
        attempt_root = output_root / f"batch_{batch_size}"
        latest_path = attempt_root / "latest.pt"
        _seed_everything(int(seed))
        model = _make_model(
            model_name,
            history=history,
            horizon=horizon,
            site_count=site_count,
            device=device,
        )
        architecture = _architecture(
            model, model_name=model_name, history=history, horizon=horizon
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        generator = torch.Generator().manual_seed(int(seed))
        fingerprint = _seed_fingerprint(
            parent_fingerprint=parent_fingerprint,
            model_name=model_name,
            task=task,
            seed=seed,
            architecture=architecture,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            anchor_interval_epochs=anchor_interval_epochs,
            selection_config=selection_config,
            tail_threshold_contract=tail_threshold_contract,
        )
        ledger: list[EpochMetricRecord] = []
        start_epoch = 1
        resumed_from_epoch = 0
        training_runtime_seconds = 0.0
        peak_cuda_memory_allocated_bytes = 0
        if latest_path.is_file():
            saved = torch.load(latest_path, map_location="cpu", weights_only=False)
            ledger = list(
                _validate_checkpoint(
                    saved,
                    fingerprint=fingerprint,
                    model_name=model_name,
                    task=task,
                    seed=seed,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                )
            )
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            saved_scaler = saved.get("gradient_scaler_state", {})
            if saved_scaler:
                scaler.load_state_dict(saved_scaler)
            _restore_rng(saved, generator)
            resumed_from_epoch = int(saved["epoch"])
            start_epoch = resumed_from_epoch + 1
            training_runtime_seconds = float(
                saved.get("training_health", {}).get("training_runtime_seconds", 0.0)
            )
            peak_cuda_memory_allocated_bytes = int(
                saved.get("training_health", {}).get(
                    "peak_cuda_memory_allocated_bytes", 0
                )
            )
            if resumed_from_epoch >= int(train_through_epoch):
                prefix = tuple(ledger[: int(train_through_epoch)])
                return {
                    "seed": int(seed),
                    "batch_size": int(batch_size),
                    "fingerprint": fingerprint,
                    "architecture": architecture,
                    "latest_checkpoint": {
                        "path": latest_path.as_posix(),
                        "sha256": sha256_file(latest_path),
                    },
                    "ledger": prefix,
                    "ledger_sha256": sha256_json(
                        [record.as_dict() for record in prefix]
                    ),
                    "resumed_from_epoch": resumed_from_epoch,
                    "training_health": dict(saved["training_health"]),
                    "trained_through_epoch": int(train_through_epoch),
                }
        else:
            anchor_zero = _checkpoint_payload(
                model_name=model_name,
                task=task,
                seed=seed,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                generator=generator,
                epoch=0,
                max_epochs=max_epochs,
                batch_size=batch_size,
                architecture=architecture,
                fingerprint=fingerprint,
                ledger=(),
                training_runtime_seconds=0.0,
                peak_cuda_memory_allocated_bytes=0,
            )
            _atomic_torch(anchor_zero, attempt_root / "anchors/epoch_0000.pt")

        loader = _loader(
            x_train,
            y_train,
            batch_size=batch_size,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        try:
            for epoch in range(start_epoch, int(train_through_epoch) + 1):
                started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                epoch_health = _train_epoch(
                    model,
                    optimizer,
                    scaler,
                    loader,
                    device=device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    clip_norm=clip_norm,
                )
                record = _epoch_record(
                    model=model,
                    epoch=epoch,
                    epoch_health=epoch_health,
                    x_checkpoint_calibration=x_checkpoint_calibration,
                    y_checkpoint_calibration=y_checkpoint_calibration,
                    checkpoint_calibration_dates=checkpoint_calibration_dates,
                    x_checkpoint_selection=x_checkpoint_selection,
                    y_checkpoint_selection=y_checkpoint_selection,
                    checkpoint_selection_dates=checkpoint_selection_dates,
                    center=center,
                    scale=scale,
                    tail_threshold_contract=tail_threshold_contract,
                    selection_config=selection_config,
                    device=device,
                    batch_size=batch_size,
                )
                ledger.append(record)
                training_runtime_seconds += time.perf_counter() - started
                if device.type == "cuda":
                    peak_cuda_memory_allocated_bytes = max(
                        peak_cuda_memory_allocated_bytes,
                        int(torch.cuda.max_memory_allocated(device)),
                    )
                payload = _checkpoint_payload(
                    model_name=model_name,
                    task=task,
                    seed=seed,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    generator=generator,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    architecture=architecture,
                    fingerprint=fingerprint,
                    ledger=ledger,
                    training_runtime_seconds=training_runtime_seconds,
                    peak_cuda_memory_allocated_bytes=peak_cuda_memory_allocated_bytes,
                )
                _atomic_torch(payload, latest_path)
                if epoch % int(anchor_interval_epochs) == 0 or epoch == max_epochs:
                    _atomic_torch(
                        payload,
                        attempt_root / f"anchors/epoch_{epoch:04d}.pt",
                    )
            completed = torch.load(latest_path, map_location="cpu", weights_only=False)
            ledger = list(
                _validate_checkpoint(
                    completed,
                    fingerprint=fingerprint,
                    model_name=model_name,
                    task=task,
                    seed=seed,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                )
            )
            if len(ledger) != int(train_through_epoch):
                raise DARTCheckpointTrainingError(
                    "DART seed did not finish the requested synchronized epoch panel"
                )
            return {
                "seed": int(seed),
                "batch_size": int(batch_size),
                "fingerprint": fingerprint,
                "architecture": architecture,
                "latest_checkpoint": {
                    "path": latest_path.as_posix(),
                    "sha256": sha256_file(latest_path),
                },
                "ledger": tuple(ledger),
                "ledger_sha256": completed["epoch_ledger_sha256"],
                "resumed_from_epoch": resumed_from_epoch,
                "training_health": dict(completed["training_health"]),
                "trained_through_epoch": int(train_through_epoch),
            }
        except torch.OutOfMemoryError as exc:
            last_oom = exc
            atomic_json(
                attempt_root / "oom_failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "FAILED_OOM_RETRY_SMALLER_BATCH",
                    "seed": int(seed),
                    "batch_size": int(batch_size),
                    "resume_fingerprint": fingerprint,
                    "error": str(exc),
                    "recorded_at_utc": utc_now(),
                },
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
    raise DARTCheckpointTrainingError(
        f"{model_name}/{task}/seed={seed} exhausted batch fallback: {last_oom}"
    )


def _anchor_path_for_epoch(seed_result: Mapping[str, Any], selected_epoch: int) -> Path:
    latest = Path(str(seed_result["latest_checkpoint"]["path"]))
    anchors = latest.parent / "anchors"
    candidates: list[tuple[int, Path]] = []
    for path in anchors.glob("epoch_*.pt"):
        try:
            epoch = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if epoch <= int(selected_epoch):
            candidates.append((epoch, path))
    if not candidates:
        raise DARTCheckpointTrainingError("no replay anchor precedes selected epoch")
    return max(candidates, key=lambda item: item[0])[1]


def _materialize_selected_checkpoint(
    *,
    seed_result: Mapping[str, Any],
    selection_result: Mapping[str, Any],
    aggregation_manifest: Mapping[str, Any],
    model_name: str,
    task: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_checkpoint_calibration: np.ndarray,
    y_checkpoint_calibration: np.ndarray,
    checkpoint_calibration_dates: Sequence[str],
    x_checkpoint_selection: np.ndarray,
    y_checkpoint_selection: np.ndarray,
    checkpoint_selection_dates: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    tail_threshold_contract: Mapping[str, Any],
    history: int,
    horizon: int,
    max_epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    weight_decay: float,
    clip_norm: float,
    selection_config: CheckpointSelectionConfig,
    device: torch.device,
    group_root: Path,
) -> dict[str, Any]:
    selected_epoch = int(selection_result["selected_epoch"])
    ledger = tuple(seed_result["ledger"])
    selected_record = next(
        (record for record in ledger if record.epoch == selected_epoch), None
    )
    if selected_record is None:
        raise DARTCheckpointTrainingError("selected epoch is absent from seed ledger")
    selected_path = group_root / f"seed_{seed}/selected_epoch_{selected_epoch:04d}.pt"
    anchor_path = _anchor_path_for_epoch(seed_result, selected_epoch)
    anchor_sha = sha256_file(anchor_path)
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
    if anchor.get("resume_fingerprint") != seed_result["fingerprint"]:
        raise DARTCheckpointTrainingError("selected-epoch replay anchor fingerprint changed")
    anchor_epoch = int(anchor["epoch"])
    batch_size = int(seed_result["batch_size"])
    model = _make_model(
        model_name,
        history=history,
        horizon=horizon,
        site_count=int(y_train.shape[1]),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(int(seed))
    model.load_state_dict(anchor["model_state"], strict=True)
    optimizer.load_state_dict(anchor["optimizer_state"])
    anchor_scaler = anchor.get("gradient_scaler_state", {})
    if anchor_scaler:
        scaler.load_state_dict(anchor_scaler)
    _restore_rng(anchor, generator)
    loader = _loader(
        x_train,
        y_train,
        batch_size=batch_size,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    replay_health: Mapping[str, Any] = {
        "training_loss": selected_record.training_loss,
        "maximum_gradient_norm": selected_record.maximum_gradient_norm,
        "nonfinite_gradient_steps": selected_record.nonfinite_gradient_steps,
        "gradient_clipped_steps": selected_record.gradient_clipped_steps,
    }
    for _epoch in range(anchor_epoch + 1, selected_epoch + 1):
        replay_health = _train_epoch(
            model,
            optimizer,
            scaler,
            loader,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clip_norm=clip_norm,
        )
    replay_record = _epoch_record(
        model=model,
        epoch=selected_epoch,
        epoch_health=replay_health,
        x_checkpoint_calibration=x_checkpoint_calibration,
        y_checkpoint_calibration=y_checkpoint_calibration,
        checkpoint_calibration_dates=checkpoint_calibration_dates,
        x_checkpoint_selection=x_checkpoint_selection,
        y_checkpoint_selection=y_checkpoint_selection,
        checkpoint_selection_dates=checkpoint_selection_dates,
        center=center,
        scale=scale,
        tail_threshold_contract=tail_threshold_contract,
        selection_config=selection_config,
        device=device,
        batch_size=batch_size,
    )
    expected_record_sha = sha256_json(selected_record.as_dict())
    replay_record_sha = sha256_json(replay_record.as_dict())
    if replay_record_sha != expected_record_sha:
        raise DARTCheckpointTrainingError(
            "selected epoch replay changed its validation prediction/health record"
        )
    payload = {
        "checkpoint_schema_version": SELECTED_CHECKPOINT_SCHEMA_VERSION,
        "namespace": "formal_dart_selected_common_epoch",
        "task": task,
        "model_name": model_name,
        "seed": int(seed),
        "selected_common_epoch": selected_epoch,
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": scaler.state_dict(),
        "dataloader_generator_state": generator.get_state(),
        "architecture": dict(seed_result["architecture"]),
        "batch_size": batch_size,
        "resume_fingerprint": seed_result["fingerprint"],
        "epoch_ledger_sha256": seed_result["ledger_sha256"],
        "selected_epoch_record": selected_record.as_dict(),
        "selected_epoch_record_sha256": expected_record_sha,
        "selection_manifest_sha256": selection_result[
            "selection_manifest_sha256"
        ],
        "seed_aggregation_manifest_sha256": aggregation_manifest[
            "manifest_sha256"
        ],
        "replay_anchor": {
            "path": anchor_path.as_posix(),
            "sha256": anchor_sha,
            "epoch": anchor_epoch,
        },
        "replay_prediction_record_bitwise_equal": True,
        "market_replay_or_action_metric_consumed": False,
        "oos_or_locked_target_consumed": False,
        **_capture_rng(generator),
        "written_at_utc": utc_now(),
    }
    record = _atomic_torch(payload, selected_path)
    return {
        **record,
        "seed": int(seed),
        "batch_size": batch_size,
        "architecture": dict(seed_result["architecture"]),
        "resume_fingerprint": seed_result["fingerprint"],
        "selected_epoch_record_sha256": expected_record_sha,
        "replay_prediction_record_bitwise_equal": True,
    }


def _load_group_manifest(path: Path, group_fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise DARTCheckpointTrainingError("DART checkpoint-group schema changed")
    if value.get("group_fingerprint") != group_fingerprint:
        raise DARTCheckpointTrainingError("DART checkpoint-group fingerprint changed")
    core = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != sha256_json(core):
        raise DARTCheckpointTrainingError("DART checkpoint-group manifest hash changed")
    if value.get("status") != "COMPLETE":
        return None
    for record in value.get("selected_checkpoints", {}).values():
        checkpoint = Path(str(record.get("path", "")))
        if not checkpoint.is_file() or sha256_file(checkpoint) != record.get("sha256"):
            raise DARTCheckpointTrainingError("selected DART checkpoint hash changed")
    return value


def _prepare_group_contract(
    *,
    task: str,
    model_name: str,
    study_seeds: Sequence[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_checkpoint_calibration: np.ndarray,
    y_checkpoint_calibration: np.ndarray,
    checkpoint_calibration_dates: Sequence[str],
    x_checkpoint_selection: np.ndarray,
    y_checkpoint_selection: np.ndarray,
    checkpoint_selection_dates: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    tail_threshold_contract: Mapping[str, Any],
    history: int,
    horizon: int,
    max_epochs: int,
    batch_sizes: Sequence[int],
    learning_rate: float,
    gradient_accumulation_steps: int,
    output_root: Path,
    resume_fingerprint: str,
    selection_config: CheckpointSelectionConfig,
    anchor_interval_epochs: int,
    minimum_epochs: int,
    evaluation_interval_epochs: int,
    stop_commit_interval_epochs: int,
    stability_evaluations: int,
    early_stopping_min_delta: float,
    weight_decay: float,
    clip_norm: float,
) -> dict[str, Any]:
    seeds = tuple(map(int, study_seeds))
    if seeds != STUDY_SEEDS:
        raise DARTCheckpointTrainingError(
            "formal DART checkpoint groups require the exact three study seeds"
        )
    if int(max_epochs) < 1 or int(anchor_interval_epochs) < 1:
        raise ValueError("DART max_epochs and anchor interval must be positive")
    if not 1 <= int(minimum_epochs) <= int(max_epochs):
        raise ValueError("minimum_epochs must lie inside the DART epoch grid")
    if int(evaluation_interval_epochs) < 1 or int(stability_evaluations) < 1:
        raise ValueError("group early-stopping intervals must be positive")
    if (
        int(stop_commit_interval_epochs) < int(evaluation_interval_epochs)
        or int(stop_commit_interval_epochs) % int(evaluation_interval_epochs) != 0
    ):
        raise ValueError(
            "group stop-commit interval must be a multiple of assessment interval"
        )
    if not math.isfinite(float(early_stopping_min_delta)) or float(
        early_stopping_min_delta
    ) < 0.0:
        raise ValueError("group early-stopping min_delta must be finite and nonnegative")
    if len(checkpoint_calibration_dates) != len(x_checkpoint_calibration):
        raise DARTCheckpointTrainingError("checkpoint-calibration date/tensor panel differs")
    if len(checkpoint_selection_dates) != len(x_checkpoint_selection):
        raise DARTCheckpointTrainingError("checkpoint-selection date/tensor panel differs")
    group_root = Path(output_root)
    forbidden_name_fragments = ("2025", "locked", "replay", "action")
    if any(fragment in str(group_root).lower() for fragment in forbidden_name_fragments):
        raise DARTCheckpointTrainingError(
            "checkpoint selection output namespace suggests forbidden evidence"
        )
    panel_contract = {
        "x_train": _array_sha256(x_train),
        "y_train": _array_sha256(y_train),
        "x_checkpoint_calibration": _array_sha256(x_checkpoint_calibration),
        "y_checkpoint_calibration": _array_sha256(y_checkpoint_calibration),
        "checkpoint_calibration_dates": sha256_json(
            list(map(str, checkpoint_calibration_dates))
        ),
        "x_checkpoint_selection": _array_sha256(x_checkpoint_selection),
        "y_checkpoint_selection": _array_sha256(y_checkpoint_selection),
        "checkpoint_selection_dates": sha256_json(
            list(map(str, checkpoint_selection_dates))
        ),
        "center": _array_sha256(center),
        "scale": _array_sha256(scale),
        "tail_threshold_contract_sha256": tail_threshold_contract.get(
            "threshold_contract_sha256"
        ),
    }
    early_stopping_policy = {
        "minimum_epochs": int(minimum_epochs),
        "evaluation_interval_epochs": int(evaluation_interval_epochs),
        "stop_commit_interval_epochs": int(stop_commit_interval_epochs),
        "stability_evaluations": int(stability_evaluations),
        "min_delta": float(early_stopping_min_delta),
        "individual_seed_early_stopping": False,
    }
    group_fingerprint = sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "parent": resume_fingerprint,
            "task": task,
            "model_name": model_name,
            "study_seeds": list(seeds),
            "history": int(history),
            "horizon": int(horizon),
            "max_epochs": int(max_epochs),
            "batch_sizes": list(map(int, batch_sizes)),
            "learning_rate": float(learning_rate),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "weight_decay": float(weight_decay),
            "clip_norm": float(clip_norm),
            "anchor_interval_epochs": int(anchor_interval_epochs),
            "synchronized_group_early_stopping": early_stopping_policy,
            "selection_config_fingerprint": selection_config.fingerprint,
            "panels": panel_contract,
            "module_sha256": sha256_file(Path(__file__)),
        }
    )
    return {
        "seeds": seeds,
        "group_root": group_root,
        "manifest_path": group_root / "checkpoint_group_manifest.json",
        "panel_contract": panel_contract,
        "early_stopping_policy": early_stopping_policy,
        "group_fingerprint": group_fingerprint,
    }


def fit_dart_energy_seed_group(
    *,
    task: str,
    model_name: str,
    requested_seed: int,
    study_seeds: Sequence[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_checkpoint_calibration: np.ndarray,
    y_checkpoint_calibration: np.ndarray,
    checkpoint_calibration_dates: Sequence[str],
    x_checkpoint_selection: np.ndarray,
    y_checkpoint_selection: np.ndarray,
    checkpoint_selection_dates: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    tail_threshold_contract: Mapping[str, Any],
    history: int,
    horizon: int,
    max_epochs: int,
    batch_sizes: Sequence[int],
    learning_rate: float,
    gradient_accumulation_steps: int,
    device: torch.device,
    output_root: Path,
    resume_fingerprint: str,
    selection_config: CheckpointSelectionConfig,
    anchor_interval_epochs: int = 5,
    minimum_epochs: int = 40,
    evaluation_interval_epochs: int = 5,
    stop_commit_interval_epochs: int = 10,
    stability_evaluations: int = 3,
    early_stopping_min_delta: float = 1.0e-8,
    weight_decay: float = 1.0e-4,
    clip_norm: float = 1.0,
) -> dict[str, Any]:
    """Train/resume three seeds and load the requested seed at one common epoch."""

    context = _prepare_group_contract(
        task=task,
        model_name=model_name,
        study_seeds=study_seeds,
        x_train=x_train,
        y_train=y_train,
        x_checkpoint_calibration=x_checkpoint_calibration,
        y_checkpoint_calibration=y_checkpoint_calibration,
        checkpoint_calibration_dates=checkpoint_calibration_dates,
        x_checkpoint_selection=x_checkpoint_selection,
        y_checkpoint_selection=y_checkpoint_selection,
        checkpoint_selection_dates=checkpoint_selection_dates,
        center=center,
        scale=scale,
        tail_threshold_contract=tail_threshold_contract,
        history=history,
        horizon=horizon,
        max_epochs=max_epochs,
        batch_sizes=batch_sizes,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_root=output_root,
        resume_fingerprint=resume_fingerprint,
        selection_config=selection_config,
        anchor_interval_epochs=anchor_interval_epochs,
        minimum_epochs=minimum_epochs,
        evaluation_interval_epochs=evaluation_interval_epochs,
        stop_commit_interval_epochs=stop_commit_interval_epochs,
        stability_evaluations=stability_evaluations,
        early_stopping_min_delta=early_stopping_min_delta,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
    )
    seeds = tuple(context["seeds"])
    if int(requested_seed) not in seeds:
        raise DARTCheckpointTrainingError(
            "formal DART checkpoint groups require the exact three study seeds"
        )
    panel_contract = dict(context["panel_contract"])
    group_fingerprint = str(context["group_fingerprint"])
    group_root = Path(context["group_root"])
    manifest_path = Path(context["manifest_path"])
    manifest = _load_group_manifest(manifest_path, group_fingerprint)
    if manifest is None:
        coordinator_path = group_root / "group_coordinator_state.json"
        if coordinator_path.is_file():
            prior_state = json.loads(coordinator_path.read_text(encoding="utf-8"))
            prior_core = {
                key: value
                for key, value in prior_state.items()
                if key != "state_sha256"
            }
            if (
                prior_state.get("group_fingerprint") != group_fingerprint
                or prior_state.get("state_sha256") != sha256_json(prior_core)
            ):
                raise DARTCheckpointTrainingError(
                    "synchronized group coordinator state changed"
                )
        assessment_epochs = list(
            range(
                int(evaluation_interval_epochs),
                int(max_epochs) + 1,
                int(evaluation_interval_epochs),
            )
        )
        if not assessment_epochs or assessment_epochs[-1] != int(max_epochs):
            assessment_epochs.append(int(max_epochs))
        assessments: list[dict[str, Any]] = []
        seed_results: dict[int, dict[str, Any]] = {}
        stop_decision: dict[str, Any] = {
            "stop": False,
            "reason": "MAX_EPOCH_NOT_YET_REACHED",
        }
        completed_common_epoch = 0
        for common_epoch in assessment_epochs:
            seed_results = {}
            for seed in seeds:
                seed_results[seed] = _fit_seed_ledger(
                    model_name=model_name,
                    task=task,
                    seed=seed,
                    x_train=np.asarray(x_train, dtype="float32"),
                    y_train=np.asarray(y_train, dtype="float32"),
                    x_checkpoint_calibration=np.asarray(
                        x_checkpoint_calibration, dtype="float32"
                    ),
                    y_checkpoint_calibration=np.asarray(
                        y_checkpoint_calibration, dtype="float64"
                    ),
                    checkpoint_calibration_dates=checkpoint_calibration_dates,
                    x_checkpoint_selection=np.asarray(
                        x_checkpoint_selection, dtype="float32"
                    ),
                    y_checkpoint_selection=np.asarray(
                        y_checkpoint_selection, dtype="float64"
                    ),
                    checkpoint_selection_dates=checkpoint_selection_dates,
                    center=np.asarray(center, dtype="float64"),
                    scale=np.asarray(scale, dtype="float64"),
                    tail_threshold_contract=tail_threshold_contract,
                    history=history,
                    horizon=horizon,
                    max_epochs=max_epochs,
                    train_through_epoch=common_epoch,
                    batch_sizes=batch_sizes,
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    weight_decay=weight_decay,
                    clip_norm=clip_norm,
                    anchor_interval_epochs=anchor_interval_epochs,
                    selection_config=selection_config,
                    device=device,
                    output_root=group_root / f"seed_{seed}",
                    parent_fingerprint=group_fingerprint,
                )
            epoch_vectors = {
                seed: tuple(record.epoch for record in seed_results[seed]["ledger"])
                for seed in seeds
            }
            exact_epoch_vector = tuple(range(1, int(common_epoch) + 1))
            aligned = all(
                epoch_vectors[seed] == exact_epoch_vector for seed in seeds
            )
            if not aligned:
                raise DARTCheckpointTrainingError(
                    "three seed ledgers are not aligned at the common epoch boundary"
                )
            completed_common_epoch = int(common_epoch)
            if common_epoch >= int(minimum_epochs):
                provisional_aggregated, provisional_aggregation = (
                    aggregate_seed_epoch_records(
                        {seed: seed_results[seed]["ledger"] for seed in seeds},
                        expected_seeds=seeds,
                    )
                )
                provisional_selection = select_checkpoint_epoch(
                    provisional_aggregated,
                    config=selection_config,
                    provisional=True,
                )
                assessments.append(
                    {
                        "assessment_epoch": int(common_epoch),
                        "study_seeds": list(seeds),
                        "aligned_exact_three_seed_ledgers": True,
                        "common_epoch_vector_sha256": sha256_json(
                            list(exact_epoch_vector)
                        ),
                        "seed_epoch_ledger_sha256": {
                            str(seed): seed_results[seed]["ledger_sha256"]
                            for seed in seeds
                        },
                        "provisional_aggregation_manifest_sha256": (
                            provisional_aggregation["manifest_sha256"]
                        ),
                        "selection_result": provisional_selection,
                    }
                )
                stop_decision = should_stop_group(
                    assessments,
                    minimum_epochs=minimum_epochs,
                    evaluation_interval_epochs=evaluation_interval_epochs,
                    consecutive_evaluations=stability_evaluations,
                    min_delta=early_stopping_min_delta,
                )
            commit_boundary = bool(
                common_epoch >= int(minimum_epochs)
                and (common_epoch - int(minimum_epochs))
                % int(stop_commit_interval_epochs)
                == 0
            )
            stop_decision = {
                **stop_decision,
                "decision_committed_only_at_group_boundary": True,
                "stop_commit_interval_epochs": int(
                    stop_commit_interval_epochs
                ),
                "current_epoch_is_commit_boundary": commit_boundary,
            }
            state_core = {
                "schema_version": SCHEMA_VERSION,
                "status": "SYNCHRONIZED_EPOCH_BOUNDARY",
                "group_fingerprint": group_fingerprint,
                "study_seeds": list(seeds),
                "completed_common_epoch": completed_common_epoch,
                "max_epochs": int(max_epochs),
                "seed_latest_checkpoints": {
                    str(seed): seed_results[seed]["latest_checkpoint"]
                    for seed in seeds
                },
                "seed_epoch_ledger_sha256": {
                    str(seed): seed_results[seed]["ledger_sha256"]
                    for seed in seeds
                },
                "resume_state_contract": {
                    "model": True,
                    "optimizer": True,
                    "amp_gradient_scaler": True,
                    "python_numpy_torch_cuda_rng": True,
                    "dataloader_generator_rng": True,
                    "epoch_health_and_hash": True,
                },
                "assessment_history": assessments,
                "stop_decision": stop_decision,
                "individual_seed_early_stopping": False,
                "market_replay_or_locked_test_consumed": False,
                "written_at_utc": utc_now(),
            }
            atomic_json(
                coordinator_path,
                {**state_core, "state_sha256": sha256_json(state_core)},
            )
            if bool(stop_decision.get("stop")) and commit_boundary:
                break
        aggregated, aggregation_manifest = aggregate_seed_epoch_records(
            {seed: seed_results[seed]["ledger"] for seed in seeds},
            expected_seeds=seeds,
        )
        selection_result = select_checkpoint_epoch(
            aggregated, config=selection_config, provisional=False
        )
        if selection_result.get("status") != "SELECTED":
            atomic_json(
                group_root / "checkpoint_group_failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": str(selection_result.get("status")),
                    "group_fingerprint": group_fingerprint,
                    "selection_result": selection_result,
                    "market_replay_or_locked_test_consumed": False,
                    "recorded_at_utc": utc_now(),
                },
            )
            raise DARTCheckpointSelectionError(
                f"no common DART checkpoint epoch: {selection_result.get('status')}"
            )
        selected_checkpoints: dict[str, dict[str, Any]] = {}
        for seed in seeds:
            selected_checkpoints[str(seed)] = _materialize_selected_checkpoint(
                seed_result=seed_results[seed],
                selection_result=selection_result,
                aggregation_manifest=aggregation_manifest,
                model_name=model_name,
                task=task,
                seed=seed,
                x_train=np.asarray(x_train, dtype="float32"),
                y_train=np.asarray(y_train, dtype="float32"),
                x_checkpoint_calibration=np.asarray(
                    x_checkpoint_calibration, dtype="float32"
                ),
                y_checkpoint_calibration=np.asarray(
                    y_checkpoint_calibration, dtype="float64"
                ),
                checkpoint_calibration_dates=checkpoint_calibration_dates,
                x_checkpoint_selection=np.asarray(
                    x_checkpoint_selection, dtype="float32"
                ),
                y_checkpoint_selection=np.asarray(
                    y_checkpoint_selection, dtype="float64"
                ),
                checkpoint_selection_dates=checkpoint_selection_dates,
                center=np.asarray(center, dtype="float64"),
                scale=np.asarray(scale, dtype="float64"),
                tail_threshold_contract=tail_threshold_contract,
                history=history,
                horizon=horizon,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=weight_decay,
                clip_norm=clip_norm,
                selection_config=selection_config,
                device=device,
                group_root=group_root,
            )
        seed_training = {
            str(seed): {
                "latest_checkpoint": seed_results[seed]["latest_checkpoint"],
                "epoch_ledger_sha256": seed_results[seed]["ledger_sha256"],
                "batch_size": seed_results[seed]["batch_size"],
                "resume_fingerprint": seed_results[seed]["fingerprint"],
                "resumed_from_epoch": seed_results[seed]["resumed_from_epoch"],
                "training_health": seed_results[seed]["training_health"],
                "trained_through_epoch": completed_common_epoch,
            }
            for seed in seeds
        }
        manifest_core = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "group_fingerprint": group_fingerprint,
            "task": task,
            "model_name": model_name,
            "study_seeds": list(seeds),
            "selected_common_epoch": int(selection_result["selected_epoch"]),
            "completed_common_epoch": completed_common_epoch,
            "maximum_epoch_budget": int(max_epochs),
            "selection_config_fingerprint": selection_config.fingerprint,
            "panel_contract": panel_contract,
            "seed_aggregation_manifest": aggregation_manifest,
            "selection_result": selection_result,
            "seed_training": seed_training,
            "selected_checkpoints": selected_checkpoints,
            "group_coordinator_state": {
                "path": coordinator_path.as_posix(),
                "sha256": sha256_file(coordinator_path),
            },
            "synchronized_early_stopping_assessments": assessments,
            "synchronized_early_stopping_decision": stop_decision,
            "all_seeds_completed_exact_common_epoch_panel": True,
            "all_seeds_exhausted_max_epoch_budget": (
                completed_common_epoch == int(max_epochs)
            ),
            "best_seed_selected": False,
            "individual_seed_early_stopping": False,
            "synchronized_group_early_stopping": bool(
                stop_decision.get("stop")
            ),
            "market_replay_or_action_metric_consumed": False,
            "oos_or_locked_target_consumed": False,
            "written_at_utc": utc_now(),
        }
        manifest = {**manifest_core, "manifest_sha256": sha256_json(manifest_core)}
        atomic_json(manifest_path, manifest)

    seed_record = manifest["selected_checkpoints"][str(int(requested_seed))]
    checkpoint_path = Path(str(seed_record["path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("checkpoint_schema_version")
        != SELECTED_CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("selected_common_epoch")
        != manifest["selected_common_epoch"]
        or int(checkpoint.get("seed", -1)) != int(requested_seed)
    ):
        raise DARTCheckpointTrainingError("selected DART checkpoint contract changed")
    model = _make_model(
        model_name,
        history=history,
        horizon=horizon,
        site_count=int(y_train.shape[1]),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    selected_record = checkpoint["selected_epoch_record"]
    return {
        "model": model,
        "selected_epoch": int(manifest["selected_common_epoch"]),
        "best_epoch": int(manifest["selected_common_epoch"]),
        "best_metric": float(np.mean(selected_record["daily_global_crps"])),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epochs_written": int(manifest["completed_common_epoch"]),
        "completed_common_epoch": int(manifest["completed_common_epoch"]),
        "resumed_from_epoch": int(
            manifest["seed_training"][str(int(requested_seed))][
                "resumed_from_epoch"
            ]
        ),
        "batch_size": int(seed_record["batch_size"]),
        "resume_fingerprint": str(seed_record["resume_fingerprint"]),
        "architecture": dict(seed_record["architecture"]),
        "training_health": dict(
            manifest["seed_training"][str(int(requested_seed))]["training_health"]
        ),
        "checkpoint_group_manifest_path": manifest_path,
        "checkpoint_group_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_selection": dict(manifest["selection_result"]),
        "seed_aggregation": dict(manifest["seed_aggregation_manifest"]),
        "selected_epoch_record": dict(selected_record),
        "synchronized_early_stopping_decision": dict(
            manifest["synchronized_early_stopping_decision"]
        ),
    }


def _context_from_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return _prepare_group_contract(
        task=str(job["task"]),
        model_name=str(job["model_name"]),
        study_seeds=job["study_seeds"],
        x_train=job["x_train"],
        y_train=job["y_train"],
        x_checkpoint_calibration=job["x_checkpoint_calibration"],
        y_checkpoint_calibration=job["y_checkpoint_calibration"],
        checkpoint_calibration_dates=job["checkpoint_calibration_dates"],
        x_checkpoint_selection=job["x_checkpoint_selection"],
        y_checkpoint_selection=job["y_checkpoint_selection"],
        checkpoint_selection_dates=job["checkpoint_selection_dates"],
        center=job["center"],
        scale=job["scale"],
        tail_threshold_contract=job["tail_threshold_contract"],
        history=int(job["history"]),
        horizon=int(job["horizon"]),
        max_epochs=int(job["max_epochs"]),
        batch_sizes=job["batch_sizes"],
        learning_rate=float(job["learning_rate"]),
        gradient_accumulation_steps=int(job["gradient_accumulation_steps"]),
        output_root=Path(job["output_root"]),
        resume_fingerprint=str(job["resume_fingerprint"]),
        selection_config=job["selection_config"],
        anchor_interval_epochs=int(job.get("anchor_interval_epochs", 5)),
        minimum_epochs=int(job.get("minimum_epochs", 40)),
        evaluation_interval_epochs=int(
            job.get("evaluation_interval_epochs", 5)
        ),
        stop_commit_interval_epochs=int(
            job.get("stop_commit_interval_epochs", 10)
        ),
        stability_evaluations=int(job.get("stability_evaluations", 3)),
        early_stopping_min_delta=float(
            job.get("early_stopping_min_delta", 1.0e-8)
        ),
        weight_decay=float(job.get("weight_decay", 1.0e-4)),
        clip_norm=float(job.get("clip_norm", 1.0)),
    )


def _advance_job_seed(
    job: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    seed: int,
    common_epoch: int,
) -> dict[str, Any]:
    return _fit_seed_ledger(
        model_name=str(job["model_name"]),
        task=str(job["task"]),
        seed=int(seed),
        x_train=np.asarray(job["x_train"], dtype="float32"),
        y_train=np.asarray(job["y_train"], dtype="float32"),
        x_checkpoint_calibration=np.asarray(
            job["x_checkpoint_calibration"], dtype="float32"
        ),
        y_checkpoint_calibration=np.asarray(
            job["y_checkpoint_calibration"], dtype="float64"
        ),
        checkpoint_calibration_dates=job["checkpoint_calibration_dates"],
        x_checkpoint_selection=np.asarray(
            job["x_checkpoint_selection"], dtype="float32"
        ),
        y_checkpoint_selection=np.asarray(
            job["y_checkpoint_selection"], dtype="float64"
        ),
        checkpoint_selection_dates=job["checkpoint_selection_dates"],
        center=np.asarray(job["center"], dtype="float64"),
        scale=np.asarray(job["scale"], dtype="float64"),
        tail_threshold_contract=job["tail_threshold_contract"],
        history=int(job["history"]),
        horizon=int(job["horizon"]),
        max_epochs=int(job["max_epochs"]),
        train_through_epoch=int(common_epoch),
        batch_sizes=job["batch_sizes"],
        learning_rate=float(job["learning_rate"]),
        gradient_accumulation_steps=int(job["gradient_accumulation_steps"]),
        weight_decay=float(job.get("weight_decay", 1.0e-4)),
        clip_norm=float(job.get("clip_norm", 1.0)),
        anchor_interval_epochs=int(job.get("anchor_interval_epochs", 5)),
        selection_config=job["selection_config"],
        device=job["device"],
        output_root=Path(context["group_root"]) / f"seed_{int(seed)}",
        parent_fingerprint=str(context["group_fingerprint"]),
    )


def pretrain_dart_energy_groups_hybrid(
    jobs: Sequence[Mapping[str, Any]],
    *,
    scheduler_root: Path,
) -> dict[str, Any]:
    """Pretrain neural groups with an efficient synchronized hybrid schedule.

    Each group advances all three seeds continuously to epoch 40 before the
    next group is loaded.  After every group reaches 40, active groups advance
    in ten-epoch global rounds (50, 60, ..., 100).  Five-epoch assessment
    records are sliced from the complete per-epoch ledgers, but stopping is
    committed only at a global round boundary.
    """

    ordered_jobs = tuple(dict(job) for job in jobs)
    if not ordered_jobs:
        raise DARTCheckpointTrainingError(
            "hybrid DART pretraining requires at least one neural group"
        )
    contexts = tuple(_context_from_job(job) for job in ordered_jobs)
    fingerprints = tuple(str(value["group_fingerprint"]) for value in contexts)
    if len(set(fingerprints)) != len(fingerprints):
        raise DARTCheckpointTrainingError(
            "hybrid DART scheduler received a duplicate neural group"
        )
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
        raise DARTCheckpointTrainingError(
            "all hybrid DART groups must share one seed/resource policy"
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
    if tuple(seeds) != STUDY_SEEDS:
        raise DARTCheckpointTrainingError("hybrid scheduler seed contract changed")
    scheduler = Path(scheduler_root)
    state_path = scheduler / "hybrid_scheduler_state.json"
    manifest_path = scheduler / "hybrid_scheduler_manifest.json"
    scheduler_fingerprint = sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "order": "warmup_to_40_then_10_epoch_active_group_rounds",
            "group_fingerprints_in_order": list(fingerprints),
            "study_seeds": list(seeds),
            "max_epochs": max_epochs,
            "minimum_epochs": minimum_epochs,
            "evaluation_interval_epochs": interval,
            "global_round_interval_epochs": global_round_interval,
            "stability_evaluations": stability,
            "early_stopping_min_delta": min_delta,
            "module_sha256": sha256_file(Path(__file__)),
        }
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if (
            manifest.get("scheduler_fingerprint") != scheduler_fingerprint
            or manifest.get("manifest_sha256") != sha256_json(core)
            or manifest.get("status") != "COMPLETE"
        ):
            raise DARTCheckpointTrainingError(
                "hybrid DART scheduler manifest changed"
            )
        return manifest

    events: list[dict[str, Any]] = []
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        core = {key: value for key, value in state.items() if key != "state_sha256"}
        if (
            state.get("scheduler_fingerprint") != scheduler_fingerprint
            or state.get("state_sha256") != sha256_json(core)
        ):
            raise DARTCheckpointTrainingError(
                "hybrid DART scheduler resume state changed"
            )
        events = list(state.get("execution_events", ()))
    else:
        preexisting = []
        for context in contexts:
            group_root = Path(context["group_root"])
            if group_root.exists() and any(group_root.rglob("latest.pt")):
                preexisting.append(group_root.as_posix())
            if Path(context["manifest_path"]).is_file():
                preexisting.append(Path(context["manifest_path"]).as_posix())
        if preexisting:
            raise DARTCheckpointTrainingError(
                "unattributed pre-scheduler neural checkpoints prevent a truthful "
                f"hybrid-schedule claim: {preexisting[:3]}"
            )
        initial_core = {
            "schema_version": SCHEMA_VERSION,
            "status": "INITIALIZED",
            "scheduler_fingerprint": scheduler_fingerprint,
            "declared_order": "warmup_to_40_then_10_epoch_active_group_rounds",
            "group_fingerprints_in_order": list(fingerprints),
            "execution_events": [],
            "individual_seed_early_stopping": False,
            "market_replay_or_locked_test_consumed": False,
            "written_at_utc": utc_now(),
        }
        atomic_json(
            state_path,
            {**initial_core, "state_sha256": sha256_json(initial_core)},
        )

    active = set(fingerprints)
    assessments: dict[str, list[dict[str, Any]]] = {
        fingerprint: [] for fingerprint in fingerprints
    }
    final_group_records: dict[str, dict[str, Any]] = {}
    seed_results_by_group: dict[str, dict[int, dict[str, Any]]] = {
        fingerprint: {} for fingerprint in fingerprints
    }
    global_round_epochs = [minimum_epochs]
    global_round_epochs.extend(
        range(
            minimum_epochs + global_round_interval,
            max_epochs + 1,
            global_round_interval,
        )
    )
    if global_round_epochs[-1] != max_epochs:
        global_round_epochs.append(max_epochs)
    last_common_epoch = 0

    def persist_state(status: str, common_epoch: int) -> None:
        state_core = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "scheduler_fingerprint": scheduler_fingerprint,
            "declared_order": "warmup_to_40_then_10_epoch_active_group_rounds",
            "group_fingerprints_in_order": list(fingerprints),
            "study_seeds": list(seeds),
            "common_epoch_boundary": int(common_epoch),
            "active_group_fingerprints": sorted(active),
            "completed_group_fingerprints": sorted(final_group_records),
            "execution_events": events,
            "execution_events_sha256": sha256_json(events),
            "individual_seed_early_stopping": False,
            "market_replay_or_locked_test_consumed": False,
            "written_at_utc": utc_now(),
        }
        atomic_json(
            state_path,
            {**state_core, "state_sha256": sha256_json(state_core)},
        )

    previous_boundary = 0
    for common_epoch in global_round_epochs:
        if not active:
            break
        last_common_epoch = int(common_epoch)
        # Efficiency contract: load one group, advance all three seeds to the
        # shared boundary, then release it before loading the next group.
        for job, context in zip(ordered_jobs, contexts, strict=True):
            fingerprint = str(context["group_fingerprint"])
            if fingerprint not in active:
                continue
            for seed in seeds:
                result = _advance_job_seed(
                    job, context, seed=int(seed), common_epoch=int(common_epoch)
                )
                seed_results_by_group[fingerprint][int(seed)] = result
                events.append(
                    {
                        "event_index": len(events),
                        "common_epoch": int(common_epoch),
                        "seed": int(seed),
                        "group_fingerprint": fingerprint,
                        "task": str(job["task"]),
                        "model_name": str(job["model_name"]),
                        "output_root": Path(job["output_root"]).as_posix(),
                        "resumed_from_epoch": int(result["resumed_from_epoch"]),
                        "trained_through_epoch": int(result["trained_through_epoch"]),
                        "training_performed_in_this_advance": (
                            int(result["resumed_from_epoch"]) < int(common_epoch)
                        ),
                        "epoch_ledger_sha256": str(result["ledger_sha256"]),
                        "latest_checkpoint_sha256": str(
                            result["latest_checkpoint"]["sha256"]
                        ),
                    }
                )
        persist_state("ADVANCING_HYBRID_GLOBAL_ROUND", common_epoch)

        completed_now: list[str] = []
        assessment_points = (
            [minimum_epochs]
            if common_epoch == minimum_epochs
            else list(
                range(
                    max(minimum_epochs, previous_boundary + interval),
                    common_epoch + 1,
                    interval,
                )
            )
        )
        for job, context in zip(ordered_jobs, contexts, strict=True):
            fingerprint = str(context["group_fingerprint"])
            if fingerprint not in active:
                continue
            seed_results = seed_results_by_group[fingerprint]
            exact = tuple(range(1, int(common_epoch) + 1))
            if set(seed_results) != set(seeds) or any(
                tuple(record.epoch for record in seed_results[seed]["ledger"])
                != exact
                for seed in seeds
            ):
                raise DARTCheckpointTrainingError(
                    "hybrid group did not reach an exact aligned three-seed panel"
                )
            decision: dict[str, Any] = {
                "stop": False,
                "reason": "GLOBAL_ROUND_STABILITY_NOT_REACHED",
            }
            for assessment_epoch in assessment_points:
                prefix_by_seed = {
                    seed: tuple(
                        record
                        for record in seed_results[seed]["ledger"]
                        if record.epoch <= int(assessment_epoch)
                    )
                    for seed in seeds
                }
                if any(
                    tuple(record.epoch for record in prefix_by_seed[seed])
                    != tuple(range(1, int(assessment_epoch) + 1))
                    for seed in seeds
                ):
                    raise DARTCheckpointTrainingError(
                        "hybrid scheduler cannot slice an exact five-epoch assessment panel"
                    )
                aggregated, aggregation = aggregate_seed_epoch_records(
                    prefix_by_seed,
                    expected_seeds=seeds,
                )
                provisional = select_checkpoint_epoch(
                    aggregated,
                    config=job["selection_config"],
                    provisional=True,
                )
                assessments[fingerprint].append(
                    {
                        "assessment_epoch": int(assessment_epoch),
                        "study_seeds": list(seeds),
                        "aligned_exact_three_seed_ledgers": True,
                        "seed_epoch_ledger_sha256": {
                            str(seed): sha256_json(
                                [record.as_dict() for record in prefix_by_seed[seed]]
                            )
                            for seed in seeds
                        },
                        "provisional_aggregation_manifest_sha256": aggregation[
                            "manifest_sha256"
                        ],
                        "selection_result": provisional,
                        "evaluated_inside_global_round_ending_epoch": int(
                            common_epoch
                        ),
                    }
                )
            decision = should_stop_group(
                assessments[fingerprint],
                minimum_epochs=minimum_epochs,
                evaluation_interval_epochs=interval,
                consecutive_evaluations=stability,
                min_delta=min_delta,
            )
            decision = {
                **decision,
                "decision_committed_only_at_global_round_boundary": True,
                "global_round_boundary_epoch": int(common_epoch),
            }
            if bool(decision.get("stop")) or common_epoch == max_epochs:
                final = fit_dart_energy_seed_group(**job)
                final_group_records[fingerprint] = {
                    "group_fingerprint": fingerprint,
                    "task": str(job["task"]),
                    "model_name": str(job["model_name"]),
                    "output_root": Path(job["output_root"]).as_posix(),
                    "completed_common_epoch": int(final["completed_common_epoch"]),
                    "selected_common_epoch": int(final["selected_epoch"]),
                    "checkpoint_group_manifest_path": Path(
                        final["checkpoint_group_manifest_path"]
                    ).as_posix(),
                    "checkpoint_group_manifest_sha256": str(
                        final["checkpoint_group_manifest_sha256"]
                    ),
                    "scheduler_stop_decision": decision,
                    "assessment_history_sha256": sha256_json(
                        assessments[fingerprint]
                    ),
                }
                completed_now.append(fingerprint)
        active.difference_update(completed_now)
        persist_state("COMMON_EPOCH_ASSESSED", common_epoch)
        previous_boundary = int(common_epoch)

    if active:
        raise DARTCheckpointTrainingError(
            "hybrid DART scheduler exhausted epochs with active groups"
        )
    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scheduler_fingerprint": scheduler_fingerprint,
        "actual_training_order": "warmup_to_40_then_10_epoch_active_group_rounds",
        "efficient_hybrid_group_scheduler": True,
        "seed_major_cross_model_round_robin": False,
        "warmup_common_epoch": int(minimum_epochs),
        "active_group_round_epochs": list(map(int, global_round_epochs[1:])),
        "assessment_interval_epochs": int(interval),
        "stop_committed_only_at_global_round_boundary": True,
        "study_seeds": list(seeds),
        "group_fingerprints_in_order": list(fingerprints),
        "group_count": len(fingerprints),
        "completed_groups": final_group_records,
        "execution_events": events,
        "execution_events_sha256": sha256_json(events),
        "individual_seed_early_stopping": False,
        "market_replay_or_locked_test_consumed": False,
        "written_at_utc": utc_now(),
    }
    manifest = {**manifest_core, "manifest_sha256": sha256_json(manifest_core)}
    atomic_json(manifest_path, manifest)
    persist_state("COMPLETE", last_common_epoch)
    return manifest


def pretrain_dart_energy_groups_round_robin(
    jobs: Sequence[Mapping[str, Any]],
    *,
    scheduler_root: Path,
) -> dict[str, Any]:
    """Compatibility alias for the superseding efficient hybrid scheduler."""

    return pretrain_dart_energy_groups_hybrid(
        jobs, scheduler_root=scheduler_root
    )


__all__ = [
    "DARTCheckpointTrainingError",
    "SCHEMA_VERSION",
    "SELECTED_CHECKPOINT_SCHEMA_VERSION",
    "fit_dart_energy_seed_group",
    "pretrain_dart_energy_groups_hybrid",
    "pretrain_dart_energy_groups_round_robin",
    "should_stop_group",
]
