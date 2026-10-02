"""Narrow AMP-overflow compatibility for the pre-OOS DART v4 resume.

The frozen DART trainer treats every non-finite gradient norm as a fatal
failure.  CUDA AMP, however, deliberately represents a detected overflow as
non-finite unscaled gradients and expects ``GradScaler.step`` to skip that
optimizer update before lowering the scale.  This module applies precisely
that documented behavior while retaining fail-closed handling for any
non-finite gradient not reported by GradScaler.

It does not change the loss, learning rate, data, model, optimizer, checkpoint
schema, epoch-selection score, or stopping policy.  Epochs containing an AMP
overflow retain ``nonfinite_gradient_steps > 0`` and are therefore ineligible
for checkpoint selection under the frozen health gate.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


FROZEN_TRAINER_SHA256 = (
    "8e43fc4f3c61be2aca7f8ec67ffc821a909aef50b23626955ca9caa2ae2d10a7"
)
POLICY_ID = "cuda_amp_detected_overflow_skip_step_v1"


class DARTAMPOverflowCompatibilityError(RuntimeError):
    """Raised when the narrow compatibility contract is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _found_inf(scaler: torch.amp.GradScaler, optimizer: Any) -> bool:
    if not scaler.is_enabled():
        return False
    values = scaler._found_inf_per_device(optimizer)  # noqa: SLF001
    return any(float(value.detach().float().cpu()) > 0.0 for value in values.values())


@contextmanager
def amp_overflow_compatibility(
    *,
    ledger_path: Path,
) -> Iterator[dict[str, Any]]:
    """Temporarily install the exact GradScaler overflow skip behavior."""

    from dcvpp.experiments import dart_checkpoint_training as frozen

    core_path = Path(frozen.__file__).resolve()
    core_sha256 = _sha256_file(core_path)
    if core_sha256 != FROZEN_TRAINER_SHA256:
        raise DARTAMPOverflowCompatibilityError(
            f"frozen DART trainer SHA changed: {core_sha256}"
        )

    destination = Path(ledger_path).resolve()
    original = frozen._train_epoch
    events: list[dict[str, Any]] = []
    invocation_started = _utc_now()

    def train_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        loader: Any,
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
                raw_loss = frozen._pinball_loss(
                    prediction, y_batch.to(device, non_blocking=True)
                )
                if not bool(torch.isfinite(raw_loss).detach().cpu()):
                    raise frozen.DARTCheckpointTrainingError(
                        "DART training loss is non-finite"
                    )
                loss = raw_loss / int(gradient_accumulation_steps)
            losses.append(float(raw_loss.detach().float().cpu()))
            scaler.scale(loss).backward()
            if (
                batch_index % int(gradient_accumulation_steps) == 0
                or batch_index == len(loader)
            ):
                scale_before = float(scaler.get_scale())
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(clip_norm)
                )
                gradient_value = float(gradient_norm.detach().float().cpu())
                if not math.isfinite(gradient_value):
                    nonfinite_gradient_steps += 1
                    if not _found_inf(scaler, optimizer):
                        raise frozen.DARTCheckpointTrainingError(
                            "DART gradient norm is non-finite and not an AMP overflow"
                        )
                    # GradScaler.step intentionally skips the optimizer update
                    # when its per-device found_inf flag is set.
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    optimizer.zero_grad(set_to_none=True)
                    events.append(
                        {
                            "recorded_at_utc": _utc_now(),
                            "policy_id": POLICY_ID,
                            "model_class": type(model).__name__,
                            "batch_index": int(batch_index),
                            "batch_count": int(len(loader)),
                            "loss_was_finite": True,
                            "gradient_norm_was_nonfinite": True,
                            "grad_scaler_found_inf": True,
                            "optimizer_step_skipped_by_grad_scaler": True,
                            "scale_before": scale_before,
                            "scale_after": scale_after,
                        }
                    )
                    continue
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_value)
                if gradient_value > float(clip_norm):
                    gradient_clipped_steps += 1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if not losses:
            raise frozen.DARTCheckpointTrainingError(
                "DART training loader produced no batches"
            )
        return {
            "training_loss": float(np.mean(losses)),
            "maximum_gradient_norm": float(maximum_gradient_norm),
            "nonfinite_gradient_steps": int(nonfinite_gradient_steps),
            "gradient_clipped_steps": int(gradient_clipped_steps),
        }

    record: dict[str, Any] = {
        "schema_version": "dcvpp.dart_amp_overflow_compatibility.v1",
        "status": "ACTIVE",
        "policy_id": POLICY_ID,
        "frozen_trainer_path": core_path.as_posix(),
        "frozen_trainer_sha256": core_sha256,
        "installed_at_utc": invocation_started,
        "scope": "paper-dart-forecast-v4 pre-OOS neural resume only",
        "allowed_change": "skip only GradScaler-confirmed overflow optimizer steps",
        "loss_model_data_optimizer_lr_and_selection_policy_changed": False,
        "unrecognized_nonfinite_gradient_remains_fatal": True,
        "overflow_epoch_selection_eligible": False,
        "events": events,
    }
    _atomic_json(destination, record)
    frozen._train_epoch = train_epoch
    try:
        yield record
    finally:
        frozen._train_epoch = original
        record["status"] = "CLOSED"
        record["closed_at_utc"] = _utc_now()
        record["event_count"] = len(events)
        _atomic_json(destination, record)


__all__ = [
    "DARTAMPOverflowCompatibilityError",
    "FROZEN_TRAINER_SHA256",
    "POLICY_ID",
    "amp_overflow_compatibility",
]
