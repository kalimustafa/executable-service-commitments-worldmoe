"""Atomic, resumable development runner for the R4-V8 TriFusion model."""

from __future__ import annotations

import inspect
import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from dcvpp.e2e_moe_r4.hashing import sha256_file, sha256_json
from dcvpp.e2e_moe_r4.physical_twin import load_frozen_physical_twin_core

from .config import ARCHITECTURE_VERSION, PHASE_ORDER, load_v8_config
from .corpus import PreloadedV8Corpus, V8ResidualAnchor
from .losses import representation_loss
from .model import TriPlaneCausalWorldModelV8
from .preflight import V8PreflightResult, assert_development_dates, prepare_v8
from .trainer import (
    V8TriFusionTrainer,
    contract_from_config,
    coverage_batches,
)


RUNNER_SCHEMA = "dcvpp.r4.v8.atomic-runner.v1"
CHECKPOINT_SCHEMA = "dcvpp.r4.v8.atomic-checkpoint.v1"
INITIALIZATION_SCHEMA = "dcvpp.r4.v8.legacy-representation-warm-start.v1"
REPAIR_INITIALIZATION_SCHEMA = "dcvpp.r4.v8.1.failed-r0-head-repair-warm-start.v1"
RolloutFn = Callable[..., Mapping[str, Any]]


class V8RunnerError(RuntimeError):
    """The requested execution cannot preserve the registered V8 contract."""


@dataclass(frozen=True)
class ResumeState:
    cycles_completed: int = 0
    optimizer_updates_completed: int = 0
    best_validation: float = float("inf")
    stale_validation_checks: int = 0
    best_cycle: int | None = None


@dataclass(frozen=True)
class RunnerComponents:
    preflight: V8PreflightResult
    corpus: Any
    model: nn.Module
    initialization_report: Mapping[str, Any]
    device: torch.device


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


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_jsonl_append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    line = json.dumps(dict(payload), sort_keys=True, default=str)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(existing + line + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _call_supported(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(callback)
    accepts_kwargs = any(
        value.kind == inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    )
    selected = kwargs if accepts_kwargs else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return callback(*args, **selected)


def _batch_dates(batch: Mapping[str, Any]) -> tuple[str, ...]:
    raw = batch.get("operating_dates", batch.get("operating_date"))
    if isinstance(raw, str):
        values: Sequence[Any] = (raw,)
    elif isinstance(raw, Sequence):
        values = raw
    else:
        raise V8RunnerError("training batch has no explicit operating-date axis")
    return assert_development_dates(values, label="training batch")


def adapt_training_batch(
    batch: Mapping[str, Any],
    *,
    anchor: V8ResidualAnchor,
    corpus: Any,
) -> dict[str, Any]:
    """Install the exact target and scale aliases expected by V8 losses."""

    result = dict(batch)
    _batch_dates(result)
    targets = result.get("targets")
    if not isinstance(result.get("representation_targets"), Mapping):
        if not isinstance(targets, Mapping):
            raise V8RunnerError("batch.targets is required for Representation")
        result["representation_targets"] = targets
    distribution_target = result.get("distribution_target")
    if distribution_target is None:
        distribution_target = result.get("scenario_target_zonal")
    if not isinstance(distribution_target, Tensor):
        raise V8RunnerError("batch.scenario_target_zonal is required for Distribution")
    result["distribution_target"] = distribution_target
    scale = anchor.channel_scale.to(
        device=distribution_target.device, dtype=torch.float32
    )
    if scale.shape != (21,) or bool((scale <= 0).any()):
        raise V8RunnerError("V8 anchor channel scale must contain 21 positive values")
    result["channel_scale"] = scale
    result["representation_scales"] = {
        "da": scale[0:4],
        "rt": scale[4:8],
        "dart": scale[8:12],
        "workload": scale[12:13],
        "pv": scale[13:17],
        "pue": scale[17:21],
    }
    rollout_targets = result.get("rollout_targets")
    if not isinstance(rollout_targets, Mapping):
        raise V8RunnerError("batch.rollout_targets is required for Policy")
    baseline_actions = rollout_targets.get("baseline_actions")
    if not isinstance(baseline_actions, Mapping):
        raise V8RunnerError("rollout_targets.baseline_actions is required")
    result["safe_anchor_actions"] = dict(baseline_actions)
    cost_scale = getattr(corpus, "fixed_training_cost_scale", None)
    if cost_scale is None or not np.isfinite(float(cost_scale)) or float(cost_scale) <= 0.0:
        raise V8RunnerError("corpus fixed_training_cost_scale is invalid")
    result["cost_scale"] = float(cost_scale)
    return result


def legacy_representation_warm_start(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load only shape-compatible legacy modality encoders.

    This is explicitly not a V7 checkpoint upgrade.  Optimizer and scheduler
    state are ignored, and the calibrated physical core is never replaced.
    """

    source = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise V8RunnerError(f"legacy Representation checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise V8RunnerError("legacy Representation checkpoint is malformed")
    if str(payload.get("phase", "representation")).lower() != "representation":
        raise V8RunnerError("legacy warm-start source is not a Representation checkpoint")
    if payload.get("capability_gate_status") != "PASSED":
        raise V8RunnerError("legacy Representation checkpoint did not pass its gate")
    if payload.get("checkpoint_role") != "capability_passed_selected_best":
        raise V8RunnerError(
            "legacy Representation checkpoint is not the selected gate-passed best"
        )
    raw_state = payload.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise V8RunnerError("legacy Representation checkpoint has no model_state")
    allowed_prefixes = (
        "market_encoder.",
        "workload_encoder.",
        "physical_stream.",
        "state_encoder.",
    )
    frozen_prefix = "physical_stream.physical_twin_core."
    current = model.state_dict()
    loaded: dict[str, Tensor] = {}
    skipped_shape: list[str] = []
    skipped_unknown: list[str] = []
    skipped_scope: list[str] = []
    for raw_name, raw_value in raw_state.items():
        name = str(raw_name)
        if name.startswith(frozen_prefix) or not name.startswith(allowed_prefixes):
            skipped_scope.append(name)
            continue
        if name not in current:
            skipped_unknown.append(name)
            continue
        tensor = torch.as_tensor(raw_value)
        if tensor.shape != current[name].shape:
            skipped_shape.append(name)
            continue
        loaded[name] = tensor
    if not loaded:
        raise V8RunnerError("legacy Representation checkpoint has no compatible encoder keys")
    model.load_state_dict(loaded, strict=False)
    physical = getattr(model, "physical_stream", None)
    if physical is not None and callable(getattr(physical, "freeze_twin", None)):
        physical.freeze_twin()
    report = {
        "schema": INITIALIZATION_SCHEMA,
        "mode": "legacy_representation_encoder_warm_start",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "source_architecture_version": payload.get("architecture_version"),
        "source_selected_epoch": payload.get(
            "selected_epoch", payload.get("epochs_completed", payload.get("epoch"))
        ),
        "target_architecture_version": ARCHITECTURE_VERSION,
        "loaded_key_count": len(loaded),
        "loaded_keys": sorted(loaded),
        "skipped_out_of_scope_count": len(skipped_scope),
        "skipped_unknown_count": len(skipped_unknown),
        "skipped_shape_count": len(skipped_shape),
        "skipped_unknown_keys": sorted(skipped_unknown),
        "skipped_shape_keys": sorted(skipped_shape),
        "physical_twin_loaded": False,
        "optimizer_loaded": False,
        "scheduler_loaded": False,
        "v7_upgrade_claimed": False,
        "created_at_utc": _utc_now(),
        "locked_2025_opened": False,
    }
    if report_path is not None:
        _atomic_json(Path(report_path).resolve(), report)
    return report


def failed_r0_repair_warm_start(
    model: nn.Module,
    checkpoint_path: str | Path,
    gate_path: str | Path,
    *,
    seed: int,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Start a new head-repair lineage from a measured failed R0 parent.

    The complete compatible neural state is copied, except for the frozen
    physical core and the newly introduced ``representation.repair`` modules.
    Optimizer, scheduler, RNG and capability status are never inherited.
    """

    source = Path(checkpoint_path).resolve()
    gate_source = Path(gate_path).resolve()
    if not source.is_file() or not gate_source.is_file():
        raise V8RunnerError("R0 repair parent checkpoint/gate is missing")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise V8RunnerError("R0 repair parent checkpoint is malformed")
    if (
        payload.get("architecture_version") != ARCHITECTURE_VERSION
        or str(payload.get("phase", "")).lower() != "r0"
        or int(payload.get("seed", -1)) != int(seed)
        or payload.get("locked_2025_opened") is not False
    ):
        raise V8RunnerError("R0 repair parent checkpoint lineage is incompatible")
    try:
        gate = json.loads(gate_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V8RunnerError("R0 repair parent gate is malformed") from exc
    if not isinstance(gate, Mapping):
        raise V8RunnerError("R0 repair parent gate must contain a JSON object")
    checkpoint_sha = sha256_file(source)
    if (
        str(gate.get("status", "")).upper() != "FAIL"
        or gate.get("passed") is not False
        or gate.get("next_phase_released") is not False
        or gate.get("selected_checkpoint") is not None
        or gate.get("locked_2025_opened") is not False
        or str(gate.get("checkpoint_id", "")) != checkpoint_sha
    ):
        raise V8RunnerError("R0 repair requires the exact measured failed parent gate")
    raw_state = payload.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise V8RunnerError("R0 repair parent checkpoint has no model state")
    current = model.state_dict()
    merged = dict(current)
    copied: list[str] = []
    fresh: list[str] = []
    frozen_prefix = "physical_stream.physical_twin_core."
    for name, target in current.items():
        if name.startswith("representation.repair."):
            fresh.append(name)
            continue
        if name.startswith(frozen_prefix):
            continue
        value = raw_state.get(name)
        if not isinstance(value, Tensor) or value.shape != target.shape:
            raise V8RunnerError(f"R0 repair parent misses compatible tensor {name}")
        merged[name] = value.to(dtype=target.dtype)
        copied.append(name)
    unknown = sorted(
        str(name)
        for name in raw_state
        if str(name) not in current and not str(name).startswith(frozen_prefix)
    )
    if unknown:
        raise V8RunnerError(
            "R0 repair parent contains unexpected model keys: " + ", ".join(unknown)
        )
    if not fresh:
        raise V8RunnerError("R0 repair model did not create fresh repair parameters")
    model.load_state_dict(merged, strict=True)
    physical = getattr(model, "physical_stream", None)
    if physical is not None and callable(getattr(physical, "freeze_twin", None)):
        physical.freeze_twin()
    reasons = tuple(str(value) for value in gate.get("scientific_gate", {}).get("reasons", ()))
    report = {
        "schema": REPAIR_INITIALIZATION_SCHEMA,
        "mode": "failed_r0_head_repair_new_lineage",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_gate": str(gate_source),
        "source_gate_status": "FAIL",
        "source_failure_reasons": reasons,
        "source_cycle": payload.get("cycles_completed"),
        "source_seed": int(seed),
        "target_architecture_version": ARCHITECTURE_VERSION,
        "target_method_revision": "v8_1_r0repair1",
        "copied_key_count": len(copied),
        "copied_keys": sorted(copied),
        "fresh_key_count": len(fresh),
        "fresh_keys": sorted(fresh),
        "physical_twin_loaded": False,
        "optimizer_loaded": False,
        "scheduler_loaded": False,
        "rng_loaded": False,
        "capability_status_inherited": False,
        "performance_inherited": False,
        "created_at_utc": _utc_now(),
        "locked_2025_opened": False,
    }
    if report_path is not None:
        _atomic_json(Path(report_path).resolve(), report)
    return report


def _physical_core(config: Mapping[str, Any], root: Path) -> nn.Module:
    data = config["data"]

    def resolve(name: str) -> Path:
        path = Path(str(data[name]))
        return (path if path.is_absolute() else root / path).resolve()

    return load_frozen_physical_twin_core(
        artifact_path=resolve("frozen_physical_twin"),
        surrogate_path=resolve("frozen_physical_surrogate"),
        calibration_manifest_path=resolve("physical_calibration_manifest"),
    )


def _default_model(
    preflight: V8PreflightResult,
    corpus: Any,
    device: torch.device,
    *,
    initialize_legacy: bool,
) -> tuple[nn.Module, Mapping[str, Any]]:
    repair_config = preflight.config.get("model", {}).get(
        "representation_repair", {}
    )
    repair_enabled = bool(
        isinstance(repair_config, Mapping) and repair_config.get("enabled", False)
    )
    model = TriPlaneCausalWorldModelV8(
        preflight.residual_anchor,
        preflight.config,
        physical_twin_core=_physical_core(
            preflight.config, preflight.repository_root
        ),
        representation_repair_statistics=getattr(
            corpus, "representation_repair_statistics", None
        ),
    ).to(device)
    if initialize_legacy:
        if repair_enabled:
            report_path = (
                preflight.resolved_paths["output_root"]
                / "initialization"
                / "failed_r0_repair_warm_start.json"
            )
            report = failed_r0_repair_warm_start(
                model,
                preflight.resolved_paths["repair_parent_checkpoint"],
                preflight.resolved_paths["repair_parent_gate"],
                seed=int(preflight.config["protocol"]["active_seed"]),
                report_path=report_path,
            )
        else:
            report_path = (
                preflight.resolved_paths["output_root"]
                / "initialization"
                / "legacy_representation_warm_start.json"
            )
            report = legacy_representation_warm_start(
                model,
                preflight.resolved_paths["representation_checkpoint"],
                report_path=report_path,
            )
    else:
        report = {
            "schema": INITIALIZATION_SCHEMA,
            "mode": "direct_predecessor_checkpoint_required",
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "legacy_representation_loaded": False,
            "locked_2025_opened": False,
        }
    return model, report


def _device(value: torch.device | str | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _components(
    config_path: str | Path,
    *,
    device: torch.device | str | None,
    corpus: Any | None,
    corpus_factory: Callable[[Mapping[str, Any], Path], Any],
    model: nn.Module | None,
    model_factory: Callable[..., Any] | None,
    materialize_anchor: bool,
    inventory_fn: Callable[[Mapping[str, Any], Path], Any] | None,
    initialize_legacy: bool,
    initialization_seed: int,
) -> RunnerComponents:
    _seed_everything(initialization_seed)
    config, root, _ = load_v8_config(config_path)
    prepared_corpus = corpus or corpus_factory(config, root)
    preflight_kwargs: dict[str, Any] = {
        "materialize_anchor": materialize_anchor,
        "corpus": prepared_corpus,
    }
    if inventory_fn is not None:
        preflight_kwargs["inventory_fn"] = inventory_fn
    preflight = prepare_v8(config_path, **preflight_kwargs)
    target_device = _device(device)
    if model is not None:
        prepared_model = model.to(target_device)
        initialization: Mapping[str, Any] = {
            "schema": INITIALIZATION_SCHEMA,
            "mode": "injected_model",
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "locked_2025_opened": False,
        }
    elif model_factory is not None:
        built = _call_supported(
            model_factory,
            preflight,
            prepared_corpus,
            target_device,
            config=preflight.config,
            anchor=preflight.residual_anchor,
        )
        if isinstance(built, tuple) and len(built) == 2:
            prepared_model, initialization = built
        else:
            prepared_model, initialization = built, {
                "schema": INITIALIZATION_SCHEMA,
                "mode": "injected_model_factory",
                "optimizer_loaded": False,
                "scheduler_loaded": False,
                "locked_2025_opened": False,
            }
        if not isinstance(prepared_model, nn.Module):
            raise V8RunnerError("model_factory did not return a torch module")
        prepared_model = prepared_model.to(target_device)
    else:
        prepared_model, initialization = _default_model(
            preflight,
            prepared_corpus,
            target_device,
            initialize_legacy=initialize_legacy,
        )
    if getattr(prepared_model, "architecture_version", ARCHITECTURE_VERSION) != ARCHITECTURE_VERSION:
        raise V8RunnerError("model has the wrong V8 architecture version")
    initialization = {
        **dict(initialization),
        "initialization_seed": int(initialization_seed),
    }
    return RunnerComponents(
        preflight,
        prepared_corpus,
        prepared_model,
        initialization,
        target_device,
    )


def _training_batch(
    corpus: Any,
    anchor: V8ResidualAnchor,
    *,
    device: torch.device,
    batch_size: int,
    max_days: int | None,
    shuffle_seed: int | None,
) -> dict[str, Any]:
    iterator = corpus.batches(
        "train",
        int(batch_size),
        device=device,
        max_days=max_days,
        shuffle_seed=shuffle_seed,
        origin=0,
    )
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise V8RunnerError("development train corpus is empty") from exc
    if max_days is None:
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise V8RunnerError("formal full-train batch unexpectedly split")
    return adapt_training_batch(batch, anchor=anchor, corpus=corpus)


def _data_lineage(preflight: V8PreflightResult) -> dict[str, Any]:
    payload = {
        "method_revision": str(
            preflight.config["project"].get("method_revision", "v8_base")
        ),
        "split_summary": dict(preflight.split_summary),
        "sources": dict(preflight.lineage_summary["sources"]),
        "residual_anchor": dict(preflight.lineage_summary["residual_anchor"]),
        "locked_2025_opened": False,
    }
    return {**payload, "digest": sha256_json(payload)}


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V8RunnerError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise V8RunnerError(f"{label} must contain a JSON object")
    return value


def _status_passed(value: Mapping[str, Any]) -> bool:
    candidates = (
        value.get("status"),
        value.get("gate_status"),
        value.get("capability_gate_status"),
    )
    return any(str(item).upper() in {"PASS", "PASSED"} for item in candidates)


def _selected_checkpoint(value: Mapping[str, Any]) -> str | None:
    containers = [value]
    for name in ("gate", "selection", "result"):
        nested = value.get(name)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        for name in (
            "selected_checkpoint",
            "selected_checkpoint_path",
            "selected_best_checkpoint",
        ):
            selected = container.get(name)
            if isinstance(selected, str) and selected.strip():
                return selected
    return None


def _load_direct_predecessor(
    *,
    model: nn.Module,
    output_root: Path,
    repository_root: Path,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    """Require an explicit PASS and selected checkpoint from the direct parent."""

    index = PHASE_ORDER.index(phase)
    if index == 0:
        raise V8RunnerError("R0 has no V8 phase predecessor")
    parent = PHASE_ORDER[index - 1]
    parent_root = output_root / f"seed={seed}" / f"phase={parent}"
    candidates = (
        parent_root / "gate.json",
        parent_root / "capability_gate.json",
        parent_root / "selection.json",
        parent_root / "capability" / "selection.json",
    )
    records = [path for path in candidates if path.is_file()]
    if not records:
        raise V8RunnerError(
            f"phase {phase} is blocked: direct predecessor {parent} has no gate"
        )
    for gate_path in records:
        gate = _read_json(gate_path, f"{parent} capability gate")
        if not _status_passed(gate):
            continue
        if gate.get("locked_2025_opened") is not False:
            raise V8RunnerError(f"{parent} gate does not prove 2025 stayed sealed")
        selected_value = _selected_checkpoint(gate)
        if selected_value is None:
            raise V8RunnerError(f"{parent} PASSED gate omitted selected checkpoint")
        selected = Path(selected_value)
        if not selected.is_absolute():
            repository_candidate = (repository_root / selected).resolve()
            selected = (
                repository_candidate
                if repository_candidate.is_file()
                else (gate_path.parent / selected).resolve()
            )
        else:
            selected = selected.resolve()
        if not selected.is_file():
            raise V8RunnerError(f"{parent} selected checkpoint is missing: {selected}")
        payload = torch.load(selected, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise V8RunnerError(f"{parent} selected checkpoint is malformed")
        if (
            payload.get("architecture_version") != ARCHITECTURE_VERSION
            or str(payload.get("phase")) != parent
            or int(payload.get("seed", -1)) != int(seed)
            or payload.get("locked_2025_opened") is not False
        ):
            raise V8RunnerError(f"{parent} selected checkpoint lineage is incompatible")
        state = payload.get("model_state")
        if not isinstance(state, Mapping):
            raise V8RunnerError(f"{parent} selected checkpoint has no model state")
        model.load_state_dict(state, strict=True)
        report = {
            "schema": "dcvpp.r4.v8.direct-phase-predecessor.v1",
            "parent_phase": parent,
            "gate_record": str(gate_path.resolve()),
            "selected_checkpoint": str(selected),
            "selected_checkpoint_sha256": sha256_file(selected),
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "locked_2025_opened": False,
        }
        return {**report, "digest": sha256_json(report)}
    raise V8RunnerError(
        f"phase {phase} is blocked: direct predecessor {parent} gate is not PASSED"
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
    }


def _initialization_digest(value: Mapping[str, Any]) -> str:
    stable = {
        key: item
        for key, item in value.items()
        if key not in {"created_at_utc", "updated_at_utc"}
    }
    return sha256_json(stable)


def _restore_rng(raw: Mapping[str, Any]) -> None:
    if "python" in raw:
        random.setstate(raw["python"])
    if "numpy" in raw:
        np.random.set_state(raw["numpy"])
    if "torch" in raw:
        torch.set_rng_state(raw["torch"])
    if torch.cuda.is_available() and raw.get("cuda"):
        torch.cuda.set_rng_state_all(raw["cuda"])


def _load_resume(
    path: Path,
    *,
    model: nn.Module,
    trainer: V8TriFusionTrainer,
    phase: str,
    seed: int,
    data_lineage: Mapping[str, Any],
    initialization_report: Mapping[str, Any],
) -> ResumeState:
    if not path.is_file():
        return ResumeState()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise V8RunnerError("resume checkpoint is malformed")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("architecture_version") != ARCHITECTURE_VERSION
        or str(payload.get("phase")) != phase
        or int(payload.get("seed", -1)) != int(seed)
    ):
        raise V8RunnerError("resume checkpoint identity does not match invocation")
    if payload.get("locked_2025_opened") is not False:
        raise V8RunnerError("resume checkpoint does not prove 2025 stayed sealed")
    prior_lineage = payload.get("data_lineage")
    if not isinstance(prior_lineage, Mapping) or prior_lineage.get("digest") != data_lineage["digest"]:
        raise V8RunnerError("resume checkpoint data lineage changed")
    prior_initialization = payload.get("initialization_report")
    if not isinstance(prior_initialization, Mapping) or _initialization_digest(
        prior_initialization
    ) != _initialization_digest(initialization_report):
        raise V8RunnerError("resume checkpoint initialization lineage changed")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer_states = payload.get("optimizer_states")
    if not isinstance(optimizer_states, Mapping) or set(optimizer_states) != set(trainer.optimizers):
        raise V8RunnerError("resume optimizer state is incomplete")
    for name, optimizer in trainer.optimizers.items():
        optimizer.load_state_dict(optimizer_states[name])
    rng = payload.get("rng_state")
    if not isinstance(rng, Mapping):
        raise V8RunnerError("resume checkpoint has no RNG state")
    _restore_rng(rng)
    return ResumeState(
        int(payload.get("cycles_completed", 0)),
        int(payload.get("optimizer_updates_completed", 0)),
        float(payload.get("best_validation", float("inf"))),
        int(payload.get("stale_validation_checks", 0)),
        (
            None
            if payload.get("best_cycle") is None
            else int(payload["best_cycle"])
        ),
    )


def _default_r0_validation(
    model: nn.Module,
    corpus: Any,
    *,
    config: Mapping[str, Any],
    anchor: V8ResidualAnchor,
    **_: Any,
) -> dict[str, Any]:
    """Fixed 64-day 2024 screen used only for R0 early stopping."""

    device = next(model.parameters()).device
    total = 0.0
    observed = 0
    dates: list[str] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for raw_batch in corpus.batches(
                "validation",
                8,
                device=device,
                shuffle_seed=None,
                max_days=64,
                origin=0,
            ):
                batch = adapt_training_batch(raw_batch, anchor=anchor, corpus=corpus)
                batch_dates = _batch_dates(batch)
                if any(not value.startswith("2024-") for value in batch_dates):
                    raise V8RunnerError(
                        "default R0 validation must use only 2024 days"
                    )
                output = model.forward_representation(
                    batch, scientific_score_dtype=torch.float32
                )
                loss = representation_loss(
                    output,
                    batch,
                    weights=config.get("losses", {}).get("representation"),
                )
                count = len(batch_dates)
                total += count * float(loss.total.detach().float().cpu())
                observed += count
                dates.extend(batch_dates)
    finally:
        model.train(was_training)
    if observed != 64 or len(set(dates)) != 64:
        raise V8RunnerError(
            f"default R0 validation requires 64 unique 2024 days, got {observed}"
        )
    metric = total / observed
    return {
        "primary_metric": metric,
        "feasible": bool(np.isfinite(metric)),
        "operating_dates": tuple(dates),
        "validation_days": observed,
        "batch_days": 8,
        "evidence_scope": "CHEAP_EARLY_STOP_ONLY",
        "formal_364_day_gate_executed": False,
        "locked_2025_opened": False,
    }


def _default_r0_repair_validation(
    model: nn.Module,
    corpus: Any,
    *,
    config: Mapping[str, Any],
    anchor: V8ResidualAnchor,
    **_: Any,
) -> dict[str, Any]:
    """Full-2024 cheap screen for the intermittent-demand repair lineage.

    This evaluates the registered repair loss on every eligible validation
    day, but does not run the 10,000-replicate scientific bootstrap.  It is
    therefore checkpoint-selection evidence only.
    """

    device = next(model.parameters()).device
    total = 0.0
    observed = 0
    dates: list[str] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for raw_batch in corpus.batches(
                "validation",
                182,
                device=device,
                shuffle_seed=None,
                max_days=None,
                origin=0,
            ):
                batch = adapt_training_batch(raw_batch, anchor=anchor, corpus=corpus)
                batch_dates = _batch_dates(batch)
                if any(not value.startswith("2024-") for value in batch_dates):
                    raise V8RunnerError(
                        "R0 repair validation must use only 2024 days"
                    )
                output = model.forward_representation(
                    batch, scientific_score_dtype=torch.float32
                )
                loss = representation_loss(
                    output,
                    batch,
                    weights=config.get("losses", {}).get("representation"),
                )
                count = len(batch_dates)
                total += count * float(loss.total.detach().float().cpu())
                observed += count
                dates.extend(batch_dates)
    finally:
        model.train(was_training)
    if observed != 364 or len(set(dates)) != 364:
        raise V8RunnerError(
            f"R0 repair validation requires 364 unique 2024 days, got {observed}"
        )
    return {
        "primary_metric": total / observed,
        "feasible": bool(np.isfinite(total / observed)),
        "operating_dates": tuple(dates),
        "validation_days": observed,
        "batch_days": 182,
        "evidence_scope": "FULL_364_CHEAP_REPAIR_SELECTION_NOT_FORMAL_GATE",
        "formal_364_day_gate_executed": False,
        "locked_2025_opened": False,
    }


def run_v8_smoke(
    config_path: str | Path,
    *,
    rollout_fn: RolloutFn,
    seed: int | None = None,
    batch_size: int = 2,
    device: torch.device | str | None = "cpu",
    corpus: Any | None = None,
    corpus_factory: Callable[[Mapping[str, Any], Path], Any] = PreloadedV8Corpus,
    model: nn.Module | None = None,
    model_factory: Callable[..., Any] | None = None,
    materialize_anchor: bool = True,
    inventory_fn: Callable[[Mapping[str, Any], Path], Any] | None = None,
) -> dict[str, Any]:
    """Run one small optimizer update for R0, D0 and P0 only."""

    registered_config, _, _ = load_v8_config(config_path)
    active_seed = int(
        registered_config["protocol"]["active_seed"] if seed is None else seed
    )
    if active_seed not in tuple(registered_config["protocol"]["seeds"]):
        raise V8RunnerError("smoke seed is not registered")
    components = _components(
        config_path,
        device=device,
        corpus=corpus,
        corpus_factory=corpus_factory,
        model=model,
        model_factory=model_factory,
        materialize_anchor=materialize_anchor,
        inventory_fn=inventory_fn,
        initialize_legacy=True,
        initialization_seed=active_seed,
    )
    batch = _training_batch(
        components.corpus,
        components.preflight.residual_anchor,
        device=components.device,
        batch_size=int(batch_size),
        max_days=int(batch_size),
        shuffle_seed=active_seed,
    )
    smoke_config = deepcopy(components.preflight.config)
    smoke_config["sampling"]["d0_warmup"] = 4
    smoke_config["sampling"]["train"] = 4
    smoke_config["sampling"]["calibration"] = 4
    trainer = V8TriFusionTrainer(
        components.model,
        smoke_config,
        rollout_fn=rollout_fn,
        origin_budget=2,
        microbatch_days=max(1, int(batch_size)),
    )
    results: list[dict[str, Any]] = []
    for index, phase in enumerate(("r0", "d0", "p0")):
        report = trainer.update(
            batch,
            stage_id=phase,
            optimizer_update_index=0,
            seed=active_seed + index,
        )
        results.append(asdict(report))
    output = {
        "schema": RUNNER_SCHEMA,
        "status": "SMOKE_PASSED",
        "seed": active_seed,
        "phases": results,
        "batch_days": len(_batch_dates(batch)),
        "training_started": False,
        "formal_checkpoint_written": False,
        "locked_2025_opened": False,
    }
    _atomic_json(
        components.preflight.resolved_paths["output_root"]
        / "smoke"
        / f"seed={active_seed}.json",
        output,
    )
    return output


def run_v8_training(
    config_path: str | Path,
    *,
    phase: str = "r0",
    seed: int | None = None,
    resume: bool = True,
    rollout_fn: RolloutFn | None = None,
    max_cycles: int | None = None,
    device: torch.device | str | None = None,
    corpus: Any | None = None,
    corpus_factory: Callable[[Mapping[str, Any], Path], Any] = PreloadedV8Corpus,
    model: nn.Module | None = None,
    model_factory: Callable[..., Any] | None = None,
    materialize_anchor: bool = True,
    inventory_fn: Callable[[Mapping[str, Any], Path], Any] | None = None,
    validation_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Train one registered phase, defaulting to R0, with cycle-boundary resume."""

    phase_id = str(phase).lower()
    if phase_id not in PHASE_ORDER:
        raise V8RunnerError(f"phase must be one of {list(PHASE_ORDER)}")
    registered_config, _, _ = load_v8_config(config_path)
    active_seed = int(
        registered_config["protocol"]["active_seed"] if seed is None else seed
    )
    if active_seed not in tuple(
        int(value) for value in registered_config["protocol"]["seeds"]
    ):
        raise V8RunnerError("training seed is not registered")
    components = _components(
        config_path,
        device=device,
        corpus=corpus,
        corpus_factory=corpus_factory,
        model=model,
        model_factory=model_factory,
        materialize_anchor=materialize_anchor,
        inventory_fn=inventory_fn,
        initialize_legacy=phase_id == "r0",
        initialization_seed=active_seed,
    )
    config = components.preflight.config
    stage = contract_from_config(config).stage(phase_id)
    requested_cycles = stage.maximum_cycles if max_cycles is None else int(max_cycles)
    if not 1 <= requested_cycles <= stage.maximum_cycles:
        raise V8RunnerError(
            f"max_cycles must lie in [1,{stage.maximum_cycles}] for {phase_id}"
        )
    if "policy" in stage.active_tasks and rollout_fn is None:
        raise V8RunnerError(f"phase {phase_id} requires rollout_fn")
    initialization_report = dict(components.initialization_report)
    if phase_id != "r0":
        initialization_report = _load_direct_predecessor(
            model=components.model,
            output_root=components.preflight.resolved_paths["output_root"],
            repository_root=components.preflight.repository_root,
            seed=active_seed,
            phase=phase_id,
        )

    # The 722-day corpus remains on host memory.  The trainer owns explicit
    # task-specific microbatch transfers to the model device.
    full_batch = _training_batch(
        components.corpus,
        components.preflight.residual_anchor,
        device=torch.device("cpu"),
        batch_size=722,
        max_days=None,
        shuffle_seed=None,
    )
    if len(_batch_dates(full_batch)) != 722:
        raise V8RunnerError("formal V8 training requires all 722 train days")
    origin_budget = int(
        config["training"].get("policy_reconditioned_origins_per_rollout", 8)
    )
    trainer = V8TriFusionTrainer(
        components.model,
        config,
        rollout_fn=rollout_fn,
        origin_budget=origin_budget,
    )
    phase_root = (
        components.preflight.resolved_paths["output_root"]
        / f"seed={active_seed}"
        / f"phase={phase_id}"
    )
    checkpoint_path = phase_root / "checkpoints" / "latest.pt"
    state_path = phase_root / "state.json"
    log_path = phase_root / "train.jsonl"
    if not resume and checkpoint_path.exists():
        raise V8RunnerError("fresh run would overwrite an existing V8 checkpoint")
    lineage = _data_lineage(components.preflight)
    resume_state = (
        _load_resume(
            checkpoint_path,
            model=components.model,
            trainer=trainer,
            phase=phase_id,
            seed=active_seed,
            data_lineage=lineage,
            initialization_report=initialization_report,
        )
        if resume
        else ResumeState()
    )
    if resume_state.cycles_completed > requested_cycles:
        raise V8RunnerError("resume checkpoint is beyond requested max_cycles")

    _atomic_json(
        state_path,
        {
            "schema": RUNNER_SCHEMA,
            "status": "TRAINING_INITIALIZED",
            "architecture_version": ARCHITECTURE_VERSION,
            "phase": phase_id,
            "seed": active_seed,
            "pid": os.getpid(),
            "device": str(components.device),
            "microbatch_days": dict(trainer.microbatch_days),
            "origin_budget": origin_budget,
            "cycles_completed": resume_state.cycles_completed,
            "optimizer_updates_completed": resume_state.optimizer_updates_completed,
            "resume": bool(resume),
            "resumed": resume_state.cycles_completed > 0,
            "checkpoint": str(checkpoint_path) if checkpoint_path.is_file() else None,
            "data_lineage_digest": lineage["digest"],
            "capability_gate_executed": False,
            "next_phase_released": False,
            "final_test_data_accessed": False,
            "locked_2025_opened": False,
            "updated_at_utc": _utc_now(),
        },
    )

    updates_completed = resume_state.optimizer_updates_completed
    best_validation = resume_state.best_validation
    stale_validation = resume_state.stale_validation_checks
    best_cycle = resume_state.best_cycle
    early_stopped = False
    latest_validation: Mapping[str, Any] | None = None
    effective_validation_fn = validation_fn
    if effective_validation_fn is None and phase_id == "r0":
        repair = config.get("model", {}).get("representation_repair", {})
        effective_validation_fn = (
            _default_r0_repair_validation
            if isinstance(repair, Mapping) and repair.get("enabled", False)
            else _default_r0_validation
        )
    patience = int(config["training"]["early_stopping"].get("patience_checks", 8))
    minimum_improvement = float(
        config["training"]["early_stopping"].get(
            "minimum_primary_improvement", 1.0e-4
        )
    )
    latest_updates: list[dict[str, Any]] = []
    for cycle in range(resume_state.cycles_completed, requested_cycles):
        pieces = coverage_batches(
            full_batch,
            cycle_index=cycle,
            expected_days=722,
            updates=4,
            seed=active_seed,
            shuffle=True,
        )
        cycle_reports: list[dict[str, Any]] = []
        for local_index, piece in enumerate(pieces):
            global_update = cycle * 4 + local_index
            horizon = stage.horizon_at(global_update)
            sample_count = trainer._sample_count(stage, global_update)
            if components.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(components.device)
                torch.cuda.synchronize(components.device)
            update_started = time.perf_counter()
            report = trainer.update(
                piece,
                stage_id=phase_id,
                optimizer_update_index=global_update,
                seed=active_seed + 1_000_003 * global_update,
            )
            if components.device.type == "cuda":
                torch.cuda.synchronize(components.device)
                peak_allocated = torch.cuda.max_memory_allocated(components.device) / 2**20
                peak_reserved = torch.cuda.max_memory_reserved(components.device) / 2**20
            else:
                peak_allocated = 0.0
                peak_reserved = 0.0
            duration = time.perf_counter() - update_started
            item = {
                **asdict(report),
                "cycle": cycle + 1,
                "optimizer_update": global_update + 1,
                "coverage_days": len(_batch_dates(piece)),
                "horizon": horizon,
                "sample_count": sample_count,
                "duration_seconds": duration,
                "cuda_peak_allocated_mb": peak_allocated,
                "cuda_peak_reserved_mb": peak_reserved,
                "created_at_utc": _utc_now(),
            }
            cycle_reports.append(item)
            _atomic_jsonl_append(log_path, item)
            updates_completed += 1
            _atomic_json(
                state_path,
                {
                    "schema": RUNNER_SCHEMA,
                    "status": "TRAINING",
                    "architecture_version": ARCHITECTURE_VERSION,
                    "phase": phase_id,
                    "seed": active_seed,
                    "pid": os.getpid(),
                    "device": str(components.device),
                    "microbatch_days": dict(trainer.microbatch_days),
                    "origin_budget": origin_budget,
                    "cycles_completed": cycle,
                    "optimizer_updates_completed": updates_completed,
                    "last_update": item,
                    "checkpoint": (
                        str(checkpoint_path) if checkpoint_path.is_file() else None
                    ),
                    "data_lineage_digest": lineage["digest"],
                    "capability_gate_executed": False,
                    "next_phase_released": False,
                    "final_test_data_accessed": False,
                    "locked_2025_opened": False,
                    "updated_at_utc": _utc_now(),
                },
            )
        latest_updates = cycle_reports
        validation_improved = False
        if effective_validation_fn is not None:
            raw_validation = _call_supported(
                effective_validation_fn,
                components.model,
                components.corpus,
                phase=phase_id,
                cycle=cycle + 1,
                seed=active_seed,
                allowed_years=(2022, 2023, 2024),
                config=config,
                anchor=components.preflight.residual_anchor,
            )
            if isinstance(raw_validation, Mapping):
                metric_raw = raw_validation.get(
                    "primary_metric", raw_validation.get("metric")
                )
                feasible = bool(raw_validation.get("feasible", True))
                validation_dates = raw_validation.get("operating_dates")
                if isinstance(validation_dates, Sequence) and not isinstance(
                    validation_dates, (str, bytes)
                ):
                    assert_development_dates(
                        validation_dates, label="validation hook"
                    )
                details = dict(raw_validation)
            else:
                metric_raw = raw_validation
                feasible = True
                details = {"primary_metric": raw_validation}
            try:
                metric = float(metric_raw)
            except (TypeError, ValueError) as exc:
                raise V8RunnerError(
                    "validation_fn must return a finite primary metric"
                ) from exc
            if not np.isfinite(metric):
                raise V8RunnerError("validation_fn returned a non-finite metric")
            improved = feasible and metric < best_validation - minimum_improvement
            if improved:
                best_validation = metric
                stale_validation = 0
                best_cycle = cycle + 1
                validation_improved = True
            else:
                stale_validation += 1
            latest_validation = {
                **details,
                "primary_metric": metric,
                "feasible": feasible,
                "improved": improved,
                "best_primary_metric": best_validation,
                "stale_checks": stale_validation,
            }
            _atomic_jsonl_append(
                log_path,
                {
                    "event": "validation",
                    "phase": phase_id,
                    "cycle": cycle + 1,
                    **latest_validation,
                    "created_at_utc": _utc_now(),
                },
            )
            if cycle + 1 >= stage.minimum_cycles and stale_validation >= patience:
                early_stopped = True
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_version": ARCHITECTURE_VERSION,
            "method_revision": str(
                config["project"].get("method_revision", "v8_base")
            ),
            "phase": phase_id,
            "seed": active_seed,
            "cycles_completed": cycle + 1,
            "optimizer_updates_completed": updates_completed,
            "best_validation": best_validation,
            "stale_validation_checks": stale_validation,
            "best_cycle": best_cycle,
            "model_state": components.model.state_dict(),
            "optimizer_states": {
                name: optimizer.state_dict()
                for name, optimizer in trainer.optimizers.items()
            },
            "rng_state": _rng_state(),
            "data_lineage": lineage,
            "initialization_report": initialization_report,
            "updated_at_utc": _utc_now(),
            "locked_2025_opened": False,
        }
        _atomic_torch(checkpoint_path, checkpoint)
        best_path = phase_root / "checkpoints" / "best.pt"
        if validation_improved:
            _atomic_torch(
                best_path,
                {
                    **checkpoint,
                    "checkpoint_role": "cheap_validation_best_candidate",
                    "validation_evidence_scope": "CHEAP_EARLY_STOP_ONLY",
                },
            )
        _atomic_json(
            state_path,
            {
                "schema": RUNNER_SCHEMA,
                "status": (
                    "TRAINING"
                    if cycle + 1 < requested_cycles
                    else "TRAINED_AWAITING_CAPABILITY_GATE"
                ),
                "architecture_version": ARCHITECTURE_VERSION,
                "method_revision": str(
                    config["project"].get("method_revision", "v8_base")
                ),
                "phase": phase_id,
                "seed": active_seed,
                "cycles_completed": cycle + 1,
                "optimizer_updates_completed": updates_completed,
                "checkpoint": str(checkpoint_path),
                "log": str(log_path),
                "latest_updates": latest_updates,
                "validation": latest_validation,
                "validation_status": (
                    "NOT_RUN"
                    if effective_validation_fn is None
                    else "MEASURED"
                ),
                "early_stopped": early_stopped,
                "best_cycle": best_cycle,
                "best_checkpoint": str(best_path) if best_path.is_file() else None,
                "data_lineage_digest": lineage["digest"],
                "resumed": resume_state.cycles_completed > 0,
                "final_test_data_accessed": False,
                "locked_2025_opened": False,
                "updated_at_utc": _utc_now(),
            },
        )
        if early_stopped:
            break
    completed_cycles = (
        cycle + 1 if "cycle" in locals() else resume_state.cycles_completed
    )
    if early_stopped:
        final_status = "EARLY_STOPPED_AWAITING_CAPABILITY_GATE"
    elif completed_cycles >= stage.minimum_cycles:
        final_status = "TRAINED_AWAITING_CAPABILITY_GATE"
    else:
        final_status = "PARTIAL_TRAINING"
    _atomic_json(
        state_path,
        {
            "schema": RUNNER_SCHEMA,
            "status": final_status,
            "architecture_version": ARCHITECTURE_VERSION,
            "method_revision": str(
                config["project"].get("method_revision", "v8_base")
            ),
            "phase": phase_id,
            "seed": active_seed,
            "cycles_completed": completed_cycles,
            "optimizer_updates_completed": updates_completed,
            "checkpoint": str(checkpoint_path),
            "log": str(log_path),
            "latest_updates": latest_updates,
            "validation": latest_validation,
            "validation_status": (
                "NOT_RUN" if effective_validation_fn is None else "MEASURED"
            ),
            "early_stopped": early_stopped,
            "best_validation": (
                None if best_validation == float("inf") else best_validation
            ),
            "stale_validation_checks": stale_validation,
            "best_cycle": best_cycle,
            "best_checkpoint": (
                str(phase_root / "checkpoints" / "best.pt")
                if (phase_root / "checkpoints" / "best.pt").is_file()
                else None
            ),
            "data_lineage_digest": lineage["digest"],
            "resumed": resume_state.cycles_completed > 0,
            "capability_gate_executed": False,
            "next_phase_released": False,
            "final_test_data_accessed": False,
            "locked_2025_opened": False,
            "updated_at_utc": _utc_now(),
        },
    )
    return {
        "schema": RUNNER_SCHEMA,
        "status": final_status,
        "architecture_version": ARCHITECTURE_VERSION,
        "method_revision": str(
            config["project"].get("method_revision", "v8_base")
        ),
        "phase": phase_id,
        "seed": active_seed,
        "cycles_completed": completed_cycles,
        "optimizer_updates_completed": updates_completed,
        "checkpoint": str(checkpoint_path),
        "state": str(state_path),
        "log": str(log_path),
        "resumed_from_cycle": resume_state.cycles_completed,
        "early_stopped": early_stopped,
        "validation_status": (
            "NOT_RUN" if effective_validation_fn is None else "MEASURED"
        ),
        "best_cycle": best_cycle,
        "best_checkpoint": (
            str(phase_root / "checkpoints" / "best.pt")
            if (phase_root / "checkpoints" / "best.pt").is_file()
            else None
        ),
        "next_phase_released": False,
        "data_lineage_digest": lineage["digest"],
        "final_test_data_accessed": False,
        "locked_2025_opened": False,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "INITIALIZATION_SCHEMA",
    "RUNNER_SCHEMA",
    "ResumeState",
    "V8RunnerError",
    "adapt_training_batch",
    "legacy_representation_warm_start",
    "run_v8_smoke",
    "run_v8_training",
]
