"""Concrete local phase runner for the approved R4-V7 lineage."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import load_v7_config
from .corpus import (
    PreloadedV7DevelopmentCorpus,
    materialize_dynamic_residual_anchor,
)
from .data import load_frozen_residual_anchor
from .distribution_gate import (
    gate_and_select_d0_distribution,
    register_d0_candidate,
)
from .model import BaselineAnchoredJointWorldPolicyMoEV7
from .preflight import fit_train_only_proper_scales
from . import losses as v7_losses
from .trainer import (
    PHASE_ORDER,
    V7JointTrainer,
    build_v7_optimizers,
    training_contract_from_config,
)


class V7RunnerError(RuntimeError):
    """Raised when a concrete phase cannot preserve its registered lineage."""


@dataclass(frozen=True)
class PhaseLineage:
    """The single gate-approved parent from which a phase may start."""

    parent_phase: str
    selected_checkpoint: str
    gate_record: str
    representation_predecessor: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeState:
    """Cycle-boundary state required for an exact, non-resetting resume."""

    cycles_completed: int = 0
    optimizer_updates_completed: int = 0
    best_cheap_validation: float = float("inf")
    best_cycle: int | None = None
    stale_cycles: int = 0


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


def _torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V7RunnerError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise V7RunnerError(f"{label} must contain a JSON object: {path}")
    return payload


def _passed_status(payload: Mapping[str, Any]) -> bool:
    """Accept only an explicit gate PASS, never an inferred training status."""

    values: list[Any] = [
        payload.get("status"),
        payload.get("gate_status"),
        payload.get("capability_gate_status"),
    ]
    gate = payload.get("gate")
    if isinstance(gate, Mapping):
        values.extend((gate.get("status"), gate.get("gate_status")))
    return any(str(value).strip().upper() in {"PASS", "PASSED"} for value in values)


def _failed_status(payload: Mapping[str, Any]) -> bool:
    values: list[Any] = [
        payload.get("status"),
        payload.get("gate_status"),
        payload.get("capability_gate_status"),
    ]
    gate = payload.get("gate")
    if isinstance(gate, Mapping):
        values.extend((gate.get("status"), gate.get("gate_status")))
    return any(
        str(value).strip().upper() in {"FAIL", "FAILED", "BLOCKED"} for value in values
    )


def _selected_checkpoint_value(payload: Mapping[str, Any]) -> str | None:
    containers: list[Mapping[str, Any]] = [payload]
    for key in ("selection", "gate", "result"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for key in (
            "selected_checkpoint",
            "selected_checkpoint_path",
            "selected_best_checkpoint",
            "best_checkpoint",
        ):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _resolve_record_path(
    raw_path: str,
    *,
    repository_root: Path,
    record_path: Path,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    repository_candidate = (repository_root / path).resolve()
    if repository_candidate.is_file():
        return repository_candidate
    return (record_path.parent / path).resolve()


def _gate_record_candidates(output_root: Path, seed: int, phase: str) -> tuple[Path, ...]:
    phase_root = output_root / f"seed={seed}" / f"phase={phase}"
    capability_root = output_root / "capability" / phase
    return (
        phase_root / "gate.json",
        phase_root / "capability_gate.json",
        phase_root / "selection.json",
        phase_root / "capability" / "d0_distribution_selection.json",
        phase_root / "gate" / "result.json",
        phase_root / "gate" / "distribution_gate.json",
        capability_root / "gate.json",
        capability_root / "selection.json",
        capability_root / "state.json",
        phase_root / "state.json",
    )


def _load_selected_phase_predecessor(
    *,
    output_root: Path,
    repository_root: Path,
    seed: int,
    phase: str,
    architecture_version: str,
) -> tuple[PhaseLineage, Mapping[str, Any]]:
    """Resolve and validate the immediately preceding gate-selected checkpoint.

    Training completion is deliberately insufficient.  A later phase can only
    inherit a checkpoint named by an explicit ``PASSED`` gate record belonging
    to the same seed and immediately preceding phase.
    """

    phase_index = PHASE_ORDER.index(phase)
    if phase_index == 0:
        raise V7RunnerError("D0 inherits Representation, not a V7 phase checkpoint")
    parent_phase = PHASE_ORDER[phase_index - 1]
    records = tuple(
        path
        for path in _gate_record_candidates(output_root, seed, parent_phase)
        if path.is_file()
    )
    if not records:
        raise V7RunnerError(
            f"phase {phase} is blocked: predecessor {parent_phase} has no gate record"
        )

    loaded_records = tuple(
        (path, _read_json(path, f"{parent_phase} gate record")) for path in records
    )
    if any(_failed_status(record) for _, record in loaded_records):
        raise V7RunnerError(
            f"phase {phase} is blocked: predecessor {parent_phase} has an explicit failed gate"
        )

    passed_without_selection: list[Path] = []
    for record_path, record in loaded_records:
        if not _passed_status(record):
            continue
        if record.get("seed") is not None and int(record["seed"]) != int(seed):
            raise V7RunnerError(f"{parent_phase} gate record belongs to a different seed")
        if record.get("phase") is not None and str(record["phase"]).lower() != parent_phase:
            raise V7RunnerError(f"{parent_phase} gate record belongs to a different phase")
        if parent_phase == "d0" and record.get("p0_release") is not True:
            raise V7RunnerError("D0 selection did not explicitly release P0")
        if record.get("locked_2025_opened") is True:
            raise V7RunnerError(f"{parent_phase} gate record opened locked 2025 data")
        selected_value = _selected_checkpoint_value(record)
        if selected_value is None:
            passed_without_selection.append(record_path)
            continue
        selected_path = _resolve_record_path(
            selected_value,
            repository_root=repository_root,
            record_path=record_path,
        )
        if not selected_path.is_file():
            raise V7RunnerError(
                f"{parent_phase} selected checkpoint is missing: {selected_path}"
            )
        payload = torch.load(selected_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise V7RunnerError(f"{parent_phase} selected checkpoint is malformed")
        if str(payload.get("phase", "")).lower() != parent_phase:
            raise V7RunnerError("selected checkpoint belongs to a non-predecessor phase")
        if int(payload.get("seed", -1)) != int(seed):
            raise V7RunnerError("selected checkpoint belongs to a different seed")
        if payload.get("architecture_version") != architecture_version:
            raise V7RunnerError("selected checkpoint has the wrong V7 architecture")
        if payload.get("locked_2025_opened") is not False:
            raise V7RunnerError("selected checkpoint does not prove locked 2025 stayed closed")
        if not isinstance(payload.get("model_state"), Mapping):
            raise V7RunnerError("selected checkpoint has no full model state")
        representation = payload.get("representation_predecessor")
        if not isinstance(representation, Mapping):
            raise V7RunnerError("selected checkpoint lost Representation lineage metadata")
        lineage = PhaseLineage(
            parent_phase=parent_phase,
            selected_checkpoint=str(selected_path),
            gate_record=str(record_path.resolve()),
            representation_predecessor=dict(representation),
        )
        return lineage, payload

    if passed_without_selection:
        raise V7RunnerError(
            f"phase {phase} is blocked: predecessor {parent_phase} passed but has no selected checkpoint"
        )
    raise V7RunnerError(
        f"phase {phase} is blocked: predecessor {parent_phase} gate is not PASSED"
    )


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise V7RunnerError("V7 formal training requires the configured CUDA GPU")
    return torch.device("cuda")


def _proper_scales(
    output_root: Path,
    anchor: Any,
    *,
    device: torch.device,
) -> dict[str, float]:
    path = output_root / "data" / "train_only_proper_scales.json"
    fit_split = f"{anchor.train_dates[0]}_to_{anchor.train_dates[-1]}_only"
    fit_days = len(anchor.train_dates)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fit_split") != fit_split or int(payload.get("fit_days", -1)) != fit_days:
            raise V7RunnerError("proper-score scale artifact has the wrong fit split")
        scales = payload.get("scales")
        if not isinstance(scales, Mapping):
            raise V7RunnerError("proper-score scale artifact is malformed")
        return {str(key): float(value) for key, value in scales.items()}
    scales = fit_train_only_proper_scales(anchor, device=device)
    _atomic_json(
        path,
        {
            "schema": "dcvpp.r4.v7-train-only-proper-scales.v1",
            "fit_split": fit_split,
            "fit_days": fit_days,
            "fit_dates": list(anchor.train_dates),
            "sample_count": 64,
            "channel_normalization": "frozen_train_residual_channel_scale",
            "scales": scales,
            "created_at_utc": _utc_now(),
        },
    )
    return scales


def _full_train_batch(
    corpus: PreloadedV7DevelopmentCorpus,
    *,
    device: torch.device,
    seed: int,
    channel_scale: torch.Tensor,
) -> dict[str, Any]:
    train_days = len(corpus.records("train"))
    iterator = corpus.batches(
        "train",
        train_days,
        device=device,
        shuffle_seed=None,
        origin=0,
    )
    batch = dict(next(iterator))
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise V7RunnerError("formal train corpus unexpectedly produced multiple batches")
    batch["channel_scale"] = channel_scale.to(device=device, dtype=torch.float32)
    rollout = batch.get("rollout_targets")
    if isinstance(rollout, Mapping) and isinstance(rollout.get("baseline_actions"), Mapping):
        batch["safe_anchor_actions"] = rollout["baseline_actions"]
    batch["coverage_seed"] = int(seed)
    return batch


def _rollout_callback(model: torch.nn.Module, phase: str):
    if phase == "d0" or phase == "calibration_restore":
        return None
    try:
        from .rollout import make_training_rollout
    except ImportError as exc:
        raise V7RunnerError("V7 differentiable rollout is not installed") from exc
    return make_training_rollout(model, phase=phase)


def _cheap_distribution_validation(
    model: BaselineAnchoredJointWorldPolicyMoEV7,
    corpus: PreloadedV7DevelopmentCorpus,
    anchor: Any,
    *,
    device: torch.device,
    proper_scales: Mapping[str, float],
    loss_weights: Mapping[str, float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    batch = next(
        corpus.batches(
            "validation",
            32,
            device=device,
            max_days=32,
            origin=0,
            shuffle_seed=None,
        )
    )
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        output = model.forward_distribution(batch, sample_count=64, seed=int(seed))
        distribution = output["distribution"]
        samples = distribution["samples"].float()
        target = batch["scenario_target_zonal"].to(
            device=samples.device, dtype=torch.float32
        )
        identity_samples = distribution.get("identity_samples", samples).float()
        v7_losses.assert_dart_identity(identity_samples)
        normalized = v7_losses.normalize_distribution_channels(
            samples,
            target,
            anchor.channel_scale.to(device),
            center=distribution.get("center").float(),
            scale=distribution.get("scale").float(),
        )
        terms = v7_losses.baseline_normalized_distribution_terms(
            normalized[0],
            normalized[1],
            baseline_scales=proper_scales,
            weights=distribution.get("weights").float(),
            center=normalized[2],
            scale=normalized[3],
            identity_samples=identity_samples,
        )
        total = v7_losses.compose_distribution_loss(terms, loss_weights)
        metrics = {
            key: float(value.mean().cpu())
            for key, value in terms.items()
            if key.startswith("raw_")
        }
    model.train(was_training)
    return float(total.cpu()), metrics


def _load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    phase: str,
    seed: int,
    state_path: Path | None = None,
    expected_lineage: Mapping[str, Any] | None = None,
) -> ResumeState:
    if not path.is_file():
        return ResumeState()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise V7RunnerError("V7 checkpoint is malformed")
    if payload.get("phase") != phase or int(payload.get("seed", -1)) != int(seed):
        raise V7RunnerError("V7 checkpoint phase/seed does not match invocation")
    if payload.get("locked_2025_opened") is not False:
        raise V7RunnerError("resume checkpoint does not prove locked 2025 stayed closed")
    if expected_lineage is not None:
        raw_lineage = payload.get("phase_lineage")
        if not isinstance(raw_lineage, Mapping):
            raise V7RunnerError("resume checkpoint has no phase-lineage record")
        for key in ("parent_phase", "selected_checkpoint", "gate_record"):
            expected = expected_lineage.get(key)
            actual = raw_lineage.get(key)
            if key in {"selected_checkpoint", "gate_record"}:
                if Path(str(actual)).resolve() != Path(str(expected)).resolve():
                    raise V7RunnerError(f"resume checkpoint changed lineage field {key}")
            elif actual != expected:
                raise V7RunnerError(f"resume checkpoint changed lineage field {key}")
    model.load_state_dict(payload["model_state"], strict=True)
    raw_optimizers = payload.get("optimizer_states", {})
    if not isinstance(raw_optimizers, Mapping):
        raise V7RunnerError("V7 checkpoint optimizer states are malformed")
    for name, optimizer in optimizers.items():
        if name not in raw_optimizers:
            raise V7RunnerError(f"V7 checkpoint is missing optimizer state {name}")
        optimizer.load_state_dict(raw_optimizers[name])

    fallback: Mapping[str, Any] = {}
    if state_path is not None and state_path.is_file():
        candidate = _read_json(state_path, "same-phase resume state")
        if (
            str(candidate.get("phase", phase)).lower() != phase
            or int(candidate.get("seed", seed)) != int(seed)
            or candidate.get("locked_2025_opened") is True
        ):
            raise V7RunnerError("same-phase resume state has incompatible provenance")
        fallback = candidate
    raw_resume = payload.get("resume_state")
    if not isinstance(raw_resume, Mapping):
        raw_resume = fallback
    cycles = int(raw_resume.get("cycles_completed", payload.get("cycles_completed", 0)))
    updates = int(
        raw_resume.get(
            "optimizer_updates_completed",
            payload.get("optimizer_updates_completed", cycles * 4),
        )
    )
    if cycles < 0 or updates != cycles * 4:
        raise V7RunnerError("resume checkpoint is not at an atomic coverage-cycle boundary")
    best_raw = raw_resume.get("best_cheap_validation")
    best = float("inf") if best_raw is None else float(best_raw)
    best_cycle_raw = raw_resume.get("best_cycle")
    best_cycle = None if best_cycle_raw is None else int(best_cycle_raw)
    stale = int(raw_resume.get("stale_cycles", 0))
    if stale < 0 or (best_cycle is not None and not 1 <= best_cycle <= cycles):
        raise V7RunnerError("resume early-stopping state is malformed")

    cpu_rng = payload.get("torch_rng_state")
    if isinstance(cpu_rng, torch.Tensor):
        torch.set_rng_state(cpu_rng)
    cuda_rng = payload.get("cuda_rng_state_all")
    if isinstance(cuda_rng, (list, tuple)) and all(
        isinstance(value, torch.Tensor) for value in cuda_rng
    ):
        torch.cuda.set_rng_state_all(list(cuda_rng))
    return ResumeState(cycles, updates, best, best_cycle, stale)


def run_phase(invocation: Mapping[str, Any]) -> dict[str, Any]:
    """Run one registered phase; callable from the V7 CLI adapter seam."""

    phase = str(invocation.get("phase", "")).lower()
    if phase not in PHASE_ORDER:
        raise V7RunnerError(f"unknown V7 phase {phase!r}")
    seed = int(invocation.get("seed", 20240813))
    resume = bool(invocation.get("resume", True))
    config_path = Path(str(invocation["config_path"])).resolve()
    config, root, resolved = load_v7_config(config_path)
    contract = training_contract_from_config(config)
    spec = contract.phase(phase)

    output_root = resolved["output_root"]
    phase_root = output_root / f"seed={seed}" / f"phase={phase}"
    checkpoint_path = phase_root / "checkpoints" / "latest.pt"
    state_path = phase_root / "state.json"
    best_path = phase_root / "checkpoints" / "best_cheap.pt"
    _atomic_json(output_root / "data" / "training_contract.json", contract.to_dict())

    if resume and state_path.is_file():
        existing_state = _read_json(state_path, "current phase state")
        if _passed_status(existing_state):
            if existing_state.get("locked_2025_opened") is not False:
                raise V7RunnerError("passed phase state does not prove locked 2025 stayed closed")
            selected = _selected_checkpoint_value(existing_state)
            if selected is None:
                raise V7RunnerError("passed phase state has no selected checkpoint")
            selected_path = _resolve_record_path(
                selected,
                repository_root=root,
                record_path=state_path,
            )
            if not selected_path.is_file():
                raise V7RunnerError("passed phase selected checkpoint is missing")
            # A passed phase is immutable.  Re-entering it must never demote the
            # gate status or reset its selected model.
            return dict(existing_state) | {"resume_noop": True}

    architecture_version = str(config["project"]["architecture_version"])
    selected_parent: tuple[PhaseLineage, Mapping[str, Any]] | None = None
    if phase != PHASE_ORDER[0]:
        # Gate lineage is resolved before loading the corpus or constructing a
        # trainable model, so a blocked phase cannot do any training work.
        selected_parent = _load_selected_phase_predecessor(
            output_root=output_root,
            repository_root=root,
            seed=seed,
            phase=phase,
            architecture_version=architecture_version,
        )

    device = _device()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    corpus = PreloadedV7DevelopmentCorpus(config, root)
    train_split = config["protocol"]["splits"]["train"]
    if not resolved["residual_anchor"].is_file():
        materialize_dynamic_residual_anchor(corpus, resolved["residual_anchor"])
    anchor = load_frozen_residual_anchor(
        resolved["residual_anchor"],
        expected_start=train_split["start"],
        expected_end=train_split["end"],
        expected_days=int(train_split["days"]),
    )
    model = BaselineAnchoredJointWorldPolicyMoEV7(
        anchor,
        config,
        physical_twin_core=corpus.physical_core,
    )
    if phase == PHASE_ORDER[0]:
        predecessor = model.load_representation_predecessor(
            resolved["representation_checkpoint"],
            expected_epoch=28,
            require_capability_passed=True,
        )
        phase_lineage = PhaseLineage(
            parent_phase="representation",
            selected_checkpoint=str(resolved["representation_checkpoint"]),
            gate_record=str(config["lineage"]["representation_state"]),
            representation_predecessor=dict(predecessor),
        )
    else:
        if selected_parent is None:
            raise AssertionError("non-D0 phase lost its selected predecessor")
        phase_lineage, selected_payload = selected_parent
        # This is the only initialization for P0 and later.  In particular, no
        # later phase calls load_representation_predecessor and therefore no
        # trained Distribution/Policy parameters can be silently reset.
        model.load_state_dict(selected_payload["model_state"], strict=True)
        predecessor = dict(phase_lineage.representation_predecessor)
    model.to(device)
    training = config["training"]
    optimizers, partition = build_v7_optimizers(
        model,
        distribution_learning_rate=float(training["distribution_learning_rate"]),
        decision_learning_rate=float(training["decision_learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    resume_state = (
        _load_checkpoint(
            checkpoint_path,
            model=model,
            optimizers=optimizers,
            phase=phase,
            seed=seed,
            state_path=state_path,
            expected_lineage=(asdict(phase_lineage) if phase != "d0" else None),
        )
        if resume
        else ResumeState()
    )
    objective_revision = int(training.get("d0_objective_revision", 1))
    if phase == "d0" and resume_state.cycles_completed:
        previous_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        previous_revision = int(previous_payload.get("training_objective_revision", 1))
        if previous_revision != objective_revision:
            # Keep the learned model/optimizer, but do not compare early-stop
            # scores produced by two different registered objectives.
            resume_state = ResumeState(
                cycles_completed=resume_state.cycles_completed,
                optimizer_updates_completed=resume_state.optimizer_updates_completed,
            )
    start_cycle = resume_state.cycles_completed
    scales = _proper_scales(output_root, anchor, device=device)
    rollout = _rollout_callback(model, phase)
    trainer = V7JointTrainer(
        model,
        optimizers,
        baseline_proper_scales=scales,
        cost_scale=corpus.fixed_training_cost_scale,
        channel_scale=anchor.channel_scale.to(device),
        partition=partition,
        rollout_fn=rollout,
        contract=contract,
        distribution_loss_weights=training.get("distribution_loss_weights"),
        policy_loss_weights=training.get("policy_loss_weights"),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        decision_microbatch_days=int(training.get("decision_microbatch_days", 1)),
        decision_autocast_dtype=(
            torch.bfloat16
            if str(training.get("mixed_precision", "")).lower() == "bf16"
            else None
        ),
    )
    batch = _full_train_batch(
        corpus,
        device=device,
        seed=seed,
        channel_scale=anchor.channel_scale,
    )
    history: list[dict[str, Any]] = []
    best_cheap = resume_state.best_cheap_validation
    best_cycle = resume_state.best_cycle
    stale_cycles = resume_state.stale_cycles
    optimizer_updates_completed = resume_state.optimizer_updates_completed
    stopped_early = False
    for cycle in range(start_cycle, spec.maximum_cycles):
        results = trainer.run_coverage_cycle(
            batch,
            phase=phase,
            coverage_cycle_index=cycle,
            seed=seed,
            shuffle=True,
        )
        history.extend(
            {
                "cycle": cycle + 1,
                "phase": item.phase,
                "update_kind": item.update_kind,
                "loss": item.loss,
                "metrics": item.metrics,
            }
            for item in results
        )
        optimizer_updates_completed += len(results)
        if optimizer_updates_completed != (cycle + 1) * 4:
            raise V7RunnerError("trainer did not finish an atomic four-update coverage cycle")
        checkpoint = {
            "schema": "dcvpp.r4.v7-phase-checkpoint.v1",
            "architecture_version": model.architecture_version,
            "phase": phase,
            "seed": seed,
            "cycles_completed": cycle + 1,
            "optimizer_updates_completed": optimizer_updates_completed,
            "model_state": model.state_dict(),
            "optimizer_states": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "proper_scales": scales,
            "representation_predecessor": predecessor,
            "phase_lineage": asdict(phase_lineage),
            "development_split_lineage": dict(corpus.split_lineage),
            "locked_2025_opened": False,
            "training_objective_revision": objective_revision,
        }
        cheap_score: float | None = None
        cheap_metrics: dict[str, float] = {}
        if phase in {"d0", "calibration_restore"}:
            cheap_score, cheap_metrics = _cheap_distribution_validation(
                model,
                corpus,
                anchor,
                device=device,
                proper_scales=scales,
                loss_weights=training["distribution_loss_weights"],
                seed=seed,
            )
            if cheap_score < best_cheap - 1.0e-4:
                best_cheap = cheap_score
                best_cycle = cycle + 1
                stale_cycles = 0
            else:
                stale_cycles += 1
        checkpoint = checkpoint | {
            "resume_state": {
                "cycles_completed": cycle + 1,
                "optimizer_updates_completed": optimizer_updates_completed,
                "best_cheap_validation": None if best_cycle is None else best_cheap,
                "best_cycle": best_cycle,
                "stale_cycles": stale_cycles,
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "updated_at_utc": _utc_now(),
        }
        if best_cycle == cycle + 1:
            candidate_payload = checkpoint | {
                "checkpoint_role": "cheap_validation_candidate",
                "cheap_validation_loss": cheap_score,
                "cheap_validation_metrics": cheap_metrics,
            }
            _torch_save(best_path, candidate_payload)
            if phase == "d0":
                candidate_path = (
                    phase_root
                    / "checkpoints"
                    / "candidates"
                    / f"cycle={cycle + 1:04d}.pt"
                )
                _torch_save(candidate_path, candidate_payload)
                register_d0_candidate(
                    phase_root / "capability" / "d0_candidates.json",
                    cycle=cycle + 1,
                    checkpoint=candidate_path,
                    cheap_validation=cheap_score,
                    cheap_metrics=cheap_metrics,
                )
        _torch_save(checkpoint_path, checkpoint)
        _atomic_json(
            state_path,
            {
                key: value
                for key, value in checkpoint.items()
                if key not in {
                    "model_state",
                    "optimizer_states",
                    "torch_rng_state",
                    "cuda_rng_state_all",
                }
            }
            | {
                "status": "TRAINING" if cycle + 1 < spec.maximum_cycles else "TRAINED_AWAITING_GATE",
                "latest_updates": history[-4:],
                "cheap_validation_loss": cheap_score,
                "cheap_validation_metrics": cheap_metrics,
                "best_cheap_validation": None if best_cycle is None else best_cheap,
                "best_cycle": best_cycle,
                "stale_cycles": stale_cycles,
                "checkpoint": str(checkpoint_path),
            },
        )
        if (
            spec.early_stopping
            and cycle + 1 >= spec.minimum_cycles
            and stale_cycles
            >= int(training.get("early_stopping_patience_cycles", 12))
        ):
            stopped_early = True
            break
    completed_cycles = (cycle + 1) if "cycle" in locals() else start_cycle
    d0_selection: dict[str, Any] | None = None
    final_status = "TRAINED_AWAITING_GATE"
    if phase == "d0":
        validation_split = config["protocol"]["splits"]["validation"]
        d0_selection = gate_and_select_d0_distribution(
            model=model,
            corpus=corpus,
            anchor=anchor,
            phase_root=phase_root,
            device=device,
            seed=seed,
            top_k=contract.pareto_top_k,
            validation_start=str(validation_split["start"]),
            validation_end=str(validation_split["end"]),
            expected_days=int(validation_split["days"]),
        )
        final_status = "PASSED" if d0_selection["status"] == "PASS" else "FAILED_GATE"
    final_payload = {
        "status": final_status,
        "phase": phase,
        "seed": seed,
        "cycles_completed": completed_cycles,
        "optimizer_updates_completed": completed_cycles * 4,
        "stopped_early": stopped_early,
        "best_cycle": best_cycle,
        "best_cheap_validation": None if best_cycle is None else best_cheap,
        "stale_cycles": stale_cycles,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "selected_checkpoint": (
            None if d0_selection is None else d0_selection["selected_checkpoint"]
        ),
        "checkpoint": str(checkpoint_path),
        "state": str(state_path),
        "phase_lineage": asdict(phase_lineage),
        "representation_predecessor": predecessor,
        "development_split_lineage": dict(corpus.split_lineage),
        "locked_2025_opened": False,
        "capability_gate": (
            None
            if d0_selection is None
            else {
                "status": d0_selection["status"],
                "selected_cycle": d0_selection["selected_cycle"],
                "selected_checkpoint": d0_selection["selected_checkpoint"],
                "selection_artifact": str(
                    phase_root / "capability" / "d0_distribution_selection.json"
                ),
                "mae_used": False,
            }
        ),
    }
    _atomic_json(state_path, final_payload | {
        "architecture_version": model.architecture_version,
        "updated_at_utc": _utc_now(),
    })
    return final_payload


__all__ = ["V7RunnerError", "run_phase"]
