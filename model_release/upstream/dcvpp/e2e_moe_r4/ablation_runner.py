"""Executable train/validation-only runner for preregistered R4 ablations.

The runner reuses the formal corpus, optimizer state machine, immutable
selected-best checkpoints, and live 179-day/17,184-origin gates.  It has its
own method-scoped lineage and output directories, and checks the completed
mainline seed-1 chain before constructing a corpus or resolving a data path.

No function in this module can select or execute 2024, and 2025 is rejected
before any data object is created.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gc
import json
import math
import subprocess
import sys

import torch
from torch import nn

from .ablation_runner_types import AblationExecutionState
from .ablations import (
    ALL_ABLATION_METHODS,
    FULL_PHASES,
    AblationContractError,
    AblationRunSpec,
    build_ablation_model,
    build_execution_plan,
    default_preregistration,
    method_registry,
    write_execution_plan_once,
)
from .artifact_io import atomic_json
from .config import FORMAL_SEEDS, load_r4_config
from .formal_protocol import ProtocolContext, prepare_protocol, protocol_status
from .hashing import sha256_file, sha256_json
from .model import R4Config
from .phase_runner import (
    _attach_a1_behavioral_baseline,
    _cheap_validator,
    _choose_physical_batch,
    _device,
    _evidence_evaluator,
)
from .physical_twin import load_frozen_physical_twin_core
from .trainer import AtomicRunStore, CheapValidationPolicy, R4PhaseTrainer, R4ProtocolError, build_phase_loss
from .training_data import PreloadedR4Corpus


ABLATION_EXECUTION_SCHEMA = "dcvpp.e2e_moe_hybrid_r4.ablation_execution.v1"


class AblationExecutionError(R4ProtocolError):
    """Fail-closed error raised before or during a registered ablation run."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AblationExecutionError(f"{label} must be a mapping")
    return value


def _ablation_root(context: ProtocolContext) -> Path:
    return Path(context.output_root).resolve() / "ablations"


def _method_root(context: ProtocolContext, method_id: str) -> Path:
    return _ablation_root(context) / f"method={method_id}"


def _seed_root(context: ProtocolContext, method_id: str, seed: int) -> Path:
    return _method_root(context, method_id) / f"seed={int(seed)}"


def _phase_root(
    context: ProtocolContext, method_id: str, seed: int, phase: str
) -> Path:
    return _seed_root(context, method_id, seed) / f"phase={phase}"


def _method_lineage(context: ProtocolContext, spec: AblationRunSpec) -> str:
    signature = sha256_json(
        {
            "mainline_lineage": context.lineage_id,
            "method": spec.to_payload(),
        }
    )
    return f"{context.lineage_id}-ablation-{spec.method_id.lower()}-{signature[:12]}"


def _store(
    context: ProtocolContext,
    spec: AblationRunSpec,
    phase: str,
) -> AtomicRunStore:
    return AtomicRunStore(
        _phase_root(context, spec.method_id, spec.seed, phase),
        lineage_id=_method_lineage(context, spec),
        config_sha256=context.config_sha256,
        config_path=context.config_path,
    )


def _assert_train_validation_only(config: Mapping[str, Any]) -> None:
    protocol = _mapping(config.get("protocol"), "protocol")
    splits = _mapping(protocol.get("splits"), "protocol.splits")
    train = _mapping(splits.get("train"), "protocol.splits.train")
    validation = _mapping(splits.get("validation"), "protocol.splits.validation")
    locked = _mapping(splits.get("locked"), "protocol.splits.locked")
    if (train.get("start"), train.get("end")) != ("2022-01-01", "2023-06-30"):
        raise AblationExecutionError("ablation train split must remain 2022..2023-H1")
    if (validation.get("start"), validation.get("end")) != (
        "2023-07-01",
        "2023-12-31",
    ):
        raise AblationExecutionError("ablation validation split must remain 2023-H2")
    if int(locked.get("year", -1)) != 2025 or locked.get("reject_path_before_open") is not True:
        raise AblationExecutionError("2025 path-before-open lock is not active")


def _mainline_seed1_pass(context: ProtocolContext) -> Mapping[str, Any]:
    """Verify all five mainline gates before constructing the ablation corpus."""

    state = protocol_status(context.config_path)
    seed1 = state.get("seed1") if isinstance(state, Mapping) else None
    if not isinstance(seed1, Mapping) or seed1.get("status") != "COMPLETE":
        raise AblationExecutionError(
            "R4 mainline seed 20240813 has not completed all five capability gates"
        )
    phases = seed1.get("phases")
    if not isinstance(phases, Mapping) or tuple(phases) != FULL_PHASES:
        raise AblationExecutionError("mainline seed1 phase chain is incomplete or reordered")
    for phase in FULL_PHASES:
        result = phases.get(phase)
        if not isinstance(result, Mapping) or result.get("status") != "COMPLETE":
            raise AblationExecutionError(f"mainline seed1 {phase} is not COMPLETE")
        gate = result.get("capability_gate")
        status = gate.get("status") if isinstance(gate, Mapping) else gate
        if status not in {"PASS", "PASSED"}:
            raise AblationExecutionError(f"mainline seed1 {phase} gate did not pass")
        if result.get("formal_selection_2024_opened") is not False:
            raise AblationExecutionError("mainline state says 2024 was opened")
    return seed1


def _verify_execution_environment(context: ProtocolContext) -> None:
    """Reuse the single measured preflight's exact Python/CUDA environment."""

    path = Path(context.output_root).resolve() / "preflight" / "metrics.json"
    if not path.is_file():
        raise AblationExecutionError("measured R4 preflight metrics are missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks") if isinstance(payload, Mapping) else None
    if not isinstance(checks, list):
        raise AblationExecutionError("measured preflight checks are malformed")
    environment = next(
        (
            item
            for item in checks
            if isinstance(item, Mapping)
            and item.get("name") == "verified_python_cuda_environment"
            and item.get("passed") is True
        ),
        None,
    )
    evidence = environment.get("evidence") if isinstance(environment, Mapping) else None
    if not isinstance(evidence, Mapping):
        raise AblationExecutionError("verified Python/CUDA preflight evidence is absent")
    expected = Path(str(evidence.get("python_executable", ""))).resolve()
    observed = Path(sys.executable).resolve()
    if expected != observed:
        raise AblationExecutionError(
            f"ablation must reuse verified Python {expected}; current interpreter is {observed}"
        )
    if evidence.get("cuda_available") is not True or not torch.cuda.is_available():
        raise AblationExecutionError("verified CUDA is unavailable in the current process")


def _canonical_plan(context: ProtocolContext) -> tuple[AblationRunSpec, ...]:
    registration = default_preregistration()
    root = _ablation_root(context)
    registration.write_once(root / "preregistration.json")
    specs = build_execution_plan(registration=registration)
    write_execution_plan_once(
        root / "canonical_execution_plan.json", specs, registration=registration
    )
    return specs


def _run_spec(
    specs: Sequence[AblationRunSpec], method_id: str, seed: int
) -> AblationRunSpec:
    for spec in specs:
        if spec.method_id == method_id and spec.seed == int(seed):
            return spec
    raise AblationExecutionError(
        f"no preregistered execution cell for method={method_id}, seed={seed}"
    )


def _apply_method_overrides(
    config: Mapping[str, Any], spec: AblationRunSpec
) -> dict[str, Any]:
    result = deepcopy(dict(config))
    phases = result["training"]["phases"]
    if not spec.routing_enabled:
        for phase in ("policy", "a2"):
            weights = phases[phase]["loss_weights"]
            weights.pop("router_balance", None)
            weights.pop("router_stability", None)
    if not spec.cvar_training_weight_enabled:
        for phase in ("policy", "a1", "a2"):
            phases[phase]["loss_weights"]["particle_cvar"] = 0.0
    if spec.scenario_count == 1:
        # Impossible diversity/entropy targets are not optimized.  The single
        # learned path is still scored on all predictive and causal metrics.
        phases["scenario"]["loss_weights"]["diversity"] = 0.0
        phases["scenario"]["loss_weights"]["weight_entropy"] = 0.0
    result["_ablation_method_id"] = spec.method_id
    result["_ablation_preregistration_sha256"] = spec.preregistration_sha256
    return result


def _resolve_data_artifact(
    config: Mapping[str, Any], repository_root: str | Path, name: str
) -> Path:
    data = _mapping(config.get("data"), "config.data")
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AblationExecutionError(f"config.data.{name} must be an explicit path")
    # Reject locked/formal year paths before filesystem metadata is queried.
    normalized = value.replace("\\", "/").lower()
    if "2024" in normalized or "2025" in normalized:
        raise AblationExecutionError(
            f"ablation input path mentions a sealed/locked year: config.data.{name}"
        )
    path = Path(value)
    root = Path(repository_root).resolve()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _build_model(
    config: Mapping[str, Any],
    context: ProtocolContext,
    spec: AblationRunSpec,
) -> nn.Module:
    torch.manual_seed(int(spec.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(spec.seed))
    core = load_frozen_physical_twin_core(
        artifact_path=_resolve_data_artifact(
            config, context.repository_root, "frozen_physical_twin"
        ),
        surrogate_path=_resolve_data_artifact(
            config, context.repository_root, "frozen_physical_surrogate"
        ),
        calibration_manifest_path=_resolve_data_artifact(
            config, context.repository_root, "physical_calibration_manifest"
        ),
    )
    base = R4Config.from_mapping(config)
    model_config = replace(
        base,
        max_grid_action=float(core.grid_import_limit_mw.max().item()),
        max_battery_action=float(core.battery_power_mw.max().item()),
        max_compute_action=float(core.capacity_work_units.max().item()),
    )
    model = build_ablation_model(
        spec.method_id,
        model_config,
        physical_twin_core=core,
    )
    model.formal_physical_lineage = core.artifact_metadata
    model.ablation_preregistration_sha256 = spec.preregistration_sha256
    model.ablation_shared_contract_sha256 = spec.shared_contract_sha256
    return model


def _selected_predecessor(
    context: ProtocolContext,
    spec: AblationRunSpec,
    phase: str,
) -> Path | None:
    phases = spec.phase_sequence
    index = phases.index(phase)
    if index == 0:
        return None
    previous = phases[index - 1]
    store = _store(context, spec, previous)
    store.load_selected_checkpoint(previous, spec.seed, map_location="cpu")
    return store.selected_best_path(previous, spec.seed)


def _load_model_state(model: nn.Module, checkpoint: Path | None) -> None:
    if checkpoint is None:
        return
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise AblationExecutionError(f"predecessor has no model_state: {checkpoint}")
    model.load_state_dict(payload["model_state"], strict=True)


class _OneScenarioStructuralGate:
    """Live 179-day gate for the preregistered one-unique-path exception."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]:
        model = kwargs.get("model")
        generator = getattr(model, "scenario_generator", None)
        if int(getattr(generator, "num_scenarios", -1)) != 1:
            raise AblationExecutionError("one-scenario gate received a non-single-code model")
        raw = dict(self.delegate(**kwargs))
        evidence_path = Path(str(raw.get("evidence_path", ""))).resolve()
        if not evidence_path.is_file() or sha256_file(evidence_path) != raw.get(
            "evidence_sha256"
        ):
            raise AblationExecutionError("one-scenario live evidence is missing or hash-mismatched")
        materialized = json.loads(evidence_path.read_text(encoding="utf-8"))
        rows = materialized.get("daily_rows")
        if not isinstance(rows, list) or len(rows) != 179:
            raise AblationExecutionError("one-scenario gate requires 179 live daily rows")
        dates = tuple(str(row.get("date", ""))[:10] for row in rows)
        if len(set(dates)) != 179 or not all(
            "2023-07-01" <= value <= "2023-12-31" for value in dates
        ):
            raise AblationExecutionError("one-scenario evidence dates are invalid")
        required_flags = (
            "finite_weighted_joint_exact20",
            "revealed_prefix_identity_passed",
        )
        if not all(row.get(flag) is True for row in rows for flag in required_flags):
            raise AblationExecutionError("one-scenario structural/causal live audit failed")
        pairwise = [float(row["model_average_pairwise_particle_distance"]) for row in rows]
        prefix = [float(row["model_revealed_prefix_identity_max_abs"]) for row in rows]
        identity = [float(row["model_da_rt_dart_identity_max_abs"]) for row in rows]
        values = (*pairwise, *prefix, *identity)
        if not all(math.isfinite(value) for value in values):
            raise AblationExecutionError("one-scenario structural metrics are non-finite")
        if max(pairwise) > 1.0e-8:
            raise AblationExecutionError(
                "one-scenario transport unexpectedly contains multiple unique paths"
            )
        if max(prefix) > 1.0e-6 or max(identity) > 1.0e-6:
            raise AblationExecutionError("one-scenario identity tolerance failed")
        metrics = {
            key: float(value)
            for key, value in _mapping(raw.get("metrics"), "one-scenario metrics").items()
        }
        metrics.update(
            {
                "unique_scenario_count": 1.0,
                "transport_scenario_count": 20.0,
                "maximum_transport_pairwise_distance": max(pairwise),
                "maximum_revealed_prefix_identity_residual": max(prefix),
                "maximum_da_rt_dart_identity_residual": max(identity),
            }
        )
        decision = {
            "schema": "dcvpp.r4.one_scenario_ablation_gate.v1",
            "status": "PASS",
            "passed": True,
            "phase": "scenario",
            "horizon": 96,
            "dates": list(dates),
            "metrics": metrics,
            "reasons": [],
            "mainline_twenty_particle_gate_claimed": False,
            "acceptance_scope": "preregistered_operational_ablation_only",
            "source_evidence": str(evidence_path),
            "source_evidence_sha256": raw["evidence_sha256"],
        }
        destination = evidence_path.with_name("one_scenario_structural_gate.json")
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != decision:
                raise AblationExecutionError("immutable one-scenario gate evidence changed")
        else:
            atomic_json(destination, decision)
        return decision


def _phase_complete(
    context: ProtocolContext, spec: AblationRunSpec, phase: str
) -> Mapping[str, Any] | None:
    store = _store(context, spec, phase)
    state = store.read_state()
    if not isinstance(state, Mapping) or state.get("status") != "COMPLETE":
        return None
    gate = state.get("capability_gate")
    status = gate.get("status") if isinstance(gate, Mapping) else gate
    if status not in {"PASS", "PASSED"}:
        raise AblationExecutionError(f"{spec.method_id}/{phase} COMPLETE lacks a PASS gate")
    store.load_selected_checkpoint(phase, spec.seed, map_location="cpu")
    return state


def run_ablation_phase(
    *,
    context: ProtocolContext,
    config: Mapping[str, Any],
    spec: AblationRunSpec,
    corpus: PreloadedR4Corpus,
    phase: str,
    resume: bool,
) -> Mapping[str, Any]:
    """Execute one real R4 trainer phase inside a method-scoped lineage."""

    if phase not in spec.phase_sequence:
        raise AblationExecutionError(f"phase {phase} is not registered for {spec.method_id}")
    if resume:
        complete = _phase_complete(context, spec, phase)
        if complete is not None:
            return {**dict(complete), "resumed_complete": True}
    phase_dir = _phase_root(context, spec.method_id, spec.seed, phase)
    phase_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(config, context, spec)
    hard_limit = int(config["model"]["parameter_budget"]["hard_stop_above_trainable"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > hard_limit:
        raise AblationExecutionError(
            f"{spec.method_id} parameter count {parameter_count} exceeds {hard_limit}"
        )
    predecessor = _selected_predecessor(context, spec, phase)
    _load_model_state(model, predecessor)
    behavioral_baseline: nn.Module | None = None
    if phase == "a1":
        if predecessor is None:
            raise AblationExecutionError("A1 requires the selected policy checkpoint")
        behavioral_baseline = deepcopy(model).eval()
        behavioral_baseline.requires_grad_(False)
    device = _device()
    model.to(device)
    if behavioral_baseline is not None:
        behavioral_baseline.to(device).eval()
    model.configure_trainable_phase(phase)
    phase_config = config["training"]["phases"][phase]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(phase_config["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    physical_batch = _choose_physical_batch(model, corpus, phase, config, device)
    store = _store(context, spec, phase)
    evaluator = _evidence_evaluator(
        phase_dir,
        phase,
        spec.seed,
        corpus,
        config,
        context.repository_root,
        _method_root(context, spec.method_id),
    )
    if spec.method_id == "one_scenario" and phase == "scenario":
        evaluator = _OneScenarioStructuralGate(evaluator)
    trainer = R4PhaseTrainer(
        model,
        optimizer,
        store=store,
        cheap_validator=_cheap_validator(
            corpus,
            config,
            device,
            behavioral_baseline=behavioral_baseline,
        ),
        phase_end_evaluator=evaluator,
        physical_batch_size=physical_batch,
        cheap_validation_policy=CheapValidationPolicy(max_days=16),
        expected_training_dates=[
            str(item["operating_date"]) for item in corpus.records("train")
        ],
        predecessor_checkpoint=predecessor,
    )
    loss = build_phase_loss(
        phase,
        config,
        fixed_training_cost_scale=(
            corpus.fixed_training_cost_scale if phase in {"policy", "a1", "a2"} else None
        ),
    )

    def batches(epoch: int, horizon: int):
        for batch in corpus.batches(
            "train",
            physical_batch,
            device=device,
            shuffle_seed=int(spec.seed) + int(epoch),
            origin=0,
        ):
            if phase == "a1":
                assert behavioral_baseline is not None
                batch = _attach_a1_behavioral_baseline(
                    batch, behavioral_baseline, horizon
                )
            yield batch

    result = trainer.run(phase, spec.seed, batches, loss, resume=resume)
    selected = store.selected_best_path(phase, spec.seed)
    selected_sha = sha256_file(selected) if selected.is_file() else None
    del trainer, optimizer, model, behavioral_baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        **asdict(result),
        "method_id": spec.method_id,
        "architecture_id": spec.architecture_id,
        "ablation_lineage_id": _method_lineage(context, spec),
        "preregistration_sha256": spec.preregistration_sha256,
        "shared_contract_sha256": spec.shared_contract_sha256,
        "physical_batch_size": physical_batch,
        "effective_batch_size": 512,
        "parameter_count": parameter_count,
        "selected_best": str(selected) if selected.is_file() else None,
        "selected_best_sha256": selected_sha,
        "predecessor_checkpoint": None if predecessor is None else str(predecessor),
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }


def _read_outer_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AblationExecutionError(f"invalid ablation state: {path}")
    return value


def _write_outer_state(
    context: ProtocolContext,
    spec: AblationRunSpec,
    *,
    status: str,
    phases: Mapping[str, Any],
    reason: str | None = None,
    source: Mapping[str, Any] | None = None,
) -> Path:
    resume_command = subprocess.list2cmdline(
        [
            sys.executable,
            str(Path(context.repository_root) / "scripts" / "run_r4_ablations.py"),
            "continue",
            spec.method_id,
            "--seed",
            str(spec.seed),
            "--config",
            context.config_path,
        ]
    )
    state = AblationExecutionState(
        schema_version=ABLATION_EXECUTION_SCHEMA,
        status=status,
        method_id=spec.method_id,
        architecture_id=spec.architecture_id,
        seed=spec.seed,
        split="train_validation_only",
        phase_sequence=spec.phase_sequence,
        phases=dict(phases),
        checkpoint_source=spec.checkpoint_source,
        ablation_lineage_id=_method_lineage(context, spec),
        preregistration_sha256=spec.preregistration_sha256,
        shared_contract_sha256=spec.shared_contract_sha256,
        reason=reason,
        source=dict(source or {}),
        resume_command=resume_command,
    )
    destination = _seed_root(context, spec.method_id, spec.seed) / "state.json"
    atomic_json(destination, state.to_payload())
    return destination


def _require_method_seed1_complete(context: ProtocolContext, method_id: str) -> None:
    path = _seed_root(context, method_id, FORMAL_SEEDS[0]) / "state.json"
    state = _read_outer_state(path)
    if not isinstance(state, Mapping) or state.get("status") != "COMPLETE":
        raise AblationExecutionError(
            f"{method_id} seed {FORMAL_SEEDS[0]} must complete before later seeds"
        )


def _complete_alias(
    context: ProtocolContext,
    spec: AblationRunSpec,
    all_specs: Sequence[AblationRunSpec],
) -> Mapping[str, Any]:
    source_method = spec.checkpoint_source.removesuffix("_same_seed")
    source_spec = next(
        (
            item
            for item in all_specs
            if item.method_id == source_method and item.seed == spec.seed
        ),
        None,
    )
    if source_spec is None:
        raise AblationExecutionError(f"missing registered source method {source_method}")
    source_state_path = _seed_root(context, source_method, spec.seed) / "state.json"
    source_state = _read_outer_state(source_state_path)
    if not isinstance(source_state, Mapping) or source_state.get("status") != "COMPLETE":
        raise AblationExecutionError(
            f"{spec.method_id} requires completed {source_method} seed {spec.seed}"
        )
    final_phase = source_spec.phase_sequence[-1]
    source_store = _store(context, source_spec, final_phase)
    source_store.load_selected_checkpoint(final_phase, spec.seed, map_location="cpu")
    checkpoint = source_store.selected_best_path(final_phase, spec.seed)
    source = {
        "method_id": source_method,
        "state": str(source_state_path),
        "state_sha256": sha256_file(source_state_path),
        "selected_phase": final_phase,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "reuse_kind": (
            "exact_architecture_alias"
            if spec.method_id == "dense_no_moe"
            else "paired_raw_and_shielded_views_from_same_execution"
        ),
    }
    path = _write_outer_state(
        context,
        spec,
        status="COMPLETE",
        phases={},
        source=source,
    )
    return {
        "status": "COMPLETE",
        "method_id": spec.method_id,
        "seed": spec.seed,
        "checkpoint_source": spec.checkpoint_source,
        "source": source,
        "state_path": str(path),
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }


def run_ablation_method(
    config_path: str | Path,
    method_id: str,
    seed: int = FORMAL_SEEDS[0],
    *,
    resume: bool = True,
    _corpus_factory: Callable[[Mapping[str, Any], str | Path], PreloadedR4Corpus] = PreloadedR4Corpus,
    _phase_executor: Callable[..., Mapping[str, Any]] = run_ablation_phase,
) -> Mapping[str, Any]:
    """Run/continue one registered method through its exact phase sequence."""

    method_id = str(method_id)
    if method_id not in method_registry():
        raise AblationExecutionError(f"unknown ablation method {method_id!r}")
    if int(seed) not in FORMAL_SEEDS:
        raise AblationExecutionError(f"unregistered ablation seed {seed}")
    context = prepare_protocol(config_path)
    config = load_r4_config(config_path)
    _assert_train_validation_only(config)
    # This is intentionally before _canonical_plan, corpus construction, model
    # construction, or artifact-path resolution.
    _mainline_seed1_pass(context)
    _verify_execution_environment(context)
    all_specs = _canonical_plan(context)
    spec = _run_spec(all_specs, method_id, int(seed))
    if int(seed) != FORMAL_SEEDS[0]:
        _require_method_seed1_complete(context, method_id)
    if not resume:
        seed_root = _seed_root(context, method_id, int(seed))
        if seed_root.exists() and any(seed_root.iterdir()):
            raise AblationExecutionError(
                f"fresh run refuses existing ablation state; use continue: {seed_root}"
            )
    if spec.checkpoint_source != "fresh":
        return _complete_alias(context, spec, all_specs)
    method_config = _apply_method_overrides(config, spec)
    # Only train/validation records are constructed, after every lock/gate.
    corpus = _corpus_factory(method_config, context.repository_root)
    train_dates = tuple(str(row["operating_date"]) for row in corpus.records("train"))
    validation_dates = tuple(
        str(row["operating_date"]) for row in corpus.records("validation")
    )
    if (
        len(train_dates) != 543
        or len(set(train_dates)) != 543
        or not all("2022-01-01" <= value <= "2023-06-30" for value in train_dates)
    ):
        raise AblationExecutionError("ablation corpus train coverage is not exact 543-day R4")
    if (
        len(validation_dates) != 179
        or len(set(validation_dates)) != 179
        or not all("2023-07-01" <= value <= "2023-12-31" for value in validation_dates)
    ):
        raise AblationExecutionError("ablation corpus validation coverage is not 179-day 2023-H2")
    seed_state_path = _seed_root(context, method_id, int(seed)) / "state.json"
    existing = _read_outer_state(seed_state_path) if resume else None
    completed_phases = (
        dict(existing.get("phases", {})) if isinstance(existing, Mapping) else {}
    )
    for phase in spec.phase_sequence:
        result = dict(
            _phase_executor(
                context=context,
                config=method_config,
                spec=spec,
                corpus=corpus,
                phase=phase,
                resume=resume,
            )
        )
        completed_phases[phase] = result
        if result.get("status") != "COMPLETE":
            status = "FAILED" if result.get("status") == "FAILED" else "PARTIAL"
            reason = str(result.get("reason") or f"{phase} did not complete")
            state_path = _write_outer_state(
                context,
                spec,
                status=status,
                phases=completed_phases,
                reason=reason,
            )
            return {
                "status": status,
                "method_id": method_id,
                "seed": int(seed),
                "stopped_phase": phase,
                "reason": reason,
                "phases": completed_phases,
                "state_path": str(state_path),
                "formal_selection_2024_opened": False,
                "locked_test_2025_opened": False,
            }
    state_path = _write_outer_state(
        context,
        spec,
        status="COMPLETE",
        phases=completed_phases,
    )
    return {
        "status": "COMPLETE",
        "method_id": method_id,
        "architecture_id": spec.architecture_id,
        "seed": int(seed),
        "phase_sequence": list(spec.phase_sequence),
        "phases": completed_phases,
        "state_path": str(state_path),
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }


def ablation_status(
    config_path: str | Path,
    method_id: str | None = None,
    seed: int | None = None,
) -> Mapping[str, Any]:
    context = prepare_protocol(config_path)
    methods = (method_id,) if method_id is not None else ALL_ABLATION_METHODS
    seeds = (int(seed),) if seed is not None else FORMAL_SEEDS
    rows: list[dict[str, Any]] = []
    for method in methods:
        for current_seed in seeds:
            path = _seed_root(context, method, current_seed) / "state.json"
            state = _read_outer_state(path)
            rows.append(
                {
                    "method_id": method,
                    "seed": current_seed,
                    "status": "NOT_STARTED" if state is None else state.get("status"),
                    "state_path": str(path),
                }
            )
    return {
        "schema": ABLATION_EXECUTION_SCHEMA,
        "cells": rows,
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }


__all__ = [
    "ABLATION_EXECUTION_SCHEMA",
    "AblationExecutionError",
    "ablation_status",
    "run_ablation_method",
    "run_ablation_phase",
]
