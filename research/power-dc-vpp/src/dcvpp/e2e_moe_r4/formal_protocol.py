"""Fail-closed formal protocol orchestration for the R4 lineages.

The archived V5 path remains executable for reproducibility.  The active V6
path is an independent Continuous Distributional lineage which may inherit
only the capability-passed Representation checkpoint and frozen train-only
contracts explicitly named by its YAML.  Neither path discovers or opens a
2024/2025 payload during preparation, preflight, or training.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import (
    ARCHITECTURE_VERSION,
    DISTRIBUTIONAL_ARCHITECTURE_VERSION,
    DISTRIBUTIONAL_EXPERIMENT_ID,
    DISTRIBUTIONAL_FORMAL_OUTPUT_ROOT,
    DISTRIBUTIONAL_PHASE_ORDER,
    DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION,
    EXPERIMENT_ID,
    FORMAL_OUTPUT_ROOT,
    TRAINING_SEMANTICS_REVISION,
    load_r4_config,
)
from .hashing import sha256_file

from .trainer import (
    AtomicRunStore,
    PHASE_ORDER,
    R4_PHASE_SPECS,
    R4_PROTOCOL_VERSION,
    R4_SEEDS,
    R4ProtocolError,
    distribution_curriculum_contract,
    phase_spec,
)


EXPECTED_CONFIG_NAME = "e2e_moe_dart_vpp_R4_hybrid_solver_free_formal.yaml"
EXPECTED_OUTPUT_ROOT = Path(FORMAL_OUTPUT_ROOT)
DISTRIBUTIONAL_EXPECTED_CONFIG_NAME = (
    "e2e_moe_dart_vpp_R4_distributional_joint_formal.yaml"
)
DISTRIBUTIONAL_EXPECTED_OUTPUT_ROOT = Path(DISTRIBUTIONAL_FORMAL_OUTPUT_ROOT)


@dataclass(frozen=True)
class _ProtocolProfile:
    config_name: str
    output_root: Path
    architecture_version: str
    experiment_id: str
    training_semantics_revision: str
    phase_order: tuple[str, ...]
    lineage_prefix: str


@dataclass(frozen=True)
class ProtocolContext:
    config_path: str
    repository_root: str
    output_root: str
    config_sha256: str
    lineage_id: str
    seeds: tuple[int, ...]
    phase_specs: Mapping[str, Mapping[str, Any]]
    phase_order: tuple[str, ...] = PHASE_ORDER
    architecture_version: str = ARCHITECTURE_VERSION
    experiment_id: str = EXPERIMENT_ID
    training_semantics_revision: str = TRAINING_SEMANTICS_REVISION
    status: str = "PREPARED"

    @property
    def is_distributional(self) -> bool:
        return self.architecture_version == DISTRIBUTIONAL_ARCHITECTURE_VERSION


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R4ProtocolError(f"{name} must be a mapping")
    return value


def _repository_root(config_path: Path) -> Path:
    if config_path.parent.name != "configs":
        raise R4ProtocolError("R4 formal config must reside in the repository configs directory")
    return config_path.parent.parent.resolve()


def _profile(config: Mapping[str, Any]) -> _ProtocolProfile:
    project = _mapping(config.get("project", {}), "project")
    architecture = str(project.get("architecture_version", ""))
    if architecture == DISTRIBUTIONAL_ARCHITECTURE_VERSION:
        return _ProtocolProfile(
            DISTRIBUTIONAL_EXPECTED_CONFIG_NAME,
            DISTRIBUTIONAL_EXPECTED_OUTPUT_ROOT,
            DISTRIBUTIONAL_ARCHITECTURE_VERSION,
            DISTRIBUTIONAL_EXPERIMENT_ID,
            DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION,
            DISTRIBUTIONAL_PHASE_ORDER,
            "r4v6-distributional",
        )
    if architecture == ARCHITECTURE_VERSION:
        return _ProtocolProfile(
            EXPECTED_CONFIG_NAME,
            EXPECTED_OUTPUT_ROOT,
            ARCHITECTURE_VERSION,
            EXPERIMENT_ID,
            TRAINING_SEMANTICS_REVISION,
            PHASE_ORDER,
            "r4-fresh",
        )
    raise R4ProtocolError(f"unregistered formal R4 architecture: {architecture!r}")


def _phase_contract_payload(
    config: Mapping[str, Any], order: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    is_distributional = (
        str(_mapping(config.get("project", {}), "project").get("architecture_version"))
        == DISTRIBUTIONAL_ARCHITECTURE_VERSION
    )
    for name in order:
        spec = phase_spec(name, config if name == "distribution" else None)
        entry = {
            **asdict(spec),
            "curriculum": [asdict(stage) for stage in spec.curriculum],
        }
        if is_distributional and name == "representation":
            configured = _mapping(
                _mapping(
                    _mapping(config.get("training", {}), "training").get("phases", {}),
                    "training.phases",
                ).get("representation", {}),
                "training.phases.representation",
            )
            entry.update(
                {
                    "mode": configured.get("mode"),
                    "selected_epoch": int(configured.get("selected_epoch", -1)),
                    "trainable": bool(configured.get("trainable", True)),
                    "epochs": 0,
                    "minimum_epochs": 0,
                }
            )
        if name == "distribution":
            entry["distribution_contract"] = distribution_curriculum_contract(
                config
            ).as_manifest()
        payload[name] = entry
    return payload


def _verify_contract(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Path, tuple[int, ...], _ProtocolProfile]:
    profile = _profile(config)
    if config_path.name != profile.config_name:
        raise R4ProtocolError(
            f"formal R4 config must be named {profile.config_name}, got {config_path.name}"
        )
    project = _mapping(config.get("project", {}), "project")
    expected_version = "R4-V6" if profile.architecture_version == DISTRIBUTIONAL_ARCHITECTURE_VERSION else "R4"
    if str(project.get("version", "")).upper() != expected_version:
        raise R4ProtocolError(f"project.version must be {expected_version}")
    output_value = project.get("output_root", str(profile.output_root).replace("\\", "/"))
    if Path(str(output_value)).as_posix() != profile.output_root.as_posix():
        raise R4ProtocolError(
            f"R4 output_root is fixed at {profile.output_root.as_posix()}"
        )

    protocol = _mapping(config.get("protocol", {}), "protocol")
    seeds = tuple(int(item) for item in protocol.get("seeds", R4_SEEDS))
    if seeds != R4_SEEDS:
        raise R4ProtocolError(f"R4 seeds are fixed and ordered as {R4_SEEDS}")
    splits = _mapping(protocol.get("splits", {}), "protocol.splits")
    train = _mapping(splits.get("train", {}), "protocol.splits.train")
    validation = _mapping(splits.get("validation", {}), "protocol.splits.validation")
    if (str(train.get("start")), str(train.get("end"))) != (
        "2022-01-01", "2023-06-30"
    ):
        raise R4ProtocolError("R4 train split must be 2022-01-01..2023-06-30")
    if (str(validation.get("start")), str(validation.get("end"))) != (
        "2023-07-01", "2023-12-31"
    ):
        raise R4ProtocolError("R4 validation split must be 2023-H2")
    formal = _mapping(
        splits.get("final_temporal_test", splits.get("formal_selection", {})),
        "protocol.splits.final_temporal_test",
    )
    if not bool(formal.get("sealed_until_seed1_complete", formal.get("sealed_until_all_methods_frozen", False))):
        raise R4ProtocolError("2024 formal selection must remain sealed until seed1 is complete")
    if tuple(protocol.get("phase_order", profile.phase_order)) != profile.phase_order:
        raise R4ProtocolError(
            f"formal phase order must be {profile.phase_order}"
        )
    if bool(project.get("initialize_from_r3", False)):
        raise R4ProtocolError("R4 requires fresh lineage and initialize_from_r3=false")
    forbidden = ("resume_from", "warm_start", "legacy_checkpoint", "r3_checkpoint")
    for key in forbidden:
        if protocol.get(key) not in (None, "", False):
            raise R4ProtocolError(f"fresh R4 lineage forbids protocol.{key}")
    if profile.architecture_version == DISTRIBUTIONAL_ARCHITECTURE_VERSION:
        if project.get("preserve_v5_read_only") is not True:
            raise R4ProtocolError("V6 must preserve V5 artifacts read-only")
        if project.get("reuse_selected_representation_only") is not True:
            raise R4ProtocolError("V6 may inherit only the selected Representation")
        if project.get("resume_fixed_particle_scenario") is not False:
            raise R4ProtocolError("V6 must not resume the failed fixed-particle Scenario")
    return _repository_root(config_path), seeds, profile


def prepare_protocol(config_path: str | Path) -> ProtocolContext:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise R4ProtocolError(f"R4 config does not exist: {path}")
    # The governing R4 validator is the single source of truth.  It validates
    # the entire schema before protocol code may inspect a data path.
    config = load_r4_config(path)
    repository_root, seeds, profile = _verify_contract(config, path)
    digest = sha256_file(path)
    lineage = f"{profile.lineage_prefix}-{digest[:16]}"
    output_root = (repository_root / profile.output_root).resolve()
    return ProtocolContext(
        config_path=str(path),
        repository_root=str(repository_root),
        output_root=str(output_root),
        config_sha256=digest,
        lineage_id=lineage,
        seeds=seeds,
        phase_specs=_phase_contract_payload(config, profile.phase_order),
        phase_order=profile.phase_order,
        architecture_version=profile.architecture_version,
        experiment_id=profile.experiment_id,
        training_semantics_revision=profile.training_semantics_revision,
    )


def _store(context: ProtocolContext) -> AtomicRunStore:
    return AtomicRunStore(
        context.output_root,
        lineage_id=context.lineage_id,
        config_sha256=context.config_sha256,
        config_path=context.config_path,
    )


def run_preflight(config_path: str | Path, resume: bool = True) -> dict[str, Any]:
    context = prepare_protocol(config_path)
    store = _store(context)
    preflight_state = Path(context.output_root) / "preflight" / "state.json"
    preflight_metrics = Path(context.output_root) / "preflight" / "metrics.json"
    if resume and preflight_state.is_file() and preflight_metrics.is_file():
        # Resume only from the dedicated measured artifact.  A later phase may
        # legitimately overwrite root/state.json and must never be mistaken for
        # preflight evidence.
        result = json.loads(preflight_state.read_text(encoding="utf-8"))
        metrics = json.loads(preflight_metrics.read_text(encoding="utf-8"))
        expected_checks = metrics.get("expected_check_names")
        actual_checks = metrics.get("checks", ())
        actual_names = [
            str(check.get("name")) for check in actual_checks
            if isinstance(check, Mapping)
        ]
        valid_checks = (
            isinstance(expected_checks, list)
            and actual_names == [str(name) for name in expected_checks]
            and all(bool(check.get("passed")) for check in actual_checks)
        ) or (
            # Backward compatibility for the already-written V5 preflight v1
            # envelope, which predates expected_check_names.  V6 never uses
            # this path.
            not context.is_distributional
            and expected_checks is None
            and len(actual_checks) == 7
            and all(bool(check.get("passed")) for check in actual_checks)
        )
        valid_identity = (
            result.get("experiment_id") == context.experiment_id
            and result.get("architecture_version") == context.architecture_version
            and result.get("phase") == "preflight"
            and result.get("lineage_id") == context.lineage_id
            and result.get("config_sha256") == context.config_sha256
            and result.get("status") == "COMPLETE"
            and result.get("capability_gate_status") == "PASSED"
            and result.get("formal_selection_2024_opened") is False
            and result.get("locked_test_2025_opened") is False
            and metrics.get("status") == "COMPLETE"
            and metrics.get("lineage_id") == context.lineage_id
            and metrics.get("config_sha256") == context.config_sha256
            and metrics.get("formal_selection_2024_opened") is False
            and metrics.get("locked_test_2025_opened") is False
            and valid_checks
        )
        if context.is_distributional:
            manifest_path = Path(
                str(result.get("frozen_preflight_manifest_path", ""))
            )
            manifest_sha = str(result.get("frozen_preflight_manifest_sha256", ""))
            valid_identity = (
                valid_identity
                and manifest_path.is_file()
                and bool(manifest_sha)
                # The manifest is small.  Large checkpoints and contracts are
                # intentionally not re-hashed on resume; their one-time hashes
                # are frozen inside this envelope and checked once by the phase
                # runner when Distribution starts.
                and sha256_file(manifest_path) == manifest_sha
            )
        if valid_identity:
            result["metrics_path"] = str(preflight_metrics.resolve())
            result["preflight_state_path"] = str(preflight_state.resolve())
            return {
                **result,
                "config_path": context.config_path,
                "output_root": context.output_root,
                "seeds": list(context.seeds),
                "phase_order": list(context.phase_order),
                "phase_specs": context.phase_specs,
                "optimizer_effective_batch_days": 543,
                "physical_microbatch_days": 32 if context.is_distributional else 512,
                "preflight_resume_mode": "small_manifest_only",
            }
    from .preflight import run_measured_preflight

    config = load_r4_config(config_path)
    result = run_measured_preflight(context, config)
    top_level = {
        **result,
        "config_path": context.config_path,
        "output_root": context.output_root,
        "seeds": list(context.seeds),
        "phase_order": list(context.phase_order),
        "phase_specs": context.phase_specs,
        "optimizer_effective_batch_days": int(
            config.get("training", {}).get("optimizer_effective_batch_days", 543)
        ),
        "physical_microbatch_days": int(
            config.get("training", {}).get(
                "physical_microbatch_days",
                config.get("training", {}).get("effective_batch_size_days", 512),
            )
        ),
        "physical_batch_fallback": list(
            config.get("training", {}).get(
                "physical_microbatch_fallback", [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            )
        ),
    }
    # AtomicRunStore provides the same strict lineage envelope used by every
    # phase.  COMPLETE is written only after all measured checks above passed.
    store.write_state(top_level)
    return top_level


PhaseRunner = Callable[..., Mapping[str, Any]]
_PHASE_RUNNER: PhaseRunner | None = None


def register_phase_runner(runner: PhaseRunner | None) -> None:
    """Register an application adapter without coupling protocol to model/data code."""

    global _PHASE_RUNNER
    _PHASE_RUNNER = runner


def run_phase(
    config_path: str | Path,
    phase: str,
    seed: int,
    resume: bool = True,
) -> dict[str, Any]:
    context = prepare_protocol(config_path)
    phase_name = str(phase).lower()
    if phase_name not in context.phase_order:
        raise R4ProtocolError(f"unknown R4 phase: {phase}")
    if int(seed) not in context.seeds:
        raise R4ProtocolError(f"unregistered R4 seed: {seed}")
    store = _store(context)
    if int(seed) != context.seeds[0]:
        from .final_evaluation import FinalProtocolError, authorize_later_seed

        try:
            authorize_later_seed(context.output_root, int(seed))
        except FinalProtocolError as exc:
            result = {
                "status": "NOT_READY",
                "phase": phase_name,
                "seed": int(seed),
                "reason": f"seed1 capability chain is incomplete: {exc}",
                "formal_selection_2024_opened": False,
            }
            store.write_state(result)
            return result
    preflight = run_preflight(config_path, resume=True)
    if preflight.get("status") != "COMPLETE":
        result = {
            "status": "NOT_READY",
            "phase": phase_name,
            "seed": int(seed),
            "reason": "measured R4 preflight has not completed successfully",
            "preflight_status": preflight.get("status"),
            "formal_selection_2024_opened": False,
            "scientific_completion": False,
        }
        store.write_state(result)
        return result
    if _PHASE_RUNNER is None:
        result = {
            "status": "NOT_READY",
            "phase": phase_name,
            "seed": int(seed),
            "reason": "R4 model/data phase runner is not connected",
            "formal_selection_2024_opened": False,
            "scientific_completion": False,
        }
        store.write_state(result)
        return result
    raw = dict(
        _PHASE_RUNNER(
            context=context,
            phase=phase_name,
            seed=int(seed),
            resume=bool(resume),
        )
    )
    gate = raw.get("capability_gate")
    gate_status = gate.get("status") if isinstance(gate, Mapping) else gate
    if raw.get("status") == "COMPLETE" and gate_status != "PASS":
        raise R4ProtocolError("phase cannot complete without an explicit capability-gate PASS")
    raw.setdefault("phase", phase_name)
    raw.setdefault("seed", int(seed))
    raw["formal_selection_2024_opened"] = False
    store.write_state(raw)
    return raw


def run_seed1(config_path: str | Path, resume: bool = True) -> dict[str, Any]:
    context = prepare_protocol(config_path)
    seed = context.seeds[0]
    phases: dict[str, Any] = {}
    for phase in context.phase_order:
        result = run_phase(config_path, phase, seed, resume=resume)
        phases[phase] = result
        if result.get("status") != "COMPLETE":
            output = {
                "status": "PARTIAL" if result.get("status") != "NOT_READY" else "NOT_READY",
                "seed": seed,
                "phases": phases,
                "reason": f"seed1 stopped at {phase}: {result.get('reason', result.get('status'))}",
                "formal_selection_2024_opened": False,
            }
            _store(context).write_state({"seed1": output, **output})
            from .reports import write_seed1_report as _write_seed1_report

            _write_seed1_report(context, {"seed1": output, **output})
            return output
    output = {
        "status": "COMPLETE",
        "seed": seed,
        "phases": phases,
        "formal_selection_2024_opened": False,
        "next_step": "freeze seed1 method before any separately authorized 2024 opening",
    }
    _store(context).write_state({"seed1": output, **output})
    from .reports import write_seed1_report as _write_seed1_report

    _write_seed1_report(context, {"seed1": output, **output})
    return output


def protocol_status(config_path: str | Path) -> dict[str, Any]:
    context = prepare_protocol(config_path)
    state = _store(context).read_state()
    if state is None:
        return {
            "status": "NOT_STARTED",
            "lineage_id": context.lineage_id,
            "formal_selection_2024_opened": False,
            "locked_test_2025_opened": False,
        }
    # Identity comes from the currently validated immutable config, not from
    # legacy defaults an older AtomicRunStore may have serialized before V6
    # support was added.  Scientific phase/gate fields remain untouched.
    return {
        **state,
        "experiment_id": context.experiment_id,
        "architecture_version": context.architecture_version,
        "training_semantics_revision": context.training_semantics_revision,
        "method": (
            "continuous_distributional_world_joint_moe_r4_v6"
            if context.is_distributional
            else state.get("method", "hybrid_ltc_transformer_top2moe_r4")
        ),
        "phase_order": list(context.phase_order),
        "output_root": context.output_root,
    }


def write_seed1_report(config_path: str | Path) -> dict[str, Any]:
    from .reports import write_seed1_report as _write_report

    context = prepare_protocol(config_path)
    return _write_report(context, protocol_status(config_path))


__all__ = [
    "EXPECTED_CONFIG_NAME",
    "EXPECTED_OUTPUT_ROOT",
    "ProtocolContext",
    "prepare_protocol",
    "protocol_status",
    "register_phase_runner",
    "run_phase",
    "run_preflight",
    "run_seed1",
    "write_seed1_report",
]
