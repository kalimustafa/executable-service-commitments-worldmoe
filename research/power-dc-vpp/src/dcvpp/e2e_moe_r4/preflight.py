"""Measured, fail-closed preflight for the formal R4 lineage.

The preflight is deliberately separate from training.  It opens exactly one
2023-H2 validation day, creates a fresh R4 model, executes all 96 rolling
origins through the analytic shield, and independently recomputes every hard
residual.  It never authorizes or opens 2024/2025 payloads.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .artifact_io import atomic_json, atomic_text, utc_now, write_state
from .config import (
    ARCHITECTURE_VERSION,
    DISTRIBUTIONAL_ARCHITECTURE_VERSION,
    DISTRIBUTIONAL_EXPERIMENT_ID,
    DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION,
    EXPERIMENT_ID,
    FORBIDDEN_ACTIVE_IMPORTS,
    TRAINING_SEMANTICS_REVISION,
)
from .hashing import sha256_file
from .constrained_runtime import (
    R4Runtime,
    load_physical_limits,
    validate_record,
)
from .data import (
    INTERVALS,
    LockedTestAccessError,
    SealedSelectionAccessError,
    discover_eligible_days,
    load_r4_day,
)
from .model import HybridCfCTransformerMoER4
from .physical_twin import (
    ArtifactBackedFrozenPhysicalTwinCore,
    build_formal_r4_model,
)
from .training_data import PreloadedR4Corpus


class PreflightBlocked(RuntimeError):
    """A required external input or CUDA device is unavailable."""


class PreflightFailed(RuntimeError):
    """A correctness invariant failed and training must not start."""


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    evidence: Mapping[str, Any]


def _config_value(config: Mapping[str, Any], dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise PreflightFailed(f"missing preflight config field {dotted}")
        current = current[part]
    return current


def _local_import_target(module: str, level: int, current: str) -> str | None:
    """Resolve a relative import inside ``dcvpp.e2e_moe_r4``."""

    package = current.split(".")[:-1]
    if level:
        trim = level - 1
        if trim > len(package):
            return None
        package = package[: len(package) - trim]
        target = ".".join((*package, module)) if module else ".".join(package)
    else:
        target = module
    prefix = "dcvpp.e2e_moe_r4"
    return target if target == prefix or target.startswith(prefix + ".") else None


def audit_active_import_graph(package_root: str | Path) -> dict[str, Any]:
    """AST-walk the complete active R4 namespace and reject solver imports.

    R4's public API and formal CLI can lazily reach model, runtime, trainer and
    evaluators.  Scanning every Python module in this *new namespace* is both
    stricter and less ambiguous than scanning whichever modules happened to be
    imported in the current interpreter.
    """

    root = Path(package_root).resolve()
    if not root.is_dir():
        raise PreflightFailed(f"R4 package root does not exist: {root}")
    modules: list[str] = []
    imports: list[dict[str, str]] = []
    forbidden: list[dict[str, str]] = []
    prefix = "dcvpp.e2e_moe_r4"
    for path in sorted(root.glob("*.py")):
        module = prefix if path.name == "__init__.py" else f"{prefix}.{path.stem}"
        modules.append(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                resolved = _local_import_target(node.module or "", node.level, module)
                names.append(resolved or (node.module or ""))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.append(node.args[0].value)
            for imported in names:
                lowered = imported.lower()
                record = {"module": module, "imported": imported}
                imports.append(record)
                top = lowered.split(".", 1)[0]
                if top in FORBIDDEN_ACTIVE_IMPORTS or any(
                    token in lowered.split(".") for token in FORBIDDEN_ACTIVE_IMPORTS
                ):
                    forbidden.append(record)
    if forbidden:
        raise PreflightFailed(f"forbidden solver imports in active R4 graph: {forbidden}")
    return {
        "package_root": str(root),
        "modules_scanned": modules,
        "module_count": len(modules),
        "import_edge_count": len(imports),
        "forbidden_imports": forbidden,
    }


def verify_temporal_path_guards(repository_root: str | Path) -> dict[str, Any]:
    """Prove 2024/2025 payload paths are rejected before ``numpy.load``.

    This is a single inexpensive guard probe.  It does not enumerate or hash a
    corpus and it does not touch either sealed year on disk.
    """

    root = Path(repository_root).resolve()
    sealed_bundle = root / "preflight_probe" / "2024-01-01.npz"
    sealed_settlement = root / "preflight_probe" / "2024-01-01-settlement.npz"
    locked_bundle = root / "preflight_probe" / "2025-01-01.npz"
    locked_settlement = root / "preflight_probe" / "2025-01-01-settlement.npz"
    with patch.object(np, "load", side_effect=AssertionError("payload was opened")) as loader:
        try:
            load_r4_day(sealed_bundle, sealed_settlement)
        except SealedSelectionAccessError as exc:
            sealed_reason = str(exc)
        else:  # pragma: no cover - a correctness catastrophe
            raise PreflightFailed("2024 path was not rejected")
        try:
            load_r4_day(locked_bundle, locked_settlement)
        except LockedTestAccessError as exc:
            locked_reason = str(exc)
        else:  # pragma: no cover - a correctness catastrophe
            raise PreflightFailed("2025 path was not rejected")
    if loader.call_count:
        raise PreflightFailed("2024/2025 path rejection occurred after payload open")
    return {
        "sealed_2024_probe_path": str(sealed_bundle),
        "locked_2025_probe_path": str(locked_bundle),
        "numpy_load_calls": 0,
        "sealed_exception": "SealedSelectionAccessError",
        "locked_exception": "LockedTestAccessError",
        "sealed_reason": sealed_reason,
        "locked_reason": locked_reason,
    }


def verify_locked_path_guard(repository_root: str | Path) -> dict[str, Any]:
    """Backward-compatible 2025 guard result used by archived V5 tests."""

    evidence = verify_temporal_path_guards(repository_root)
    return {
        "probe_path": evidence["locked_2025_probe_path"],
        "numpy_load_calls": 0,
        "exception": evidence["locked_exception"],
        "reason": evidence["locked_reason"],
    }


def inspect_cuda_environment() -> tuple[torch.device, dict[str, Any]]:
    """Return the usable formal CUDA device and measured environment facts."""

    evidence: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_below_3_13": sys.version_info[:2] < (3, 13),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not evidence["python_below_3_13"]:
        raise PreflightBlocked("formal R4 requires Python below 3.13")
    if not torch.cuda.is_available():
        raise PreflightBlocked("formal R4 requires a usable CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        properties = torch.cuda.get_device_properties(device)
        probe = torch.ones((32, 32), device=device)
        checksum = float((probe @ probe).sum().item())
        torch.cuda.synchronize(device)
    except (RuntimeError, AssertionError) as exc:
        raise PreflightBlocked(f"CUDA allocation/compute probe failed: {exc}") from exc
    evidence.update(
        {
            "device": str(device),
            "device_name": properties.name,
            "device_total_memory_bytes": int(properties.total_memory),
            "cuda_compute_probe_checksum": checksum,
        }
    )
    return device, evidence


def _validation_pair(config: Mapping[str, Any], repository_root: Path) -> tuple[Path, Path, str]:
    bundle_root = repository_root / str(_config_value(config, "data.selection_bundle_root"))
    settlement_root = repository_root / str(
        _config_value(config, "data.official_settlement_root")
    )
    index_path = repository_root / str(_config_value(config, "data.canonical_selection_index"))
    for label, path in (
        ("selection bundle root", bundle_root),
        ("official settlement root", settlement_root),
        ("canonical selection index", index_path),
    ):
        if not path.exists():
            raise PreflightBlocked(f"{label} is unavailable: {path}")
    bundles = discover_eligible_days(
        bundle_root, "2023-07-01", "2023-12-31", metadata_only=True
    )
    settlements = discover_eligible_days(
        settlement_root, "2023-07-01", "2023-12-31", metadata_only=True
    )
    bundle_by_date = {item.operating_date: item.path for item in bundles}
    settlement_by_date = {item.operating_date: item.path for item in settlements}
    common = sorted(set(bundle_by_date).intersection(settlement_by_date))
    if not common:
        raise PreflightBlocked("no paired 2023-H2 bundle and official settlement were found")
    operating_date = common[0]
    return bundle_by_date[operating_date], settlement_by_date[operating_date], operating_date


class _RecordingR4Policy(nn.Module):
    """Delegate to the real model while recording formal recurrent inputs."""

    def __init__(
        self,
        model: HybridCfCTransformerMoER4,
        normalized_issue_batch: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.normalized_issue_batch = dict(normalized_issue_batch)
        self.state_inputs: list[torch.Tensor] = []
        self.scenario_counts: list[int] = []

    def prepare_issue_context(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        del batch
        return self.model.prepare_issue_context(self.normalized_issue_batch)

    def _rolling_state(self, session_input: Mapping[str, Any]) -> dict[str, Any]:
        state_value = session_input.get("state", {})
        rolling = dict(state_value) if isinstance(state_value, Mapping) else {}
        for key in (
            "current_soc", "backlog", "deadline_pressure", "previous_grid",
            "previous_battery", "previous_compute", "previous_migration",
            "rt_innovation", "workload_innovation", "pv_innovation",
        ):
            if key in session_input:
                rolling[key] = session_input[key]
        previous = session_input.get("previous_action")
        if isinstance(previous, Mapping):
            for key, value in previous.items():
                rolling[f"previous_{key}"] = value
        return rolling

    def forward_session(self, session_input: Mapping[str, Any]) -> Mapping[str, Any]:
        context = session_input["issue_context"]
        reference = context["market_tokens"]
        state_input = self.model._state_tensor(  # formal white-box equivalence audit
            self._rolling_state(session_input),
            reference,
            origin=session_input.get("origin", 0),
        )
        self.state_inputs.append(state_input.detach().cpu())
        output = self.model.forward_session(session_input)
        particles = output["scenarios"]["particles_zonal"]
        self.scenario_counts.append(int(particles.shape[1]))
        return output


def _liquid_replay_evidence(
    model: HybridCfCTransformerMoER4,
    inputs: list[torch.Tensor],
    tolerance: float,
) -> dict[str, Any]:
    if len(inputs) != INTERVALS:
        raise PreflightFailed(f"Liquid audit received {len(inputs)} rather than 96 inputs")
    sequence = torch.stack(inputs, dim=1).to(next(model.parameters()).device)
    with torch.inference_mode():
        replay_outputs, replay_state = model.state_encoder.replay(sequence)
        incremental_state = None
        incremental_outputs: list[torch.Tensor] = []
        for origin in range(INTERVALS):
            output, incremental_state = model.state_encoder.step(
                sequence[:, origin], incremental_state, dt=0.25
            )
            incremental_outputs.append(output)
        incremental = torch.stack(incremental_outputs, dim=1)
    assert incremental_state is not None
    output_error = float((incremental - replay_outputs).abs().max().item())
    state_error = float((incremental_state - replay_state).abs().max().item())
    if max(output_error, state_error) > tolerance:
        raise PreflightFailed(
            "incremental Liquid state differs from full-prefix replay: "
            f"outputs={output_error}, state={state_error}, tolerance={tolerance}"
        )
    return {
        "origins": INTERVALS,
        "input_shape": list(sequence.shape),
        "maximum_output_absolute_error": output_error,
        "final_state_absolute_error": state_error,
        "tolerance": tolerance,
    }


def _artifact_state(
    *, status: str, capability_status: str, reason: str, context: Any
) -> dict[str, Any]:
    architecture = str(getattr(context, "architecture_version", ARCHITECTURE_VERSION))
    is_distributional = architecture == DISTRIBUTIONAL_ARCHITECTURE_VERSION
    return {
        "experiment_id": (
            DISTRIBUTIONAL_EXPERIMENT_ID if is_distributional else EXPERIMENT_ID
        ),
        "architecture_version": architecture,
        "training_semantics_revision": (
            DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION
            if is_distributional
            else TRAINING_SEMANTICS_REVISION
        ),
        "method": (
            "continuous_distributional_world_joint_moe_r4_v6"
            if is_distributional
            else "hybrid_ltc_transformer_top2moe_r4"
        ),
        "seed": int(context.seeds[0]),
        "phase": "preflight",
        "split": (
            "train_contracts_only_no_validation_payloads"
            if is_distributional
            else "2023_h2_validation_only"
        ),
        "predecessor_checkpoint": (
            "capability_passed_representation_epoch28"
            if is_distributional
            else None
        ),
        "capability_gate_status": capability_status,
        "status": status,
        "resume_command": (
            "python scripts/run_e2e_moe_r4_formal.py preflight --resume "
            f"--config {json.dumps(context.config_path)}"
        ),
        "reason": reason,
        "lineage_id": context.lineage_id,
        "config_sha256": context.config_sha256,
        "config_path": context.config_path,
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "scientific_completion": False,
    }


def _resolved_config_path(
    config: Mapping[str, Any], repository_root: Path, dotted: str
) -> Path:
    value = _config_value(config, dotted)
    if not isinstance(value, str) or not value.strip():
        raise PreflightFailed(f"{dotted} must be an explicit artifact path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise PreflightBlocked(f"{label} is unavailable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PreflightFailed(f"{label} must contain one JSON mapping")
    return payload


def _same_resolved_path(left: Any, right: Path) -> bool:
    if not isinstance(left, str) or not left.strip():
        return False
    return Path(left).resolve() == right.resolve()


def _v6_historical_lineage_evidence(
    config: Mapping[str, Any], repository_root: Path, output_root: Path
) -> dict[str, Any]:
    """Validate the one allowed predecessor and preserve the V5 failure.

    Only small JSON envelopes are read here.  The large selected checkpoint is
    opened and hashed once by the preflight contract materializer.
    """

    checkpoint_path = _resolved_config_path(
        config, repository_root, "lineage.representation_selected_checkpoint"
    )
    representation_state_path = _resolved_config_path(
        config, repository_root, "lineage.representation_state"
    )
    normalization_path = _resolved_config_path(
        config, repository_root, "lineage.train_only_normalization"
    )
    failed_scenario_path = _resolved_config_path(
        config, repository_root, "lineage.v5_failed_scenario_state"
    )
    residual_contract_path = _resolved_config_path(
        config, repository_root, "lineage.residual_score_contract"
    )
    for label, path in (
        ("selected Representation checkpoint", checkpoint_path),
        ("selected Representation state", representation_state_path),
        ("train-only normalization", normalization_path),
        ("failed V5 Scenario state", failed_scenario_path),
        ("train-only residual score contract", residual_contract_path),
    ):
        if not path.is_file():
            raise PreflightBlocked(f"{label} is unavailable: {path}")

    representation = _json_mapping(
        representation_state_path, "selected Representation state"
    )
    gate = representation.get("capability_gate")
    if not isinstance(gate, Mapping):
        raise PreflightFailed("selected Representation state lacks its capability gate")
    if (
        representation.get("status") != "COMPLETE"
        or representation.get("capability_gate_status") != "PASSED"
        or gate.get("status") != "PASS"
        or int(gate.get("evaluated_days", -1)) != 179
        or int(representation.get("selected_candidate_epoch", -1)) != 28
        or not _same_resolved_path(representation.get("checkpoint"), checkpoint_path)
    ):
        raise PreflightFailed(
            "V6 predecessor is not the 179-day capability-passed epoch-28 Representation"
        )

    normalization = _json_mapping(normalization_path, "train-only normalization")
    normalization_sha = sha256_file(normalization_path)
    if (
        normalization.get("fit_split") != "train"
        or int(normalization.get("train_days", -1)) != 543
        or int(normalization.get("validation_days", -1)) != 179
        or representation.get("normalization_sha256") != normalization_sha
    ):
        raise PreflightFailed("inherited normalization is not the frozen 543-day train contract")

    residual = _json_mapping(residual_contract_path, "residual score contract")
    if (
        residual.get("fit_split") != "train"
        or int(residual.get("fit_days", -1)) != 543
        or int(residual.get("representation_checkpoint_epoch", -1)) != 28
        or residual.get("representation_normalization_sha256") != normalization_sha
        or residual.get("formal_selection_2024_opened") is not False
        or residual.get("locked_test_2025_opened") is not False
    ):
        raise PreflightFailed("inherited residual score contract has invalid split/lineage")

    failed = _json_mapping(failed_scenario_path, "failed V5 Scenario state")
    failed_gate = failed.get("capability_gate")
    if not isinstance(failed_gate, Mapping):
        raise PreflightFailed("V5 Scenario state lacks its formal capability gate")
    if (
        failed.get("status") != "FAILED"
        or failed.get("capability_gate_status") != "FAILED"
        or failed_gate.get("status") != "FAIL"
        or int(failed_gate.get("evaluated_days", -1)) != 179
        or int(failed.get("epochs_completed", -1)) != 200
        or failed.get("selected_candidate_epoch") is not None
    ):
        raise PreflightFailed(
            "the V5 fixed-particle Scenario must remain the recorded 200-epoch formal failure"
        )
    metrics = failed_gate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise PreflightFailed("V5 Scenario failure lacks measured metrics")
    if not (
        float(metrics.get("energy_score_delta_one_sided_upper", float("-inf"))) >= 0.0
        and float(metrics.get("absolute_coverage_error_90", 0.0)) > 0.03
        and float(metrics.get("absolute_coverage_error_95", 0.0)) > 0.03
        and float(metrics.get("tail_score_noninferiority_one_sided_upper", float("-inf")))
        > 0.0
    ):
        raise PreflightFailed("V5 Scenario state no longer records the governing failure")

    superseded_path = output_root / "data" / "v5_fixed20_scenario_superseded.json"
    superseded = {
        "schema": "dcvpp.r4.v6-fixed20-scenario-superseded.v1",
        "source_state": str(failed_scenario_path),
        "source_state_sha256": sha256_file(failed_scenario_path),
        "source_architecture": str(failed.get("architecture_version")),
        "source_phase": "scenario",
        "source_epochs_completed": 200,
        "source_formal_gate_status": "FAIL",
        "source_selected_candidate_epoch": None,
        "read_only": True,
        "resume_allowed": False,
        "checkpoint_predecessor_allowed": False,
        "superseded_by": DISTRIBUTIONAL_ARCHITECTURE_VERSION,
        "supersede_reason": str(
            _config_value(config, "lineage.supersede_reason")
        ),
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }
    if superseded_path.is_file():
        if _json_mapping(superseded_path, "V5 superseded record") != superseded:
            raise PreflightFailed("frozen V5 superseded record changed")
    else:
        atomic_json(superseded_path, superseded)

    return {
        "representation_state_path": str(representation_state_path),
        "representation_state_sha256": sha256_file(representation_state_path),
        "representation_checkpoint_path": str(checkpoint_path),
        "selected_epoch": 28,
        "representation_gate": "PASS",
        "representation_gate_days": 179,
        "normalization_path": str(normalization_path),
        "normalization_sha256": normalization_sha,
        "residual_score_contract_path": str(residual_contract_path),
        "residual_score_contract_sha256": sha256_file(residual_contract_path),
        "v5_failed_scenario_state_path": str(failed_scenario_path),
        "v5_failed_scenario_gate": "FAIL",
        "v5_failed_scenario_epochs": 200,
        "v5_superseded_record_path": str(superseded_path.resolve()),
        "v5_superseded_record_sha256": sha256_file(superseded_path),
    }


def _jsonable(value: Any) -> Any:
    """Normalize a preparation summary before it enters the frozen manifest."""

    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _freeze_v6_preflight_manifest(
    context: Any,
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> tuple[Path, str, Mapping[str, Any]]:
    path = Path(context.output_root) / "preflight" / "frozen_preflight_manifest.json"
    # The already-frozen v1 manifest binds the Representation state through
    # ``historical_lineage``.  Keep that byte-stable contract instead of
    # mutating it merely to duplicate the same small state record at the top
    # level.  New runners accept the historical-lineage binding below.
    prepared_for_manifest = dict(prepared)
    prepared_for_manifest.pop("source_representation_state", None)
    artifacts = prepared.get("frozen_artifacts")
    required = {
        "representation_predecessor_import",
        "distribution_support",
        "distribution_training_contract",
        "distribution_sampling_contract",
    }
    if not isinstance(artifacts, Mapping) or not required.issubset(artifacts):
        raise PreflightFailed(
            "V6 contract preparation did not return all frozen Distribution artifacts"
        )
    for name in sorted(required):
        record = artifacts[name]
        if not isinstance(record, Mapping):
            raise PreflightFailed(f"frozen artifact {name} is not a mapping")
        artifact_path = Path(str(record.get("path", "")))
        digest = str(record.get("sha256", ""))
        if not artifact_path.is_file() or len(digest) != 64:
            raise PreflightFailed(f"frozen artifact {name} is missing path/SHA-256")

    payload = {
        "schema": "dcvpp.e2e_moe_distributional_r4.frozen-preflight.v1",
        "experiment_id": str(context.experiment_id),
        "architecture_version": str(context.architecture_version),
        "training_semantics_revision": str(context.training_semantics_revision),
        "lineage_id": str(context.lineage_id),
        "status": "FROZEN_BEFORE_TRAINING",
        "config_path": str(context.config_path),
        "config_sha256": str(context.config_sha256),
        "phase_order": list(context.phase_order),
        "hash_policy": str(_config_value(config, "audit.hash_policy")),
        "large_checkpoint_hash_policy": str(
            _config_value(config, "audit.hash_large_checkpoint")
        ),
        "repeat_full_corpus_hash_each_epoch": bool(
            _config_value(config, "audit.repeat_full_corpus_hash_each_epoch")
        ),
        "historical_lineage": _jsonable(lineage),
        # Stable top-level interface consumed once by the Distribution runner.
        "frozen_artifacts": _jsonable(artifacts),
        "normalization": _jsonable(prepared.get("normalization", {})),
        "source_checkpoint": _jsonable(prepared.get("source_checkpoint", {})),
        "source_residual_contract": _jsonable(
            prepared.get("source_residual_contract", {})
        ),
        "model_contract": _jsonable(prepared.get("model_contract", {})),
        "data_contract": _jsonable(prepared.get("data_contract", {})),
        "prepared_contracts": _jsonable(prepared_for_manifest),
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "solver_free": True,
        "external_solver": False,
    }
    if path.is_file():
        existing = _json_mapping(path, "frozen V6 preflight manifest")
        if existing != payload:
            raise PreflightFailed("frozen V6 preflight manifest changed")
    else:
        atomic_json(path, payload)
    return path.resolve(), sha256_file(path), payload


_V6_PREFLIGHT_CHECK_NAMES = [
    "active_import_graph_has_no_external_solver",
    "sealed_2024_and_locked_2025_rejected_before_open",
    "verified_python_cuda_environment",
    "capability_passed_epoch28_representation_only",
    "failed_v5_fixed20_scenario_preserved_as_superseded",
    "v6_distribution_contracts_frozen_once",
    "v6_static_architecture_and_sampling_contract",
]


def _run_distributional_preflight(
    context: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Run V6 architecture/lineage preflight without opening validation data."""

    output_root = Path(context.output_root)
    preflight_root = output_root / "preflight"
    checks: list[PreflightCheck] = []
    started = utc_now()
    manifest_path: Path | None = None
    manifest_sha: str | None = None
    try:
        import_evidence = audit_active_import_graph(
            Path(context.repository_root) / "src" / "dcvpp" / "e2e_moe_r4"
        )
        checks.append(
            PreflightCheck(
                _V6_PREFLIGHT_CHECK_NAMES[0], True, import_evidence
            )
        )
        lock_evidence = verify_temporal_path_guards(context.repository_root)
        checks.append(
            PreflightCheck(_V6_PREFLIGHT_CHECK_NAMES[1], True, lock_evidence)
        )
        _, environment = inspect_cuda_environment()
        checks.append(
            PreflightCheck(_V6_PREFLIGHT_CHECK_NAMES[2], True, environment)
        )

        lineage = _v6_historical_lineage_evidence(
            config, Path(context.repository_root), output_root
        )
        checks.append(
            PreflightCheck(
                _V6_PREFLIGHT_CHECK_NAMES[3],
                True,
                {
                    "selected_epoch": 28,
                    "formal_gate": "PASS",
                    "evaluated_days": 179,
                    "checkpoint_path": lineage["representation_checkpoint_path"],
                    "normalization_sha256": lineage["normalization_sha256"],
                },
            )
        )
        checks.append(
            PreflightCheck(
                _V6_PREFLIGHT_CHECK_NAMES[4],
                True,
                {
                    "source_state": lineage["v5_failed_scenario_state_path"],
                    "epochs": 200,
                    "formal_gate": "FAIL",
                    "read_only": True,
                    "resume_allowed": False,
                    "superseded_record": lineage["v5_superseded_record_path"],
                },
            )
        )

        # Delayed import avoids a module cycle: phase_runner imports the public
        # ProtocolContext type but this preparation function is called only
        # after formal_protocol finished importing.
        from .phase_runner import prepare_v6_distribution_preflight_contracts

        prepared = prepare_v6_distribution_preflight_contracts(context, config=config)
        if not isinstance(prepared, Mapping):
            raise PreflightFailed("V6 preflight contract preparation returned no mapping")
        manifest_path, manifest_sha, manifest = _freeze_v6_preflight_manifest(
            context, config, lineage, prepared
        )
        checks.append(
            PreflightCheck(
                _V6_PREFLIGHT_CHECK_NAMES[5],
                True,
                {
                    "frozen_manifest_path": str(manifest_path),
                    "frozen_manifest_sha256": manifest_sha,
                    "frozen_artifacts": manifest["prepared_contracts"]["frozen_artifacts"],
                    "large_files_hashed_once": True,
                    "per_epoch_rehash": False,
                },
            )
        )

        model_contract = prepared.get("model_contract")
        if not isinstance(model_contract, Mapping):
            raise PreflightFailed("V6 preflight preparation lacks model_contract")
        parameter_count = int(model_contract.get("parameter_count", -1))
        minimum = int(_config_value(config, "model.parameter_budget.expected_min_trainable"))
        maximum = int(_config_value(config, "model.parameter_budget.expected_max_trainable"))
        if not minimum <= parameter_count <= maximum:
            raise PreflightFailed(
                f"V6 parameter count {parameter_count} lies outside [{minimum},{maximum}]"
            )
        expected_budgets = {
            "train": 32,
            "cheap_validation": 64,
            "formal_probability": 256,
            "formal_empirical_risk": 512,
            "ablation": [20, 64, 128, 256, 512],
        }
        actual_budgets = {
            key: _config_value(config, f"sampling.{key}") for key in expected_budgets
        }
        if actual_budgets != expected_budgets:
            raise PreflightFailed("V6 sampling budgets differ from the frozen contract")
        if (
            model_contract.get("architecture_version")
            != DISTRIBUTIONAL_ARCHITECTURE_VERSION
            or model_contract.get("policy_requires_samples") is not False
            or model_contract.get("fixed20_head_loaded") is not False
        ):
            raise PreflightFailed("V6 static world/policy contract is not N-independent")
        checks.append(
            PreflightCheck(
                _V6_PREFLIGHT_CHECK_NAMES[6],
                True,
                {
                    **_jsonable(model_contract),
                    "sampling_budgets": actual_budgets,
                    "physical_microbatch_days": 32,
                    "optimizer_effective_batch_days": 543,
                },
            )
        )
        status, gate_status, reason = (
            "COMPLETE",
            "PASSED",
            "all V6 architecture and frozen-contract preflight checks passed",
        )
    except PreflightBlocked as exc:
        status, gate_status, reason = "BLOCKED", "NOT_RUN", str(exc)
    except (PreflightFailed, ValueError, RuntimeError, OSError, KeyError) as exc:
        status, gate_status, reason = "FAILED", "FAILED", str(exc)
        checks.append(
            PreflightCheck(
                "terminal_failure",
                False,
                {"exception_type": type(exc).__name__, "reason": str(exc)},
            )
        )
    except Exception as exc:  # fail closed while retaining diagnostics
        status, gate_status, reason = "FAILED", "FAILED", str(exc)
        checks.append(
            PreflightCheck(
                "unexpected_terminal_failure",
                False,
                {
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    metrics = {
        "schema_version": "dcvpp.e2e_moe_distributional_r4.preflight.v1",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "status": status,
        "reason": reason,
        "lineage_id": context.lineage_id,
        "config_sha256": context.config_sha256,
        "expected_check_names": list(_V6_PREFLIGHT_CHECK_NAMES),
        "checks": [asdict(item) for item in checks],
        "validation_payloads_opened": 0,
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }
    atomic_json(preflight_root / "metrics.json", metrics)
    atomic_text(
        preflight_root / "runtime.log",
        "\n".join(
            [f"R4-V6 PRECHECK {started}"]
            + [
                f"{'PASS' if item.passed else 'FAIL'} {item.name}: {dict(item.evidence)}"
                for item in checks
            ]
            + [f"TERMINAL {status}: {reason}"]
        )
        + "\n",
    )
    artifact_state = _artifact_state(
        status=status,
        capability_status=gate_status,
        reason=reason,
        context=context,
    )
    artifact_state.update(
        {
            "preflight_kind": "architecture_static_and_train_only_contract_freeze",
            "validation_payloads_opened": 0,
            "frozen_preflight_manifest_path": (
                None if manifest_path is None else str(manifest_path)
            ),
            "frozen_preflight_manifest_sha256": manifest_sha,
        }
    )
    write_state(preflight_root / "state.json", artifact_state)
    return {
        **artifact_state,
        "checks": metrics["checks"],
        "metrics_path": str((preflight_root / "metrics.json").resolve()),
        "runtime_log_path": str((preflight_root / "runtime.log").resolve()),
        "preflight_state_path": str((preflight_root / "state.json").resolve()),
    }


def run_measured_preflight(context: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute and persist the single real R4 preflight.

    The returned ``status`` is ``COMPLETE`` only when every check passes.
    Missing data/device is ``BLOCKED``; violated invariants are ``FAILED``.
    """

    if (
        str(config.get("project", {}).get("architecture_version", ""))
        == DISTRIBUTIONAL_ARCHITECTURE_VERSION
    ):
        return _run_distributional_preflight(context, config)

    output_root = Path(context.output_root)
    preflight_root = output_root / "preflight"
    checks: list[PreflightCheck] = []
    started = utc_now()
    try:
        import_evidence = audit_active_import_graph(
            Path(context.repository_root) / "src" / "dcvpp" / "e2e_moe_r4"
        )
        checks.append(PreflightCheck("active_import_graph_has_no_external_solver", True, import_evidence))
        lock_evidence = verify_locked_path_guard(context.repository_root)
        checks.append(PreflightCheck("locked_2025_path_before_open", True, lock_evidence))
        device, environment = inspect_cuda_environment()
        checks.append(PreflightCheck("verified_python_cuda_environment", True, environment))

        bundle_path, settlement_path, operating_date = _validation_pair(
            config, Path(context.repository_root)
        )
        if not operating_date.startswith("2023-"):
            raise PreflightFailed("preflight day is not in 2023-H2")
        physical_path = Path(context.repository_root) / str(
            _config_value(config, "data.frozen_physical_twin")
        )
        if not physical_path.is_file():
            raise PreflightBlocked(f"frozen physical artifact is unavailable: {physical_path}")
        physical = load_physical_limits(physical_path)
        if not physical.frozen:
            raise PreflightFailed("physical limits are not frozen")
        # Fit and freeze the exact train-only normalization before any neural
        # rolling inference.  The same corpus object also supplies the seven
        # causal history tokens used by formal training, so preflight and
        # deployment cannot silently diverge on input semantics.
        corpus = PreloadedR4Corpus(config, context.repository_root)
        normalization_path = output_root / "data" / "train_only_normalization.json"
        atomic_json(
            normalization_path,
            {
                **corpus.statistics.as_json(),
                "fixed_training_cost_scale": corpus.fixed_training_cost_scale,
                "physical_sha256": corpus.physical_sha256,
                "physical_twin_lineage": dict(corpus.physical_lineage),
                "train_days": len(corpus.records("train")),
                "validation_days": len(corpus.records("validation")),
            },
        )
        model = build_formal_r4_model(config, context.repository_root).to(device).eval()
        core = model.physical_stream.physical_twin_core
        day = load_r4_day(
            bundle_path,
            settlement_path,
            capacity_work_units=physical.capacity_work_units,
            physical_twin_core=core,
        )
        if day.split != "validation":
            raise PreflightFailed(f"preflight opened non-validation split {day.split}")
        checks.append(
            PreflightCheck(
                "fixed_temporal_split_and_issue_time_provenance",
                True,
                {
                    "operating_date": day.operating_date,
                    "split": day.split,
                    "bundle_path": str(bundle_path),
                    "settlement_path": str(settlement_path),
                    "feature_provenance_keys": sorted(day.feature_provenance),
                },
            )
        )

        parameter_count = int(model.parameter_count)
        hard_max = int(_config_value(config, "model.parameter_budget.hard_stop_above_trainable"))
        if parameter_count > hard_max:
            raise PreflightFailed(
                f"fresh R4 has {parameter_count} trainable parameters, above {hard_max}"
            )
        if not isinstance(core, ArtifactBackedFrozenPhysicalTwinCore):
            raise PreflightFailed(
                "formal R4 did not instantiate the artifact-backed physical twin"
            )
        if any(parameter.requires_grad for parameter in core.parameters()):
            raise PreflightFailed("fresh R4 physical-twin core is trainable")
        if core.training:
            raise PreflightFailed("fresh R4 physical-twin core is not in eval mode")
        physical_lineage = dict(core.artifact_metadata)
        if (
            physical_lineage.get("fit_split") != "train"
            or physical_lineage.get("selection_split") != "validation"
            or physical_lineage.get("locked_test_used_to_fit_or_select") is not False
        ):
            raise PreflightFailed("physical-twin split lineage is invalid")
        if dict(day.physical_twin_metadata) != physical_lineage:
            raise PreflightFailed("runtime day physical-twin lineage differs from model core")
        checks.append(
            PreflightCheck(
                "fresh_r4_model_and_frozen_physical",
                True,
                {
                    "architecture_version": model.architecture_version,
                    "trainable_parameter_count": parameter_count,
                    "device": str(device),
                    "physical_calibration_id": physical.calibration_id,
                    "physical_source_hashes": dict(physical.source_hashes),
                    "artifact_backed_physical_twin": True,
                    "physical_artifact_lineage": physical_lineage,
                    "model_frozen_twin_trainable_parameters": 0,
                },
            )
        )

        issue_record = next(
            item for item in corpus.records("validation")
            if str(item["operating_date"]) == operating_date
        )
        issue_keys = (
            "market", "workload", "physical_static", "physical_dynamic",
            "historical_market", "historical_market_available", "day_of_week",
            "month", "holiday", "dst", "forecast_anchors",
            "forecast_residual_caps", "forecast_residual_cap_metadata",
        )

        def add_batch_axis(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.unsqueeze(0)
            if isinstance(value, Mapping):
                return {name: add_batch_axis(item) for name, item in value.items()}
            return value

        normalized_issue_batch = {
            key: add_batch_axis(issue_record[key])
            for key in issue_keys
        }
        recorder = _RecordingR4Policy(model, normalized_issue_batch)
        runtime_config = dict(_config_value(config, "runtime"))
        runtime_config["allow_sealed_selection"] = False
        runtime_config["inference_mode"] = True
        result = R4Runtime(
            physical,
            runtime_config,
            physical_twin_core=corpus.physical_core,
        ).run_day(day, recorder)
        if result.get("status") != "COMPLETE" or int(result.get("completed_origins", -1)) != INTERVALS:
            raise PreflightFailed(
                f"real R4 rolling day did not complete 96 origins: {result.get('status')}"
            )
        tolerance = float(_config_value(config, "runtime.validation_tolerance"))
        independent = [
            validate_record(record, physical=physical, day=day, tolerance=tolerance)
            for record in result["records"]
        ]
        if len(independent) != INTERVALS or not all(item["passed"] for item in independent):
            raise PreflightFailed("independent per-origin residual validation failed")
        if recorder.scenario_counts != [20] * INTERVALS:
            raise PreflightFailed("fresh R4 did not emit exactly 20 particles at every origin")
        if result.get("day_ahead_frozen_after_origin_zero") is not True:
            raise PreflightFailed("day-ahead schedule changed after origin zero")
        residual_keys = [key for key in independent[0] if key != "passed"]
        maximum_residual = max(
            float(item[key]) for item in independent for key in residual_keys
        )
        checks.append(
            PreflightCheck(
                "one_complete_96_origin_r4_day_and_independent_residual_validation",
                True,
                {
                    "operating_date": day.operating_date,
                    "completed_origins": INTERVALS,
                    "independently_validated_origins": len(independent),
                    "exact_scenario_count_each_origin": 20,
                    "day_ahead_frozen_after_origin_zero": True,
                    "maximum_independent_residual": maximum_residual,
                    "validation_tolerance": tolerance,
                },
            )
        )
        liquid_tolerance = float(
            _config_value(
                config,
                "model.rolling_state.incremental_prefix_replay_equivalence_tolerance",
            )
        )
        replay = _liquid_replay_evidence(model, recorder.state_inputs, liquid_tolerance)
        checks.append(PreflightCheck("incremental_liquid_state_matches_prefix_replay", True, replay))
        status, gate_status, reason = "COMPLETE", "PASSED", "all measured preflight checks passed"
    except PreflightBlocked as exc:
        status, gate_status, reason = "BLOCKED", "NOT_RUN", str(exc)
    except (PreflightFailed, ValueError, RuntimeError, OSError, KeyError) as exc:
        status, gate_status, reason = "FAILED", "FAILED", str(exc)
        checks.append(
            PreflightCheck(
                "terminal_failure",
                False,
                {"exception_type": type(exc).__name__, "reason": str(exc)},
            )
        )
    except Exception as exc:  # fail closed while retaining diagnostics
        status, gate_status, reason = "FAILED", "FAILED", str(exc)
        checks.append(
            PreflightCheck(
                "unexpected_terminal_failure",
                False,
                {
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    metrics = {
        "schema_version": "dcvpp.e2e_moe_hybrid_r4.preflight.v1",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "status": status,
        "reason": reason,
        "lineage_id": context.lineage_id,
        "config_sha256": context.config_sha256,
        "expected_check_names": [item.name for item in checks],
        "checks": [asdict(item) for item in checks],
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }
    atomic_json(preflight_root / "metrics.json", metrics)
    atomic_text(
        preflight_root / "runtime.log",
        "\n".join(
            [f"R4 PRECHECK {started}"]
            + [f"{'PASS' if item.passed else 'FAIL'} {item.name}: {dict(item.evidence)}" for item in checks]
            + [f"TERMINAL {status}: {reason}"]
        )
        + "\n",
    )
    artifact_state = _artifact_state(
        status=status,
        capability_status=gate_status,
        reason=reason,
        context=context,
    )
    write_state(preflight_root / "state.json", artifact_state)
    return {
        **artifact_state,
        "checks": metrics["checks"],
        "metrics_path": str((preflight_root / "metrics.json").resolve()),
        "runtime_log_path": str((preflight_root / "runtime.log").resolve()),
        "preflight_state_path": str((preflight_root / "state.json").resolve()),
    }


__all__ = [
    "PreflightBlocked",
    "PreflightCheck",
    "PreflightFailed",
    "audit_active_import_graph",
    "inspect_cuda_environment",
    "run_measured_preflight",
    "verify_locked_path_guard",
    "verify_temporal_path_guards",
]
