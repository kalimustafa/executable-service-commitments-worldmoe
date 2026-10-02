"""Exact reuse of the registered baseline and seasonal-naive A2 solve."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from dcvpp.core import atomic_json, sha256_json, utc_now
from dcvpp.optimization.stochastic_milp import array_sha256


SCHEMA_VERSION = "dart_equivalent_baseline_reuse_v1"
REGISTERED_CANDIDATE = "seasonal_naive"
REGISTERED_EXPERIMENT = "A2"
BASELINE_EXPERIMENT = "BASELINE"


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    return value


def _callable_identity(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    return {
        "module": str(getattr(value, "__module__", "unknown")),
        "qualified_name": str(
            getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        ),
    }


def operational_solver_input_sha256(
    *,
    bundle: Any,
    physical: Any,
    settlement: Any,
    forcing: Any,
    seed: int,
    config: Mapping[str, Any],
    first_stage_solver: Any,
    recourse_solver: Any,
    component_cost_hook: Any,
) -> str:
    """Hash every mathematical input to the first-stage and recourse solves."""

    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "policy_id": "B6",
            "seed": int(seed),
            "bundle": {
                "operating_date": str(bundle.operating_date),
                "regime": str(bundle.regime),
                "kappa": float(bundle.kappa),
                "initial_count": int(bundle.initial_count),
                "reduced_count": int(bundle.reduced_count),
                "zones": list(map(str, bundle.zones)),
                "timestamps_utc": array_sha256(bundle.timestamps_utc),
                "da_price": array_sha256(bundle.da_price),
                "rt_price": array_sha256(bundle.rt_price),
                "workload": array_sha256(bundle.workload),
                "pv": array_sha256(bundle.pv),
                "physical_multiplier": array_sha256(bundle.physical_multiplier),
                "probabilities": array_sha256(bundle.probabilities),
                "selected_parent_ids": array_sha256(bundle.selected_parent_ids),
                "full_parent_mapping": array_sha256(bundle.full_parent_mapping),
                "reduction_mse": float(bundle.reduction_mse),
                "scenario_seed": int(bundle.seed),
                "mixture_stress_mass": float(bundle.mixture_stress_mass),
                "nominal_component_count": int(bundle.nominal_component_count),
                "stress_component_count": int(bundle.stress_component_count),
                "decision_value_evaluation": bool(bundle.decision_value_evaluation),
            },
            "physical": {
                "calibration_id": str(physical.calibration_id),
                "twin_ids": list(map(str, physical.twin_ids)),
                "transfer_regime": str(physical.transfer_regime),
                "zones": list(map(str, physical.zones)),
                "idle_mw": array_sha256(physical.idle_mw),
                "work_mw_per_unit": array_sha256(physical.work_mw_per_unit),
                "capacity_work_units": array_sha256(physical.capacity_work_units),
                "grid_import_limit_mw": array_sha256(physical.grid_import_limit_mw),
                "battery_power_mw": array_sha256(physical.battery_power_mw),
                "battery_energy_mwh": array_sha256(physical.battery_energy_mwh),
            },
            "settlement": str(settlement.input_sha256),
            "forcing": str(forcing.input_sha256),
            "config": _plain(config),
            "first_stage_solver": _callable_identity(first_stage_solver),
            "recourse_solver": _callable_identity(recourse_solver),
            "component_cost_hook": _callable_identity(component_cost_hook),
        }
    )


def _eligible(candidate_id: str, experiment_id: str) -> bool:
    return experiment_id == BASELINE_EXPERIMENT or (
        candidate_id == REGISTERED_CANDIDATE and experiment_id == REGISTERED_EXPERIMENT
    )


def _cache_path(root: Path, *, day: str, seed: int, fingerprint: str) -> Path:
    return (
        Path(root)
        / "equivalent_baseline_seasonal_a2"
        / str(day)
        / f"seed_{int(seed)}"
        / f"{fingerprint}.json"
    )


def _adapt_telemetry(
    result: Mapping[str, Any],
    *,
    actionability: Any,
    model_complexity: float,
    forecast_time_seconds: float,
    fingerprint: str,
    source_candidate_id: str,
    source_experiment_id: str,
    target_candidate_id: str,
    target_experiment_id: str,
    reused: bool,
) -> dict[str, Any]:
    output = deepcopy(dict(result))
    output["gate_time_feasible_flexible_energy_mwh"] = float(
        0.0 if actionability is None else actionability.total_feasible_mwh
    )
    output["model_complexity"] = float(max(model_complexity, 0.0))
    output["forecast_time_seconds"] = float(max(forecast_time_seconds, 0.0))
    output["equivalent_operational_solve_reuse"] = {
        "schema_version": SCHEMA_VERSION,
        "mathematical_solver_input_sha256": fingerprint,
        "source_candidate_id": source_candidate_id,
        "source_experiment_id": source_experiment_id,
        "target_candidate_id": target_candidate_id,
        "target_experiment_id": target_experiment_id,
        # This is a scientific role, not a per-call cache-hit counter.  Keeping
        # it target/source-derived makes checkpoint identity stable on resume.
        "underlying_milp_reused": bool(
            target_candidate_id == REGISTERED_CANDIDATE
            and target_experiment_id == REGISTERED_EXPERIMENT
            and (
                source_candidate_id != target_candidate_id
                or source_experiment_id != target_experiment_id
            )
        ),
        "labels_or_telemetry_in_solver_input_hash": False,
        "scientific_estimand_changed": False,
    }
    return output


def _run_with_equivalent_cache(
    original: Callable[..., Mapping[str, Any]], cache_root: Path, **kwargs: Any
) -> dict[str, Any]:
    candidate_id = str(kwargs["candidate_id"])
    experiment_id = str(kwargs["experiment_id"])
    if not _eligible(candidate_id, experiment_id):
        return dict(original(**kwargs))
    fingerprint = operational_solver_input_sha256(
        bundle=kwargs["bundle"],
        physical=kwargs["physical"],
        settlement=kwargs["settlement"],
        forcing=kwargs["forcing"],
        seed=int(kwargs["seed"]),
        config=kwargs["config"],
        first_stage_solver=kwargs["first_stage_solver"],
        recourse_solver=kwargs["recourse_solver"],
        component_cost_hook=kwargs.get("component_cost_hook"),
    )
    path = _cache_path(
        cache_root,
        day=str(kwargs["bundle"].operating_date),
        seed=int(kwargs["seed"]),
        fingerprint=fingerprint,
    )
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("schema_version") != SCHEMA_VERSION
            or cached.get("mathematical_solver_input_sha256") != fingerprint
            or cached.get("result_sha256") != sha256_json(cached.get("result"))
        ):
            raise ValueError(f"equivalent operational-solve cache is invalid: {path}")
        source_result = dict(cached["result"])
        source_candidate = str(cached["source_candidate_id"])
        source_experiment = str(cached["source_experiment_id"])
        reused = True
    else:
        source_result = dict(original(**kwargs))
        source_candidate = candidate_id
        source_experiment = experiment_id
        reused = False
        atomic_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE",
                "created_at_utc": utc_now(),
                "mathematical_solver_input_sha256": fingerprint,
                "source_candidate_id": source_candidate,
                "source_experiment_id": source_experiment,
                "result": source_result,
                "result_sha256": sha256_json(source_result),
            },
        )
    return _adapt_telemetry(
        source_result,
        actionability=kwargs.get("actionability"),
        model_complexity=float(kwargs.get("model_complexity", 0.0)),
        forecast_time_seconds=float(kwargs.get("forecast_time_seconds", 0.0)),
        fingerprint=fingerprint,
        source_candidate_id=source_candidate,
        source_experiment_id=source_experiment,
        target_candidate_id=candidate_id,
        target_experiment_id=experiment_id,
        reused=reused,
    )


def _wrap_runner(
    real_evidence: Any,
    runner: Callable[..., Mapping[str, Any]],
    cache_root: Path,
) -> Callable[..., Mapping[str, Any]]:
    def wrapped(*args: Any) -> Mapping[str, Any]:
        original_external = real_evidence._run_external_cell

        def cached_external(**kwargs: Any) -> dict[str, Any]:
            return _run_with_equivalent_cache(original_external, cache_root, **kwargs)

        real_evidence._run_external_cell = cached_external
        try:
            return runner(*args)
        finally:
            real_evidence._run_external_cell = original_external

    return wrapped


def _install_panel_wrapper(
    real_evidence: Any, original_panel: Callable[..., Mapping[str, Any]]
) -> Callable[..., Mapping[str, Any]]:
    def panel_with_reuse(
        panel: Mapping[str, Any], checkpoint_root: Path, **kwargs: Any
    ) -> dict[str, Any]:
        cache_root = Path(checkpoint_root) / "equivalent_solver_cache"
        candidate_runner = kwargs["candidate_runner"]
        baseline_runner = kwargs["baseline_runner"]

        def candidate(cell: Any) -> Mapping[str, Any]:
            if cell.candidate_id == REGISTERED_CANDIDATE and cell.experiment_id == REGISTERED_EXPERIMENT:
                return _wrap_runner(real_evidence, candidate_runner, cache_root)(cell)
            return candidate_runner(cell)

        forwarded = dict(kwargs)
        forwarded["candidate_runner"] = candidate
        forwarded["baseline_runner"] = _wrap_runner(
            real_evidence, baseline_runner, cache_root
        )
        return dict(original_panel(panel, checkpoint_root, **forwarded))

    return panel_with_reuse


@contextmanager
def equivalent_baseline_reuse_runtime(
    *, ledger_path: Path | None = None
) -> Iterator[Mapping[str, Any]]:
    """Patch the production panel boundary without touching frozen modules."""

    from dcvpp.experiments import dart_real_evidence as real_evidence

    original_panel = real_evidence.run_dart_replay_panel
    real_evidence.run_dart_replay_panel = _install_panel_wrapper(
        real_evidence, original_panel
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ACTIVE",
        "activated_at_utc": utc_now(),
        "registered_pair": [
            "DART_BASELINE_B6_SEASONAL_NAIVE_V1",
            "seasonal_naive/A2",
        ],
        "full_mathematical_input_hash_required": True,
        "selection_only": True,
        "locked_2025_opened": False,
    }
    if ledger_path is not None:
        atomic_json(ledger_path, record)
    try:
        yield record
    finally:
        real_evidence.run_dart_replay_panel = original_panel
        record.update({"status": "INACTIVE", "deactivated_at_utc": utc_now()})
        if ledger_path is not None:
            atomic_json(ledger_path, record)


__all__ = [
    "SCHEMA_VERSION",
    "_install_panel_wrapper",
    "_run_with_equivalent_cache",
    "equivalent_baseline_reuse_runtime",
    "operational_solver_input_sha256",
]
