from __future__ import annotations

"""Solver-free orchestration for E2E-MoE-DART-VPP R3.

This module is the only command-line entrypoint for the active R3 variant.  It
never imports or invokes a mathematical-programming backend.  Neural decisions
are generated once at every rolling origin by ``model.forward_origin`` and are
then executed through :class:`NeuralConstrainedRuntime`.
"""

from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
import psutil
import torch
import typer

from dcvpp.core import load_yaml

from .constrained_runtime import (
    INTERVALS,
    SCENARIOS,
    NeuralConstrainedRuntime,
    R3DataError,
    load_frozen_physical,
    load_selection_day,
    records_frame,
)


app = typer.Typer(no_args_is_help=True, add_completion=False)
TRAINING_PHASES = ("representation", "scenario", "policy", "a1", "a2")
PHASES = (*TRAINING_PHASES, "selection_evaluation")
_PARENT_PHASE = {
    "scenario": "representation",
    "policy": "scenario",
    "a1": "policy",
    "a2": "a1",
}


def _root_from_config(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    return parent.parent if parent.name == "configs" else Path.cwd().resolve()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _assert_solver_free_config(cfg: Mapping[str, Any]) -> None:
    runtime = cfg.get("runtime")
    if not isinstance(runtime, Mapping):
        raise typer.BadParameter("solver-free config requires a runtime mapping")
    if runtime.get("backend") != "neural_constraint_shield":
        raise typer.BadParameter("runtime.backend must be neural_constraint_shield")
    if runtime.get("external_solver") is not False:
        raise typer.BadParameter("runtime.external_solver must be false")
    output_root = str(cfg.get("project", {}).get("output_root", ""))
    if not output_root or "gurobi" in output_root.lower():
        raise typer.BadParameter(
            "solver-free execution requires a distinct non-Gurobi output_root"
        )
    if int(cfg.get("project", {}).get("locked_test_year", 2025)) != 2025:
        raise typer.BadParameter("locked_test_year must remain 2025")
    if int(cfg.get("rolling", {}).get("intervals_per_day", 0)) != INTERVALS:
        raise typer.BadParameter("rolling.intervals_per_day must be 96")
    if int(cfg.get("rolling", {}).get("scenario_count", 0)) != SCENARIOS:
        raise typer.BadParameter("rolling.scenario_count must be 20")


def _run_dir(
    root: Path,
    cfg: Mapping[str, Any],
    phase: str,
    split: str,
    *,
    resume: bool,
) -> Path:
    output = root / str(cfg["project"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"run_id={phase}_{split}_"
    existing = sorted(
        (path for path in output.glob(f"{prefix}*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if resume and existing:
        return existing[-1]
    run = output / f"{prefix}{_timestamp()}"
    suffix = 0
    while run.exists():
        suffix += 1
        run = output / f"{prefix}{_timestamp()}_{suffix:02d}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoints").mkdir()
    return run


def _initialize_run(run: Path, config_path: Path) -> None:
    target = run / "config_used.yaml"
    if not target.exists():
        shutil.copyfile(config_path, target)
    (run / "checkpoints").mkdir(exist_ok=True)
    (run / "runtime.log").touch(exist_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _scoped_run_dir(
    root: Path,
    cfg: Mapping[str, Any],
    phase: str,
    split: str,
    *,
    resume: bool,
    scope: Mapping[str, Any],
) -> Path:
    """Resume only an artifact run created for the same evaluation scope."""

    run = _run_dir(root, cfg, phase, split, resume=resume)
    if not resume:
        return run
    state = _read_json(run / "state.json")
    if state is None or all(state.get(key) == value for key, value in scope.items()):
        return run
    # A different day/month is a different scientific run.  Preserve the old
    # evidence instead of replacing its state and metrics in place.
    return _run_dir(root, cfg, phase, split, resume=False)


def _write_state(run: Path, payload: Mapping[str, Any]) -> None:
    temporary = run / "state.json.tmp"
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(run / "state.json")


def _write_metrics(run: Path, rows: pd.DataFrame | list[Mapping[str, Any]]) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame([{"status": "NO_METRICS"}])
    temporary = run / "metrics.tmp.parquet"
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as handle:
        json_path = Path(handle.name)
        frame.to_json(handle, orient="records", date_format="iso")
    parquet_path = json_path.with_suffix(".parquet")
    try:
        code = (
            "import sys,pandas as pd,pyarrow as pa,pyarrow.parquet as pq;"
            "f=pd.read_json(sys.argv[1],orient='records');"
            "pq.write_table(pa.Table.from_pandas(f,preserve_index=False),"
            "sys.argv[2],compression='zstd')"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(json_path), str(parquet_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        with parquet_path.open("rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
    finally:
        json_path.unlink(missing_ok=True)
        parquet_path.unlink(missing_ok=True)
    temporary.replace(run / "metrics.parquet")


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        json_path = Path(handle.name)
    try:
        code = (
            "import sys,pandas as pd;"
            "pd.read_parquet(sys.argv[1]).to_json(sys.argv[2],"
            "orient='records',date_format='iso')"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(path), str(json_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return pd.read_json(json_path, orient="records")
    except (OSError, ValueError, subprocess.SubprocessError):
        return pd.DataFrame()
    finally:
        json_path.unlink(missing_ok=True)


def _write_summary(run: Path, title: str, payload: Mapping[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _device_snapshot() -> dict[str, Any]:
    process = psutil.Process()
    memory = psutil.virtual_memory()
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_logical_count": psutil.cpu_count(),
        "process_cpu_percent": process.cpu_percent(interval=None),
        "process_rss_bytes": process.memory_info().rss,
        "system_ram_used_bytes": memory.used,
        "system_ram_available_bytes": memory.available,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_allocated_bytes": 0,
        "gpu_reserved_bytes": 0,
    }
    if torch.cuda.is_available():
        result.update(
            gpu_name=torch.cuda.get_device_name(0),
            gpu_allocated_bytes=torch.cuda.memory_allocated(0),
            gpu_reserved_bytes=torch.cuda.memory_reserved(0),
        )
    return result


def _paths(root: Path, cfg: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    data = cfg["data"]
    return (
        root / str(data["selection_bundle_root"]),
        root / str(data["realized_settlement_root"]),
        root / str(data["physical_artifact"]),
    )


def _lineage_complete(state: Mapping[str, Any], phase: str) -> bool:
    parent = _PARENT_PHASE.get(phase)
    if parent is None:
        return True
    predecessor = state.get("predecessor", {})
    if not isinstance(predecessor, Mapping) or predecessor.get("phase") != parent:
        return False
    run_dir = predecessor.get("run_dir")
    if not run_dir:
        return False
    parent_state = _read_json(Path(str(run_dir)) / "state.json")
    return bool(
        parent_state is not None
        and parent_state.get("status") == "COMPLETE"
        and _lineage_complete(parent_state, parent)
    )


def _latest_checkpoint(
    root: Path,
    cfg: Mapping[str, Any],
    *,
    phase: str = "a2",
) -> tuple[Path | None, dict[str, Any] | None]:
    output = root / str(cfg["project"]["output_root"])
    candidates = sorted(
        (
            path
            for path in output.glob(f"run_id={phase}_selection_*")
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        training = cfg.get("training", {})
        formal = training.get("formal", {}) if isinstance(training, Mapping) else {}
        is_formal = bool(
            isinstance(training, Mapping)
            and (
                str(training.get("mode", "pilot")).lower() == "formal"
                or (isinstance(formal, Mapping) and formal.get("enabled", False))
            )
        )
        checkpoint = candidate / "checkpoints" / (
            "best.pt" if is_formal else "latest.pt"
        )
        state = _read_json(candidate / "state.json")
        if (
            checkpoint.exists()
            and state is not None
            and state.get("status") == "COMPLETE"
            and int(state.get("completed_epochs", 0)) > 0
            and _lineage_complete(state, phase)
        ):
            return checkpoint, state
    return None, None


def _model_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from .model import E2EMoER3Config

    values: dict[str, Any] = {
        "market_dim": 4,
        "workload_dim": 2,
        "physical_dim": 4,
        "state_dim": 4,
    }
    aliases = {
        "experts": "num_experts",
        "refinement_steps": "policy_refinement_steps",
    }
    allowed = set(E2EMoER3Config.__dataclass_fields__)
    for key, value in cfg.get("model", {}).items():
        normalized = aliases.get(str(key), str(key))
        if normalized in allowed:
            values[normalized] = value
    policy = cfg.get("policy", {})
    if isinstance(policy, Mapping) and "refinement_steps" in policy:
        values["policy_refinement_steps"] = policy["refinement_steps"]
    return values


def _load_a2_model(
    root: Path, cfg: Mapping[str, Any]
) -> tuple[torch.nn.Module | None, Path | None, str | None]:
    checkpoint, _ = _latest_checkpoint(root, cfg, phase="a2")
    if checkpoint is None:
        return None, None, "no lineage-complete COMPLETE A2 checkpoint is available"
    from .model import E2EMoEDARTVPP, E2EMoER3Config

    model = E2EMoEDARTVPP(E2EMoER3Config(**_model_kwargs(cfg)))
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
            raise ValueError("checkpoint has no model_state_dict")
        model.load_state_dict(payload["model_state_dict"])
    except Exception as exc:
        return None, checkpoint, f"invalid A2 checkpoint: {type(exc).__name__}: {exc}"
    requested_device = str(cfg.get("training", {}).get("device", "auto"))
    device = (
        torch.device("cuda")
        if requested_device == "auto" and torch.cuda.is_available()
        else torch.device("cpu")
        if requested_device == "auto"
        else torch.device(requested_device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        return None, checkpoint, f"requested evaluation device {device} is unavailable"
    try:
        model.to(device)
    except (RuntimeError, ValueError) as exc:
        return None, checkpoint, f"cannot place A2 model on {device}: {exc}"
    model.eval()
    return model, checkpoint, None


def _issue_time_inputs(day: Any, physical: Any) -> dict[str, torch.Tensor]:
    da = np.asarray(day.da_price, dtype=np.float32).T
    rt_pool = np.asarray(day.rt_price_pool, dtype=np.float32)
    rt = rt_pool.mean(axis=0).T
    market = np.stack(
        (
            da.mean(axis=1),
            rt.mean(axis=1),
            (da - rt).mean(axis=1),
            rt_pool.std(axis=(0, 1)),
        ),
        axis=-1,
    )
    workload_pool = np.asarray(day.workload_pool, dtype=np.float32)
    workload = np.stack(
        (workload_pool.mean(axis=0), workload_pool.std(axis=0)), axis=-1
    )
    physical_input = np.broadcast_to(
        np.asarray(day.physical_multiplier, dtype=np.float32), (INTERVALS, 4)
    ).copy()
    energy = np.maximum(
        np.asarray(physical.battery_energy_mwh, dtype=np.float32), 1e-6
    )
    state = np.zeros((INTERVALS, 4), dtype=np.float32)
    state[0] = np.asarray(day.initial_soc_mwh, dtype=np.float32) / energy
    return {
        "market": torch.from_numpy(market).unsqueeze(0),
        "workload": torch.from_numpy(workload).unsqueeze(0),
        "physical": torch.from_numpy(physical_input).unsqueeze(0),
        "state": torch.from_numpy(state).unsqueeze(0),
    }


def _particles_to_joint(
    particles: np.ndarray,
    day: Any,
) -> dict[str, np.ndarray]:
    if particles.shape != (SCENARIOS, INTERVALS, 6):
        raise ValueError("model scenarios must have shape [20,96,6]")
    rt_base = np.asarray(day.rt_price_pool, dtype=float)[:SCENARIOS]
    rt_center = rt_base.mean(axis=1, keepdims=True)
    rt_scale = np.divide(
        particles[:, None, :, 1],
        np.where(np.abs(rt_center) > 1e-6, rt_center, 1.0),
    )
    pv_base = np.asarray(day.pv_pool, dtype=float)[:SCENARIOS]
    pv_center = pv_base.mean(axis=1, keepdims=True)
    pv_scale = np.divide(
        np.maximum(particles[:, None, :, 4], 0.0),
        np.where(pv_center > 1e-6, pv_center, 1.0),
    )
    pue = np.maximum(particles[:, :, 5].mean(axis=1), 1e-6)
    pue /= max(float(pue.mean()), 1e-6)
    return {
        "rt": rt_base * rt_scale,
        "workload": np.maximum(particles[:, :, 3], 0.0),
        "pv": np.maximum(pv_base * pv_scale, 0.0),
        "physical": pue[:, None]
        * np.asarray(day.physical_multiplier, dtype=float)[None, :],
    }


def _learned_policy_adapter(
    model: torch.nn.Module,
    day: Any,
    physical: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
    """Create one causal ``forward_origin`` call per runtime callback."""

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    inputs = {
        name: value.to(device) for name, value in _issue_time_inputs(day, physical).items()
    }
    prepared_context = (
        model.prepare_origin_context(inputs)  # type: ignore[attr-defined]
        if hasattr(model, "prepare_origin_context")
        else None
    )
    energy = np.maximum(np.asarray(physical.battery_energy_mwh, dtype=np.float32), 1e-6)
    grid_limit = np.asarray(physical.grid_import_limit_mw, dtype=np.float32)
    battery_limit = np.asarray(physical.battery_power_mw, dtype=np.float32)
    work_capacity = np.asarray(physical.capacity_work_units, dtype=np.float32)
    periods = torch.arange(INTERVALS, device=device).unsqueeze(0)
    scenario_cache: dict[str, Any] = {}
    state_history = np.zeros((INTERVALS, 4), dtype=np.float32)

    def callback(context: Mapping[str, Any]) -> Mapping[str, Any]:
        origin = int(context["origin"])
        # Origin zero was evaluated once below to obtain the issue-time learned
        # particles.  Reuse that exact state-conditioned action so a 96-origin
        # day contains exactly 96 forward_origin calls, not 97.
        if origin == 0 and "origin_zero_action" in scenario_cache:
            return scenario_cache.pop("origin_zero_action")
        soc = np.clip(
            np.asarray(context["soc_mwh"], dtype=np.float32),
            0.0,
            energy,
        )
        queue = np.asarray(context["queue_age_buckets"], dtype=np.float32)
        backlog_total = float(queue.sum())
        deadline = float(queue[0]) if queue.size else 0.0
        previous = context.get("previous_action")
        if isinstance(previous, Mapping):
            previous_action = {
                "grid": torch.as_tensor(
                    np.asarray(previous.get("grid", np.zeros(4)), dtype=np.float32),
                    device=device,
                ).unsqueeze(0),
                "battery": torch.as_tensor(
                    np.asarray(previous.get("battery", np.zeros(4)), dtype=np.float32),
                    device=device,
                ).unsqueeze(0),
                "compute": torch.as_tensor(
                    np.asarray(previous.get("compute", np.zeros(4)), dtype=np.float32),
                    device=device,
                ).unsqueeze(0),
                "migration": torch.zeros(
                    (1, 4, 4), dtype=torch.float32, device=device
                ),
            }
        else:
            previous_action = {
                "grid": torch.zeros((1, 4), device=device),
                "battery": torch.zeros((1, 4), device=device),
                "compute": torch.zeros((1, 4), device=device),
                "migration": torch.zeros((1, 4, 4), device=device),
            }
        state_history[origin] = soc / energy
        causal_state = torch.from_numpy(state_history.copy()).unsqueeze(0).to(device)
        call_inputs = {**inputs, "state": causal_state}
        revealed = periods.le(origin)
        with torch.inference_mode():
            output = model.forward_origin(  # type: ignore[attr-defined]
                call_inputs,
                origin=origin,
                current_soc=torch.from_numpy(soc).unsqueeze(0).to(device),
                backlog=torch.full(
                    (1, 4), backlog_total / 4.0, device=device
                ),
                deadline_pressure=torch.full(
                    (1, 4), deadline / 4.0, device=device
                ),
                previous_action=previous_action,
                revealed_mask=revealed,
                prepared_context=prepared_context,
            )
        actions = output["actions"]
        raw: dict[str, Any] = {
            "grid": actions["grid"][0].detach().cpu().numpy() * grid_limit,
            "battery": actions["battery"][0].detach().cpu().numpy()
            * battery_limit,
            "compute": actions["compute"][0].detach().cpu().numpy()
            * work_capacity,
            # The empirical selection contract is known-zero migration.
            "migration": np.zeros((4, 4), dtype=float),
            "fallback": bool(
                float(output["fallback_confidence"][0].detach().cpu()) < 0.5
                or float(output["feasibility"][0].detach().cpu()) < 0.5
            ),
        }
        if origin == 0:
            raw["day_ahead"] = (
                actions["day_ahead"][0].detach().cpu().numpy()
                * grid_limit[:, None]
            )
            particles = output["scenario"]["particles"][0].detach().cpu().numpy()
            weights = output["scenario"]["weights"][0].detach().cpu().numpy()
            scenario_cache["joint"] = _particles_to_joint(particles, day)
            scenario_cache["weights"] = weights / max(float(weights.sum()), 1e-12)
            scenario_cache["origin_zero_action"] = raw
        return raw

    # A single origin-zero dry callback would mutate runtime state, so obtain
    # the learned scenario distribution with an explicit origin-zero forward.
    zero_context: dict[str, Any] = {
        "origin": 0,
        "soc_mwh": np.asarray(day.initial_soc_mwh, dtype=np.float32),
        "queue_age_buckets": np.asarray(
            [0.0, 0.0, 0.0, 0.0, float(day.realized_workload[0])],
            dtype=np.float32,
        ),
        "previous_action": None,
    }
    callback(zero_context)
    return scenario_cache["joint"], scenario_cache["weights"], callback


def _blocked_day(date: str, reason: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "operating_date": date,
        "reason": reason,
        "completed_origins": 0,
        "scenario_count": SCENARIOS,
        "records": [],
        "summary": {},
    }


def _failed_day(date: str, reason: str) -> dict[str, Any]:
    result = _blocked_day(date, reason)
    result["status"] = "FAILED"
    return result


def _run_day(
    cfg: Mapping[str, Any],
    root: Path,
    run: Path,
    date: str,
    *,
    origins: int,
    formal: bool,
    model: torch.nn.Module | None,
) -> dict[str, Any]:
    if str(date).startswith("2025"):
        raise R3DataError("2025 remains closed")
    if model is None:
        return _blocked_day(date, "a COMPLETE A2 model is required for formal rolling")
    bundle_root, settlement_root, physical_path = _paths(root, cfg)
    settlement_path = settlement_root / f"{date}.npz"
    try:
        day = load_selection_day(
            bundle_root / f"{date}.npz",
            realized_settlement_path=settlement_path if formal else None,
        )
        physical = load_frozen_physical(physical_path)
    except (OSError, R3DataError, KeyError, ValueError) as exc:
        return _blocked_day(date, f"{type(exc).__name__}: {exc}")
    try:
        joint, weights, callback = _learned_policy_adapter(model, day, physical)
        runtime = NeuralConstrainedRuntime(
            physical, cfg["runtime"], logfile=run / "runtime.log"
        )
        return runtime.run_day(
            day,
            policy_callback=callback,
            joint_scenarios=joint,
            scenario_weights=weights,
            max_origins=origins,
            formal=formal,
        )
    except Exception as exc:
        return _failed_day(date, f"{type(exc).__name__}: {exc}")


def _month_candidates(bundle_root: Path, month: str | None) -> list[Path]:
    if month is not None:
        if month.startswith("2025"):
            raise typer.BadParameter("2025 remains closed")
        if len(month) != 7 or month[:4] not in {"2022", "2023", "2024"}:
            raise typer.BadParameter("month must be YYYY-MM within 2022-2024")
        return sorted(bundle_root.glob(f"{month}-*.npz"))
    return sorted(
        path
        for year in (2022, 2023, 2024)
        for path in bundle_root.glob(f"{year}-*.npz")
    )


def _evaluate_impl(
    *,
    cfg: Mapping[str, Any],
    root: Path,
    run: Path,
    month: str | None,
    resume: bool,
    model: torch.nn.Module | None,
    checkpoint: Path | None,
) -> dict[str, Any]:
    bundle_root, _, _ = _paths(root, cfg)
    candidates = _month_candidates(bundle_root, month)
    rows: list[dict[str, Any]] = []
    resumed_existing_run = resume and (run / "state.json").exists()
    if resume:
        prior = _read_metrics(run / "metrics.parquet")
        if {"operating_date", "origin"}.issubset(prior.columns):
            rows = prior.loc[
                prior["operating_date"].notna() & prior["origin"].notna()
            ].to_dict(orient="records")
    counts: dict[str, int] = {}
    for row in rows:
        date = str(row.get("operating_date", ""))[:10]
        if date:
            counts[date] = counts.get(date, 0) + int(not bool(row.get("rejected", False)))
    completed_dates = {date for date, count in counts.items() if count == INTERVALS}
    required_metric_columns = {
        "day_weighted_particle_expected_cost_usd",
        "day_weighted_particle_cvar_95_usd",
        "device_process_rss_bytes",
    }
    if rows and not required_metric_columns.issubset(set(prior.columns)):
        # Metric schema changed: rebuild every requested day rather than
        # calling an old row set complete with missing formal evidence.
        rows = []
        completed_dates = set()
    started = time.perf_counter()
    start_device = _device_snapshot()
    status = "PARTIAL"
    reason: str | None = None
    prior_state = _read_json(run / "state.json")
    resume_compatible = bool(
        prior_state is None
        or (
            prior_state.get("phase") == "selection_evaluation"
            and prior_state.get("split") == "selection"
            and prior_state.get("month") == month
        )
    )
    if not resume_compatible:
        status = "BLOCKED"
        reason = (
            "resume scope mismatch: existing selection evaluation was created "
            f"for month={prior_state.get('month')!r}, requested month={month!r}"
        )
    elif model is None:
        status = "BLOCKED"
        reason = "no lineage-complete COMPLETE A2 checkpoint is available"
    elif not candidates:
        status = "BLOCKED"
        reason = f"no selection bundles match {month or '2022-2024'}"
    else:
        for candidate in candidates:
            date = candidate.stem
            if date in completed_dates:
                continue
            # Resume discards an incomplete prior day before rerunning it.
            rows = [row for row in rows if str(row.get("operating_date", ""))[:10] != date]
            result = _run_day(
                cfg, root, run, date, origins=INTERVALS, formal=True, model=model
            )
            day_summary = result.get("summary", {})
            device_day = _device_snapshot()
            day_rows = result.get("records", [])
            for row in day_rows:
                row.update(
                    day_weighted_particle_expected_cost_usd=day_summary.get(
                        "weighted_particle_expected_cost_usd"
                    ),
                    day_weighted_particle_cvar_95_usd=day_summary.get(
                        "weighted_particle_cvar_95_usd"
                    ),
                    day_total_cost_usd=day_summary.get("total_cost_usd"),
                    device_process_rss_bytes=device_day.get("process_rss_bytes"),
                    device_system_ram_available_bytes=device_day.get(
                        "system_ram_available_bytes"
                    ),
                    device_gpu_allocated_bytes=device_day.get("gpu_allocated_bytes"),
                    device_gpu_reserved_bytes=device_day.get("gpu_reserved_bytes"),
                )
            rows.extend(day_rows)
            _write_metrics(run, rows)
            if result.get("status") == "COMPLETE" and int(
                result.get("completed_origins", 0)
            ) == INTERVALS:
                completed_dates.add(date)
                continue
            result_status = str(result.get("status", "PARTIAL"))
            status = (
                result_status
                if result_status in {"BLOCKED", "FAILED"}
                else "PARTIAL"
            )
            reason = f"day {date}: {result.get('reason', result.get('status'))}"
            break
        else:
            status = (
                "COMPLETE"
                if len(completed_dates) == len(candidates)
                and all(date.stem in completed_dates for date in candidates)
                else "PARTIAL"
            )
    completed_requested = {
        candidate.stem for candidate in candidates if candidate.stem in completed_dates
    }
    complete_rows = [
        row
        for row in rows
        if str(row.get("operating_date", ""))[:10] in completed_requested
        and not bool(row.get("rejected", False))
    ]
    resume_command = (
        ".\\.venv\\Scripts\\python.exe scripts\\run_e2e_moe_r3.py evaluate "
        "--split selection --full-rolling --config "
        "configs\\e2e_moe_dart_vpp_R3_solver_free.yaml --resume"
        + (f" --month {month}" if month else "")
    )
    frame = pd.DataFrame(complete_rows)
    numeric = lambda name: pd.to_numeric(  # noqa: E731 - compact audit helper
        frame[name], errors="coerce"
    ) if name in frame else pd.Series(dtype=float)
    daily = (
        frame.groupby("operating_date", sort=True)
        if not frame.empty and "operating_date" in frame
        else None
    )
    audit_summary: dict[str, Any] = {
        "feasibility_rate": float(frame.get("passed", pd.Series(dtype=bool)).mean())
        if not frame.empty
        else 0.0,
        "rejected_origin_count": int(
            frame.get("rejected", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "exogenous_infeasible_origin_count": int(
            frame.get("exogenous_infeasible", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "safe_policy_origin_count": int(
            (frame.get("mode", pd.Series(dtype=str)) == "certified_safe_fallback").sum()
        ),
        "total_cost_usd": float(numeric("total_cost_usd").sum()),
        "mean_daily_cost_usd": float(
            daily["total_cost_usd"].sum().mean()
        ) if daily is not None else None,
        "mean_daily_weighted_particle_expected_cost_usd": float(
            daily["day_weighted_particle_expected_cost_usd"].first().mean()
        ) if daily is not None and "day_weighted_particle_expected_cost_usd" in frame else None,
        "mean_daily_weighted_particle_cvar_95_usd": float(
            daily["day_weighted_particle_cvar_95_usd"].first().mean()
        ) if daily is not None and "day_weighted_particle_cvar_95_usd" in frame else None,
        "sla_shortfall_work_units": float(
            numeric("sla_shortfall_work_units").sum()
        ),
        "policy_inference_mean_ms": float(
            1000.0 * numeric("policy_inference_seconds").mean()
        ),
        "policy_inference_p95_ms": float(
            1000.0 * numeric("policy_inference_seconds").quantile(0.95)
        ),
        "constraint_shield_mean_ms": float(
            1000.0 * numeric("shield_seconds").mean()
        ),
        "max_energy_balance_residual_mw": float(
            numeric("energy_balance_max_abs_mw").abs().max()
        ),
        "max_soc_transition_residual_mwh": float(
            numeric("soc_transition_max_abs_mwh").abs().max()
        ),
        "max_sla_balance_residual_work_units": float(
            numeric("sla_balance_max_abs_work_units").abs().max()
        ),
        "max_simultaneous_charge_discharge_mw": float(
            numeric("battery_simultaneous_charge_discharge_max_mw").abs().max()
        ),
        "max_migration_violation_work_units": float(
            numeric("migration_max_violation_work_units").abs().max()
        ),
    }
    return {
        "status": status,
        "phase": "selection_evaluation",
        "split": "selection",
        "month": month,
        "candidate_day_count": len(candidates),
        "completed_day_count": len(completed_requested),
        "completed_origin_count": len(complete_rows),
        "resumed_existing_run": resumed_existing_run,
        "reason": reason,
        "model_checkpoint": None if checkpoint is None else str(checkpoint),
        "external_solver": False,
        "runtime_backend": "neural_constraint_shield",
        "audit_summary": audit_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "device_start": start_device,
        "device_end": _device_snapshot(),
        "locked_test_year_accessed": False,
        "selection_years": [2022, 2023, 2024],
        "resume_command": resume_command,
    }


@app.command()
def preflight(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume: bool = typer.Option(True),
) -> None:
    """Validate solver-free imports, data seams, shapes and the 2025 gate."""

    cfg = load_yaml(config)
    _assert_solver_free_config(cfg)
    root = _root_from_config(config)
    run = _run_dir(root, cfg, "preflight", "selection", resume=resume)
    _initialize_run(run, config)
    bundle_root, settlement_root, physical_path = _paths(root, cfg)
    pilot_date = f"{cfg['data']['pilot_month']}-01"
    checks: dict[str, Any] = {
        "runtime_import": True,
        "external_solver": False,
        "intervals": INTERVALS,
        "scenarios": SCENARIOS,
        "locked_test_year_accessed": False,
        "device": _device_snapshot(),
    }
    status = "COMPLETE"
    reason: str | None = None
    try:
        physical = load_frozen_physical(physical_path)
        day = load_selection_day(
            bundle_root / f"{pilot_date}.npz",
            realized_settlement_path=settlement_root / f"{pilot_date}.npz",
        )
        checks.update(
            physical_sites=len(physical.zones),
            day_valid=True,
            settlement_present=day.realized_rt_price is not None,
        )
    except (OSError, ValueError, KeyError) as exc:
        status = "BLOCKED"
        reason = f"{type(exc).__name__}: {exc}"
    state = {"status": status, "phase": "preflight", "reason": reason, **checks}
    _write_state(run, state)
    _write_metrics(
        run, [{key: value for key, value in state.items() if not isinstance(value, dict)}]
    )
    _write_summary(run, "R3 solver-free preflight", state)
    typer.echo(json.dumps(state, indent=2, default=str))


@app.command()
def train(
    phase: str = typer.Option(...),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume: bool = typer.Option(True),
) -> None:
    """Run one resumable solver-free neural phase."""

    phase = phase.lower().strip()
    if phase not in PHASES:
        raise typer.BadParameter(f"phase must be one of {PHASES}")
    cfg = load_yaml(config)
    _assert_solver_free_config(cfg)
    root = _root_from_config(config)
    if phase == "selection_evaluation":
        selected_month = str(cfg["data"]["pilot_month"])
        run = _scoped_run_dir(
            root,
            cfg,
            phase,
            "selection",
            resume=resume,
            scope={
                "phase": "selection_evaluation",
                "split": "selection",
                "month": selected_month,
            },
        )
        _initialize_run(run, config)
        model, checkpoint, blocked = _load_a2_model(root, cfg)
        state = _evaluate_impl(
            cfg=cfg,
            root=root,
            run=run,
            month=selected_month,
            resume=resume,
            model=None if blocked is not None else model,
            checkpoint=checkpoint,
        )
        if blocked is not None:
            state["status"] = "BLOCKED"
            state["reason"] = blocked
        _write_state(run, state)
        if not (run / "metrics.parquet").exists():
            _write_metrics(run, [])
        _write_summary(run, "R3 solver-free selection evaluation", state)
        typer.echo(json.dumps(state, indent=2, default=str))
        return
    run = _run_dir(root, cfg, phase, "selection", resume=resume)
    _initialize_run(run, config)
    existing = _read_json(run / "state.json")
    if existing is not None and existing.get("status") == "COMPLETE":
        typer.echo(json.dumps(existing, indent=2, default=str))
        return
    from .trainer import run_training_phase

    state = run_training_phase(
        phase=phase,
        config=cfg,
        repository_root=root,
        run_dir=run,
        resume=resume,
    )
    typer.echo(json.dumps(state, indent=2, default=str))


@app.command("rolling-pilot")
def rolling_pilot(
    date: str = typer.Option(...),
    origins: int = typer.Option(INTERVALS, min=1, max=INTERVALS),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume: bool = typer.Option(True),
) -> None:
    """Run one formal day through 96 causal policy/shield origins."""

    if date.startswith("2025"):
        raise typer.BadParameter("2025 remains closed")
    cfg = load_yaml(config)
    _assert_solver_free_config(cfg)
    root = _root_from_config(config)
    run = _scoped_run_dir(
        root,
        cfg,
        "rolling_pilot",
        "selection",
        resume=resume,
        scope={
            "phase": "rolling_pilot",
            "operating_date": date,
            "requested_origins": origins,
        },
    )
    _initialize_run(run, config)
    existing = _read_json(run / "state.json")
    if (
        resume
        and existing is not None
        and existing.get("status") == "COMPLETE"
        and existing.get("operating_date") == date
        and int(existing.get("completed_origins", 0)) == origins
    ):
        typer.echo(json.dumps(existing, indent=2, default=str))
        return
    model, checkpoint, blocked = _load_a2_model(root, cfg)
    result = (
        _blocked_day(date, str(blocked))
        if blocked is not None
        else _run_day(
            cfg,
            root,
            run,
            date,
            origins=origins,
            formal=True,
            model=model,
        )
    )
    state = {key: value for key, value in result.items() if key != "records"}
    state.update(
        phase="rolling_pilot",
        requested_origins=origins,
        model_checkpoint=None if checkpoint is None else str(checkpoint),
        external_solver=False,
        locked_test_year_accessed=False,
    )
    _write_state(run, state)
    _write_metrics(run, records_frame(result))
    _write_summary(run, "R3 solver-free rolling pilot", state)
    typer.echo(json.dumps(state, indent=2, default=str))


@app.command()
def evaluate(
    split: str = typer.Option("selection"),
    month: Optional[str] = typer.Option(None),
    full_rolling: bool = typer.Option(True, "--full-rolling/--no-full-rolling"),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume: bool = typer.Option(True),
) -> None:
    """Run a resumable month or the full 2022--2024 selection period."""

    if split != "selection":
        raise typer.BadParameter("only the 2022-2024 selection split is open")
    if not full_rolling:
        raise typer.BadParameter("formal evaluation requires all 96 origins")
    cfg = load_yaml(config)
    _assert_solver_free_config(cfg)
    root = _root_from_config(config)
    run = _scoped_run_dir(
        root,
        cfg,
        "selection_evaluation",
        split,
        resume=resume,
        scope={
            "phase": "selection_evaluation",
            "split": split,
            "month": month,
        },
    )
    _initialize_run(run, config)
    existing = _read_json(run / "state.json")
    if (
        resume
        and existing is not None
        and existing.get("status") == "COMPLETE"
        and existing.get("month") == month
    ):
        typer.echo(json.dumps(existing, indent=2, default=str))
        return
    model, checkpoint, blocked = _load_a2_model(root, cfg)
    state = _evaluate_impl(
        cfg=cfg,
        root=root,
        run=run,
        month=month,
        resume=resume,
        model=None if blocked is not None else model,
        checkpoint=checkpoint,
    )
    if blocked is not None:
        state["status"] = "BLOCKED"
        state["reason"] = blocked
    _write_state(run, state)
    if not (run / "metrics.parquet").exists():
        _write_metrics(run, [])
    _write_summary(run, "R3 solver-free selection evaluation", state)
    typer.echo(json.dumps(state, indent=2, default=str))


def _invoke_train_phase(phase: str, config: Path, resume: bool) -> dict[str, Any]:
    cfg = load_yaml(config)
    root = _root_from_config(config)
    # A lineage-complete phase is immutable and reusable even when a newer
    # failed/partial attempt exists.  This keeps run-all from resuming the wrong
    # directory after a transient failure.
    checkpoint, completed_state = _latest_checkpoint(root, cfg, phase=phase)
    if resume and checkpoint is not None and completed_state is not None:
        return completed_state
    run = _run_dir(root, cfg, phase, "selection", resume=resume)
    _initialize_run(run, config)
    existing = _read_json(run / "state.json")
    if existing is not None and existing.get("status") == "COMPLETE":
        return existing
    from .trainer import run_training_phase

    return run_training_phase(
        phase=phase,
        config=cfg,
        repository_root=root,
        run_dir=run,
        resume=resume,
    )


@app.command("run-all")
def run_all(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume: bool = typer.Option(True),
    pilot_date: Optional[str] = typer.Option(None),
    month: Optional[str] = typer.Option(None),
) -> None:
    """Run training, one day, then the requested one-month pilot in order."""

    cfg = load_yaml(config)
    _assert_solver_free_config(cfg)
    root = _root_from_config(config)
    date = pilot_date or f"{cfg['data']['pilot_month']}-01"
    requested_month = month or str(cfg["data"]["pilot_month"])
    if date.startswith("2025") or requested_month.startswith("2025"):
        raise typer.BadParameter("2025 remains closed")
    phases: dict[str, Any] = {}
    for phase in TRAINING_PHASES:
        state = _invoke_train_phase(phase, config, resume)
        phases[phase] = state.get("status")
        if state.get("status") != "COMPLETE":
            output = {
                "status": "PARTIAL",
                "stopped_after": phase,
                "phase_statuses": phases,
                "reason": state.get("reason", state.get("blocked_reason")),
            }
            typer.echo(json.dumps(output, indent=2, default=str))
            return
    model, checkpoint, blocked = _load_a2_model(root, cfg)
    pilot_run = _scoped_run_dir(
        root,
        cfg,
        "rolling_pilot",
        "selection",
        resume=resume,
        scope={
            "phase": "rolling_pilot",
            "operating_date": date,
            "requested_origins": INTERVALS,
        },
    )
    _initialize_run(pilot_run, config)
    prior_pilot = _read_json(pilot_run / "state.json")
    reuse_pilot = bool(
        resume
        and prior_pilot is not None
        and prior_pilot.get("status") == "COMPLETE"
        and prior_pilot.get("operating_date") == date
        and int(prior_pilot.get("requested_origins", 0)) == INTERVALS
    )
    if reuse_pilot:
        day_result = {**prior_pilot, "records": []}
    else:
        day_result = (
            _blocked_day(date, str(blocked))
            if blocked is not None
            else _run_day(
                cfg,
                root,
                pilot_run,
                date,
                origins=INTERVALS,
                formal=True,
                model=model,
            )
        )
    day_state = {key: value for key, value in day_result.items() if key != "records"}
    day_state.update(
        phase="rolling_pilot",
        requested_origins=INTERVALS,
        model_checkpoint=None if checkpoint is None else str(checkpoint),
        locked_test_year_accessed=False,
    )
    if not reuse_pilot:
        _write_state(pilot_run, day_state)
        _write_metrics(pilot_run, records_frame(day_result))
        _write_summary(pilot_run, "R3 solver-free rolling pilot", day_state)
    if day_result.get("status") != "COMPLETE":
        typer.echo(
            json.dumps(
                {
                    "status": "PARTIAL",
                    "stopped_after": "rolling_pilot",
                    "phase_statuses": phases,
                    "rolling_status": day_result.get("status"),
                    "reason": day_result.get("reason"),
                },
                indent=2,
                default=str,
            )
        )
        return
    evaluation_run = _scoped_run_dir(
        root,
        cfg,
        "selection_evaluation",
        "selection",
        resume=resume,
        scope={
            "phase": "selection_evaluation",
            "split": "selection",
            "month": requested_month,
        },
    )
    _initialize_run(evaluation_run, config)
    evaluation = _evaluate_impl(
        cfg=cfg,
        root=root,
        run=evaluation_run,
        month=requested_month,
        resume=resume,
        model=model,
        checkpoint=checkpoint,
    )
    _write_state(evaluation_run, evaluation)
    if not (evaluation_run / "metrics.parquet").exists():
        _write_metrics(evaluation_run, [])
    _write_summary(
        evaluation_run, "R3 solver-free selection evaluation", evaluation
    )
    typer.echo(
        json.dumps(
            {
                "status": evaluation["status"],
                "phase_statuses": phases,
                "rolling_status": day_result["status"],
                "month_status": evaluation["status"],
                "month": requested_month,
                "resume_command": evaluation["resume_command"],
            },
            indent=2,
            default=str,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
