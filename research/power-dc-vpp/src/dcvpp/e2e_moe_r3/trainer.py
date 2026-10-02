"""Resumable solver-free training for the decision-native E2E-MoE R3 model.

The trainer consumes the materialized 2022--2024 issue-time bundles and the
frozen physical twin.  Policy learning uses differentiable settlement, state
transition, risk, SLA, and constraint-intervention objectives.  The deployment
path is a neural rolling policy followed by a deterministic constraint shield;
no mathematical-programming solver or solver-generated teacher participates in
training, inference, repair, or evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dcvpp.core import atomic_json, atomic_text, sha256_file, sha256_json, utc_now
from .constrained_runtime import FrozenPhysicalArtifact, NYISO_ZONES

from .losses import (
    cvar_loss,
    energy_score_loss,
    marginal_scenario_loss,
    migration_cost_loss,
    routing_balance_loss,
    routing_stability_loss,
    scenario_diversity_loss,
    scenario_weight_entropy_loss,
    tail_coverage_loss,
    variogram_score_loss,
)


PHASES = (
    "representation",
    "scenario",
    "policy",
    "a1",
    "a2",
    "selection_evaluation",
)
ALLOWED_YEARS = frozenset({2022, 2023, 2024})
INTERVALS = 96
EMPIRICAL_SCENARIOS = 100
LEARNED_SCENARIOS = 20
PHASE_PREDECESSOR: dict[str, str] = {
    "scenario": "representation",
    "policy": "scenario",
    "a1": "policy",
    "a2": "a1",
    "selection_evaluation": "a2",
}


def _checkpoint_lineage_complete(
    state: Mapping[str, Any],
    phase: str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> bool:
    """Verify the entire predecessor chain, including formal seed identity."""

    if expected is not None:
        for key in (
            "training_regime",
            "experiment_id",
            "method",
            "seed",
            "protocol_fingerprint",
        ):
            if expected.get(key) is not None and state.get(key) != expected.get(key):
                return False
    parent = PHASE_PREDECESSOR.get(phase)
    if parent is None:
        return True
    predecessor = state.get("predecessor", {})
    if not isinstance(predecessor, Mapping) or predecessor.get("phase") != parent:
        return False
    run_dir = predecessor.get("run_dir")
    if not run_dir:
        return False
    state_path = Path(str(run_dir)) / "state.json"
    try:
        parent_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        parent_state.get("status") == "COMPLETE"
        and _checkpoint_lineage_complete(parent_state, parent, expected=expected)
    )

# These are the exact scientific scopes from the R3 contract.  Alias handling
# below accommodates harmless naming differences, but never broadens A1/A2.
A1_TRAINABLE_GROUPS = frozenset(
    {"dart_head", "dart_scenario_adapter", "dart_calibration"}
)
A2_TRAINABLE_GROUPS = frozenset(
    {
        "market_encoder",
        "workload_encoder",
        "physical_residual_adapter",
        "bounded_residual_adapter",
        "uncertainty_adapter",
        "moe_fusion",
        "moe_router",
        "experts",
        "scenario_head",
        "risk_head",
        "policy",
        "feasibility_head",
        "value_head",
    }
)

_PHASE_TRAINABLE_GROUPS: dict[str, frozenset[str]] = {
    "representation": frozenset(
        {
            "market_encoder",
            "workload_encoder",
            "physical_encoder",
            "state_encoder",
            "physical_residual_adapter",
            "forecast_heads",
            "da_head",
            "rt_head",
            "dart_head",
            "workload_head",
            "pv_head",
            "pue_head",
        }
    ),
    "scenario": frozenset(
        {"moe_fusion", "moe_router", "experts", "scenario_head", "scenario_adapter", "risk_head"}
    ),
    "policy": frozenset(
        {"risk_head", "policy", "policy_head", "feasibility_head", "value_head"}
    ),
    "a1": A1_TRAINABLE_GROUPS,
    "a2": A2_TRAINABLE_GROUPS,
    "selection_evaluation": frozenset(),
}

_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "forecast_heads": ("forecast_heads", "forecast_head", "forecasters"),
    "scenario_head": ("scenario_head", "joint_scenario_head", "scenarios"),
    "scenario_adapter": ("scenario_adapter",),
    "dart_scenario_adapter": ("dart_scenario_adapter", "dart_adapter"),
    "dart_calibration": ("dart_calibration", "dart_calibrator"),
    "moe_router": ("moe_router", "router"),
    "physical_residual_adapter": (
        "physical_residual_adapter",
        "bounded_residual_adapter",
    ),
    "bounded_residual_adapter": (
        "bounded_residual_adapter",
        "physical_residual_adapter",
    ),
    "experts": ("experts", "moe_experts"),
    "risk_head": ("risk_head", "decision_risk_head"),
    "policy": ("policy", "policy_head", "rolling_policy"),
    "policy_head": ("policy_head", "policy", "rolling_policy"),
    "feasibility_head": ("feasibility_head", "fallback_head"),
}

REUSE_ANNOTATIONS = {
    "daily_bundle_loader": (
        "dcvpp.experiments.dart_counterfactual_input_materialization."
        "materialize_selection_counterfactual_inputs"
    ),
    "daily_bundle_validation": "dcvpp.e2e_moe_r3.constrained_runtime.load_selection_day",
    "dart_sign": "DA minus RT, identical to the existing settlement contract",
    "physical_contract": "dcvpp.e2e_moe_r3.constrained_runtime.FrozenPhysicalArtifact",
    "settlement_boundary": "dcvpp.optimization.replay_engine.settle_realized_components",
    "audited_runtime": "dcvpp.e2e_moe_r3.constrained_runtime.NeuralConstrainedRuntime",
}


class TrainingContractError(RuntimeError):
    """Raised when phase, data, scope, or gradient evidence violates R3."""


def _canonical_dart(da_lbmp: Any, rt_lbmp: Any) -> np.ndarray:
    """Canonical DART sign without importing an optimization package."""

    da = np.asarray(da_lbmp, dtype=float)
    rt = np.asarray(rt_lbmp, dtype=float)
    if da.shape != rt.shape:
        raise ValueError("DA and RT arrays must have identical shapes")
    return da - rt


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value


def _nested_get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _parse_year(date_text: str) -> int:
    try:
        year = datetime.fromisoformat(str(date_text)[:10]).year
    except ValueError as exc:
        raise TrainingContractError(f"invalid operating_date: {date_text!r}") from exc
    if year not in ALLOWED_YEARS:
        raise TrainingContractError(
            f"R3 training is sealed to 2022-2024; rejected {date_text!r}"
        )
    return year


def _reject_locked_path(path: Path) -> None:
    for part in path.parts:
        lowered = part.lower()
        if lowered == "2025" or lowered.startswith("2025-") or "locked_2025" in lowered:
            raise TrainingContractError(f"refusing to access locked 2025 path: {path}")


def discover_daily_bundles(
    root: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    max_days: int | None = None,
) -> list[Path]:
    """Return a deterministic list of selection NPZs, rejecting 2025 by path.

    Only filenames that are ISO operating dates are accepted.  The rejection
    occurs before any NPZ is opened, which makes the held-out boundary explicit.
    """

    root = Path(root).resolve()
    _reject_locked_path(root)
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(root.glob("*.npz"))
    else:
        raise FileNotFoundError(f"daily bundle seam does not exist: {root}")

    selected: list[Path] = []
    for path in candidates:
        _reject_locked_path(path)
        try:
            date_text = datetime.fromisoformat(path.stem[:10]).date().isoformat()
        except ValueError as exc:
            raise TrainingContractError(
                f"daily bundle filename must start with YYYY-MM-DD: {path.name}"
            ) from exc
        _parse_year(date_text)
        if start_date is not None and date_text < str(start_date)[:10]:
            continue
        if end_date is not None and date_text > str(end_date)[:10]:
            continue
        selected.append(path)

    if max_days is not None:
        if int(max_days) < 1:
            raise ValueError("max_days must be positive when supplied")
        selected = selected[: int(max_days)]
    if not selected:
        raise TrainingContractError("no 2022-2024 daily bundles matched the request")
    return selected


def select_training_bundles(
    paths: Sequence[Path],
    *,
    max_days: int | None,
    decision_stratified: bool,
) -> tuple[list[Path], dict[str, str]]:
    """Select a deterministic cross-year pilot subset and expose its regime.

    Representation/scenario phases use evenly spaced dates.  Decision phases
    score an evenly spaced 96-day candidate panel using issue-time ensembles
    only, then retain distinct normal, spike, negative-price, high-PV,
    workload-burst, tight-SLA and volatile days.  No realized settlement value
    enters this selection, and empirical migration is not relabelled as a
    congestion regime because the selection contract fixes it to zero.
    """

    ordered = sorted(Path(path).resolve() for path in paths)
    if max_days is None or int(max_days) >= len(ordered):
        return ordered, {path.stem[:10]: "all_available" for path in ordered}
    count = int(max_days)
    if count < 1:
        raise TrainingContractError("training max_days must be positive")
    if not decision_stratified:
        indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        selected = [ordered[int(index)] for index in indices]
        return selected, {
            path.stem[:10]: f"temporal_stratum_{position + 1}"
            for position, path in enumerate(selected)
        }

    panel_count = min(len(ordered), max(96, count * 8))
    panel_indices = np.linspace(0, len(ordered) - 1, panel_count).round().astype(int)
    panel = [ordered[int(index)] for index in panel_indices]
    rows: list[dict[str, Any]] = []
    for path in panel:
        _reject_locked_path(path)
        with np.load(path, allow_pickle=False) as bundle:
            if bool(bundle["locked_test"].item()):
                raise TrainingContractError(f"locked_test=true in {path}")
            rt = np.asarray(bundle["rt_price_pool"], dtype=float)
            pv = np.asarray(bundle["pv_pool"], dtype=float)
            workload = np.asarray(bundle["workload_pool"], dtype=float)
        if not (np.isfinite(rt).all() and np.isfinite(pv).all() and np.isfinite(workload).all()):
            raise TrainingContractError(f"non-finite regime panel values in {path}")
        workload_q95 = float(np.quantile(workload, 0.95))
        rows.append(
            {
                "path": path,
                "rt_high": float(np.quantile(rt, 0.995)),
                "rt_low": float(np.quantile(rt, 0.005)),
                "rt_volatility": float(np.std(rt)),
                "pv_high": float(np.quantile(pv, 0.99)),
                "workload_high": workload_q95,
                "tight_sla": workload_q95
                + float(np.quantile(np.abs(np.diff(workload, axis=-1)), 0.99)),
            }
        )
    frame = pd.DataFrame(rows)
    chosen: list[tuple[Path, str]] = []

    def add(index: int, label: str) -> None:
        path = Path(frame.loc[index, "path"])
        if all(existing != path for existing, _ in chosen):
            chosen.append((path, label))

    add(int(frame["rt_high"].idxmax()), "price_spike")
    add(int(frame["rt_low"].idxmin()), "negative_price")
    add(int(frame["pv_high"].idxmax()), "high_pv")
    add(int(frame["workload_high"].idxmax()), "workload_burst")
    add(int(frame["tight_sla"].idxmax()), "tight_sla")
    add(int(frame["rt_volatility"].idxmax()), "price_volatility")
    normalized = frame[["rt_high", "rt_low", "rt_volatility", "pv_high", "workload_high"]]
    centered = (normalized - normalized.median()) / normalized.std(ddof=0).replace(0.0, 1.0)
    add(int(centered.abs().sum(axis=1).idxmin()), "normal")
    while len(chosen) < count:
        chosen_dates = [pd.Timestamp(path.stem[:10]) for path, _ in chosen]
        candidates: list[tuple[int, int]] = []
        for position, path in enumerate(panel):
            if any(existing == path for existing, _ in chosen):
                continue
            date = pd.Timestamp(path.stem[:10])
            distance = min(abs((date - anchor).days) for anchor in chosen_dates)
            candidates.append((int(distance), int(position)))
        if not candidates:
            break
        _, farthest_position = max(candidates)
        add(farthest_position, "temporal_diversity")
    chosen = chosen[:count]
    selected = sorted(path for path, _ in chosen)
    labels = {path.stem[:10]: label for path, label in chosen}
    return selected, labels


def load_frozen_physical_artifact(
    path: str | Path,
) -> tuple[FrozenPhysicalArtifact, dict[str, Any]]:
    """Load the immutable physical JSON through the existing research contract."""

    path = Path(path).resolve()
    _reject_locked_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # The canonical file contains ``frozen: true``.  Wrapping avoids the legacy
    # loader treating that boolean itself as the artifact mapping.
    physical = FrozenPhysicalArtifact.from_mapping({"frozen_artifact": payload})
    return physical, {
        "path": str(path),
        "sha256": sha256_file(path),
        "calibration_id": physical.calibration_id,
        "twin_ids": list(physical.twin_ids),
        "zones": list(physical.zones),
        "frozen": bool(physical.frozen),
        "reused_symbol": REUSE_ANNOTATIONS["physical_contract"],
    }


def _torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location=map_location)

class R3DailyBundleDataset(Dataset[dict[str, Any]]):
    """PyTorch view of the existing 2022-2024 daily NPZ seam.

    Input convention:
      * market ``[T,4]`` = DA mean, RT mean, canonical DART mean, RT dispersion
      * workload ``[T,2]`` = ensemble mean and standard deviation
      * physical ``[T,4]`` = transferred physical multiplier by zone
      * state ``[T,4]`` = initial SOC divided by frozen battery capacity

    The scenario target has six scalar joint variables in the canonical order
    DA, RT, DART, workload, PV, PUE.  Zonal price/PV targets remain separately
    available under ``targets`` for zonal forecast heads.
    """

    scenario_variables = ("da", "rt", "dart", "workload", "pv", "pue")

    def __init__(
        self,
        paths: Sequence[str | Path],
        physical: FrozenPhysicalArtifact,
        *,
        realized_settlement_root: str | Path | None = None,
    ) -> None:
        if not paths:
            raise ValueError("paths must not be empty")
        self.paths = tuple(Path(path).resolve() for path in paths)
        self.physical_artifact = physical
        self.realized_settlement_root = (
            None
            if realized_settlement_root is None
            else Path(realized_settlement_root).resolve()
        )
        if self.realized_settlement_root is not None:
            _reject_locked_path(self.realized_settlement_root)
        self.battery_energy = np.maximum(
            np.asarray(physical.battery_energy_mwh, dtype=np.float32), 1e-6
        )
        self._memory_cache: tuple[dict[str, Any], ...] | None = None
        if tuple(physical.zones) != tuple(NYISO_ZONES):
            raise TrainingContractError("frozen physical zones do not match NYISO_ZONES")
        for path in self.paths:
            _reject_locked_path(path)

    def __len__(self) -> int:
        return len(self.paths)

    def preload(self) -> "R3DailyBundleDataset":
        """Decode the fixed formal panel once instead of once per epoch.

        The cached samples stay on host memory and remain immutable by
        convention.  This removes repeated NPZ decompression and settlement
        reads from the critical GPU training loop without changing sample
        values, order, or sealed-year gates.
        """

        if self._memory_cache is None:
            self._memory_cache = tuple(self._load_item(index) for index in range(len(self.paths)))
        return self

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._memory_cache is not None:
            return self._memory_cache[index]
        return self._load_item(index)

    def _load_item(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        _reject_locked_path(path)
        with np.load(path, allow_pickle=False) as bundle:
            date_text = str(bundle["operating_date"].item())[:10]
            _parse_year(date_text)
            if bool(bundle["locked_test"].item()):
                raise TrainingContractError(f"locked_test=true in {path}")
            zones = tuple(str(value) for value in np.asarray(bundle["zones"]).tolist())
            if zones != tuple(NYISO_ZONES):
                raise TrainingContractError(f"unexpected zone order in {path}: {zones}")

            da_z_t = np.asarray(bundle["da_price"], dtype=np.float32)
            rt_s_z_t = np.asarray(bundle["rt_price_pool"], dtype=np.float32)
            workload_s_t = np.asarray(bundle["workload_pool"], dtype=np.float32)
            pv_s_z_t = np.asarray(bundle["pv_pool"], dtype=np.float32)
            workload_t = np.asarray(bundle["workload"], dtype=np.float32)
            pv_z_t = np.asarray(bundle["pv"], dtype=np.float32)
            multiplier_z = np.asarray(bundle["physical_multiplier"], dtype=np.float32)
            soc_z = np.asarray(bundle["initial_soc_mwh_by_zone"], dtype=np.float32)
            timestamps = np.asarray(bundle["timestamps_utc"])

        realized_da_z_t = da_z_t
        realized_rt_z_t: np.ndarray | None = None
        if self.realized_settlement_root is not None:
            settlement_path = self.realized_settlement_root / f"{date_text}.npz"
            _reject_locked_path(settlement_path)
            if not settlement_path.exists():
                raise TrainingContractError(
                    f"official settlement day is missing: {settlement_path}"
                )
            with np.load(settlement_path, allow_pickle=False) as settlement:
                settlement_timestamps = np.asarray(settlement["timestamps_utc"])
                if not np.array_equal(timestamps, settlement_timestamps):
                    raise TrainingContractError(
                        f"settlement timestamps do not match {path.name}"
                    )
                realized_da_z_t = np.asarray(
                    settlement["da_lbmp"], dtype=np.float32
                )
                realized_rt_z_t = np.asarray(
                    settlement["rt_lbmp"], dtype=np.float32
                )
            if realized_da_z_t.shape != (4, INTERVALS) or realized_rt_z_t.shape != (
                4,
                INTERVALS,
            ):
                raise TrainingContractError(
                    f"official settlement arrays must be [4,96] for {date_text}"
                )
            if not (
                np.isfinite(realized_da_z_t).all()
                and np.isfinite(realized_rt_z_t).all()
            ):
                raise TrainingContractError(
                    f"official settlement contains non-finite values for {date_text}"
                )

        expected = {
            "da_price": (4, INTERVALS),
            "rt_price_pool": (EMPIRICAL_SCENARIOS, 4, INTERVALS),
            "workload_pool": (EMPIRICAL_SCENARIOS, INTERVALS),
            "pv_pool": (EMPIRICAL_SCENARIOS, 4, INTERVALS),
            "workload": (INTERVALS,),
            "pv": (4, INTERVALS),
            "physical_multiplier": (4,),
            "initial_soc": (4,),
        }
        observed = {
            "da_price": da_z_t.shape,
            "rt_price_pool": rt_s_z_t.shape,
            "workload_pool": workload_s_t.shape,
            "pv_pool": pv_s_z_t.shape,
            "workload": workload_t.shape,
            "pv": pv_z_t.shape,
            "physical_multiplier": multiplier_z.shape,
            "initial_soc": soc_z.shape,
        }
        for name, shape in expected.items():
            if observed[name] != shape:
                raise TrainingContractError(
                    f"{path.name}:{name} has shape {observed[name]}, expected {shape}"
                )
        arrays = (da_z_t, rt_s_z_t, workload_s_t, pv_s_z_t, workload_t, pv_z_t,
                  multiplier_z, soc_z)
        if not all(np.isfinite(array).all() for array in arrays):
            raise TrainingContractError(f"non-finite daily bundle values in {path}")

        da_t_z = da_z_t.T
        issue_rt_t_z = rt_s_z_t.mean(axis=0).T
        issue_dart_t_z = _canonical_dart(da_t_z, issue_rt_t_z).astype(np.float32)
        rt_dispersion_t = rt_s_z_t.std(axis=(0, 1)).astype(np.float32)
        market = np.stack(
            (
                da_t_z.mean(axis=1),
                issue_rt_t_z.mean(axis=1),
                issue_dart_t_z.mean(axis=1),
                rt_dispersion_t,
            ),
            axis=-1,
        ).astype(np.float32)
        workload_features = np.stack(
            (workload_s_t.mean(axis=0), workload_s_t.std(axis=0)), axis=-1
        ).astype(np.float32)
        physical_input = np.broadcast_to(multiplier_z, (INTERVALS, 4)).copy()
        state = np.broadcast_to(soc_z / self.battery_energy, (INTERVALS, 4)).copy()
        pue_t_z = physical_input.copy()

        # Existing bundles contain 100 coherent joint rows.  Aggregate only the
        # zone dimension so the new head's six named variables remain coherent.
        da_s_t = np.broadcast_to(
            da_t_z.mean(axis=-1)[None, :], (EMPIRICAL_SCENARIOS, INTERVALS)
        )
        rt_s_t = rt_s_z_t.mean(axis=1)
        dart_s_t = _canonical_dart(da_s_t, rt_s_t).astype(np.float32)
        pv_s_t = pv_s_z_t.mean(axis=1)
        pue_s_t = np.broadcast_to(
            multiplier_z.mean(dtype=np.float32),
            (EMPIRICAL_SCENARIOS, INTERVALS),
        )
        scenario_target = np.stack(
            (da_s_t, rt_s_t, dart_s_t, workload_s_t, pv_s_t, pue_s_t), axis=-1
        ).astype(np.float32)

        target_da_t_z = realized_da_z_t.T
        target_rt_t_z = (
            issue_rt_t_z if realized_rt_z_t is None else realized_rt_z_t.T
        )
        target_dart_t_z = _canonical_dart(
            target_da_t_z, target_rt_t_z
        ).astype(np.float32)
        targets = {
            "da": torch.from_numpy(target_da_t_z.copy()),
            "rt": torch.from_numpy(target_rt_t_z.copy()),
            "dart": torch.from_numpy(target_dart_t_z.copy()),
            "workload": torch.from_numpy(workload_t[:, None].copy()),
            "pv": torch.from_numpy(pv_z_t.T.copy()),
            # The transferred multiplier is the physical/PUE proxy already
            # frozen upstream; it is not recalibrated here.
            "pue": torch.from_numpy(pue_t_z.copy()),
        }
        return {
            "market": torch.from_numpy(market),
            "workload": torch.from_numpy(workload_features),
            "physical": torch.from_numpy(physical_input),
            "state": torch.from_numpy(state),
            "targets": targets,
            "scenario_target": torch.from_numpy(scenario_target),
            "decision": {
                "da_price": torch.from_numpy(realized_da_z_t.copy()),
                # Selection forcing is revealed sequentially in deployment;
                # it is a realized training target, never a model input.
                "rt_price": torch.from_numpy(target_rt_t_z.T.copy()),
                "workload": torch.from_numpy(workload_t.copy()),
                "pv": torch.from_numpy(pv_z_t.copy()),
                "physical_multiplier": torch.from_numpy(multiplier_z.copy()),
                "initial_soc": torch.from_numpy(soc_z.copy()),
                "idle_mw": torch.from_numpy(
                    np.asarray(self.physical_artifact.idle_mw, dtype=np.float32)
                ),
                "work_mw_per_unit": torch.from_numpy(
                    np.asarray(
                        self.physical_artifact.work_mw_per_unit,
                        dtype=np.float32,
                    )
                ),
                "capacity_work_units": torch.from_numpy(
                    np.asarray(
                        self.physical_artifact.capacity_work_units,
                        dtype=np.float32,
                    )
                ),
                "grid_limit_mw": torch.from_numpy(
                    np.asarray(
                        self.physical_artifact.grid_import_limit_mw,
                        dtype=np.float32,
                    )
                ),
                "battery_power_mw": torch.from_numpy(
                    np.asarray(
                        self.physical_artifact.battery_power_mw,
                        dtype=np.float32,
                    )
                ),
                "battery_energy_mwh": torch.from_numpy(
                    np.asarray(
                        self.physical_artifact.battery_energy_mwh,
                        dtype=np.float32,
                    )
                ),
            },
            "operating_date": date_text,
            "bundle_path": str(path),
        }


def collect_device_metrics(device: str | torch.device | None = None) -> dict[str, Any]:
    """Return lightweight CPU/RAM/GPU process telemetry without a background thread."""

    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metrics: dict[str, Any] = {
        "monitor_timestamp_utc": utc_now(),
        "device": str(resolved),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()
        metrics.update(
            {
                "process_rss_bytes": int(process.memory_info().rss),
                "system_memory_total_bytes": int(memory.total),
                "system_memory_available_bytes": int(memory.available),
                "system_memory_percent": float(memory.percent),
            }
        )
    except (ImportError, OSError):
        metrics["process_rss_bytes"] = None

    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        metrics.update(
            {
                "gpu_index": int(index),
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "gpu_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "gpu_max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            }
        )
    return metrics


def _parameters_from_group(value: Any) -> list[nn.Parameter]:
    if isinstance(value, nn.Parameter):
        return [value]
    if isinstance(value, nn.Module):
        return list(value.parameters())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        parameters: list[nn.Parameter] = []
        for item in value:
            parameters.extend(_parameters_from_group(item))
        return parameters
    raise TrainingContractError(
        "parameter_groups() values must be Parameters, Modules, or iterables thereof"
    )


def _resolve_allowed_names(
    requested: Iterable[str], available: Iterable[str]
) -> tuple[set[str], set[str]]:
    available_set = set(available)
    resolved: set[str] = set()
    missing: set[str] = set()
    for canonical in requested:
        candidates = _GROUP_ALIASES.get(canonical, (canonical,))
        hits = available_set.intersection(candidates)
        if hits:
            resolved.update(hits)
        else:
            missing.add(canonical)
    return resolved, missing


def configure_phase_parameters(model: nn.Module, phase: str) -> dict[str, Any]:
    """Apply and verify the exact trainable parameter scope for a phase."""

    if phase not in PHASES:
        raise TrainingContractError(f"unknown training phase {phase!r}; expected {PHASES}")
    if hasattr(model, "set_phase"):
        model.set_phase(phase)  # type: ignore[attr-defined]
    # The concrete R3 model exposes phase-level named groups so that no
    # parameter is duplicated between module collections.  Use that contract
    # directly when available, then independently verify the applied scope.
    if hasattr(model, "parameter_names_for_phase"):
        allowed_parameter_names = set(
            map(str, model.parameter_names_for_phase(phase))  # type: ignore[attr-defined]
        )
        if hasattr(model, "configure_trainable_phase"):
            configured = set(
                map(str, model.configure_trainable_phase(phase))  # type: ignore[attr-defined]
            )
            if configured != allowed_parameter_names:
                raise TrainingContractError(
                    "model phase methods disagree on their parameter allowlist"
                )
        else:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name in allowed_parameter_names)
        observed = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if observed != allowed_parameter_names:
            raise TrainingContractError(
                "model did not apply its exact phase allowlist: "
                f"added={sorted(observed - allowed_parameter_names)}, "
                f"removed={sorted(allowed_parameter_names - observed)}"
            )
        all_names = {name for name, _ in model.named_parameters()}
        unknown = allowed_parameter_names.difference(all_names)
        if unknown:
            raise TrainingContractError(f"phase allowlist names unknown parameters: {sorted(unknown)}")
        if hasattr(model, "named_parameter_groups"):
            named_groups = model.named_parameter_groups()  # type: ignore[attr-defined]
            if isinstance(named_groups, Mapping) and phase in named_groups:
                reported = set(map(str, named_groups[phase]))
                if reported != allowed_parameter_names:
                    raise TrainingContractError(
                        "named_parameter_groups() disagrees with parameter_names_for_phase()"
                    )
        a1_prefixes = ("dart_head.", "dart_scenario_adapter.", "dart_calibration.")
        a2_prefixes = (
            "market_encoder.",
            "workload_encoder.",
            "physical_residual_adapter.",
            "uncertainty_adapter.",
            "moe_fusion.",
            "moe.router.",
            "moe.experts.",
            "scenario_head.",
            "risk_head.",
            "policy.",
            "feasibility_head.",
            "value_head.",
        )
        if phase == "a1":
            forbidden = sorted(
                name for name in observed if not name.startswith(a1_prefixes)
            )
            if forbidden:
                raise TrainingContractError(f"A1 scope contains forbidden parameters: {forbidden}")
        if phase == "a2":
            forbidden = sorted(
                name for name in observed if not name.startswith(a2_prefixes)
            )
            if forbidden:
                raise TrainingContractError(f"A2 scope contains forbidden parameters: {forbidden}")
        if phase != "selection_evaluation" and not observed:
            raise TrainingContractError(f"phase {phase!r} resolved no trainable parameters")
        if phase == "selection_evaluation" and observed:
            raise TrainingContractError("selection_evaluation must freeze every parameter")
        return {
            "phase": phase,
            "scope_source": "model.parameter_names_for_phase/configure_trainable_phase",
            "requested_groups": [phase],
            "resolved_groups": [phase],
            "missing_requested_groups": [],
            "trainable_parameter_names": sorted(observed),
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "frozen_parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
            ),
        }
    if not hasattr(model, "parameter_groups"):
        raise TrainingContractError(
            "R3 model must expose parameter_groups() for auditable phase scopes"
        )
    raw_groups = model.parameter_groups()  # type: ignore[attr-defined]
    if not isinstance(raw_groups, Mapping):
        raise TrainingContractError("model.parameter_groups() must return a mapping")
    groups = {
        str(name): _parameters_from_group(value) for name, value in raw_groups.items()
    }
    requested = _PHASE_TRAINABLE_GROUPS[phase]
    allowed_names, missing = _resolve_allowed_names(requested, groups)

    owner_names: dict[int, set[str]] = {}
    known_parameters: dict[int, nn.Parameter] = {}
    for group_name, parameters in groups.items():
        for parameter in parameters:
            owner_names.setdefault(id(parameter), set()).add(group_name)
            known_parameters[id(parameter)] = parameter
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    unknown = model_parameter_ids.difference(known_parameters)
    if unknown:
        named = [name for name, parameter in model.named_parameters() if id(parameter) in unknown]
        raise TrainingContractError(
            f"parameter_groups() omits model parameters, so scope is unauditable: {named}"
        )

    trainable_ids: set[int] = set()
    for parameter_id, owners in owner_names.items():
        allowed_owners = owners.intersection(allowed_names)
        forbidden_owners = owners.difference(allowed_names)
        if allowed_owners and forbidden_owners:
            raise TrainingContractError(
                "a parameter belongs to both allowed and frozen groups: "
                f"allowed={sorted(allowed_owners)}, frozen={sorted(forbidden_owners)}"
            )
        if allowed_owners:
            trainable_ids.add(parameter_id)

    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in trainable_ids)

    observed_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if phase != "selection_evaluation" and not observed_trainable:
        raise TrainingContractError(
            f"phase {phase!r} resolved no trainable parameters; missing groups={sorted(missing)}"
        )
    if phase == "selection_evaluation" and observed_trainable:
        raise TrainingContractError("selection_evaluation must freeze every parameter")

    return {
        "phase": phase,
        "requested_groups": sorted(requested),
        "resolved_groups": sorted(allowed_names),
        "missing_requested_groups": sorted(missing),
        "trainable_parameter_names": sorted(observed_trainable),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "frozen_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        ),
    }


def assert_parameter_scope(model: nn.Module, scope: Mapping[str, Any]) -> None:
    expected = set(map(str, scope["trainable_parameter_names"]))
    observed = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if observed != expected:
        raise TrainingContractError(
            "trainable scope changed after configuration: "
            f"added={sorted(observed - expected)}, removed={sorted(expected - observed)}"
        )


def _snapshot_frozen(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def _assert_frozen_unchanged(model: nn.Module, before: Mapping[str, Tensor]) -> None:
    current = dict(model.named_parameters())
    changed = [
        name
        for name, value in before.items()
        if name not in current or not torch.equal(value, current[name].detach().cpu())
    ]
    if changed:
        raise TrainingContractError(f"frozen parameters changed: {changed}")


def _find_mapping(outputs: Mapping[str, Any], aliases: Sequence[str]) -> Mapping[str, Any]:
    for alias in aliases:
        value = outputs.get(alias)
        if isinstance(value, Mapping):
            return value
    return outputs


def _find_tensor(container: Mapping[str, Any], aliases: Sequence[str]) -> Tensor | None:
    for alias in aliases:
        value = container.get(alias)
        if isinstance(value, Tensor):
            return value
    return None


_FORECAST_ALIASES: dict[str, tuple[str, ...]] = {
    "da": ("da", "da_forecast", "day_ahead_price"),
    "rt": ("rt", "rt_forecast", "real_time_price"),
    "dart": ("dart", "dart_forecast", "da_minus_rt"),
    "workload": ("workload", "workload_forecast", "load"),
    "pv": ("pv", "pv_forecast", "solar"),
    "pue": ("pue", "pue_forecast", "physical_multiplier"),
}


def _align_prediction(prediction: Tensor, target: Tensor, *, name: str) -> tuple[Tensor, Tensor]:
    prediction = prediction.float()
    target = target.float()
    while prediction.ndim > 1 and prediction.shape[-1] == 1 and prediction.ndim > target.ndim:
        prediction = prediction.squeeze(-1)
    while target.ndim > 1 and target.shape[-1] == 1 and target.ndim > prediction.ndim:
        target = target.squeeze(-1)
    if prediction.shape == target.shape:
        return prediction, target
    if prediction.ndim == target.ndim - 1 and prediction.shape == target.shape[:-1]:
        return prediction, target.mean(dim=-1)
    if (
        prediction.ndim == target.ndim
        and prediction.shape[:-1] == target.shape[:-1]
        and prediction.shape[-1] == 1
    ):
        return prediction, target.mean(dim=-1, keepdim=True)
    if prediction.numel() == target.numel():
        return prediction, target.reshape_as(prediction)
    raise TrainingContractError(
        f"cannot align {name} prediction {tuple(prediction.shape)} with target "
        f"{tuple(target.shape)}"
    )


def _normalized_mse(prediction: Tensor, target: Tensor) -> Tensor:
    scale = target.detach().std(unbiased=False).clamp_min(1.0)
    return F.mse_loss(prediction / scale, target / scale)


def _forecast_losses(
    outputs: Mapping[str, Any], targets: Mapping[str, Tensor], names: Iterable[str]
) -> dict[str, Tensor]:
    forecasts = _find_mapping(outputs, ("forecasts", "forecast", "forecast_heads"))
    losses: dict[str, Tensor] = {}
    for name in names:
        prediction = _find_tensor(forecasts, _FORECAST_ALIASES[name])
        if prediction is None or name not in targets:
            continue
        aligned_prediction, aligned_target = _align_prediction(
            prediction, targets[name], name=name
        )
        losses[f"forecast_{name}"] = _normalized_mse(
            aligned_prediction, aligned_target
        )
    return losses


def _scenario_losses(
    outputs: Mapping[str, Any], scenario_target: Tensor
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    scenario = _find_mapping(outputs, ("scenario", "scenarios", "joint_scenarios"))
    particles = _find_tensor(scenario, ("particles", "scenario_particles"))
    weights = _find_tensor(scenario, ("weights", "scenario_weights"))
    if particles is None:
        return {}, {"scenario_features_used": 0}
    if particles.ndim != 4:
        raise TrainingContractError(
            f"scenario particles must be [B,20,T,D], got {tuple(particles.shape)}"
        )
    if particles.shape[1] != LEARNED_SCENARIOS and particles.shape[2] == LEARNED_SCENARIOS:
        particles = particles.transpose(1, 2)
    if particles.shape[1] != LEARNED_SCENARIOS:
        raise TrainingContractError(
            f"R3 requires exactly 20 learned particles, got {particles.shape[1]}"
        )
    if scenario_target.ndim != 4 or scenario_target.shape[1] != EMPIRICAL_SCENARIOS:
        raise TrainingContractError("scenario target must have shape [B,100,T,D]")
    if particles.shape[0] != scenario_target.shape[0] or particles.shape[2] != scenario_target.shape[2]:
        raise TrainingContractError(
            "scenario particle batch/horizon does not match the daily bundle target"
        )
    features = min(particles.shape[-1], scenario_target.shape[-1])
    if features < 1:
        raise TrainingContractError("scenario particles have no comparable variables")
    particles = particles[..., :features].float()
    target = scenario_target[..., :features].float()
    scale = target.detach().std(dim=(0, 1, 2), unbiased=False).clamp_min(1.0)
    particles_flat = (particles / scale).flatten(start_dim=2)
    target_indices = torch.linspace(
        0,
        EMPIRICAL_SCENARIOS - 1,
        LEARNED_SCENARIOS,
        device=target.device,
    ).round().long()
    target_flat = (target[:, target_indices] / scale).flatten(start_dim=2)

    if weights is None:
        normalized_weights = particles.new_full(
            (particles.shape[0], LEARNED_SCENARIOS), 1.0 / LEARNED_SCENARIOS
        )
    else:
        if weights.shape != (particles.shape[0], LEARNED_SCENARIOS):
            raise TrainingContractError(
                f"scenario weights must be [B,20], got {tuple(weights.shape)}"
            )
        if bool((weights.detach() < 0).any()):
            normalized_weights = torch.softmax(weights, dim=-1)
        else:
            normalized_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    cross_distance = torch.cdist(particles_flat, target_flat, p=2)
    self_distance = torch.cdist(particles_flat, particles_flat, p=2)
    target_weight = 1.0 / LEARNED_SCENARIOS
    energy_score = (
        (normalized_weights.unsqueeze(-1) * cross_distance).sum(dim=(1, 2))
        * target_weight
        - 0.5
        * (
            normalized_weights.unsqueeze(-1)
            * normalized_weights.unsqueeze(-2)
            * self_distance
        ).sum(dim=(1, 2))
    ).mean()
    particle_delta = particles[:, :, 1:] - particles[:, :, :-1]
    target_delta = target[:, target_indices, 1:] - target[:, target_indices, :-1]
    variogram = _normalized_mse(
        particle_delta.mean(dim=1), target_delta.mean(dim=1)
    )
    entropy_gap = (
        (normalized_weights * normalized_weights.clamp_min(1e-8).log()).sum(dim=-1)
        + math.log(LEARNED_SCENARIOS)
    ).mean()
    # The 100-row bundle is an empirical distribution.  Deterministically use
    # twenty spread-out rows as observations for the implemented proper scores.
    aligned_target = target[:, target_indices].mean(dim=1)
    marginal = marginal_scenario_loss(
        particles, normalized_weights, aligned_target
    )
    tail = tail_coverage_loss(
        particles, normalized_weights, aligned_target
    )
    diversity = scenario_diversity_loss(particles)
    return {
        "scenario_marginal": marginal,
        "scenario_energy_score": energy_score,
        "scenario_variogram": variogram,
        "scenario_tail_coverage": tail,
        "scenario_diversity": diversity,
        "scenario_weight_entropy_gap": entropy_gap,
    }, {
        "scenario_features_used": int(features),
        "scenario_feature_alignment": list(
            R3DailyBundleDataset.scenario_variables[:features]
        ),
    }


def _decision_native_losses(
    outputs: Mapping[str, Any],
    *,
    include_routing: bool,
) -> dict[str, Tensor]:
    """Latent risk/routing regularizers before the causal constrained rollout."""

    losses: dict[str, Tensor] = {}
    scenario = _find_mapping(outputs, ("scenario", "scenarios"))
    risk = _find_mapping(outputs, ("risk", "decision_risk"))
    actions = _find_mapping(outputs, ("actions", "policy"))
    routing = _find_mapping(outputs, ("routing",))
    particles = _find_tensor(scenario, ("particles", "scenario_particles"))
    scenario_weights = _find_tensor(scenario, ("weights", "scenario_weights"))
    if particles is not None and scenario_weights is not None:
        scenario_costs = _find_tensor(risk, ("scenario_costs",))
        if scenario_costs is not None:
            eta = _find_tensor(risk, ("eta",))
            losses["decision_cvar"] = cvar_loss(
                scenario_costs,
                scenario_weights,
                alpha=0.95,
                eta=eta,
            )
        losses["scenario_diversity"] = scenario_diversity_loss(particles)
        losses["scenario_weight_entropy"] = scenario_weight_entropy_loss(
            scenario_weights
        )
    expected_cost = _find_tensor(risk, ("expected_cost", "scenario_expected_cost"))
    value = _find_tensor(outputs, ("value",))
    if expected_cost is not None and value is not None:
        prediction, target = _align_prediction(
            value, expected_cost.detach(), name="decision_value_consistency"
        )
        losses["decision_value_consistency"] = F.smooth_l1_loss(
            prediction, target
        )

    migration = _find_tensor(actions, ("migration",))
    if migration is not None:
        losses["decision_migration"] = migration_cost_loss(migration)
    if include_routing:
        probabilities = _find_tensor(
            routing, ("router_probabilities", "probabilities")
        )
        expert_load = _find_tensor(routing, ("expert_load",))
        if probabilities is not None:
            losses["router_balance"] = routing_balance_loss(
                probabilities, expert_load
            )
            clean = _find_tensor(routing, ("clean_router_probabilities",))
            if clean is not None:
                losses["router_stability"] = routing_stability_loss(
                    probabilities, clean
                )
    return losses


def _solver_free_rollout_losses(
    model: nn.Module,
    model_inputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    physical: FrozenPhysicalArtifact,
    *,
    horizon: int,
    start_origin: int,
) -> tuple[dict[str, Tensor], dict[str, Any], dict[str, Tensor]]:
    """Differentiate a causal rolling chunk through policy and settlement.

    This is deliberately a neural/state rollout, not an embedded optimizer.
    Hard deployment feasibility is independently rechecked by
    :mod:`constrained_runtime`; training uses smooth bounded parameterizations
    and explicit violation penalties so gradients remain informative near a
    constraint boundary.
    """

    if not hasattr(model, "forward_origin"):
        raise TrainingContractError("solver-free policy requires model.forward_origin")
    decision = batch.get("decision")
    if not isinstance(decision, Mapping):
        raise TrainingContractError("batch is missing solver-free decision targets")
    reference = model_inputs["market"]
    batch_size = int(reference.shape[0])
    sites = len(NYISO_ZONES)
    total_horizon = int(model_inputs["market"].shape[1])
    start = max(0, min(int(start_origin), total_horizon - 1))
    stop = min(start + max(1, int(horizon)), total_horizon)

    def tensor(name: str) -> Tensor:
        value = decision.get(name)
        if not isinstance(value, Tensor):
            raise TrainingContractError(f"decision target {name!r} is missing")
        return value.to(reference)

    da_price = tensor("da_price")
    rt_price = tensor("rt_price")
    realized_workload = tensor("workload")
    realized_pv = tensor("pv")
    multiplier = tensor("physical_multiplier")
    initial_soc = tensor("initial_soc")
    idle = tensor("idle_mw")
    work_coeff = tensor("work_mw_per_unit")
    capacity = tensor("capacity_work_units")
    grid_limit = tensor("grid_limit_mw")
    battery_power = tensor("battery_power_mw")
    battery_energy = tensor("battery_energy_mwh")

    expected_z_t = (batch_size, sites, total_horizon)
    for name, value in (("da_price", da_price), ("rt_price", rt_price), ("pv", realized_pv)):
        if value.shape != expected_z_t:
            raise TrainingContractError(
                f"decision {name} must be {expected_z_t}, got {tuple(value.shape)}"
            )

    soc = initial_soc.clamp(min=0.0)
    # The deployment shield uses five aggregate EDF age buckets for the
    # four-interval deadline.  Training mirrors that state transition while
    # retaining differentiability through the selected rollout chunk.
    deadline_steps = 4
    queue_prior = reference.new_zeros((batch_size, deadline_steps + 1))
    previous_action: dict[str, Tensor] = {
        "grid": reference.new_zeros((batch_size, sites)),
        "battery": reference.new_zeros((batch_size, sites)),
        "compute": reference.new_zeros((batch_size, sites)),
        "migration": reference.new_zeros((batch_size, sites, sites)),
    }
    day_ahead: Tensor | None = None
    costs: list[Tensor] = []
    particle_costs: list[Tensor] = []
    particle_weights: list[Tensor] = []
    sla_shortfalls: list[Tensor] = []
    feasibility_violations: list[Tensor] = []
    battery_throughput: list[Tensor] = []
    migration_magnitudes: list[Tensor] = []
    shield_interventions: list[Tensor] = []
    feasibility_predictions: list[Tensor] = []
    fallback_predictions: list[Tensor] = []
    action_signatures: list[Tensor] = []
    periods = torch.arange(total_horizon, device=reference.device).unsqueeze(0)
    last_output: Mapping[str, Any] | None = None
    eta_c = 0.95
    eta_d = 0.95
    interval_hours = 0.25
    prepared_context: Mapping[str, Tensor] | None = None
    prepare = getattr(model, "prepare_origin_context", None)
    if callable(prepare):
        candidate_context = prepare(model_inputs)
        if not isinstance(candidate_context, Mapping):
            raise TrainingContractError(
                "model.prepare_origin_context must return a tensor mapping"
            )
        prepared_context = candidate_context

    def capped_waterfill(raw: Tensor, caps: Tensor, total: Tensor) -> Tensor:
        """Fixed-pass differentiable capped-simplex allocation over four sites."""

        allocation = torch.zeros_like(raw)
        remaining_capacity = caps
        remaining = total.clamp_min(0.0)
        scores = raw.clamp_min(0.0) + 0.01 * caps + 1e-6
        for _ in range(sites):
            active = remaining_capacity.gt(1e-8).to(raw.dtype)
            shares = scores * active
            shares = shares / shares.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            proposed = remaining.unsqueeze(-1) * shares
            addition = torch.minimum(proposed, remaining_capacity)
            allocation = allocation + addition
            remaining_capacity = (remaining_capacity - addition).clamp_min(0.0)
            remaining = (remaining - addition.sum(dim=-1)).clamp_min(0.0)
        return allocation

    def exact_weighted_cvar(cost: Tensor, weight: Tensor, alpha: float = 0.95) -> Tensor:
        """Exact forward CVaR for the learned discrete 20-particle law."""

        normalized = weight.clamp_min(0.0)
        normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ordered_cost, order = torch.sort(cost, dim=-1)
        ordered_weight = torch.gather(normalized, -1, order)
        cumulative = ordered_weight.cumsum(dim=-1)
        index = cumulative.ge(alpha).to(torch.int64).argmax(dim=-1, keepdim=True)
        eta = torch.gather(ordered_cost, -1, index).squeeze(-1)
        excess = F.relu(cost - eta.unsqueeze(-1))
        return eta + (normalized * excess).sum(dim=-1) / (1.0 - alpha)

    for origin in range(0, stop):
        # Prefix execution reconstructs the causal rolling state but is
        # detached.  Only the selected short chunk retains an autograd graph.
        # Preserve a gradient path for the once-issued DA schedule at t=0,
        # while detaching all other warm-up transitions before a later chunk.
        keep_graph = origin == 0 or origin >= start
        gradient_context = nullcontext() if keep_graph else torch.no_grad()
        with gradient_context:
            arrivals = realized_workload[:, origin].clamp_min(0.0)
            queue_available = queue_prior.clone()
            queue_available[:, -1] = queue_available[:, -1] + arrivals
            backlog_total = queue_available.sum(dim=-1)
            backlog = backlog_total.unsqueeze(-1).expand(-1, sites) / float(sites)
            age_weights = torch.linspace(
                1.0,
                0.2,
                deadline_steps + 1,
                device=reference.device,
                dtype=reference.dtype,
            )
            pressure = (queue_available * age_weights).sum(dim=-1)
            deadline_pressure = pressure.unsqueeze(-1).expand(-1, sites) / float(sites)
            revealed = periods.le(origin).expand(batch_size, -1)
            origin_kwargs = {
                "origin": origin,
                "current_soc": soc,
                "backlog": backlog,
                "deadline_pressure": deadline_pressure,
                "previous_action": previous_action,
                "revealed_mask": revealed,
            }
            if prepared_context is not None:
                origin_kwargs["prepared_context"] = prepared_context
            output = model.forward_origin(  # type: ignore[attr-defined]
                model_inputs, **origin_kwargs
            )
            last_output = output
            raw = _find_mapping(output, ("actions", "policy"))
            if day_ahead is None:
                candidate = _find_tensor(raw, ("day_ahead",))
                if candidate is None or candidate.shape != expected_z_t:
                    raise TrainingContractError(
                        "forward_origin at t0 must emit day_ahead [B,4,96]"
                    )
                day_ahead = candidate * grid_limit.unsqueeze(-1)

            raw_compute = _find_tensor(raw, ("compute",))
            raw_battery = _find_tensor(raw, ("battery",))
            raw_grid = _find_tensor(raw, ("grid", "real_time"))
            migration = _find_tensor(raw, ("migration",))
            if raw_compute is None or raw_battery is None or raw_grid is None or migration is None:
                raise TrainingContractError("forward_origin omitted a required policy action")

            raw_work = raw_compute.clamp(0.0, 1.0) * capacity
            remaining = float(total_horizon - origin - 1)
            terminal_reserve = 0.5 * battery_energy
            guaranteed_charge = torch.minimum(
                battery_power,
                (grid_limit - multiplier * idle).clamp_min(0.0),
            )
            reachable_floor = torch.maximum(
                torch.zeros_like(soc),
                terminal_reserve
                - remaining * interval_hours * eta_c * guaranteed_charge,
            )
            required_terminal_charge = torch.minimum(
                battery_power,
                ((reachable_floor - soc) / (interval_hours * eta_c)).clamp_min(0.0),
            )
            pv_available = realized_pv[:, :, origin].clamp_min(0.0)
            available_for_compute = (
                grid_limit + pv_available - required_terminal_charge
            ).clamp_min(0.0)
            compute_caps = capacity.clone()
            positive_work = work_coeff.gt(1e-8)
            power_caps = (
                (available_for_compute / multiplier.clamp_min(1e-6) - idle)
                / work_coeff.clamp_min(1e-6)
            ).clamp_min(0.0)
            compute_caps = torch.where(
                positive_work, torch.minimum(compute_caps, power_caps), compute_caps
            )
            max_service = torch.minimum(backlog_total, compute_caps.sum(dim=-1))
            guaranteed_future = capacity.sum(dim=-1)
            required_now = reference.new_zeros(batch_size)
            for remaining_deadline in range(deadline_steps + 1):
                due_prefix = queue_available[:, : remaining_deadline + 1].sum(dim=-1)
                required_now = torch.maximum(
                    required_now,
                    due_prefix - float(remaining_deadline) * guaranteed_future,
                )
            required_now = torch.minimum(required_now.clamp_min(0.0), max_service)
            desired_service = torch.maximum(
                torch.minimum(raw_work.sum(dim=-1), max_service), required_now
            )
            work = capped_waterfill(raw_work, compute_caps, desired_service)
            load = multiplier * (idle + work_coeff * work)
            raw_signed_battery = raw_battery.clamp(-1.0, 1.0) * battery_power

            lo_soc = torch.maximum(
                -battery_power,
                -eta_d * (soc - reachable_floor) / interval_hours,
            )
            hi_soc = torch.minimum(
                battery_power,
                (battery_energy - soc) / (interval_hours * eta_c),
            )
            lo_energy = -load
            hi_energy = grid_limit + pv_available - load
            battery_lower = torch.maximum(lo_soc, lo_energy)
            battery_upper = torch.minimum(hi_soc, hi_energy)
            battery_interval_violation = F.relu(battery_lower - battery_upper)
            safe_lower = torch.minimum(battery_lower, battery_upper)
            safe_upper = torch.maximum(battery_lower, battery_upper)
            battery = torch.maximum(
                torch.minimum(raw_signed_battery, safe_upper), safe_lower
            )
            charge = F.relu(battery)
            discharge = F.relu(-battery)
            net_load = load + battery
            grid_lower = torch.maximum(torch.zeros_like(net_load), net_load - pv_available)
            grid_upper = torch.minimum(grid_limit, net_load)
            grid_interval_violation = F.relu(grid_lower - grid_upper)
            safe_grid_lower = torch.minimum(grid_lower, grid_upper)
            safe_grid_upper = torch.maximum(grid_lower, grid_upper)
            raw_grid_mw = raw_grid.clamp(0.0, 1.0) * grid_limit
            grid = torch.maximum(
                torch.minimum(raw_grid_mw, safe_grid_upper), safe_grid_lower
            )
            pv_used = net_load - grid
            grid_excess = F.relu(grid - grid_limit)
            pv_violation = F.relu(-pv_used) + F.relu(pv_used - pv_available)
            soc_next = soc + interval_hours * (eta_c * charge - discharge / eta_d)
            soc_violation = F.relu(-soc_next) + F.relu(soc_next - battery_energy)
            # Exact bound closure removes sub-ULP negative values before the
            # next strict causal-policy state check.  The pre-clamp residual is
            # retained above as training evidence.
            soc_next = torch.maximum(
                torch.zeros_like(soc_next), torch.minimum(soc_next, battery_energy)
            )
            energy_residual = grid + pv_used + discharge - load - charge

            served = work.sum(dim=-1)
            remaining_service = served
            queue_rows: list[Tensor] = []
            for bucket in range(deadline_steps + 1):
                served_here = torch.minimum(queue_available[:, bucket], remaining_service)
                queue_rows.append((queue_available[:, bucket] - served_here).clamp_min(0.0))
                remaining_service = (remaining_service - served_here).clamp_min(0.0)
            queue_after = torch.stack(queue_rows, dim=-1)
            sla_shortfall = queue_after[:, 0]
            queue_after = torch.cat(
                (torch.zeros_like(queue_after[:, :1]), queue_after[:, 1:]), dim=-1
            )
            queue_next = torch.cat(
                (queue_after[:, 1:], torch.zeros_like(queue_after[:, :1])), dim=-1
            )
            intervention = (
                ((raw_signed_battery - battery) / battery_power.clamp_min(1e-6))
                .square()
                .mean(dim=-1)
                + ((raw_grid_mw - grid) / grid_limit.clamp_min(1e-6))
                .square()
                .mean(dim=-1)
                + ((raw_work - work) / capacity.clamp_min(1e-6))
                .square()
                .mean(dim=-1)
                + grid_excess.mean(dim=-1)
                + pv_violation.mean(dim=-1)
            )
            interval_cost = interval_hours * (
                da_price[:, :, origin] * day_ahead[:, :, origin]
                + rt_price[:, :, origin]
                * (grid - day_ahead[:, :, origin])
            ).sum(dim=-1)
            interval_cost = interval_cost + interval_hours * (charge + discharge).sum(dim=-1)

            scenario = _find_mapping(output, ("scenario", "scenarios"))
            particles = _find_tensor(scenario, ("particles", "scenario_particles"))
            weights_at_origin = _find_tensor(scenario, ("weights", "scenario_weights"))
            if particles is None or weights_at_origin is None:
                raise TrainingContractError("forward_origin omitted learned scenarios")
            if particles.shape != (
                batch_size,
                LEARNED_SCENARIOS,
                total_horizon,
                len(R3DailyBundleDataset.scenario_variables),
            ):
                raise TrainingContractError(
                    "rolling scenarios must be [B,20,96,6], got "
                    f"{tuple(particles.shape)}"
                )
            scenario_da = particles[:, :, origin, 0]
            scenario_rt = particles[:, :, origin, 1]
            scenario_workload = particles[:, :, origin, 3].clamp_min(0.0)
            scenario_pv = particles[:, :, origin, 4].clamp_min(0.0)
            scenario_pue = particles[:, :, origin, 5].clamp(0.1, 5.0)
            scenario_load = scenario_pue.unsqueeze(-1) * (
                idle.unsqueeze(1) + work_coeff.unsqueeze(1) * work.unsqueeze(1)
            )
            scenario_net = scenario_load + battery.unsqueeze(1)
            scenario_grid_uncapped = (
                scenario_net - scenario_pv.unsqueeze(-1)
            ).clamp_min(0.0)
            scenario_grid = torch.minimum(
                scenario_grid_uncapped, grid_limit.unsqueeze(1)
            )
            scenario_grid_excess = F.relu(
                scenario_grid_uncapped - grid_limit.unsqueeze(1)
            ).sum(dim=-1)
            scenario_sla = F.relu(scenario_workload - served.unsqueeze(-1))
            particle_interval_cost = interval_hours * (
                scenario_da * day_ahead[:, :, origin].sum(dim=-1).unsqueeze(-1)
                + scenario_rt
                * (
                    scenario_grid.sum(dim=-1)
                    - day_ahead[:, :, origin].sum(dim=-1).unsqueeze(-1)
                )
            )
            particle_interval_cost = (
                particle_interval_cost
                + interval_hours * (charge + discharge).sum(dim=-1).unsqueeze(-1)
                + 1000.0 * scenario_sla
                + 10000.0 * scenario_grid_excess
            )

            if origin >= start:
                costs.append(interval_cost)
                sla_shortfalls.append(sla_shortfall)
                feasibility_violations.append(
                    energy_residual.abs().amax(dim=-1)
                    + grid_excess.amax(dim=-1)
                    + soc_violation.amax(dim=-1)
                    + battery_interval_violation.amax(dim=-1)
                    + grid_interval_violation.amax(dim=-1)
                )
                battery_throughput.append((charge + discharge).mean(dim=-1))
                migration_magnitudes.append(migration.abs().mean(dim=(-1, -2)))
                shield_interventions.append(intervention)
                feasibility_probability = _find_tensor(
                    output, ("feasibility",)
                )
                fallback_probability = _find_tensor(
                    output, ("fallback_confidence",)
                )
                if feasibility_probability is None or fallback_probability is None:
                    raise TrainingContractError(
                        "forward_origin omitted feasibility/fallback confidence"
                    )
                feasibility_predictions.append(feasibility_probability)
                fallback_predictions.append(fallback_probability)
                particle_costs.append(particle_interval_cost)
                particle_weights.append(weights_at_origin)
                action_signatures.append(
                    torch.cat(
                        (
                            day_ahead[:, :, origin],
                            grid,
                            battery,
                            work,
                        ),
                        dim=-1,
                    )
                )
            soc = soc_next
            queue_prior = queue_next
            previous_action = {
                "grid": grid,
                "battery": battery,
                "compute": work,
                # The empirical selection contract fixes migration to zero.
                "migration": torch.zeros_like(migration),
            }
            if origin < start:
                soc = soc.detach()
                queue_prior = queue_prior.detach()
                previous_action = {
                    name: value.detach() for name, value in previous_action.items()
                }

    if last_output is None or day_ahead is None:
        raise TrainingContractError("empty solver-free rollout chunk")
    daily_cost = torch.stack(costs, dim=-1).sum(dim=-1)
    violation = torch.stack(feasibility_violations, dim=-1)
    scenario_cost = torch.stack(particle_costs, dim=-1).sum(dim=-1)
    scenario_weights = torch.stack(particle_weights, dim=1).mean(dim=1)
    scenario_weights = scenario_weights / scenario_weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    expected_particle_cost = (scenario_weights * scenario_cost).sum(dim=-1)
    particle_cvar = exact_weighted_cvar(scenario_cost, scenario_weights, alpha=0.95)
    realized_scale = daily_cost.detach().abs().mean().clamp_min(100.0)
    particle_scale = scenario_cost.detach().abs().mean().clamp_min(100.0)
    losses: dict[str, Tensor] = {
        "decision_realized_cost": daily_cost.mean() / realized_scale,
        "decision_expected_particle_cost": expected_particle_cost.mean() / particle_scale,
        "decision_cvar": particle_cvar.mean() / particle_scale,
        "decision_sla": torch.stack(sla_shortfalls, dim=-1).mean(),
        "decision_feasibility": violation.square().mean(),
        "decision_battery_degradation": torch.stack(
            battery_throughput, dim=-1
        ).mean(),
        "decision_migration": torch.stack(migration_magnitudes, dim=-1).mean(),
        "decision_constraint_intervention": torch.stack(
            shield_interventions, dim=-1
        ).mean(),
    }
    intervention_tensor = torch.stack(shield_interventions, dim=-1)
    feasibility_prediction = torch.stack(feasibility_predictions, dim=-1).clamp(
        1e-6, 1.0 - 1e-6
    )
    fallback_prediction = torch.stack(fallback_predictions, dim=-1).clamp(
        1e-6, 1.0 - 1e-6
    )
    # Self-supervised calibration: a raw action that needs little shield
    # intervention is feasible and should be accepted.  Large projection
    # magnitude supplies a smooth fallback target.  Detaching the target keeps
    # the confidence heads from changing the policy merely to relabel itself.
    safe_policy_target = (
        intervention_tensor.detach() / (1.0 + intervention_tensor.detach())
    ).clamp(0.0, 1.0)
    # Avoid a threshold race: a merely moderate correction should calibrate
    # acceptance confidence, while only a large shield projection drives the
    # conservative branch below the 0.5 deployment threshold.
    safe_policy_target = (0.5 * safe_policy_target).clamp(0.0, 1.0)
    feasibility_target = (1.0 - safe_policy_target).clamp(0.0, 1.0)
    losses["decision_confidence_calibration"] = (
        F.binary_cross_entropy(feasibility_prediction, feasibility_target)
        # ``fallback_confidence`` is the legacy field name for confidence in
        # accepting the neural proposal.  Low confidence triggers safe policy.
        + F.binary_cross_entropy(fallback_prediction, feasibility_target)
    )
    value = _find_tensor(last_output, ("value",))
    if value is not None:
        prediction, target = _align_prediction(
            value, daily_cost.detach(), name="policy_value_consistency"
        )
        losses["decision_value_consistency"] = F.smooth_l1_loss(
            torch.sign(prediction) * torch.log1p(prediction.abs()),
            torch.sign(target) * torch.log1p(target.abs()),
        )
    action_signature = torch.stack(action_signatures, dim=1)
    return losses, {
        "decision_rollout_start_origin": start,
        "decision_rollout_origins": stop - start,
        "decision_external_solver": False,
        "decision_daily_cost_usd": float(daily_cost.detach().mean().item()),
        "decision_particle_expected_cost_usd": float(
            expected_particle_cost.detach().mean().item()
        ),
        "decision_particle_cvar_95_usd": float(particle_cvar.detach().mean().item()),
        "decision_mean_safe_policy_target": float(safe_policy_target.mean().item()),
        "decision_mean_acceptance_confidence": float(
            fallback_prediction.detach().mean().item()
        ),
    }, {
        "daily_cost": daily_cost,
        "particle_costs": scenario_cost,
        "scenario_weights": scenario_weights,
        "action_signature": action_signature,
    }


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device=device, non_blocking=device.type == "cuda")
        elif isinstance(value, Mapping):
            moved[key] = {
                nested_key: nested_value.to(
                    device=device, non_blocking=device.type == "cuda"
                )
                if isinstance(nested_value, Tensor)
                else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _pin_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Pin a complete host batch so non-blocking CUDA copies are real."""

    pinned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            pinned[key] = value.pin_memory()
        elif isinstance(value, Mapping):
            pinned[key] = {
                nested_key: nested_value.pin_memory()
                if isinstance(nested_value, Tensor)
                else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            pinned[key] = value
    return pinned


def _add_batch_dimension(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Create a one-day batch without Torch's unstable Windows stack path."""

    result: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, Tensor):
            result[key] = value.unsqueeze(0)
        elif isinstance(value, Mapping):
            result[key] = {
                nested_key: nested_value.unsqueeze(0)
                if isinstance(nested_value, Tensor)
                else nested_value
                for nested_key, nested_value in value.items()
            }
        elif key in {"operating_date", "bundle_path"}:
            result[key] = [value]
        else:
            result[key] = value
    return result


def _collate_days(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate a small number of complete days without a worker process."""

    if not samples:
        raise TrainingContractError("cannot collate an empty day batch")

    def collate(values: Sequence[Any], *, name: str) -> Any:
        first = values[0]
        if isinstance(first, Tensor):
            if not all(isinstance(value, Tensor) for value in values):
                raise TrainingContractError(f"mixed tensor/non-tensor field {name}")
            return torch.stack(list(values), dim=0)
        if isinstance(first, Mapping):
            keys = tuple(first)
            if not all(isinstance(value, Mapping) and tuple(value) == keys for value in values):
                raise TrainingContractError(f"inconsistent nested batch field {name}")
            return {
                key: collate([value[key] for value in values], name=f"{name}.{key}")
                for key in keys
            }
        if name in {"operating_date", "bundle_path"}:
            return [str(value) for value in values]
        if all(value == first for value in values[1:]):
            return first
        return list(values)

    keys = tuple(samples[0])
    if not all(tuple(sample) == keys for sample in samples):
        raise TrainingContractError("daily samples have inconsistent fields")
    return {key: collate([sample[key] for sample in samples], name=key) for key in keys}


def _rng_state() -> dict[str, Any]:
    """Capture every RNG needed for an epoch-boundary exact resume."""

    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    if "python" in payload:
        random.setstate(payload["python"])
    if "numpy" in payload:
        np.random.set_state(payload["numpy"])
    if "torch_cpu" in payload:
        torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError)
        and ("out of memory" in message or "not enough memory" in message)
    )


def _finite_gradient_evidence(model: nn.Module) -> dict[str, Any]:
    tensors = 0
    nonzero_tensors = 0
    squared_norm = 0.0
    max_abs = 0.0
    non_finite: list[str] = []
    nonzero_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        tensors += 1
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            non_finite.append(name)
            continue
        absolute = gradient.abs()
        if bool((absolute > 0).any()):
            nonzero_tensors += 1
            nonzero_names.append(name)
        max_abs = max(max_abs, float(absolute.max().item()))
        squared_norm += float(torch.sum(gradient.double() ** 2).item())
    if non_finite:
        raise TrainingContractError(f"non-finite gradients: {non_finite}")
    if tensors == 0:
        raise TrainingContractError("no trainable parameter received a gradient")
    if nonzero_tensors == 0:
        raise TrainingContractError("all trainable gradients are zero")
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm):
        raise TrainingContractError("gradient norm is non-finite")
    return {
        "gradient_tensor_count": tensors,
        "nonzero_gradient_tensor_count": nonzero_tensors,
        "gradient_l2_norm": norm,
        "gradient_max_abs": max_abs,
        "gradients_finite": True,
        "nonzero_gradient_parameter_names_json": json.dumps(
            nonzero_names, ensure_ascii=True, separators=(",", ":")
        ),
    }


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write Parquet in an isolated process outside the Unicode workspace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tmp", dir=path.parent, delete=False) as handle:
        target_temporary = Path(handle.name)
    try:
        with tempfile.TemporaryDirectory(prefix="dcvpp_r3_parquet_") as ascii_dir:
            ascii_root = Path(ascii_dir)
            json_path = ascii_root / "records.json"
            parquet_path = ascii_root / "metrics.parquet"
            json_path.write_text(
                json.dumps(
                    _plain(frame.to_dict(orient="records")),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="ascii",
            )
            writer = (
                "import json,sys,pandas as pd,pyarrow as pa,pyarrow.parquet as pq;"
                "rows=json.load(open(sys.argv[1],encoding='ascii'));"
                "frame=pd.DataFrame(rows);"
                "table=pa.Table.from_pandas(frame,preserve_index=False);"
                "stream=open(sys.argv[2],'wb');"
                "pq.write_table(table,stream,compression='zstd');stream.close()"
            )
            completed = subprocess.run(
                [sys.executable, "-c", writer, str(json_path), str(parquet_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise TrainingContractError(
                    "isolated Parquet writer failed: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            with parquet_path.open("rb") as source, target_temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        os.replace(target_temporary, path)
    finally:
        if target_temporary.exists():
            target_temporary.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pt", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class R3Trainer:
    """Run one resumable R3 phase with sealed inputs and auditable artifacts."""

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        *,
        repository_root: str | Path,
        run_dir: str | Path,
        phase: str,
        split: str = "selection",
        device: str | torch.device | None = None,
        selection_evaluator: Callable[..., Mapping[str, Any] | None] | None = None,
        validation_evaluator: Callable[..., Mapping[str, Any] | None] | None = None,
        resume: bool = True,
    ) -> None:
        if phase not in PHASES:
            raise TrainingContractError(f"unknown phase {phase!r}; expected {PHASES}")
        self.model = model
        self.config = dict(config)
        self.repository_root = Path(repository_root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.phase = phase
        self.split = str(split)
        requested_device = device or _nested_get(config, "training", "device", default=None)
        if requested_device == "auto":
            requested_device = None
        self.device = torch.device(
            requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingContractError("CUDA requested but torch.cuda.is_available() is false")
        self.selection_evaluator = selection_evaluator
        self.validation_evaluator = validation_evaluator
        self.resume = bool(resume)
        self.state_path = self.run_dir / "state.json"
        self.metrics_path = self.run_dir / "metrics.parquet"
        self.summary_path = self.run_dir / "summary.md"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_path = self.checkpoint_dir / "latest.pt"
        self.best_checkpoint_path = self.checkpoint_dir / "best.pt"

    def _formal_config(self) -> Mapping[str, Any]:
        value = _nested_get(self.config, "training", "formal", default={})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TrainingContractError("training.formal must be a mapping")
        return value

    def _is_formal(self) -> bool:
        formal = self._formal_config()
        return bool(formal.get("enabled", False)) or str(
            _nested_get(self.config, "training", "mode", default="pilot")
        ).lower() == "formal"

    def _experiment_id(self) -> str:
        return str(
            _nested_get(
                self.config,
                "protocol",
                "experiment_id",
                default=_nested_get(
                    self.config,
                    "project",
                    "experiment_id",
                    default=("e2e_moe_r3_formal" if self._is_formal() else "e2e_moe_r3_pilot"),
                ),
            )
        )

    def _method(self) -> str:
        return str(
            _nested_get(
                self.config,
                "protocol",
                "method",
                default=_nested_get(
                    self.config,
                    "project",
                    "variant",
                    default="solver_free_constrained_policy",
                ),
            )
        )

    def _protocol_splits(self) -> dict[str, dict[str, str]]:
        """Return the fixed formal train/validation windows without opening 2024."""

        defaults = {
            "train": {"start": "2022-01-01", "end": "2023-06-30"},
            "validation": {"start": "2023-07-01", "end": "2023-12-31"},
        }
        result: dict[str, dict[str, str]] = {}
        aliases = {
            "train": ("train_start_date", "train_end_date"),
            "validation": ("validation_start_date", "validation_end_date"),
        }
        for split_name, (start_alias, end_alias) in aliases.items():
            configured = _nested_get(
                self.config, "protocol", "splits", split_name, default={}
            )
            if configured is None:
                configured = {}
            if not isinstance(configured, Mapping):
                raise TrainingContractError(
                    f"protocol.splits.{split_name} must be a mapping"
                )
            start = str(
                configured.get(
                    "start",
                    _nested_get(
                        self.config,
                        "data",
                        start_alias,
                        default=defaults[split_name]["start"],
                    ),
                )
            )[:10]
            end = str(
                configured.get(
                    "end",
                    _nested_get(
                        self.config,
                        "data",
                        end_alias,
                        default=defaults[split_name]["end"],
                    ),
                )
            )[:10]
            if _parse_year(start) not in {2022, 2023} or _parse_year(end) not in {
                2022,
                2023,
            }:
                raise TrainingContractError(
                    f"formal {split_name} window must remain within 2022-2023"
                )
            if start > end:
                raise TrainingContractError(
                    f"formal {split_name} start is after its end"
                )
            result[split_name] = {"start": start, "end": end}
        if result["train"]["end"] >= result["validation"]["start"]:
            raise TrainingContractError("formal train and validation windows overlap")
        selection = _nested_get(
            self.config, "protocol", "splits", "formal_selection", default={}
        )
        if selection is None:
            selection = {}
        if not isinstance(selection, Mapping):
            raise TrainingContractError("protocol.splits.formal_selection must be a mapping")
        selection_start = str(
            selection.get(
                "start",
                _nested_get(
                    self.config,
                    "data",
                    "formal_selection_start_date",
                    default="2024-01-01",
                ),
            )
        )[:10]
        selection_end = str(
            selection.get(
                "end",
                _nested_get(
                    self.config,
                    "data",
                    "formal_selection_end_date",
                    default="2024-12-31",
                ),
            )
        )[:10]
        if _parse_year(selection_start) != 2024 or _parse_year(selection_end) != 2024:
            raise TrainingContractError("formal selection window is sealed to 2024")
        if selection_start > selection_end:
            raise TrainingContractError("formal selection start is after its end")
        result["formal_selection"] = {
            "start": selection_start,
            "end": selection_end,
        }
        return result

    def _protocol_fingerprint(self) -> str:
        return sha256_json(
            {
                "schema": "dcvpp.e2e_moe_r3.formal_protocol.v1",
                "experiment_id": self._experiment_id(),
                "method": self._method(),
                "training_semantics_revision": _nested_get(
                    self.config,
                    "protocol",
                    "training_semantics_revision",
                    default="legacy_unversioned",
                ),
                "splits": self._protocol_splits() if self._is_formal() else None,
                "seeds": _plain(
                    _nested_get(self.config, "protocol", "seeds", default=[self._seed()])
                ),
            }
        )

    def _artifact_root(self) -> Path:
        value = _nested_get(
            self.config,
            "project",
            "output_root",
            default="artifacts/e2e_moe_solver_free_r3",
        )
        path = Path(value)
        return (path if path.is_absolute() else self.repository_root / path).resolve()

    def _predecessor_checkpoint(self) -> tuple[Path | None, dict[str, Any]]:
        predecessor = PHASE_PREDECESSOR.get(self.phase)
        if predecessor is None:
            return None, {"required": False}
        root = self._artifact_root()
        expected_identity = (
            {
                "training_regime": "formal",
                "experiment_id": self._experiment_id(),
                "method": self._method(),
                "seed": self._seed(),
                "protocol_fingerprint": self._protocol_fingerprint(),
            }
            if self._is_formal()
            else None
        )
        candidates = sorted(
            (
                path
                for path in root.glob(f"run_id={predecessor}_{self.split}_*")
                if path.is_dir() and path.resolve() != self.run_dir
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            state_path = candidate / "state.json"
            # Formal phases must inherit the validation-selected checkpoint,
            # never merely the final epoch.  Pilot runs retain their historical
            # latest.pt lineage for backward compatibility.
            checkpoint_path = candidate / "checkpoints" / (
                "best.pt" if self._is_formal() else "latest.pt"
            )
            if not state_path.exists() or not checkpoint_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                state.get("status") == "COMPLETE"
                and state.get("phase") == predecessor
                and int(state.get("completed_epochs", 0)) > 0
                and _checkpoint_lineage_complete(
                    state, predecessor, expected=expected_identity
                )
            ):
                required_parent = PHASE_PREDECESSOR.get(predecessor)
                lineage = state.get("predecessor", {})
                if required_parent is not None and (
                    not isinstance(lineage, Mapping)
                    or lineage.get("phase") != required_parent
                    or not lineage.get("checkpoint_sha256")
                ):
                    continue
                return checkpoint_path, {
                    "required": True,
                    "phase": predecessor,
                    "run_dir": str(candidate),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "state_sha256": sha256_file(state_path),
                    **(expected_identity or {}),
                }
        return None, {
            "required": True,
            "phase": predecessor,
            "reason": f"no COMPLETE {predecessor} checkpoint for split {self.split}",
        }

    def _blocked_state(
        self,
        *,
        reason: str,
        inputs: Mapping[str, Any],
        scope: Mapping[str, Any],
        predecessor: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = {
            "schema_version": "dcvpp.e2e_moe_r3.training_state.v2",
            "status": "BLOCKED",
            "training_regime": "formal" if self._is_formal() else "pilot",
            "experiment_id": self._experiment_id(),
            "method": self._method(),
            "phase": self.phase,
            "split": self.split,
            "seed": self._seed(),
            "active_seed": self._seed(),
            "protocol_seeds": _plain(
                _nested_get(self.config, "protocol", "seeds", default=[self._seed()])
            ),
            "protocol_fingerprint": self._protocol_fingerprint(),
            "training_semantics_revision": _nested_get(
                self.config,
                "protocol",
                "training_semantics_revision",
                default="legacy_unversioned",
            ),
            "split_manifest_sha256": inputs.get("split_manifest_sha256"),
            "protocol_splits": self._protocol_splits() if self._is_formal() else None,
            "completed_epochs": 0,
            "reason": reason,
            "scope": scope,
            "inputs": inputs,
            "predecessor": predecessor,
            "locked_2025_opened": False,
            "formal_2024_opened": False,
            "external_solver": False,
            "resumable": True,
            "resume_command": self._resume_command(),
            "updated_at_utc": utc_now(),
        }
        metrics = pd.DataFrame(
            [
                {
                    "phase": self.phase,
                    "split": self.split,
                    "epoch": 0,
                    "event": "blocked",
                    "blocked_reason": reason,
                    "locked_2025_opened": False,
                    **collect_device_metrics(self.device),
                }
            ]
        )
        _atomic_parquet(self.metrics_path, metrics)
        self._write_state(state)
        atomic_text(self.summary_path, self._summary(state, metrics))
        return state

    def _phase_config(self) -> Mapping[str, Any]:
        phase_config = _nested_get(self.config, "training", self.phase, default={})
        if self.phase == "policy" and not phase_config:
            phase_config = _nested_get(
                self.config, "training", "policy_training", default={}
            )
        if not isinstance(phase_config, Mapping):
            raise TrainingContractError(f"training.{self.phase} must be a mapping")
        return phase_config

    def _input_paths(self) -> tuple[Path, Path]:
        daily_value = (
            _nested_get(self.config, "data", "daily_bundle_root", default=None)
            or _nested_get(self.config, "data", "selection_bundle_root", default=None)
            or _nested_get(self.config, "paths", "daily_bundle_root", default=None)
            or self.repository_root
            / "artifacts"
            / "dart_value_select"
            / "counterfactual_inputs"
            / "v1"
            / "selection"
            / "daily_bundles"
        )
        physical_value = (
            _nested_get(self.config, "physical_twin", "artifact_path", default=None)
            or _nested_get(self.config, "data", "physical_artifact", default=None)
            or _nested_get(self.config, "paths", "frozen_physical_artifact", default=None)
            or self.repository_root
            / "artifacts"
            / "research"
            / "physical"
            / "run_id=paper-physical-v1"
            / "frozen_physical_artifact.json"
        )
        daily = Path(daily_value)
        physical = Path(physical_value)
        if not daily.is_absolute():
            daily = self.repository_root / daily
        if not physical.is_absolute():
            physical = self.repository_root / physical
        return daily.resolve(), physical.resolve()

    def _epochs(self) -> int:
        phase_config = self._phase_config()
        formal_defaults = {
            "representation": 500,
            "scenario": 500,
            "policy": 500,
            "a1": 500,
            "a2": 500,
        }
        default_epochs = (
            formal_defaults.get(self.phase, 1)
            if self._is_formal()
            else _nested_get(self.config, "training", "smoke_epochs", default=1)
        )
        value = phase_config.get("epochs", default_epochs)
        epochs = int(value)
        if epochs < 1:
            raise TrainingContractError("epochs must be at least one")
        if self._is_formal() and self.phase in {"policy", "a1", "a2"} and epochs < 4:
            raise TrainingContractError(
                "formal decision phases require at least four epochs for 8->24->48->96"
            )
        # Formal runs opt into their frozen protocol budget; pilot runs retain
        # the conservative smoke ceiling.
        max_epochs = int(
            _nested_get(
                self.config,
                "training",
                "max_phase_epochs",
                default=512 if self._is_formal() else 32,
            )
        )
        if epochs > max_epochs:
            raise TrainingContractError(
                f"requested {epochs} epochs exceeds max_phase_epochs={max_epochs}"
            )
        return epochs

    def _seed(self) -> int:
        seed = int(
            _nested_get(
                self.config,
                "training",
                "seed",
                default=_nested_get(self.config, "run", "seed", default=20240813),
            )
        )
        configured = _nested_get(self.config, "protocol", "seeds", default=None)
        if self._is_formal() and configured is not None:
            if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
                raise TrainingContractError("protocol.seeds must be an integer sequence")
            allowed = tuple(int(value) for value in configured)
            if seed not in allowed:
                raise TrainingContractError(
                    f"active formal seed {seed} is not present in protocol.seeds={allowed}"
                )
        return seed

    def _build_inputs(
        self,
    ) -> tuple[
        R3DailyBundleDataset,
        R3DailyBundleDataset | None,
        FrozenPhysicalArtifact,
        dict[str, Any],
    ]:
        daily_root, physical_path = self._input_paths()
        phase_config = self._phase_config()
        max_days = None if self._is_formal() else phase_config.get(
            "max_days", _nested_get(self.config, "training", "smoke_days", default=1)
        )
        all_paths = discover_daily_bundles(
            daily_root,
            start_date=phase_config.get("start_date"),
            end_date=phase_config.get("end_date"),
            max_days=None,
        )
        decision_stratified = self.phase in {"policy", "a1", "a2"}
        validation_paths: list[Path] = []
        if self._is_formal():
            windows = self._protocol_splits()
            paths = [
                path
                for path in all_paths
                if windows["train"]["start"] <= path.stem[:10] <= windows["train"]["end"]
            ]
            validation_paths = [
                path
                for path in all_paths
                if windows["validation"]["start"]
                <= path.stem[:10]
                <= windows["validation"]["end"]
            ]
            if not paths or not validation_paths:
                raise TrainingContractError(
                    "formal protocol requires non-empty explicit train and validation dates"
                )
            regime_labels = {path.stem[:10]: "all_eligible_train" for path in paths}
        else:
            paths, regime_labels = select_training_bundles(
                all_paths,
                max_days=None if max_days is None else int(max_days),
                decision_stratified=decision_stratified,
            )
        physical, physical_annotation = load_frozen_physical_artifact(physical_path)
        settlement_value = _nested_get(
            self.config, "data", "realized_settlement_root", default=None
        )
        settlement_root: Path | None = None
        if settlement_value not in (None, ""):
            settlement_root = Path(str(settlement_value))
            if not settlement_root.is_absolute():
                settlement_root = self.repository_root / settlement_root
        dataset = R3DailyBundleDataset(
            paths,
            physical,
            realized_settlement_root=settlement_root,
        )
        validation_dataset = (
            R3DailyBundleDataset(
                validation_paths,
                physical,
                realized_settlement_root=settlement_root,
            )
            if validation_paths
            else None
        )
        canonical_index = _nested_get(
            self.config, "data", "realized_settlement_index", default=None
        )
        canonical_index_path: Path | None = None
        if canonical_index not in (None, ""):
            canonical_index_path = Path(str(canonical_index))
            if not canonical_index_path.is_absolute():
                canonical_index_path = self.repository_root / canonical_index_path
        train_dates = [path.stem[:10] for path in paths]
        validation_dates = [path.stem[:10] for path in validation_paths]
        manifest = {
            "schema": "dcvpp.e2e_moe_r3.split_manifest.v1",
            "formal": self._is_formal(),
            "protocol_splits": self._protocol_splits() if self._is_formal() else None,
            "train_count": len(train_dates),
            "train_first_date": train_dates[0] if train_dates else None,
            "train_last_date": train_dates[-1] if train_dates else None,
            "train_date_index_sha256": sha256_json(train_dates),
            "validation_count": len(validation_dates),
            "validation_first_date": validation_dates[0] if validation_dates else None,
            "validation_last_date": validation_dates[-1] if validation_dates else None,
            "validation_date_index_sha256": sha256_json(validation_dates),
            "daily_bundle_root": str(daily_root),
            "canonical_index_sha256": (
                sha256_file(canonical_index_path)
                if canonical_index_path is not None and canonical_index_path.exists()
                else None
            ),
        }
        split_manifest_sha256 = sha256_json(manifest)
        annotation = {
            "daily_bundle_root": str(daily_root),
            "daily_bundle_count": len(paths),
            "daily_bundle_dates": None if self._is_formal() else train_dates,
            "daily_bundle_hashes": (
                None
                if self._is_formal()
                else {path.name: sha256_file(path) for path in paths}
            ),
            "validation_bundle_count": len(validation_paths),
            "validation_bundle_dates": None if self._is_formal() else validation_dates,
            "split_manifest": manifest,
            "split_manifest_sha256": split_manifest_sha256,
            "protocol_fingerprint": self._protocol_fingerprint(),
            "training_sample_strategy": (
                "issue_time_regime_stratified"
                if decision_stratified
                else "cross_year_temporal_stratified"
            ),
            "training_sample_regimes": dict(regime_labels),
            "physical": physical_annotation,
            "realized_settlement_root": (
                None if settlement_root is None else str(settlement_root.resolve())
            ),
            "locked_2025_opened": False,
            "reuse": dict(REUSE_ANNOTATIONS),
        }
        return dataset, validation_dataset, physical, annotation

    def _fingerprint(
        self,
        inputs: Mapping[str, Any],
        scope: Mapping[str, Any],
        predecessor: Mapping[str, Any] | None = None,
    ) -> str:
        return sha256_json(
            {
                "schema": "dcvpp.e2e_moe_r3.trainer.v2",
                "training_regime": "formal" if self._is_formal() else "pilot",
                "experiment_id": self._experiment_id(),
                "method": self._method(),
                "seed": self._seed(),
                "protocol_fingerprint": self._protocol_fingerprint(),
                "phase": self.phase,
                "split": self.split,
                "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "config": _plain(self.config),
                "inputs": _plain(inputs),
                "predecessor": _plain(predecessor or {"required": False}),
                "scope": {
                    "requested_groups": scope["requested_groups"],
                    "resolved_groups": scope["resolved_groups"],
                    "trainable_parameter_names": scope["trainable_parameter_names"],
                },
            }
        )

    def _read_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: Mapping[str, Any]) -> None:
        atomic_json(self.state_path, _plain(state))

    def _metrics(self) -> pd.DataFrame:
        if self.metrics_path.exists():
            with self.metrics_path.open("rb") as stream:
                frame = pd.read_parquet(stream)
            return frame
        return pd.DataFrame()

    def _summary(self, state: Mapping[str, Any], metrics: pd.DataFrame) -> str:
        completed = int(state.get("completed_epochs", 0))
        latest_loss = None
        if not metrics.empty and "total_loss" in metrics:
            finite = pd.to_numeric(metrics["total_loss"], errors="coerce").dropna()
            if not finite.empty:
                latest_loss = float(finite.iloc[-1])
        lines = [
            "# E2E-MoE R3 phase summary",
            "",
            f"- Status: `{state.get('status')}`",
            f"- Phase: `{self.phase}`",
            f"- Split: `{self.split}`",
            f"- Completed smoke epochs: `{completed}`",
            f"- Latest total loss: `{latest_loss}`",
            f"- Device: `{self.device}`",
            "- 2025 opened: `false`",
            "- Deployment: state-conditioned neural policy plus deterministic constraint shield",
            "- External mathematical-programming solver: `none`",
            "",
            "## Reused seams",
            "",
        ]
        for name, symbol in REUSE_ANNOTATIONS.items():
            lines.append(f"- `{name}`: `{symbol}`")
        if state.get("delegated"):
            lines.extend(
                [
                    "",
                    "> Selection evaluation was delegated to the audited constraint-shielded runtime.",
                ]
            )
        return "\n".join(lines) + "\n"

    def _delegate_selection(
        self,
        dataset: R3DailyBundleDataset,
        physical: FrozenPhysicalArtifact,
        inputs: Mapping[str, Any],
        scope: Mapping[str, Any],
    ) -> dict[str, Any]:
        fingerprint = self._fingerprint(inputs, scope)
        existing = self._read_state()
        if existing is not None:
            if existing.get("fingerprint") != fingerprint:
                raise TrainingContractError("selection evaluation resume fingerprint mismatch")
            if existing.get("status") == "COMPLETE":
                return existing
        result: Mapping[str, Any] = {}
        delegated = self.selection_evaluator is None
        if self.selection_evaluator is not None:
            returned = self.selection_evaluator(
                model=self.model,
                dataset=dataset,
                physical=physical,
                run_dir=self.run_dir,
                split=self.split,
            )
            if returned is not None:
                result = returned
        terminal_status = str(result.get("status", "BLOCKED"))
        if self.selection_evaluator is None:
            terminal_status = "BLOCKED"
            result = {
                "status": "BLOCKED",
                "reason": "no formal selection evaluator was supplied",
            }
        elif terminal_status not in {"COMPLETE", "PARTIAL", "BLOCKED", "FAILED"}:
            terminal_status = "FAILED"
        numeric = {
            str(key): float(value)
            for key, value in result.items()
            if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
        }
        row = {
            "phase": self.phase,
            "split": self.split,
            "epoch": 0,
            "event": "selection_evaluation",
            "delegated": delegated,
            "locked_2025_opened": False,
            **numeric,
            **collect_device_metrics(self.device),
        }
        metrics = pd.DataFrame([row])
        _atomic_parquet(self.metrics_path, metrics)
        state = {
            "schema_version": "dcvpp.e2e_moe_r3.training_state.v1",
            "status": terminal_status,
            "phase": self.phase,
            "split": self.split,
            "fingerprint": fingerprint,
            "completed_epochs": 0,
            "delegated": delegated,
            "evaluation_result": _plain(result),
            "locked_2025_opened": False,
            "updated_at_utc": utc_now(),
            "reuse": dict(REUSE_ANNOTATIONS),
        }
        self._write_state(state)
        atomic_text(self.summary_path, self._summary(state, metrics))
        return state

    def _losses(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        rollout_losses: Mapping[str, Tensor] | None = None,
        rollout_metadata: Mapping[str, Any] | None = None,
        rollout_aux: Mapping[str, Tensor] | None = None,
        baseline_aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        targets = batch["targets"]
        metadata: dict[str, Any] = {}
        if self.phase == "representation":
            losses = _forecast_losses(outputs, targets, _FORECAST_ALIASES)
        elif self.phase == "scenario":
            losses, metadata = _scenario_losses(outputs, batch["scenario_target"])
            losses.update(
                _decision_native_losses(outputs, include_routing=True)
            )
        elif self.phase == "policy":
            losses = _decision_native_losses(outputs, include_routing=True)
            if rollout_losses is None:
                raise TrainingContractError(
                    "policy training requires a causal solver-free rollout"
                )
            losses.update(rollout_losses)
            # The rolling value head is trained against the realized chunk
            # target inside ``rollout_losses``; remove the static latent proxy
            # to avoid overwriting that physically grounded objective.
            losses.pop("decision_value_consistency", None)
            losses.update(rollout_losses)
        elif self.phase == "a1":
            losses = _forecast_losses(outputs, targets, ("dart",))
            scenario_losses, metadata = _scenario_losses(
                outputs, batch["scenario_target"]
            )
            # The DART-specific scenario adapter is trainable only in A1.  The
            # joint score supplies it a real distributional gradient while all
            # non-DART scenario parameters remain frozen by the exact scope.
            losses.update(scenario_losses)
            if rollout_losses is None or rollout_aux is None or baseline_aux is None:
                raise TrainingContractError(
                    "A1 requires current and frozen-policy causal rollouts"
                )
            losses.update(rollout_losses)
            current_cost = rollout_aux["daily_cost"]
            baseline_cost = baseline_aux["daily_cost"].detach()
            baseline_scale = baseline_cost.abs().mean().clamp_min(100.0)
            margin = 0.01 * baseline_cost.abs().mean().detach()
            losses["a1_baseline_consistency"] = F.smooth_l1_loss(
                rollout_aux["action_signature"] / baseline_scale.sqrt(),
                baseline_aux["action_signature"].detach() / baseline_scale.sqrt(),
            )
            losses["a1_noninferiority"] = F.relu(
                current_cost - baseline_cost - margin
            ).mean() / baseline_scale
            metadata.update(
                {
                    "a1_baseline_daily_cost_usd": float(
                        baseline_cost.mean().item()
                    ),
                    "a1_cost_delta_usd": float(
                        (current_cost.detach() - baseline_cost).mean().item()
                    ),
                    "a1_noninferiority_margin_usd": float(margin.item()),
                }
            )
        elif self.phase == "a2":
            losses = _forecast_losses(outputs, targets, _FORECAST_ALIASES)
            scenario_losses, metadata = _scenario_losses(
                outputs, batch["scenario_target"]
            )
            losses.update(scenario_losses)
            losses.update(
                _decision_native_losses(outputs, include_routing=True)
            )
            if rollout_losses is None:
                raise TrainingContractError(
                    "A2 training requires a causal solver-free rollout"
                )
            losses.update(rollout_losses)
        else:
            raise TrainingContractError(f"phase {self.phase} has no training loss")
        if not losses:
            raise TrainingContractError(
                f"phase {self.phase} found no compatible differentiable outputs"
            )
        if rollout_metadata is not None:
            metadata.update(rollout_metadata)
        return losses, metadata

    def _loss_weights(self) -> dict[str, float]:
        """Resolve scientific objective weights from the solver-free config."""

        configured = _nested_get(self.config, "training", "loss_weights", default={})
        if configured is None:
            configured = {}
        if not isinstance(configured, Mapping):
            raise TrainingContractError("training.loss_weights must be a mapping")
        weights = {str(name): float(value) for name, value in configured.items()}
        if self.phase == "policy":
            policy = _nested_get(
                self.config, "training", "policy_training", "losses", default={}
            )
            if isinstance(policy, Mapping):
                aliases = {
                    "decision_realized_cost": "realized_settlement",
                    "decision_expected_particle_cost": "expected_particle_cost",
                    "decision_cvar": "particle_cvar",
                    "decision_sla": "sla",
                    "decision_battery_degradation": "battery_degradation",
                    "decision_migration": "migration",
                    "decision_constraint_intervention": "constraint_intervention",
                    "decision_confidence_calibration": "confidence_calibration",
                    "decision_value_consistency": "value_consistency",
                    "router_balance": "router_balance",
                    "router_stability": "router_stability",
                }
                weights.update(
                    {
                        loss_name: float(policy[config_name])
                        for loss_name, config_name in aliases.items()
                        if config_name in policy
                    }
                )
        elif self.phase == "a1":
            a1 = _nested_get(self.config, "training", "a1", "losses", default={})
            if isinstance(a1, Mapping):
                supervised = float(a1.get("supervised", 1.0))
                for name in (
                    "forecast_dart",
                    "scenario_marginal",
                    "scenario_energy_score",
                    "scenario_variogram",
                    "scenario_tail_coverage",
                    "scenario_diversity",
                    "scenario_weight_entropy_gap",
                ):
                    weights[name] = supervised
                decision = float(a1.get("realized_decision", 1.0))
                for name in (
                    "decision_realized_cost",
                    "decision_expected_particle_cost",
                    "decision_cvar",
                    "decision_sla",
                    "decision_feasibility",
                    "decision_constraint_intervention",
                ):
                    weights[name] = decision
                weights["a1_baseline_consistency"] = float(
                    a1.get("baseline_consistency", 0.5)
                )
                weights["a1_noninferiority"] = float(a1.get("noninferiority", 1.0))
        elif self.phase == "a2":
            risk = _nested_get(self.config, "risk", default={})
            if isinstance(risk, Mapping):
                weights.update(
                    {
                        "decision_realized_cost": float(
                            risk.get("expected_cost_weight", 1.0)
                        ),
                        "decision_expected_particle_cost": float(
                            risk.get("expected_cost_weight", 1.0)
                        ),
                        "decision_cvar": float(risk.get("cvar_weight", 0.25)),
                        "decision_sla": float(risk.get("sla_weight", 10.0)),
                        "decision_migration": float(
                            risk.get("migration_weight", 1.0)
                        ),
                        "decision_feasibility": float(
                            risk.get("feasibility_weight", 20.0)
                        ),
                        "decision_confidence_calibration": float(
                            risk.get("feasibility_weight", 20.0)
                        ),
                    }
                )
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise TrainingContractError("all loss weights must be finite and nonnegative")
        return weights

    def _curriculum(self) -> tuple[int, ...]:
        if self.phase not in {"policy", "a1", "a2"}:
            return ()
        phase_config = self._phase_config()
        formal = self._formal_config()
        value = (
            formal.get("curriculum", (8, 24, 48, 96))
            if self._is_formal()
            else phase_config.get(
                "short_horizon_curriculum",
                _nested_get(
                    self.config,
                    "training",
                    "short_horizon_curriculum",
                    default=(8,),
                ),
            )
        )
        if isinstance(value, (int, float)):
            result = (int(value),)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = tuple(int(item) for item in value)
        else:
            raise TrainingContractError("short_horizon_curriculum must be an integer sequence")
        if not result or any(item < 1 or item > INTERVALS for item in result):
            raise TrainingContractError("rollout curriculum horizons must lie in [1,96]")
        if self._is_formal() and result != (8, 24, 48, 96):
            raise TrainingContractError(
                "formal decision curriculum is fixed to [8,24,48,96]"
            )
        return result

    def _rollout_horizon(self, epoch: int, target_epochs: int) -> int | None:
        curriculum = self._curriculum()
        if not curriculum:
            return None
        if self._is_formal():
            # Equal-width stages, with a hard final 96-origin gradient stage.
            index = min(len(curriculum) - 1, (epoch * len(curriculum)) // target_epochs)
            return INTERVALS if epoch == target_epochs - 1 else curriculum[index]
        return curriculum[min(epoch, len(curriculum) - 1)]

    def _batch_fallback(self) -> tuple[int, ...]:
        if not self._is_formal() and not bool(
            _nested_get(
                self.config, "training", "enable_batched_pilot", default=False
            )
        ):
            return (1,)
        formal = self._formal_config()
        microbatch = int(
            formal.get(
                "microbatch_size",
                _nested_get(self.config, "training", "microbatch_size", default=512),
            )
        )
        value = formal.get("batch_size_fallback")
        if value is None:
            value = _nested_get(
                self.config, "training", "batch_size_fallback", default=None
            )
        if value in (None, (4, 2, 1), [4, 2, 1]):
            # The old pilot-only setting is deliberately upgraded in formal
            # mode; it is much too small for complete-date neural training.
            value = (512, 256, 128, 64, 32, 4, 2, 1)
        if isinstance(value, int):
            result = (int(value),)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = tuple(int(item) for item in value)
        else:
            raise TrainingContractError("batch_size_fallback must be an integer sequence")
        if not result or any(item < 1 for item in result):
            raise TrainingContractError("batch_size_fallback values must be positive")
        if self._is_formal():
            if result[0] != microbatch or result[-3:] != (4, 2, 1):
                raise TrainingContractError(
                    "formal batch fallback must begin at microbatch_size and end 4->2->1"
                )
            if any(left <= right for left, right in zip(result, result[1:])):
                raise TrainingContractError(
                    "formal batch fallback must be strictly decreasing"
                )
        return result

    def _gradient_accumulation(self, active_batch_size: int | None = None) -> int:
        if not self._is_formal() and not bool(
            _nested_get(
                self.config, "training", "enable_batched_pilot", default=False
            )
        ):
            return 1
        if self._is_formal():
            target = int(
                self._formal_config().get(
                    "microbatch_size",
                    _nested_get(
                        self.config, "training", "microbatch_size", default=512
                    ),
                )
            )
            batch = target if active_batch_size is None else int(active_batch_size)
            if target < 1 or batch < 1:
                raise TrainingContractError("formal effective/microbatch size must be positive")
            return max(1, math.ceil(target / batch))
        value = self._formal_config().get(
            "gradient_accumulation",
            _nested_get(self.config, "training", "gradient_accumulation", default=4),
        )
        result = int(value)
        if result < 1:
            raise TrainingContractError("gradient_accumulation must be positive")
        return result

    def _early_stopping(self) -> tuple[int, float]:
        phase_defaults = {
            "representation": 5,
            "scenario": 5,
            "policy": 7,
            "a1": 5,
            "a2": 7,
        }
        phase_config = self._phase_config()
        patience = int(
            phase_config.get(
                "early_stopping_patience",
                self._formal_config().get(
                    "early_stopping_patience", phase_defaults.get(self.phase, 5)
                ),
            )
        )
        min_delta = float(
            phase_config.get(
                "early_stopping_min_delta",
                self._formal_config().get("early_stopping_min_delta", 1e-6),
            )
        )
        if patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
            raise TrainingContractError("invalid early-stopping configuration")
        return patience, min_delta

    def _resume_command(self) -> str:
        configured = _nested_get(self.config, "protocol", "resume_command", default=None)
        if configured:
            return str(configured)
        config_path = str(
            _nested_get(
                self.config,
                "protocol",
                "config_path",
                default="configs/e2e_moe_dart_vpp_R3_solver_free.yaml",
            )
        )
        return (
            f'python scripts/run_e2e_moe_r3.py train --phase {self.phase} '
            f'--config "{config_path}" --resume'
        )

    def _compute_batch(
        self,
        batch: Mapping[str, Any],
        physical: FrozenPhysicalArtifact,
        *,
        rollout_horizon: int | None,
        baseline_model: nn.Module | None,
        trainable: Sequence[nn.Parameter],
        check_a1_path: bool,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
        model_batch = {
            key: batch[key] for key in ("market", "workload", "physical", "state")
        }
        outputs = self.model(model_batch)
        if not isinstance(outputs, Mapping):
            raise TrainingContractError("model forward must return a mapping")
        rollout_losses: dict[str, Tensor] | None = None
        rollout_metadata: dict[str, Any] | None = None
        rollout_aux: dict[str, Tensor] | None = None
        baseline_aux: dict[str, Tensor] | None = None
        if rollout_horizon is not None:
            rollout_losses, rollout_metadata, rollout_aux = _solver_free_rollout_losses(
                self.model,
                model_batch,
                batch,
                physical,
                horizon=rollout_horizon,
                # Prefix-zero keeps every chunk causal and makes the final stage
                # exactly the complete 96-origin operating day.
                start_origin=0,
            )
            if baseline_model is not None:
                with torch.no_grad():
                    _, _, baseline_aux = _solver_free_rollout_losses(
                        baseline_model,
                        model_batch,
                        batch,
                        physical,
                        horizon=rollout_horizon,
                        start_origin=0,
                    )
        losses, metadata = self._losses(
            outputs,
            batch,
            rollout_losses=rollout_losses,
            rollout_metadata=rollout_metadata,
            rollout_aux=rollout_aux,
            baseline_aux=baseline_aux,
        )
        if check_a1_path:
            assert rollout_losses is not None
            path_gradients = torch.autograd.grad(
                sum(rollout_losses.values()),
                tuple(trainable),
                retain_graph=True,
                allow_unused=True,
            )
            names = {
                id(parameter): name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            missing = [
                names[id(parameter)]
                for parameter, gradient in zip(trainable, path_gradients)
                if gradient is None
                or not bool(torch.isfinite(gradient).all())
                or not bool((gradient.abs() > 0).any())
            ]
            if missing:
                raise TrainingContractError(
                    "A1 decision objective has no finite nonzero path to the exact "
                    f"trainable scope: {missing}"
                )
            metadata["a1_decision_gradient_parameter_names_json"] = json.dumps(
                [names[id(parameter)] for parameter in trainable],
                ensure_ascii=True,
                separators=(",", ":"),
            )
        weights = self._loss_weights()
        total = sum(loss * float(weights.get(name, 1.0)) for name, loss in losses.items())
        if not bool(torch.isfinite(total)):
            raise TrainingContractError("non-finite training loss")
        return total, losses, metadata

    def _validation_epoch(
        self,
        dataset: R3DailyBundleDataset,
        physical: FrozenPhysicalArtifact,
        *,
        rollout_horizon: int | None,
        batch_size: int,
        baseline_model: nn.Module | None = None,
        run_hard_validation: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate without gradients or state mutation; never select on test data."""

        was_training = self.model.training
        self.model.eval()
        sums: dict[str, float] = {}
        batches = 0
        samples = 0
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                raw = _collate_days(
                    [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
                )
                if self.device.type == "cuda":
                    raw = _pin_batch(raw)
                batch = _move_batch(raw, self.device)
                total, losses, _ = self._compute_batch(
                    batch,
                    physical,
                    rollout_horizon=rollout_horizon,
                    baseline_model=baseline_model,
                    trainable=(),
                    check_a1_path=False,
                )
                count = int(batch["market"].shape[0])
                sums["validation_loss"] = sums.get("validation_loss", 0.0) + float(
                    total.item()
                ) * count
                for name, loss in losses.items():
                    sums[f"validation_{name}"] = sums.get(
                        f"validation_{name}", 0.0
                    ) + float(loss.item()) * count
                batches += 1
                samples += count
        if batches < 1 or samples < 1:
            raise TrainingContractError("validation dataset produced no batches")
        metrics = {name: value / samples for name, value in sums.items()}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise TrainingContractError("non-finite validation metric")
        selection: Mapping[str, Any] | None = None
        # Deployment checkpoint eligibility is meaningful only for the formal
        # 96-origin stage.  Keep the model in eval mode here so noisy Top-2
        # routing cannot make checkpoint selection stochastic.
        if run_hard_validation and self.validation_evaluator is not None and (
            not self._is_formal()
            or self.phase not in {"policy", "a1", "a2"}
            or rollout_horizon == INTERVALS
        ):
            selection = self.validation_evaluator(
                model=self.model,
                dataset=dataset,
                physical=physical,
                phase=self.phase,
                device=self.device,
            )
        if self.phase in {"policy", "a1", "a2"} and self._is_formal():
            if selection is None:
                # Generic differentiable validation loss is logged, but cannot
                # masquerade as deployment feasibility/checkpoint eligibility.
                metrics["checkpoint_eligible"] = False
                if run_hard_validation:
                    metrics["checkpoint_ineligible_reason"] = (
                        "formal decision checkpoint requires validation_evaluator hard-shield metrics"
                    )
                else:
                    metrics["checkpoint_selection_stage"] = (
                        "provisional_cheap_validation"
                    )
            else:
                feasibility = float(selection.get("post_shield_feasibility_fraction", 0.0))
                sla_ok = bool(selection.get("sla_noninferiority_passed", False))
                cost = float(selection.get("operating_cost_usd", math.nan))
                cvar = float(selection.get("cvar_95_usd", math.nan))
                correction = float(selection.get("corrective_magnitude", math.nan))
                decision_metrics_finite = all(
                    math.isfinite(value) for value in (cost, cvar, correction)
                )
                eligible = feasibility == 1.0 and sla_ok and decision_metrics_finite
                metrics.update(
                    {
                        "checkpoint_eligible": eligible,
                        "validation_post_shield_feasibility_fraction": feasibility,
                        "validation_sla_noninferiority_passed": sla_ok,
                        "validation_operating_cost_usd": cost if math.isfinite(cost) else None,
                        "validation_cvar_95_usd": cvar if math.isfinite(cvar) else None,
                        "validation_corrective_magnitude": (
                            correction if math.isfinite(correction) else None
                        ),
                    }
                )
        else:
            metrics["checkpoint_eligible"] = True
        if was_training:
            self.model.train()
        return float(metrics["validation_loss"]), metrics

    def _is_phase_end_hard_gate(self) -> bool:
        return self._is_formal() and self.phase in {"policy", "a1", "a2"}

    def _candidate_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"candidate_epoch={int(epoch):03d}.pt"

    def _recover_hard_gate_candidates(
        self, state: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Reconcile atomic candidate files after an epoch-boundary interruption."""

        fingerprint = state.get("fingerprint")
        discovered: dict[int, dict[str, Any]] = {}
        raw = state.get("candidate_registry", [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                if isinstance(item, Mapping) and "epoch" in item:
                    discovered[int(item["epoch"])] = dict(item)

        # A crash can occur after latest.pt is atomically replaced but before
        # its K=96 candidate copy and registry entry are committed. Recreate
        # that copy before scanning the candidate directory.
        if self.checkpoint_path.exists():
            latest = _torch_load(self.checkpoint_path, map_location="cpu")
            if (
                isinstance(latest, Mapping)
                and latest.get("fingerprint") == fingerprint
                and latest.get("full_96_origin_gradient_epoch") is True
            ):
                epoch = int(latest.get("completed_epochs", 0))
                path = self._candidate_path(epoch)
                if epoch > 0 and not path.exists():
                    _atomic_torch_save(
                        path,
                        {
                            **latest,
                            "checkpoint_role": (
                                "provisional_k96_hard_gate_candidate"
                            ),
                        },
                    )

        for path in sorted(self.checkpoint_dir.glob("candidate_epoch=*.pt")):
            payload = _torch_load(path, map_location="cpu")
            if (
                not isinstance(payload, Mapping)
                or payload.get("fingerprint") != fingerprint
                or payload.get("full_96_origin_gradient_epoch") is not True
            ):
                continue
            epoch = int(payload.get("completed_epochs", 0))
            if epoch < 1:
                continue
            discovered[epoch] = {
                "epoch": epoch,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": sha256_file(path),
                "validation_loss": payload.get("validation_loss"),
            }
        return [discovered[epoch] for epoch in sorted(discovered)]

    @staticmethod
    def _hard_gate_result(
        selection: Mapping[str, Any],
        *,
        epoch: int,
        checkpoint: Path,
    ) -> dict[str, Any]:
        independent = selection.get("independent_validation", {})
        independently_valid = bool(
            isinstance(independent, Mapping)
            and independent.get("all_passed") is True
        )
        feasibility = float(
            selection.get("post_shield_feasibility_fraction", 0.0)
        )
        sla_ok = bool(selection.get("sla_noninferiority_passed", False))
        cost = float(selection.get("operating_cost_usd", math.nan))
        cvar = float(selection.get("cvar_95_usd", math.nan))
        correction = float(selection.get("corrective_magnitude", math.nan))
        finite = all(math.isfinite(value) for value in (cost, cvar, correction))
        eligible = bool(
            selection.get("status") == "COMPLETE"
            and independently_valid
            and feasibility == 1.0
            and sla_ok
            and finite
        )
        return {
            "epoch": int(epoch),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "status": str(selection.get("status", "FAILED")),
            "independent_validation_all_passed": independently_valid,
            "post_shield_feasibility_fraction": feasibility,
            "sla_noninferiority_passed": sla_ok,
            "operating_cost_usd": cost if math.isfinite(cost) else None,
            "cvar_95_usd": cvar if math.isfinite(cvar) else None,
            "corrective_magnitude": correction if math.isfinite(correction) else None,
            "eligible": eligible,
            "validation_day_count": selection.get("validation_day_count"),
            "completed_day_count": selection.get("completed_day_count"),
            "independent_validation": (
                dict(independent) if isinstance(independent, Mapping) else {}
            ),
        }

    def _hard_gate_metric_row(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "split": self.split,
            "epoch": int(result["epoch"]),
            "event": "formal_hard_gate_candidate",
            "checkpoint_eligible": bool(result["eligible"]),
            **{
                f"validation_{key}": value
                for key, value in result.items()
                if key
                in {
                    "post_shield_feasibility_fraction",
                    "sla_noninferiority_passed",
                    "operating_cost_usd",
                    "cvar_95_usd",
                    "corrective_magnitude",
                }
            },
        }

    @staticmethod
    def _hard_gate_better(
        candidate: Mapping[str, Any],
        incumbent: Mapping[str, Any] | None,
        *,
        min_delta: float,
    ) -> bool:
        if not bool(candidate.get("eligible", False)):
            return False
        if incumbent is None:
            return True
        candidate_score = (
            float(candidate["operating_cost_usd"]),
            float(candidate["cvar_95_usd"]),
            float(candidate["corrective_magnitude"]),
            int(candidate["epoch"]),
        )
        incumbent_score = (
            float(incumbent["operating_cost_usd"]),
            float(incumbent["cvar_95_usd"]),
            float(incumbent["corrective_magnitude"]),
            int(incumbent["epoch"]),
        )
        if candidate_score[0] < incumbent_score[0] - min_delta:
            return True
        return math.isclose(
            candidate_score[0],
            incumbent_score[0],
            rel_tol=0.0,
            abs_tol=min_delta,
        ) and candidate_score[1:] < incumbent_score[1:]

    def _run_phase_end_hard_gate(
        self,
        *,
        state: dict[str, Any],
        metrics: pd.DataFrame,
        dataset: R3DailyBundleDataset,
        physical: FrozenPhysicalArtifact,
        min_delta: float,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Hard-gate every K=96 candidate after gradient training has stopped."""

        if self.validation_evaluator is None:
            raise TrainingContractError(
                "formal decision phase requires validation_evaluator at phase-end hard gate"
            )
        candidates = self._recover_hard_gate_candidates(state)
        if not candidates:
            raise TrainingContractError(
                "formal decision phase produced no K=96 hard-gate candidate"
            )
        results_raw = state.get("hard_gate_results", [])
        results_by_epoch: dict[int, dict[str, Any]] = {}
        if isinstance(results_raw, Sequence) and not isinstance(
            results_raw, (str, bytes)
        ):
            for item in results_raw:
                if isinstance(item, Mapping) and "epoch" in item:
                    results_by_epoch[int(item["epoch"])] = dict(item)
        results = list(results_by_epoch.values())
        candidate_by_epoch = {int(item["epoch"]): item for item in candidates}
        results = [
            item
            for item in results
            if int(item.get("epoch", -1)) in candidate_by_epoch
            and item.get("checkpoint_sha256")
            == candidate_by_epoch[int(item["epoch"])].get("checkpoint_sha256")
        ]
        results.sort(key=lambda item: int(item["epoch"]))
        completed_epochs = {int(item["epoch"]) for item in results}
        state["stage"] = "HARD_GATE"
        state["candidate_registry"] = candidates
        state["hard_gate_candidate_count"] = len(candidates)
        state["hard_gate_results"] = results
        state["hard_gate_next_index"] = len(results)
        state["updated_at_utc"] = utc_now()
        self._write_state(state)

        # State is committed before Parquet for safe gate progress. Reconcile
        # the narrow crash window where a completed result exists in state but
        # its metrics row was not yet atomically replaced.
        metric_epochs: set[int] = set()
        if not metrics.empty and {"event", "epoch"}.issubset(metrics.columns):
            metric_epochs = {
                int(value)
                for value in metrics.loc[
                    metrics["event"] == "formal_hard_gate_candidate", "epoch"
                ].dropna()
            }
        recovered_rows = [
            self._hard_gate_metric_row(result)
            for result in results
            if int(result["epoch"]) not in metric_epochs
        ]
        if recovered_rows:
            metrics = pd.concat(
                [metrics, pd.DataFrame(recovered_rows)], ignore_index=True
            )
            _atomic_parquet(self.metrics_path, metrics)

        for index, candidate in enumerate(candidates):
            epoch = int(candidate["epoch"])
            if epoch in completed_epochs:
                continue
            path = Path(str(candidate["checkpoint"])).resolve()
            if not path.exists() or sha256_file(path) != candidate.get(
                "checkpoint_sha256"
            ):
                raise TrainingContractError(
                    f"hard-gate candidate is missing or changed: {path}"
                )
            payload = _torch_load(path, map_location=self.device)
            if not isinstance(payload, Mapping) or payload.get("fingerprint") != state.get(
                "fingerprint"
            ):
                raise TrainingContractError(
                    f"hard-gate candidate payload is invalid: {path}"
                )
            self.model.load_state_dict(payload["model_state_dict"])
            self.model.to(self.device).eval()
            selection = self.validation_evaluator(
                model=self.model,
                dataset=dataset,
                physical=physical,
                phase=self.phase,
                device=self.device,
            )
            if not isinstance(selection, Mapping):
                raise TrainingContractError(
                    "validation_evaluator hard gate must return a mapping"
                )
            result = self._hard_gate_result(
                selection, epoch=epoch, checkpoint=path
            )
            results.append(result)
            results.sort(key=lambda item: int(item["epoch"]))
            completed_epochs.add(epoch)
            state["hard_gate_results"] = results
            state["hard_gate_next_index"] = index + 1
            state["updated_at_utc"] = utc_now()
            self._write_state(state)
            gate_row = self._hard_gate_metric_row(result)
            metrics = pd.concat([metrics, pd.DataFrame([gate_row])], ignore_index=True)
            _atomic_parquet(self.metrics_path, metrics)

        best: Mapping[str, Any] | None = None
        for result in results:
            if self._hard_gate_better(result, best, min_delta=min_delta):
                best = result
        if best is None:
            raise TrainingContractError(
                "formal decision phase has no feasible SLA-noninferior hard-gate candidate"
            )
        selected_path = Path(str(best["checkpoint"])).resolve()
        selected_payload = _torch_load(selected_path, map_location="cpu")
        if not isinstance(selected_payload, Mapping):
            raise TrainingContractError("selected hard-gate checkpoint is invalid")
        selected_score = [
            float(best["operating_cost_usd"]),
            float(best["cvar_95_usd"]),
            float(best["corrective_magnitude"]),
        ]
        _atomic_torch_save(
            self.best_checkpoint_path,
            {
                **selected_payload,
                "checkpoint_role": "hard_gate_selected_best",
                "best_epoch": int(best["epoch"]),
                "best_validation_loss": selected_payload.get("validation_loss"),
                "best_validation_decision_score": selected_score,
                "hard_gate_result": dict(best),
            },
        )
        state["best_epoch"] = int(best["epoch"])
        state["best_validation_loss"] = selected_payload.get(
            "validation_loss", state.get("best_validation_loss")
        )
        state["best_validation_decision_score"] = selected_score
        state["best_checkpoint_sha256"] = sha256_file(self.best_checkpoint_path)
        state["hard_gate_next_index"] = len(candidates)
        state["hard_gate_complete"] = True
        state["updated_at_utc"] = utc_now()
        self._write_state(state)
        return state, metrics

    def run(self) -> dict[str, Any]:
        """Execute or resume the configured phase and return its final state."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        allowed_existing = {
            "config_used.yaml",  # owned by the CLI, not written here
            "runtime.log",  # owned by the audited rolling runtime
            self.state_path.name,
            self.metrics_path.name,
            self.summary_path.name,
            self.checkpoint_dir.name,
        }
        unexpected = sorted(
            path.name for path in self.run_dir.iterdir() if path.name not in allowed_existing
        )
        if unexpected:
            raise TrainingContractError(
                f"run directory contains artifacts outside the R3 contract: {unexpected}"
            )

        if not self.resume and (
            self.state_path.exists()
            or self.metrics_path.exists()
            or self.checkpoint_path.exists()
        ):
            raise TrainingContractError(
                "resume=False refuses to overwrite an initialized run directory"
            )

        scope = configure_phase_parameters(self.model, self.phase)
        dataset, validation_dataset, physical, input_annotation = self._build_inputs()
        if self._is_formal():
            # Formal panels are fixed and fit comfortably in host RAM. Decode
            # them once so the GPU is not starved by 722 repeated NPZ reads per
            # epoch (543 train + 179 validation).
            dataset.preload()
            if validation_dataset is not None:
                validation_dataset.preload()
        predecessor_path, predecessor = self._predecessor_checkpoint()
        if predecessor.get("required") and predecessor_path is None:
            return self._blocked_state(
                reason=str(predecessor["reason"]),
                inputs=input_annotation,
                scope=scope,
                predecessor=predecessor,
            )
        if predecessor_path is not None:
            predecessor_payload = _torch_load(predecessor_path, map_location="cpu")
            if not isinstance(predecessor_payload, Mapping) or "model_state_dict" not in predecessor_payload:
                return self._blocked_state(
                    reason=f"invalid predecessor checkpoint: {predecessor_path}",
                    inputs=input_annotation,
                    scope=scope,
                    predecessor=predecessor,
                )
            self.model.load_state_dict(predecessor_payload["model_state_dict"])
        if self.phase == "selection_evaluation":
            self.model.to(self.device)
            self.model.eval()
            return self._delegate_selection(dataset, physical, input_annotation, scope)

        fingerprint = self._fingerprint(input_annotation, scope, predecessor)
        existing = self._read_state()
        if existing is not None:
            if existing.get("phase") != self.phase or existing.get("split") != self.split:
                raise TrainingContractError("run directory belongs to a different phase/split")
            if existing.get("fingerprint") != fingerprint:
                raise TrainingContractError("resume fingerprint mismatch")
            if existing.get("status") == "COMPLETE":
                return existing

        seed = self._seed()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model.to(self.device)
        baseline_model: nn.Module | None = None
        if self.phase == "a1":
            # This immutable copy is the pre-A1 solver-free policy baseline.
            # It supplies paired behavior/cost targets without any teacher or
            # external optimizer.
            baseline_model = copy.deepcopy(self.model).to(self.device)
            baseline_model.eval()
            for parameter in baseline_model.parameters():
                parameter.requires_grad_(False)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        phase_config = self._phase_config()
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(phase_config.get("learning_rate", 1e-3)),
            weight_decay=float(phase_config.get("weight_decay", 1e-4)),
        )
        completed_epochs = int(existing.get("completed_epochs", 0)) if existing else 0
        best_epoch = int(existing.get("best_epoch", 0)) if existing else 0
        existing_best_loss = (
            existing.get("best_validation_loss") if existing else None
        )
        best_validation_loss = (
            float(existing_best_loss)
            if existing_best_loss is not None
            else math.inf
        )
        patience_counter = int(existing.get("early_stopping_counter", 0)) if existing else 0
        active_batch_size = int(
            existing.get("active_batch_size", self._batch_fallback()[0])
            if existing
            else self._batch_fallback()[0]
        )
        if existing is not None and existing.get("status") not in {
            "RUNNING",
            "FAILED",
            "PARTIAL",
        }:
            raise TrainingContractError(
                f"cannot resume non-terminal training state {existing.get('status')!r}"
            )
        if completed_epochs and not self.checkpoint_path.exists():
            raise TrainingContractError("state requests resume but checkpoints/latest.pt is missing")
        if self.checkpoint_path.exists():
            checkpoint = _torch_load(self.checkpoint_path, map_location=self.device)
            if not isinstance(checkpoint, Mapping) or checkpoint.get("fingerprint") != fingerprint:
                raise TrainingContractError("checkpoint fingerprint mismatch")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            completed_epochs = int(checkpoint.get("completed_epochs", completed_epochs))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            best_validation_loss = float(
                checkpoint.get("best_validation_loss", best_validation_loss)
            )
            patience_counter = int(
                checkpoint.get("early_stopping_counter", patience_counter)
            )
            active_batch_size = int(
                checkpoint.get("active_batch_size", active_batch_size)
            )
            saved_rng = checkpoint.get("rng_state")
            if isinstance(saved_rng, Mapping):
                _restore_rng_state(saved_rng)
        else:
            checkpoint = None

        target_epochs = self._epochs()
        patience, min_delta = self._early_stopping()
        curriculum = self._curriculum()
        protocol_seeds = _nested_get(
            self.config, "protocol", "seeds", default=[seed]
        )
        split_manifest_sha256 = input_annotation.get("split_manifest_sha256")

        phase_end_hard_gate = self._is_phase_end_hard_gate()
        resume_hard_gate = bool(
            phase_end_hard_gate
            and existing is not None
            and existing.get("stage") == "HARD_GATE"
        )
        state = {
            "schema_version": "dcvpp.e2e_moe_r3.training_state.v2",
            "status": "RUNNING",
            "training_regime": "formal" if self._is_formal() else "pilot",
            "experiment_id": self._experiment_id(),
            "method": self._method(),
            "phase": self.phase,
            "split": self.split,
            "fingerprint": fingerprint,
            "config_protocol_fingerprint": self._protocol_fingerprint(),
            "protocol_fingerprint": self._protocol_fingerprint(),
            "training_semantics_revision": _nested_get(
                self.config,
                "protocol",
                "training_semantics_revision",
                default="legacy_unversioned",
            ),
            "split_manifest_sha256": split_manifest_sha256,
            "protocol_splits": self._protocol_splits() if self._is_formal() else None,
            "completed_epochs": completed_epochs,
            "target_epochs": target_epochs,
            "seed": seed,
            "active_seed": seed,
            "protocol_seeds": _plain(protocol_seeds),
            "device": str(self.device),
            "scope": scope,
            "inputs": input_annotation,
            "predecessor": predecessor,
            "predecessor_checkpoint": predecessor.get("checkpoint"),
            "predecessor_checkpoint_sha256": predecessor.get("checkpoint_sha256"),
            "predecessor_seed": predecessor.get("seed"),
            "external_solver": False,
            "locked_2025_opened": False,
            "formal_2024_opened": False,
            "curriculum": list(curriculum),
            "curriculum_stage_horizons": [
                self._rollout_horizon(epoch, target_epochs)
                for epoch in range(target_epochs)
            ],
            "batch_size_fallback": list(self._batch_fallback()),
            "active_batch_size": active_batch_size,
            "gradient_accumulation": self._gradient_accumulation(active_batch_size),
            "target_effective_batch_size": self._batch_fallback()[0],
            "early_stopping_target": "validation_loss",
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "early_stopping_counter": patience_counter,
            "best_epoch": best_epoch,
            "best_validation_loss": (
                None if not math.isfinite(best_validation_loss) else best_validation_loss
            ),
            "stage": "HARD_GATE" if resume_hard_gate else "TRAINING",
            "checkpoint_selection_strategy": (
                "phase_end_all_k96_hard_gate"
                if phase_end_hard_gate
                else "per_epoch_validation_loss"
            ),
            "candidate_registry": (
                list(existing.get("candidate_registry", [])) if existing else []
            ),
            "hard_gate_results": (
                list(existing.get("hard_gate_results", [])) if existing else []
            ),
            "hard_gate_next_index": int(
                existing.get("hard_gate_next_index", 0) if existing else 0
            ),
            "latest_checkpoint": str(self.checkpoint_path),
            "best_checkpoint": str(self.best_checkpoint_path),
            "resumable": True,
            "finite_metrics_verified": bool(
                existing.get("finite_metrics_verified", False) if existing else False
            ),
            "finite_nonzero_gradients_verified": bool(
                existing.get("finite_nonzero_gradients_verified", False)
                if existing
                else False
            ),
            "checkpoint_finite_verified": bool(
                existing.get("checkpoint_finite_verified", False) if existing else False
            ),
            "full_96_origin_gradient_epoch_completed": bool(
                existing.get("full_96_origin_gradient_epoch_completed", False)
                if existing
                else False
            ),
            "resume_command": self._resume_command(),
            "updated_at_utc": utc_now(),
        }
        self._write_state(state)
        metrics = self._metrics()
        frozen_before = _snapshot_frozen(self.model) if self.phase in {"a1", "a2"} else {}

        try:
            stop_early = bool(existing.get("early_stopped", False)) if existing else False
            training_end = completed_epochs if resume_hard_gate else target_epochs
            for epoch in range(completed_epochs, training_end):
                indices = list(range(len(dataset)))
                random.Random(seed + epoch).shuffle(indices)
                self.model.train()
                if self.phase == "a1":
                    # A1 is a paired non-inferiority update against a frozen
                    # policy.  Disable router exploration noise so both sides
                    # see the identical deterministic information set.
                    self.model.eval()
                monitor_before = collect_device_metrics(self.device)
                sums: dict[str, float] = {}
                batches = 0
                samples = 0
                gradient_evidence: dict[str, Any] = {}
                phase_metadata: dict[str, Any] = {}
                rollout_horizon = self._rollout_horizon(epoch, target_epochs)
                fallback_values = self._batch_fallback()
                if active_batch_size not in fallback_values:
                    active_batch_size = fallback_values[0]
                accumulation = self._gradient_accumulation(active_batch_size)
                optimizer.zero_grad(set_to_none=True)
                cursor = 0
                microbatches_since_step = 0
                samples_since_step = 0
                a1_path_checked = False
                while cursor < len(indices):
                    count = min(active_batch_size, len(indices) - cursor)
                    chosen = indices[cursor : cursor + count]
                    try:
                        raw_batch = _collate_days([dataset[index] for index in chosen])
                        if self.device.type == "cuda":
                            raw_batch = _pin_batch(raw_batch)
                        batch = _move_batch(raw_batch, self.device)
                        total, losses, batch_metadata = self._compute_batch(
                            batch,
                            physical,
                            rollout_horizon=rollout_horizon,
                            baseline_model=baseline_model,
                            trainable=trainable,
                            check_a1_path=self.phase == "a1" and not a1_path_checked,
                        )
                        # Sum per-sample means; normalize gradients by the actual
                        # number of distinct days in this accumulation group at
                        # step time, including the final partial group.
                        (total * float(count)).backward()
                    except Exception as exc:
                        if not _is_oom(exc):
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        position = fallback_values.index(active_batch_size)
                        if position + 1 >= len(fallback_values):
                            raise TrainingContractError(
                                "CUDA OOM persisted through the configured formal batch fallback"
                            ) from exc
                        active_batch_size = fallback_values[position + 1]
                        accumulation = self._gradient_accumulation(active_batch_size)
                        microbatches_since_step = 0
                        samples_since_step = 0
                        continue
                    a1_path_checked = a1_path_checked or self.phase == "a1"
                    cursor += count
                    samples += count
                    batches += 1
                    microbatches_since_step += 1
                    samples_since_step += count
                    sums["total_loss"] = sums.get("total_loss", 0.0) + float(
                        total.item()
                    ) * count
                    for name, loss in losses.items():
                        sums[name] = sums.get(name, 0.0) + float(
                            loss.detach().item()
                        ) * count
                    phase_metadata.update(batch_metadata)
                    # A formal coverage cycle has 543 days. With B512 the last
                    # 31 days must join the first microbatch's gradient before
                    # the optimizer step; otherwise the tiny tail receives an
                    # unintended second full-strength update and creates a GPU
                    # utilization cliff. OOM fallback still honors its dynamic
                    # accumulation target.
                    coverage_cycle_complete = cursor >= len(indices)
                    should_step = (
                        coverage_cycle_complete
                        if self._is_formal() and active_batch_size == fallback_values[0]
                        else microbatches_since_step >= accumulation
                        or coverage_cycle_complete
                    )
                    if should_step:
                        if samples_since_step < 1:
                            raise TrainingContractError("empty gradient accumulation group")
                        for parameter in trainable:
                            if parameter.grad is not None:
                                parameter.grad.div_(float(samples_since_step))
                        gradient_evidence = _finite_gradient_evidence(self.model)
                        clip_norm = float(
                            phase_config.get(
                                "gradient_clip_norm",
                                _nested_get(
                                    self.config,
                                    "training",
                                    "gradient_clip_norm",
                                    default=10.0,
                                ),
                            )
                        )
                        torch.nn.utils.clip_grad_norm_(trainable, max_norm=clip_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        microbatches_since_step = 0
                        samples_since_step = 0
                        assert_parameter_scope(self.model, scope)

                if batches < 1:
                    raise TrainingContractError("training DataLoader produced no batches")
                if self.phase in {"a1", "a2"}:
                    _assert_frozen_unchanged(self.model, frozen_before)
                monitor_after = collect_device_metrics(self.device)
                validation_values: dict[str, Any] = {}
                validation_loss = math.nan
                if validation_dataset is not None:
                    validation_loss, validation_values = self._validation_epoch(
                        validation_dataset,
                        physical,
                        rollout_horizon=rollout_horizon,
                        batch_size=active_batch_size,
                        baseline_model=baseline_model,
                        run_hard_validation=not phase_end_hard_gate,
                    )
                is_full_horizon_epoch = rollout_horizon == INTERVALS
                eligible = bool(validation_values.get("checkpoint_eligible", False))
                if validation_dataset is None:
                    eligible = True
                    validation_loss = sums["total_loss"] / samples
                if self._is_formal() and self.phase in {"policy", "a1", "a2"}:
                    eligible = eligible and is_full_horizon_epoch
                if phase_end_hard_gate:
                    # Cheap differentiable validation controls patience only
                    # within the comparable K=96 stage. Deployment selection
                    # is deferred to the phase-end all-candidate hard gate.
                    eligible = False
                    improved = bool(
                        is_full_horizon_epoch
                        and validation_loss < best_validation_loss - min_delta
                    )
                elif (
                    self._is_formal()
                    and self.phase in {"policy", "a1", "a2"}
                    and eligible
                ):
                    # Formal checkpoint ordering: feasibility/SLA are eligibility
                    # gates; then operating cost, with CVaR and corrective action
                    # magnitude as deterministic close-tie breakers.
                    candidate_score = (
                        float(validation_values["validation_operating_cost_usd"]),
                        float(validation_values["validation_cvar_95_usd"]),
                        float(validation_values["validation_corrective_magnitude"]),
                    )
                    prior_score_raw = state.get("best_validation_decision_score")
                    prior_score = (
                        tuple(float(value) for value in prior_score_raw)
                        if isinstance(prior_score_raw, Sequence)
                        else (math.inf, math.inf, math.inf)
                    )
                    cost_close = math.isclose(
                        candidate_score[0],
                        prior_score[0],
                        rel_tol=0.0,
                        abs_tol=min_delta,
                    )
                    improved = candidate_score[0] < prior_score[0] - min_delta or (
                        cost_close and candidate_score[1:] < prior_score[1:]
                    )
                    if improved:
                        state["best_validation_decision_score"] = list(candidate_score)
                else:
                    improved = eligible and validation_loss < best_validation_loss - min_delta
                if improved:
                    best_validation_loss = validation_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                elif is_full_horizon_epoch or self.phase not in {"policy", "a1", "a2"}:
                    # Curriculum stages before K=96 cannot consume patience.
                    patience_counter += 1
                row: dict[str, Any] = {
                    "phase": self.phase,
                    "split": self.split,
                    "epoch": epoch + 1,
                    "event": "formal_epoch" if self._is_formal() else "smoke_epoch",
                    "batch_count": batches,
                    "sample_count": samples,
                    "external_solver": False,
                    "locked_2025_opened": False,
                    "full_horizon_intervals": INTERVALS,
                    "rollout_horizon": rollout_horizon,
                    "full_96_origin_gradient_epoch": is_full_horizon_epoch,
                    "active_batch_size": active_batch_size,
                    "gradient_accumulation": accumulation,
                    "target_effective_batch_size": fallback_values[0],
                    "train_dataset_complete_coverage": samples == len(dataset),
                    "validation_dataset_count": (
                        0 if validation_dataset is None else len(validation_dataset)
                    ),
                    "best_epoch": best_epoch,
                    "best_validation_loss": (
                        None
                        if not math.isfinite(best_validation_loss)
                        else best_validation_loss
                    ),
                    "early_stopping_counter": patience_counter,
                    "checkpoint_eligible": eligible,
                    "checkpoint_improved": improved,
                    **{name: value / samples for name, value in sums.items()},
                    **validation_values,
                    **gradient_evidence,
                    **phase_metadata,
                }
                for key, value in monitor_before.items():
                    row[f"before_{key}"] = value
                for key, value in monitor_after.items():
                    row[f"after_{key}"] = value
                metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
                metrics_to_write = metrics
                _atomic_parquet(self.metrics_path, metrics_to_write)
                checkpoint_payload = {
                        "schema_version": "dcvpp.e2e_moe_r3.checkpoint.v2",
                        "phase": self.phase,
                        "split": self.split,
                        "fingerprint": fingerprint,
                        "completed_epochs": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "seed": seed,
                        "training_regime": "formal" if self._is_formal() else "pilot",
                        "experiment_id": self._experiment_id(),
                        "method": self._method(),
                        "protocol_fingerprint": self._protocol_fingerprint(),
                        "split_manifest_sha256": split_manifest_sha256,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_validation_loss,
                        "early_stopping_counter": patience_counter,
                        "active_batch_size": active_batch_size,
                        "rollout_horizon": rollout_horizon,
                        "full_96_origin_gradient_epoch": is_full_horizon_epoch,
                        "gradient_evidence": gradient_evidence,
                        "rng_state": _rng_state(),
                        "sampler_state": {
                            "epoch": epoch + 1,
                            "next_cursor": 0,
                            "permutation_seed": seed + epoch + 1,
                        },
                    }
                _atomic_torch_save(self.checkpoint_path, checkpoint_payload)
                if improved and not phase_end_hard_gate:
                    _atomic_torch_save(self.best_checkpoint_path, checkpoint_payload)
                if phase_end_hard_gate and is_full_horizon_epoch:
                    candidate_path = self._candidate_path(epoch + 1)
                    candidate_payload = {
                        **checkpoint_payload,
                        "validation_loss": validation_loss,
                        "checkpoint_role": "provisional_k96_hard_gate_candidate",
                    }
                    _atomic_torch_save(candidate_path, candidate_payload)
                    candidate_sha256 = sha256_file(candidate_path)
                    registry = [
                        dict(item)
                        for item in state.get("candidate_registry", [])
                        if isinstance(item, Mapping)
                        and int(item.get("epoch", -1)) != epoch + 1
                    ]
                    registry.append(
                        {
                            "epoch": epoch + 1,
                            "checkpoint": str(candidate_path),
                            "checkpoint_sha256": candidate_sha256,
                            "validation_loss": validation_loss,
                        }
                    )
                    registry.sort(key=lambda item: int(item["epoch"]))
                    state["candidate_registry"] = registry
                    checkpoint_payload["hard_gate_candidate"] = {
                        "epoch": epoch + 1,
                        "checkpoint": str(candidate_path),
                        "checkpoint_sha256": candidate_sha256,
                        "validation_loss": validation_loss,
                    }
                    # Commit the registry-bearing resume checkpoint only after
                    # the immutable candidate exists. The prior atomic latest
                    # remains a valid fallback if this replace is interrupted.
                    _atomic_torch_save(self.checkpoint_path, checkpoint_payload)
                state["completed_epochs"] = epoch + 1
                state["checkpoint"] = str(self.checkpoint_path)
                state["latest_checkpoint_sha256"] = sha256_file(self.checkpoint_path)
                state["best_checkpoint_sha256"] = (
                    sha256_file(self.best_checkpoint_path)
                    if self.best_checkpoint_path.exists()
                    else None
                )
                state["best_epoch"] = best_epoch
                state["best_validation_loss"] = (
                    None if not math.isfinite(best_validation_loss) else best_validation_loss
                )
                state["early_stopping_counter"] = patience_counter
                state["active_batch_size"] = active_batch_size
                state["finite_metrics_verified"] = True
                state["finite_nonzero_gradients_verified"] = bool(
                    gradient_evidence.get("gradients_finite", False)
                    and gradient_evidence.get("nonzero_gradient_tensor_count", 0) > 0
                )
                state["checkpoint_finite_verified"] = True
                state["resumable"] = True
                state["full_96_origin_gradient_epoch_completed"] = bool(
                    state.get("full_96_origin_gradient_epoch_completed", False)
                    or is_full_horizon_epoch
                )
                state["updated_at_utc"] = utc_now()
                self._write_state(state)
                if (
                    validation_dataset is not None
                    and patience_counter >= patience
                    and (is_full_horizon_epoch or self.phase not in {"policy", "a1", "a2"})
                ):
                    stop_early = True
                    state["early_stopped"] = True
                    break

            if phase_end_hard_gate:
                if not state.get("full_96_origin_gradient_epoch_completed", False):
                    raise TrainingContractError(
                        "formal decision phase cannot complete before one full 96-origin gradient epoch"
                    )
                state["stage"] = "HARD_GATE"
                state["early_stopped"] = bool(stop_early)
                state["updated_at_utc"] = utc_now()
                self._write_state(state)
                if validation_dataset is None:
                    raise TrainingContractError(
                        "formal decision hard gate requires the validation dataset"
                    )
                state, metrics = self._run_phase_end_hard_gate(
                    state=state,
                    metrics=metrics,
                    dataset=validation_dataset,
                    physical=physical,
                    min_delta=min_delta,
                )
            elif validation_dataset is not None and not self.best_checkpoint_path.exists():
                raise TrainingContractError("no finite validation-best checkpoint was produced")
            if self.best_checkpoint_path.exists():
                best_payload = _torch_load(self.best_checkpoint_path, map_location="cpu")
                if not isinstance(best_payload, Mapping):
                    raise TrainingContractError("best checkpoint payload is invalid")
                for key, value in best_payload.get("model_state_dict", {}).items():
                    if isinstance(value, Tensor) and not bool(torch.isfinite(value).all()):
                        raise TrainingContractError(
                            f"best checkpoint contains non-finite tensor {key}"
                        )
                state["best_checkpoint_finite_verified"] = True
            state["status"] = "COMPLETE"
            state["stage"] = "COMPLETE"
            state["early_stopped"] = bool(stop_early)
            state["updated_at_utc"] = utc_now()
            if self.phase in {"a1", "a2"}:
                state["frozen_parameter_integrity"] = "VERIFIED_UNCHANGED"
            self._write_state(state)
            atomic_text(self.summary_path, self._summary(state, metrics))
            return state
        except Exception as exc:
            state["status"] = "FAILED"
            # HARD_GATE is an independently resumable post-training stage.
            # Preserve its marker so resume never re-enters gradient training.
            if phase_end_hard_gate and state.get("stage") == "HARD_GATE":
                state["hard_gate_resumable"] = True
            state["error_type"] = type(exc).__name__
            state["error"] = str(exc)
            state["updated_at_utc"] = utc_now()
            self._write_state(state)
            atomic_text(self.summary_path, self._summary(state, metrics))
            raise


def run_phase(
    model: nn.Module | None,
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_dir: str | Path,
    phase: str,
    split: str = "selection",
    device: str | torch.device | None = None,
    selection_evaluator: Callable[..., Mapping[str, Any] | None] | None = None,
    validation_evaluator: Callable[..., Mapping[str, Any] | None] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Functional entry point used by the CLI."""
    if model is None:
        from .model import E2EMoEDARTVPP, E2EMoER3Config

        model_config = _nested_get(config, "model", default={})
        if not isinstance(model_config, Mapping):
            raise TrainingContractError("model configuration must be a mapping")
        known_fields = set(E2EMoER3Config.__dataclass_fields__)
        aliases = {
            "experts": "num_experts",
        }
        kwargs: dict[str, Any] = {
            "market_dim": 4,
            "workload_dim": 2,
            "physical_dim": 4,
            "state_dim": 4,
        }
        for key, value in model_config.items():
            normalized = aliases.get(str(key), str(key))
            if normalized in known_fields:
                kwargs[normalized] = value
        model = E2EMoEDARTVPP(E2EMoER3Config(**kwargs))
    return R3Trainer(
        model,
        config,
        repository_root=repository_root,
        run_dir=run_dir,
        phase=phase,
        split=split,
        device=device,
        selection_evaluator=selection_evaluator,
        validation_evaluator=validation_evaluator,
        resume=resume,
    ).run()


def run_training_phase(
    *,
    phase: str,
    config: Mapping[str, Any],
    repository_root: str | Path,
    run_dir: str | Path,
    resume: bool = True,
    model: nn.Module | None = None,
    split: str = "selection",
    device: str | torch.device | None = None,
    selection_evaluator: Callable[..., Mapping[str, Any] | None] | None = None,
    validation_evaluator: Callable[..., Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """CLI-compatible entry point; constructs the compact model when omitted."""

    return run_phase(
        model,
        config,
        repository_root=repository_root,
        run_dir=run_dir,
        phase=phase,
        split=split,
        device=device,
        selection_evaluator=selection_evaluator,
        validation_evaluator=validation_evaluator,
        resume=resume,
    )


__all__ = [
    "A1_TRAINABLE_GROUPS",
    "A2_TRAINABLE_GROUPS",
    "ALLOWED_YEARS",
    "PHASES",
    "R3DailyBundleDataset",
    "R3Trainer",
    "TrainingContractError",
    "assert_parameter_scope",
    "collect_device_metrics",
    "configure_phase_parameters",
    "discover_daily_bundles",
    "load_frozen_physical_artifact",
    "run_phase",
    "run_training_phase",
]
