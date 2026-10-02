from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


NYISO_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
CONTROLLED_EVIDENCE_ROLES = {
    "observed",
    "derived",
    "reconstructed",
    "replayed",
    "assumed",
    "oracle",
    "counterfactual",
}


def _float_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class FrozenPhysicalArtifact:
    """Immutable optimization-facing view of a calibrated/transfer twin.

    The caller is responsible for freezing and hashing the upstream artifacts.
    This interface deliberately accepts only the small surrogate needed by the
    MILP; no optimizer is allowed to refit it.
    """

    calibration_id: str
    twin_ids: tuple[str, ...]
    transfer_regime: str
    idle_mw: np.ndarray
    work_mw_per_unit: np.ndarray
    capacity_work_units: np.ndarray
    grid_import_limit_mw: np.ndarray
    battery_power_mw: np.ndarray
    battery_energy_mwh: np.ndarray
    physical_multiplier_pool: np.ndarray
    zones: tuple[str, ...] = NYISO_ZONES
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.frozen:
            raise ValueError("physical artifact must be frozen before optimization")
        if tuple(self.zones) != NYISO_ZONES:
            raise ValueError(f"zones must be exactly {NYISO_ZONES}")
        if len(self.twin_ids) != len(NYISO_ZONES) or len(set(self.twin_ids)) != len(self.twin_ids):
            raise ValueError("one unique twin_id is required for each NYISO zone")
        z = len(NYISO_ZONES)
        for name in (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
        ):
            object.__setattr__(self, name, _float_array(getattr(self, name), name=name, shape=(z,)))
        pool = np.asarray(self.physical_multiplier_pool, dtype=float)
        if pool.ndim != 2 or pool.shape[1] != z or pool.shape[0] < 100:
            raise ValueError("physical_multiplier_pool must have shape (at least 100, 4)")
        if not np.isfinite(pool).all() or (pool <= 0).any():
            raise ValueError("physical multipliers must be finite and strictly positive")
        object.__setattr__(self, "physical_multiplier_pool", pool)
        for name in ("idle_mw", "work_mw_per_unit", "capacity_work_units", "grid_import_limit_mw"):
            if (getattr(self, name) < 0).any():
                raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenPhysicalArtifact":
        frozen = payload.get("frozen_artifact", payload.get("frozen", payload))
        if isinstance(frozen, cls):
            return frozen
        if not isinstance(frozen, Mapping):
            raise TypeError("physical_result must contain a mapping or FrozenPhysicalArtifact")
        return cls(
            calibration_id=str(frozen["calibration_id"]),
            twin_ids=tuple(map(str, frozen["twin_ids"])),
            transfer_regime=str(frozen.get("transfer_regime", "frozen_transfer_ensemble")),
            idle_mw=frozen["idle_mw"],
            work_mw_per_unit=frozen["work_mw_per_unit"],
            capacity_work_units=frozen["capacity_work_units"],
            grid_import_limit_mw=frozen["grid_import_limit_mw"],
            battery_power_mw=frozen.get("battery_power_mw", np.zeros(4)),
            battery_energy_mwh=frozen.get("battery_energy_mwh", np.zeros(4)),
            physical_multiplier_pool=frozen["physical_multiplier_pool"],
            zones=tuple(map(str, frozen.get("zones", NYISO_ZONES))),
            source_hashes=dict(frozen.get("source_hashes", {})),
            frozen=bool(frozen.get("frozen", True)),
        )


@dataclass(frozen=True)
class FrozenForecastArtifact:
    """Forecast-distribution inputs available at one day-ahead issue time."""

    issue_time_utc: str
    operating_date: str
    forecast_id: str
    timestamps_utc: np.ndarray
    da_price: np.ndarray
    rt_price_pool: np.ndarray
    workload_pool: np.ndarray
    pv_pool: np.ndarray
    zones: tuple[str, ...] = NYISO_ZONES
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    locked_test: bool = True

    def __post_init__(self) -> None:
        if tuple(self.zones) != NYISO_ZONES:
            raise ValueError(f"zones must be exactly {NYISO_ZONES}")
        timestamps = np.asarray(self.timestamps_utc)
        if timestamps.ndim != 1 or len(timestamps) == 0:
            raise ValueError("timestamps_utc must be a nonempty one-dimensional array")
        if len(np.unique(timestamps)) != len(timestamps):
            raise ValueError("timestamps_utc must be unique")
        t = len(timestamps)
        z = len(NYISO_ZONES)
        da = _float_array(self.da_price, name="da_price", shape=(z, t))
        rt = np.asarray(self.rt_price_pool, dtype=float)
        work = np.asarray(self.workload_pool, dtype=float)
        pv = np.asarray(self.pv_pool, dtype=float)
        n = rt.shape[0] if rt.ndim == 3 else -1
        if n < 100 or rt.shape != (n, z, t):
            raise ValueError("rt_price_pool must have shape (at least 100, 4, horizon)")
        if work.shape != (n, t):
            raise ValueError(f"workload_pool must have shape {(n, t)}, got {work.shape}")
        if pv.shape != (n, z, t):
            raise ValueError(f"pv_pool must have shape {(n, z, t)}, got {pv.shape}")
        if not np.isfinite(rt).all() or not np.isfinite(work).all() or not np.isfinite(pv).all():
            raise ValueError("forecast pools contain non-finite values")
        if (work < 0).any() or (pv < 0).any():
            raise ValueError("workload and PV pools must be nonnegative")
        object.__setattr__(self, "timestamps_utc", timestamps)
        object.__setattr__(self, "da_price", da)
        object.__setattr__(self, "rt_price_pool", rt)
        object.__setattr__(self, "workload_pool", work)
        object.__setattr__(self, "pv_pool", pv)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenForecastArtifact":
        frozen = payload.get("frozen_artifact", payload.get("frozen", payload))
        if isinstance(frozen, cls):
            return frozen
        if not isinstance(frozen, Mapping):
            raise TypeError("forecast day must be a mapping or FrozenForecastArtifact")
        return cls(
            issue_time_utc=str(frozen["issue_time_utc"]),
            operating_date=str(frozen["operating_date"]),
            forecast_id=str(frozen["forecast_id"]),
            timestamps_utc=frozen["timestamps_utc"],
            da_price=frozen["da_price"],
            rt_price_pool=frozen["rt_price_pool"],
            workload_pool=frozen["workload_pool"],
            pv_pool=frozen["pv_pool"],
            zones=tuple(map(str, frozen.get("zones", NYISO_ZONES))),
            source_hashes=dict(frozen.get("source_hashes", {})),
            locked_test=bool(frozen.get("locked_test", True)),
        )


@dataclass(frozen=True)
class ScenarioBundle:
    operating_date: str
    regime: str
    kappa: float
    initial_count: int
    reduced_count: int
    zones: tuple[str, ...]
    timestamps_utc: np.ndarray
    da_price: np.ndarray
    rt_price: np.ndarray
    workload: np.ndarray
    pv: np.ndarray
    physical_multiplier: np.ndarray
    probabilities: np.ndarray
    selected_parent_ids: np.ndarray
    full_parent_mapping: np.ndarray
    reduction_mse: float
    seed: int
    mixture_stress_mass: float = 0.0
    nominal_component_count: int = 0
    stress_component_count: int = 0
    full_parent_components: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype="U32")
    )
    full_parent_source_ids: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=int)
    )
    generation_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    decision_value_evaluation: bool = False
    evidence_role: str = "counterfactual"
    source_relationship: str = "counterfactual_using_third_party_referenced_public_data"
    measurement_relationship: str = "not_measured_by_this_study"

    def validate(self) -> None:
        w, z, t = self.rt_price.shape
        if tuple(self.zones) != NYISO_ZONES:
            raise ValueError("scenario bundle must contain the four registered NYISO zones")
        if self.initial_count != 100:
            raise ValueError("paper scenarios require exactly 100 initial draws")
        allowed_counts = {10, 20, 40, 80, 100}
        if self.reduced_count != w or self.reduced_count not in allowed_counts:
            raise ValueError("scenario count must be one of 10, 20, 40, 80, 100")
        if self.reduced_count == 100 and not self.decision_value_evaluation:
            raise ValueError(
                "100-scenario bundles are restricted to full-population decision-value evaluation"
            )
        if self.workload.shape != (w, t) or self.pv.shape != (w, z, t):
            raise ValueError("scenario block shapes are inconsistent")
        if self.physical_multiplier.shape != (w, z):
            raise ValueError("physical multiplier shape is inconsistent")
        if self.da_price.shape != (z, t):
            raise ValueError("day-ahead price shape is inconsistent")
        if self.probabilities.shape != (w,) or not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("scenario probabilities must be nonnegative and sum to one")
        if (self.probabilities < 0).any():
            raise ValueError("scenario probabilities must be nonnegative")
        selected_parent_ids = np.asarray(self.selected_parent_ids, dtype=int)
        full_parent_mapping = np.asarray(self.full_parent_mapping, dtype=int)
        if selected_parent_ids.shape != (w,):
            raise ValueError("selected_parent_ids must contain one parent for each representative")
        if full_parent_mapping.shape != (self.initial_count,):
            raise ValueError("full_parent_mapping must map every initial scenario")
        if (full_parent_mapping < 0).any() or (full_parent_mapping >= w).any():
            raise ValueError("full_parent_mapping contains an invalid representative index")
        object.__setattr__(self, "selected_parent_ids", selected_parent_ids)
        object.__setattr__(self, "full_parent_mapping", full_parent_mapping)
        components = np.asarray(self.full_parent_components, dtype="U32")
        source_ids = np.asarray(self.full_parent_source_ids, dtype=int)
        if components.size == 0:
            components = np.full(self.initial_count, self.regime, dtype="U32")
        if source_ids.size == 0:
            source_ids = np.arange(self.initial_count, dtype=int)
        if components.shape != (self.initial_count,) or source_ids.shape != (
            self.initial_count,
        ):
            raise ValueError("full-parent provenance arrays must match initial_count")
        if (source_ids < 0).any():
            raise ValueError("full_parent_source_ids must be nonnegative")
        object.__setattr__(self, "full_parent_components", components)
        object.__setattr__(self, "full_parent_source_ids", source_ids)
        if self.evidence_role != "counterfactual":
            raise ValueError("NYISO scenario outcomes must be labeled counterfactual")
        if not 0.0 <= self.mixture_stress_mass <= 1.0:
            raise ValueError("mixture_stress_mass must be between zero and one")
        if self.mixture_stress_mass > 0 or (
            self.nominal_component_count or self.stress_component_count
        ):
            if self.nominal_component_count + self.stress_component_count != self.reduced_count:
                raise ValueError("mixture component counts must sum to the reduced scenario count")

    def save(self, path: Path) -> Path:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            operating_date=np.asarray(self.operating_date),
            regime=np.asarray(self.regime),
            kappa=np.asarray(self.kappa),
            initial_count=np.asarray(self.initial_count),
            reduced_count=np.asarray(self.reduced_count),
            zones=np.asarray(self.zones),
            timestamps_utc=self.timestamps_utc,
            da_price=self.da_price,
            rt_price=self.rt_price,
            workload=self.workload,
            pv=self.pv,
            physical_multiplier=self.physical_multiplier,
            probabilities=self.probabilities,
            selected_parent_ids=self.selected_parent_ids,
            full_parent_mapping=self.full_parent_mapping,
            reduction_mse=np.asarray(self.reduction_mse),
            seed=np.asarray(self.seed),
            mixture_stress_mass=np.asarray(self.mixture_stress_mass),
            nominal_component_count=np.asarray(self.nominal_component_count),
            stress_component_count=np.asarray(self.stress_component_count),
            full_parent_components=self.full_parent_components,
            full_parent_source_ids=self.full_parent_source_ids,
            generation_diagnostics_json=np.asarray(
                json.dumps(dict(self.generation_diagnostics), sort_keys=True, default=str)
            ),
            decision_value_evaluation=np.asarray(self.decision_value_evaluation),
            evidence_role=np.asarray(self.evidence_role),
            source_relationship=np.asarray(self.source_relationship),
            measurement_relationship=np.asarray(self.measurement_relationship),
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ScenarioBundle":
        with np.load(path, allow_pickle=False) as data:
            bundle = cls(
                operating_date=str(data["operating_date"].item()),
                regime=str(data["regime"].item()),
                kappa=float(data["kappa"].item()),
                initial_count=int(data["initial_count"].item()),
                reduced_count=int(data["reduced_count"].item()),
                zones=tuple(map(str, data["zones"].tolist())),
                timestamps_utc=data["timestamps_utc"],
                da_price=data["da_price"],
                rt_price=data["rt_price"],
                workload=data["workload"],
                pv=data["pv"],
                physical_multiplier=data["physical_multiplier"],
                probabilities=data["probabilities"],
                selected_parent_ids=data["selected_parent_ids"],
                full_parent_mapping=data["full_parent_mapping"],
                reduction_mse=float(data["reduction_mse"].item()),
                seed=int(data["seed"].item()),
                mixture_stress_mass=(
                    float(data["mixture_stress_mass"].item()) if "mixture_stress_mass" in data.files else 0.0
                ),
                nominal_component_count=(
                    int(data["nominal_component_count"].item()) if "nominal_component_count" in data.files else 0
                ),
                stress_component_count=(
                    int(data["stress_component_count"].item()) if "stress_component_count" in data.files else 0
                ),
                full_parent_components=(
                    data["full_parent_components"]
                    if "full_parent_components" in data.files
                    else np.asarray([], dtype="U32")
                ),
                full_parent_source_ids=(
                    data["full_parent_source_ids"]
                    if "full_parent_source_ids" in data.files
                    else np.asarray([], dtype=int)
                ),
                generation_diagnostics=(
                    json.loads(str(data["generation_diagnostics_json"].item()))
                    if "generation_diagnostics_json" in data.files
                    else {}
                ),
                decision_value_evaluation=(
                    bool(data["decision_value_evaluation"].item())
                    if "decision_value_evaluation" in data.files
                    else False
                ),
                evidence_role=str(data["evidence_role"].item()),
                source_relationship=str(data["source_relationship"].item()),
                measurement_relationship=str(data["measurement_relationship"].item()),
            )
        bundle.validate()
        return bundle


@dataclass(frozen=True)
class XcelTariffDefinition:
    """Published-rule counterfactual; never an observed event-call series."""

    source_sha256: str
    zone: str
    energy_usd_per_mwh: float
    demand_usd_per_mw_day: float
    cpp_adder_usd_per_mwh: float
    cpp_event_mask: np.ndarray
    observed_event_calls: bool = False
    evidence_role: str = "counterfactual"
    source_relationship: str = "tariff_defined_counterfactual"
    measurement_relationship: str = "not_measured_by_this_study"

    def __post_init__(self) -> None:
        if self.zone not in NYISO_ZONES:
            raise ValueError("Xcel tariff counterfactual zone must map to a registered twin")
        if self.observed_event_calls:
            raise ValueError("this interface does not accept observed Xcel event calls")
        mask = np.asarray(self.cpp_event_mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("cpp_event_mask must be one-dimensional")
        object.__setattr__(self, "cpp_event_mask", mask)
        if self.evidence_role != "counterfactual":
            raise ValueError("Xcel tariff outcomes must be counterfactual")
