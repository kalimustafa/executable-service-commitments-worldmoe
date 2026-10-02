"""Artifact-backed frozen physical twin for formal R4.

The research physical branch emits a frozen transfer artifact, an accepted
piecewise surrogate, and the calibration manifest that binds both to the NLR
calibration lineage.  This module reconstructs that *numerical* twin with
ordinary torch operations.  It deliberately contains no trainable parameter
and it is not a replacement learned from R4 data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .hashing import sha256_file


NYISO_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
DEFAULT_HORIZON = 96
DEFAULT_HIDDEN_DIM = 128


class PhysicalTwinArtifactError(ValueError):
    """Raised when the three frozen physical artifacts do not form one lineage."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalTwinArtifactError(f"{label} must be a JSON object")
    return value


def _read_json(path: str | Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise PhysicalTwinArtifactError(f"{label} does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhysicalTwinArtifactError(f"cannot read {label}: {resolved}") from error
    return resolved, _mapping(value, label)


def _array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise PhysicalTwinArtifactError(
            f"{label} must have shape {shape}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise PhysicalTwinArtifactError(f"{label} contains non-finite values")
    if positive and not (result > 0.0).all():
        raise PhysicalTwinArtifactError(f"{label} must be strictly positive")
    if nonnegative and not (result >= 0.0).all():
        raise PhysicalTwinArtifactError(f"{label} must be nonnegative")
    return result


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        raise PhysicalTwinArtifactError(f"{label} must be true")


def _validate_split_metadata(manifest: Mapping[str, Any]) -> None:
    primary = _mapping(manifest.get("primary"), "calibration manifest primary")
    for name in ("facility_model", "pue_model"):
        model = _mapping(primary.get(name), f"primary.{name}")
        if model.get("fit_split") != "train":
            raise PhysicalTwinArtifactError(f"primary.{name}.fit_split must be train")
        if model.get("selection_split") != "validation":
            raise PhysicalTwinArtifactError(
                f"primary.{name}.selection_split must be validation"
            )


class ArtifactBackedFrozenPhysicalTwinCore(nn.Module):
    """Frozen torch reconstruction of the accepted NLR transfer surrogate.

    Parameters are supplied only by :func:`load_frozen_physical_twin_core`.
    ``physical_multiplier`` is a static positive four-zone transfer multiplier
    with shape ``[B,4]``. ``issue_utilization`` is a causal issue-time plan with
    shape ``[B,96,4]``.  Utilization 0 and 1 map to the first and last calibrated
    IT-power knots; values outside that interval use the registered endpoint
    linear extrapolation.

    The returned mapping contains ``tokens [B,96,H]``, ``pue [B,96,4]``,
    ``facility_load_mw [B,96,4]``, ``it_power_mw [B,96,4]``, and the effective
    ``utilization [B,96,4]``.  Tokens are a deterministic fixed feature lift,
    not a claim that a 128-dimensional neural projection was NLR-calibrated.
    """

    _FEATURE_WIDTH = 32
    formal_artifact_backed = True

    def __init__(
        self,
        *,
        artifact_values: Mapping[str, np.ndarray],
        it_power_knots_kw: np.ndarray,
        facility_power_knots_kw: np.ndarray,
        metadata: Mapping[str, Any],
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise PhysicalTwinArtifactError("hidden_dim must be positive")
        if int(horizon) <= 0:
            raise PhysicalTwinArtifactError("horizon must be positive")
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.num_sites = len(NYISO_ZONES)
        self._artifact_metadata = dict(metadata)

        # Preserve the published JSON numerics exactly in the stored artifact;
        # forward casts them to the incoming computation dtype/device.
        def buffer(name: str, value: Any, *, dtype: torch.dtype = torch.float64) -> None:
            self.register_buffer(name, torch.as_tensor(np.asarray(value).copy(), dtype=dtype))

        for name in (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
            "physical_multiplier_pool",
        ):
            buffer(name, artifact_values[name])
        buffer("it_power_knots_kw", it_power_knots_kw)
        buffer("facility_power_knots_kw", facility_power_knots_kw)
        slopes = np.diff(facility_power_knots_kw) / np.diff(it_power_knots_kw)
        buffer("facility_slopes_kw_per_it_kw", slopes)

        peak = (
            artifact_values["idle_mw"]
            + artifact_values["work_mw_per_unit"]
            * artifact_values["capacity_work_units"]
        )
        base_idle_mw = float(facility_power_knots_kw[0]) / 1000.0
        zone_scale = artifact_values["idle_mw"] / base_idle_mw
        buffer("peak_facility_mw", peak)
        buffer("zone_scale", zone_scale)
        feature_index = np.arange(self.hidden_dim, dtype=np.int64) % self._FEATURE_WIDTH
        buffer("feature_index", feature_index, dtype=torch.long)
        self.train(False)

    @property
    def artifact_metadata(self) -> Mapping[str, Any]:
        return dict(self._artifact_metadata)

    @property
    def calibration_id(self) -> str:
        return str(self._artifact_metadata["calibration_id"])

    def get_extra_state(self) -> Mapping[str, Any]:
        """Persist immutable lineage metadata beside the numeric buffers."""

        return dict(self._artifact_metadata)

    def set_extra_state(self, state: Any) -> None:
        observed = _mapping(state, "physical twin checkpoint metadata")
        if dict(observed) != self._artifact_metadata:
            raise RuntimeError(
                "physical twin checkpoint lineage differs from the loaded artifacts"
            )

    def train(self, mode: bool = True) -> "ArtifactBackedFrozenPhysicalTwinCore":
        # ``Module.train`` may recurse from the enclosing R4 model.  The core is
        # a frozen numerical artifact and therefore always remains in eval mode.
        super().train(False)
        return self

    def _piecewise_facility_kw(self, it_power_kw: Tensor) -> Tensor:
        knots_x = self.it_power_knots_kw.to(
            device=it_power_kw.device, dtype=it_power_kw.dtype
        )
        knots_y = self.facility_power_knots_kw.to(
            device=it_power_kw.device, dtype=it_power_kw.dtype
        )
        slopes = self.facility_slopes_kw_per_it_kw.to(
            device=it_power_kw.device, dtype=it_power_kw.dtype
        )
        # Interior search boundaries select the first/last segment outside the
        # knot support, which implements linear endpoint extrapolation.
        segment = torch.searchsorted(knots_x[1:-1], it_power_kw.contiguous())
        x0 = knots_x[segment]
        y0 = knots_y[segment]
        prediction = y0 + slopes[segment] * (it_power_kw - x0)
        return torch.maximum(prediction, it_power_kw)

    @staticmethod
    def _validate_inputs(physical_multiplier: Tensor, utilization: Tensor) -> None:
        if physical_multiplier.ndim != 2 or physical_multiplier.shape[-1] != 4:
            raise ValueError("physical_multiplier must be [batch,4]")
        if utilization.ndim != 3 or utilization.shape[-1] != 4:
            raise ValueError("issue_utilization must be [batch,time,4]")
        if utilization.shape[0] != physical_multiplier.shape[0]:
            raise ValueError("physical_multiplier and issue_utilization batch sizes differ")
        if not bool(torch.isfinite(physical_multiplier).all()):
            raise ValueError("physical_multiplier contains non-finite values")
        if not bool((physical_multiplier > 0.0).all()):
            raise ValueError("physical_multiplier must be strictly positive")
        if not bool(torch.isfinite(utilization).all()):
            raise ValueError("issue_utilization contains non-finite values")

    def forward(
        self,
        physical_multiplier: Tensor,
        issue_utilization: Tensor,
    ) -> dict[str, Tensor]:
        self._validate_inputs(physical_multiplier, issue_utilization)
        if issue_utilization.shape[1] != self.horizon:
            raise ValueError(
                f"issue_utilization must have {self.horizon} intervals, "
                f"got {issue_utilization.shape[1]}"
            )
        if physical_multiplier.device != issue_utilization.device:
            raise ValueError("physical_multiplier and issue_utilization must share a device")

        computation_dtype = (
            torch.float64
            if physical_multiplier.dtype == torch.float64
            or issue_utilization.dtype == torch.float64
            else torch.float32
        )
        multiplier = physical_multiplier.to(dtype=computation_dtype)
        utilization = issue_utilization.to(dtype=computation_dtype)
        x = self.it_power_knots_kw.to(
            device=utilization.device, dtype=computation_dtype
        )
        it_reference_kw = x[0] + utilization * (x[-1] - x[0])
        if not bool((it_reference_kw > 0.0).all()):
            raise ValueError("issue_utilization extrapolates to non-positive IT power")
        facility_reference_kw = self._piecewise_facility_kw(it_reference_kw)
        scale = self.zone_scale.to(
            device=utilization.device, dtype=computation_dtype
        )[None, None, :]
        multiplier_t = multiplier[:, None, :]
        base_it_mw = it_reference_kw * scale / 1000.0
        base_facility_mw = facility_reference_kw * scale / 1000.0
        it_power_mw = base_it_mw * multiplier_t
        facility_load_mw = base_facility_mw * multiplier_t
        pue = facility_load_mw / it_power_mw

        peak = self.peak_facility_mw.to(
            device=utilization.device, dtype=computation_dtype
        )[None, None, :]
        grid = self.grid_import_limit_mw.to(
            device=utilization.device, dtype=computation_dtype
        )[None, None, :]
        battery_power = self.battery_power_mw.to(
            device=utilization.device, dtype=computation_dtype
        )[None, None, :]
        battery_energy = self.battery_energy_mwh.to(
            device=utilization.device, dtype=computation_dtype
        )[None, None, :]
        expanded = lambda value: value.expand(utilization.shape[0], self.horizon, -1)
        features = torch.cat(
            (
                utilization,
                multiplier_t.expand_as(utilization) - 1.0,
                it_power_mw / peak,
                facility_load_mw / peak,
                pue - 1.0,
                expanded(grid / peak),
                expanded(battery_power / peak),
                expanded(battery_energy / (4.0 * peak)),
            ),
            dim=-1,
        )
        tokens = torch.index_select(
            features,
            dim=-1,
            index=self.feature_index.to(device=features.device),
        )
        return {
            "tokens": tokens,
            "pue": pue,
            "facility_load_mw": facility_load_mw,
            "it_power_mw": it_power_mw,
            "base_facility_power_mw": base_facility_mw,
            "base_it_power_mw": base_it_mw,
            "utilization": utilization,
        }

    def extra_repr(self) -> str:
        return (
            f"calibration_id={self.calibration_id!r}, horizon={self.horizon}, "
            f"hidden_dim={self.hidden_dim}, trainable_parameters=0"
        )


def load_frozen_physical_twin_core(
    *,
    artifact_path: str | Path,
    surrogate_path: str | Path,
    calibration_manifest_path: str | Path,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    horizon: int = DEFAULT_HORIZON,
) -> ArtifactBackedFrozenPhysicalTwinCore:
    """Load and cryptographically bind the three frozen physical artifacts."""

    artifact_file, artifact = _read_json(artifact_path, "frozen physical artifact")
    surrogate_file, surrogate = _read_json(surrogate_path, "piecewise surrogate")
    manifest_file, manifest = _read_json(
        calibration_manifest_path, "physical calibration manifest"
    )
    artifact_sha256 = sha256_file(artifact_file)
    surrogate_sha256 = sha256_file(surrogate_file)
    manifest_sha256 = sha256_file(manifest_file)

    _require_true(artifact.get("frozen"), "frozen physical artifact.frozen")
    _require_true(surrogate.get("frozen"), "piecewise surrogate.frozen")
    _require_true(manifest.get("frozen"), "physical calibration manifest.frozen")
    if artifact.get("synthetic") is not False or surrogate.get("synthetic") is not False:
        raise PhysicalTwinArtifactError("formal physical artifacts must be non-synthetic")

    calibration_id = str(artifact.get("calibration_id", "")).strip()
    if not calibration_id or str(manifest.get("calibration_id", "")) != calibration_id:
        raise PhysicalTwinArtifactError(
            "physical artifact and calibration manifest calibration_id differ"
        )
    if tuple(map(str, artifact.get("zones", ()))) != NYISO_ZONES:
        raise PhysicalTwinArtifactError(f"zones must be exactly {NYISO_ZONES}")

    artifact_sources = _mapping(
        artifact.get("source_hashes"), "frozen physical artifact.source_hashes"
    )
    manifest_sources = _mapping(
        manifest.get("source_hashes"), "physical calibration manifest.source_hashes"
    )
    declared_hashes = (
        artifact_sources.get("piecewise_surrogate"),
        manifest_sources.get("piecewise_surrogate"),
        manifest.get("surrogate_sha256"),
    )
    if any(str(value) != surrogate_sha256 for value in declared_hashes):
        raise PhysicalTwinArtifactError(
            "piecewise surrogate SHA-256 is not bound by both artifact manifests"
        )

    acceptance = _mapping(surrogate.get("acceptance"), "surrogate.acceptance")
    _require_true(
        acceptance.get("accepted_for_optimization"),
        "surrogate.acceptance.accepted_for_optimization",
    )
    if acceptance.get("status") != "accepted":
        raise PhysicalTwinArtifactError("surrogate acceptance status must be accepted")
    if acceptance.get("extrapolation_policy") != "linear_endpoint_extrapolation":
        raise PhysicalTwinArtifactError(
            "surrogate must register linear endpoint extrapolation"
        )
    if acceptance.get("locked_test_used_to_fit_or_select") is not False:
        raise PhysicalTwinArtifactError(
            "surrogate locked test must not be used to fit or select"
        )
    if surrogate.get("distillation_split") != "train":
        raise PhysicalTwinArtifactError("surrogate distillation_split must be train")
    if surrogate.get("selection_split") != "validation":
        raise PhysicalTwinArtifactError("surrogate selection_split must be validation")
    domain = _mapping(surrogate.get("domain_behavior"), "surrogate.domain_behavior")
    if (
        domain.get("below_first_knot")
        != "linear_endpoint_extrapolation_using_first_segment"
        or domain.get("above_last_knot")
        != "linear_endpoint_extrapolation_using_last_segment"
        or domain.get("facility_power_floor") != "it_power_kw"
    ):
        raise PhysicalTwinArtifactError("surrogate domain behavior is incompatible")
    _validate_split_metadata(manifest)

    primary = _mapping(manifest.get("primary"), "calibration manifest primary")
    primary_surrogate = _mapping(primary.get("surrogate"), "primary.surrogate")
    if dict(primary_surrogate) != dict(surrogate):
        raise PhysicalTwinArtifactError(
            "piecewise surrogate payload differs from the calibrated primary payload"
        )

    knots_x = _array(
        surrogate.get("it_power_knots_kw"), label="it_power_knots_kw", positive=True
    )
    knots_y = _array(
        surrogate.get("facility_power_knots_kw"),
        label="facility_power_knots_kw",
        positive=True,
    )
    utilization_knots = _array(
        surrogate.get("utilization_knots"), label="utilization_knots"
    )
    if knots_x.ndim != 1 or len(knots_x) < 2 or knots_y.shape != knots_x.shape:
        raise PhysicalTwinArtifactError("physical power knots must be matching 1-D arrays")
    if utilization_knots.shape != knots_x.shape:
        raise PhysicalTwinArtifactError("utilization knots do not match power knots")
    if not (np.diff(knots_x) > 0.0).all() or not (np.diff(knots_y) >= 0.0).all():
        raise PhysicalTwinArtifactError("physical power knots must be monotone")
    expected_utilization = (knots_x - knots_x[0]) / (knots_x[-1] - knots_x[0])
    if not np.allclose(utilization_knots, expected_utilization, rtol=0.0, atol=1e-9):
        raise PhysicalTwinArtifactError(
            "utilization knots are inconsistent with the registered IT-power axis"
        )
    if not (knots_y >= knots_x).all():
        raise PhysicalTwinArtifactError("facility power knots cannot be below IT power")

    arrays = {
        name: _array(
            artifact.get(name),
            label=f"frozen physical artifact.{name}",
            shape=(4,),
            nonnegative=True,
        )
        for name in (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
        )
    }
    arrays["physical_multiplier_pool"] = _array(
        artifact.get("physical_multiplier_pool"),
        label="frozen physical artifact.physical_multiplier_pool",
        positive=True,
    )
    if arrays["physical_multiplier_pool"].ndim != 2 or arrays[
        "physical_multiplier_pool"
    ].shape[1] != 4:
        raise PhysicalTwinArtifactError("physical_multiplier_pool must be [draws,4]")
    if not (arrays["capacity_work_units"] > 0.0).all():
        raise PhysicalTwinArtifactError("capacity_work_units must be strictly positive")

    artifact_peak = arrays["idle_mw"] + (
        arrays["work_mw_per_unit"] * arrays["capacity_work_units"]
    )
    base_idle_mw = knots_y[0] / 1000.0
    zone_scale = arrays["idle_mw"] / base_idle_mw
    expected_peak = zone_scale * knots_y[-1] / 1000.0
    if not np.allclose(artifact_peak, expected_peak, rtol=1e-6, atol=1e-9):
        raise PhysicalTwinArtifactError(
            "artifact idle/work/capacity values do not reproduce surrogate endpoints"
        )

    metadata = {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "artifact_path": str(artifact_file),
        "artifact_sha256": artifact_sha256,
        "surrogate_path": str(surrogate_file),
        "surrogate_sha256": surrogate_sha256,
        "calibration_manifest_path": str(manifest_file),
        "calibration_manifest_sha256": manifest_sha256,
        "surrogate_model": "frozen_piecewise_linear_reference_weather",
        "endpoint_extrapolation": "linear",
        "fit_split": "train",
        "selection_split": "validation",
        "locked_test_used_to_fit_or_select": False,
    }
    return ArtifactBackedFrozenPhysicalTwinCore(
        artifact_values=arrays,
        it_power_knots_kw=knots_x,
        facility_power_knots_kw=knots_y,
        metadata=metadata,
        hidden_dim=hidden_dim,
        horizon=horizon,
    )


def build_formal_r4_model(
    config: Mapping[str, Any],
    repository_root: str | Path,
) -> nn.Module:
    """Build the only model factory authorized for formal R4 execution.

    The tensor-only compatibility constructor remains useful for isolated unit
    tests.  Training, measured preflight and deployment instead call this
    factory so an artifact-backed frozen core and physical action scales are
    part of the checkpoint lineage from initialization onward.
    """

    data = _mapping(config.get("data"), "R4 config.data")
    root = Path(repository_root).resolve()

    def resolved(name: str) -> Path:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            raise PhysicalTwinArtifactError(
                f"R4 config.data.{name} must be an explicit artifact path"
            )
        path = Path(value)
        return (root / path).resolve() if not path.is_absolute() else path.resolve()

    core = load_frozen_physical_twin_core(
        artifact_path=resolved("frozen_physical_twin"),
        surrogate_path=resolved("frozen_physical_surrogate"),
        calibration_manifest_path=resolved("physical_calibration_manifest"),
    )
    # Local imports avoid a module cycle while keeping the calibrated factory
    # colocated with the artifact validation contract.
    from .model import HybridCfCTransformerMoER4, R4Config

    base = R4Config.from_mapping(config)
    cfg = replace(
        base,
        max_grid_action=float(core.grid_import_limit_mw.max().item()),
        max_battery_action=float(core.battery_power_mw.max().item()),
        max_compute_action=float(core.capacity_work_units.max().item()),
    )
    model = HybridCfCTransformerMoER4(cfg, physical_twin_core=core)
    model.formal_physical_lineage = core.artifact_metadata
    return model


__all__ = [
    "ArtifactBackedFrozenPhysicalTwinCore",
    "PhysicalTwinArtifactError",
    "build_formal_r4_model",
    "load_frozen_physical_twin_core",
]
