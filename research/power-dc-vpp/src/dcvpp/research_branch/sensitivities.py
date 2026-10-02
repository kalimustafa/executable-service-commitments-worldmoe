from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import (
    FrozenPhysicalArtifact,
    NYISO_ZONES,
    ScenarioBundle,
)


REQUIRED_DIMENSIONS = (
    "flexible_share",
    "deadline_steps",
    "migration_eligibility",
    "migration_bandwidth",
    "transfer_energy_checkpoint_overhead",
    "workload_scale",
    "battery_duration",
    "pv_capacity",
    "pue_slope",
    "physical_parameter_ensemble",
    "climate_support_distance",
    "site_capacity_hardware_transfer",
    "rt_volatility",
    "negative_price_clipping",
    "rt_sell_back_haircut",
    "scenario_count",
    "risk_alpha",
    "risk_weight_frontier",
    "adverse_coupling",
    "stress_scenario_mass",
    "ramp_concentration",
)


def _slug(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        value = f"{value:.8g}"
    return re.sub(r"[^A-Za-z0-9]+", "p", str(value)).strip("p").lower() or "zero"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant(
    dimension: str,
    level: str,
    value: Any,
    *,
    solve: Mapping[str, Any] | None = None,
    scenario: Mapping[str, Any] | None = None,
    physical: Mapping[str, Any] | None = None,
    policy_id: str = "B6",
    regime: str = "independent",
    reduced_count: int = 20,
    constraint_variant: str = "base",
    baseline: bool = False,
    grid_source: str = "conservative_three_point_declared_assumption",
) -> dict[str, Any]:
    consumed_by = [
        key
        for key, payload in (
            ("solve_policy", solve),
            ("scenario_bundle_transform", scenario),
            ("frozen_physical_reconstruction", physical),
        )
        if payload
    ]
    if dimension == "scenario_count":
        consumed_by.append("scenario_reduction_count")
    if dimension == "adverse_coupling":
        consumed_by.append("scenario_regime_selector")
    if constraint_variant != "base":
        consumed_by.append("optimization_constraint_variant")
    return {
        "dimension": dimension,
        "variant_id": f"ofat__{dimension}__{level}__{_slug(value)}",
        "level": level,
        "value": value,
        "baseline": baseline,
        "policy_id": policy_id,
        "regime": regime,
        "reduced_count": int(reduced_count),
        "constraint_variant": constraint_variant,
        "solve_overrides": dict(solve or {}),
        "scenario_transforms": dict(scenario or {}),
        "physical_transforms": dict(physical or {}),
        "grid_source": grid_source,
        "consumed_by": consumed_by,
    }


def build_ofat_variants(
    settings: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the preregistered sparse one-factor-at-a-time design.

    Explicit ranges in section 14.4 use their endpoints and configured base.
    Dimensions without numerical grids use conservative declared three-point
    grids.  Only the preregistered study seed executes these exploratory cells;
    the three-seed reference design remains separate.
    """

    custom = settings.get("ofat_sensitivities", {})
    custom = custom if isinstance(custom, Mapping) else {}
    workload_cfg = config.get("workload", {})
    workload_cfg = workload_cfg if isinstance(workload_cfg, Mapping) else {}
    flexible_base = float(
        settings.get(
            "flexible_share",
            workload_cfg.get("flexible_share_cap", custom.get("flexible_share_base", 0.30)),
        )
    )
    risk_alpha = float(settings.get("cvar_alpha", 0.95))
    risk_weight = float(settings.get("risk_weight", 0.50))
    h6 = settings.get("h6_constraints", {})
    h6 = h6 if isinstance(h6, Mapping) else {}
    ramp = float(h6.get("ramp_limit_mw_per_step", 2.0))
    concentration = float(h6.get("concentration_share_limit", 0.55))
    bandwidth_base = float(
        settings.get("migration_bandwidth_base_work_units_per_interval", 1.0)
    )
    transfer_energy_base = float(
        settings.get("transfer_energy_mwh_per_work_unit", 0.00002)
    )
    checkpoint_base = float(settings.get("checkpoint_overhead_fraction", 0.02))
    support_thresholds = list(
        custom.get(
            "climate_support_distance_thresholds",
            settings.get(
                "climate_support_distance_thresholds", [0.05, 0.20, 0.30]
            ),
        )
    )
    if (
        len(support_thresholds) != 3
        or not all(np.isfinite(float(value)) for value in support_thresholds)
        or not (
            0.0
            <= float(support_thresholds[0])
            < float(support_thresholds[1])
            < float(support_thresholds[2])
        )
    ):
        raise ValueError(
            "climate_support_distance_thresholds must be three strictly increasing nonnegative values"
        )
    support_thresholds = [float(value) for value in support_thresholds]

    variants: list[dict[str, Any]] = []

    def add_three(
        dimension: str,
        values: list[Any],
        baseline_index: int,
        factory,
        *,
        source: str = "conservative_three_point_declared_assumption",
    ) -> None:
        for index, value in enumerate(values):
            level = ("low", "base", "high")[index]
            variants.append(
                factory(
                    dimension,
                    level,
                    value,
                    baseline=index == baseline_index,
                    grid_source=source,
                )
            )

    add_three(
        "flexible_share",
        [0.10, flexible_base, 0.60],
        1,
        lambda d, l, v, **kw: _variant(d, l, v, solve={"flexible_share": v}, **kw),
        source="contract_range_endpoints_plus_configured_base",
    )
    add_three(
        "deadline_steps",
        [16, 48, 95],
        2,
        lambda d, l, v, **kw: _variant(d, l, v, solve={"deadline_steps": v}, **kw),
    )
    add_three(
        "migration_eligibility",
        [0.0, 0.5, 1.0],
        2,
        lambda d, l, v, **kw: _variant(d, l, v, solve={"migration_eligible": v}, **kw),
    )
    add_three(
        "migration_bandwidth",
        [0.25, 1.0, 4.0],
        1,
        lambda d, l, v, **kw: _variant(
            d,
            l,
            v,
            solve={
                "migration_bandwidth_base_work_units_per_interval": bandwidth_base,
                "migration_bandwidth_multiplier": v,
            },
            **kw,
        ),
        source="contract_range_endpoints_plus_unit_base",
    )
    add_three(
        "transfer_energy_checkpoint_overhead",
        [0.5, 1.0, 2.0],
        1,
        lambda d, l, v, **kw: _variant(
            d,
            l,
            v,
            solve={
                "transfer_energy_mwh_per_work_unit": transfer_energy_base * v,
                "checkpoint_overhead_fraction": checkpoint_base * v,
            },
            **kw,
        ),
    )
    add_three(
        "workload_scale",
        [0.5, 1.0, 2.0],
        1,
        lambda d, l, v, **kw: _variant(d, l, v, scenario={"workload_scale": v}, **kw),
        source="contract_range_endpoints_plus_unit_base",
    )
    add_three(
        "battery_duration",
        [0.0, 4.0, 8.0],
        1,
        lambda d, l, v, **kw: _variant(
            d, l, v, physical={"battery_duration_hours": v}, **kw
        ),
        source="contract_range_endpoints_plus_four_hour_base",
    )
    add_three(
        "pv_capacity",
        [0.0, 1.0, 1.5],
        1,
        lambda d, l, v, **kw: _variant(
            d, l, v, scenario={"pv_capacity_multiplier": v}, **kw
        ),
        source="contract_range_endpoints_plus_unit_base",
    )
    add_three(
        "pue_slope",
        [0.5, 1.0, 2.0],
        1,
        lambda d, l, v, **kw: _variant(
            d, l, v, solve={"pue_slope_multiplier": v}, **kw
        ),
        source="contract_range_endpoints_plus_unit_base",
    )
    add_three(
        "physical_parameter_ensemble",
        ["p10", "scenario", "p90"],
        1,
        lambda d, l, v, **kw: _variant(
            d, l, v, scenario={"physical_ensemble_selector": v}, **kw
        ),
    )
    add_three(
        "climate_support_distance",
        support_thresholds,
        1,
        lambda d, l, v, **kw: _variant(
            d,
            l,
            v,
            physical={
                "support_distance_threshold": v,
                "out_of_support_capacity_derating": 0.0,
            },
            **kw,
        ),
        source="declared_three_point_standardized_mean_temperature_distance_thresholds",
    )
    add_three(
        "site_capacity_hardware_transfer",
        [0.8, 1.0, 1.2],
        1,
        lambda d, l, v, **kw: _variant(
            d,
            l,
            v,
            physical={
                "site_capacity_multiplier": v,
                "hardware_transfer_multiplier": v,
            },
            **kw,
        ),
    )
    add_three(
        "rt_volatility",
        [0.5, 1.0, 1.5],
        1,
        lambda d, l, v, **kw: _variant(
            d, l, v, scenario={"rt_volatility_scale": v}, **kw
        ),
    )
    add_three(
        "negative_price_clipping",
        [None, -50.0, 0.0],
        0,
        lambda d, l, v, **kw: _variant(
            d, l, v, scenario={"negative_price_floor_usd_per_mwh": v}, **kw
        ),
    )
    add_three(
        "rt_sell_back_haircut",
        [0.0, 0.15, 0.30],
        0,
        lambda d, l, v, **kw: _variant(
            d, l, v, solve={"rt_sell_back_haircut": v}, **kw
        ),
    )

    for count in (10, 20, 40, 80):
        variants.append(
            _variant(
                "scenario_count",
                "base" if count == 20 else f"count_{count}",
                count,
                reduced_count=count,
                baseline=count == 20,
                grid_source="contract_explicit_grid",
            )
        )
    add_three(
        "risk_alpha",
        [0.90, risk_alpha, 0.99],
        1,
        lambda d, l, v, **kw: _variant(d, l, v, solve={"cvar_alpha": v}, **kw),
        source="contract_explicit_grid",
    )
    add_three(
        "risk_weight_frontier",
        [0.0, risk_weight, 1.0],
        1,
        lambda d, l, v, **kw: _variant(d, l, v, solve={"risk_weight": v}, **kw),
    )
    for level, value, regime in (
        ("base", 0.0, "independent"),
        ("mild", 0.3, "kappa_0_3"),
        ("strong", 0.6, "kappa_0_6"),
    ):
        variants.append(
            _variant(
                "adverse_coupling",
                level,
                value,
                regime=regime,
                baseline=value == 0.0,
                grid_source="contract_explicit_grid",
            )
        )
    add_three(
        "stress_scenario_mass",
        [0.0, 0.20, 0.40],
        1,
        lambda d, l, v, **kw: _variant(
            d,
            l,
            v,
            solve={"stress_mass": v},
            policy_id="B7",
            regime="kappa_0_6",
            **kw,
        ),
        source="contract_range_endpoints_plus_configured_base",
    )
    variants.extend(
        [
            _variant(
                "ramp_concentration",
                "off",
                "off",
                solve={"ramp_limit_mw_per_step": None, "concentration_share_limit": None},
                baseline=True,
            ),
            _variant(
                "ramp_concentration",
                "configured",
                "configured",
                solve={
                    "ramp_limit_mw_per_step": ramp,
                    "concentration_share_limit": concentration,
                },
                constraint_variant="ramp_concentration",
            ),
            _variant(
                "ramp_concentration",
                "strict",
                "strict",
                solve={
                    "ramp_limit_mw_per_step": 0.5 * ramp,
                    "concentration_share_limit": max(0.25, concentration - 0.10),
                },
                constraint_variant="ramp_concentration_strict",
            ),
        ]
    )

    dimensions = {item["dimension"] for item in variants}
    if dimensions != set(REQUIRED_DIMENSIONS):
        raise RuntimeError(
            f"OFAT sensitivity design mismatch: missing={set(REQUIRED_DIMENSIONS)-dimensions}, "
            f"extra={dimensions-set(REQUIRED_DIMENSIONS)}"
        )
    ids = [str(item["variant_id"]) for item in variants]
    if len(ids) != len(set(ids)):
        raise RuntimeError("OFAT sensitivity variant IDs are not unique")
    return variants


def apply_ofat_variant(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    variant: Mapping[str, Any],
    *,
    support_distance_by_zone: Mapping[str, float] | None,
) -> tuple[ScenarioBundle, FrozenPhysicalArtifact, dict[str, Any]]:
    scenario = dict(variant.get("scenario_transforms", {}))
    physical_transform = dict(variant.get("physical_transforms", {}))
    solve_overrides = dict(variant.get("solve_overrides", {}))

    da_price = bundle.da_price
    rt_price = bundle.rt_price
    workload = bundle.workload
    pv = bundle.pv
    physical_multiplier = bundle.physical_multiplier

    if "workload_scale" in scenario:
        workload = workload * float(scenario["workload_scale"])
        solve_overrides["workload_scale"] = float(scenario["workload_scale"])
    if "pv_capacity_multiplier" in scenario:
        pv = pv * float(scenario["pv_capacity_multiplier"])
        solve_overrides["pv_capacity_multiplier"] = float(
            scenario["pv_capacity_multiplier"]
        )
    if "rt_volatility_scale" in scenario:
        scale = float(scenario["rt_volatility_scale"])
        rt_price = bundle.da_price[None, :, :] + scale * (
            bundle.rt_price - bundle.da_price[None, :, :]
        )
        solve_overrides["rt_volatility_scale"] = scale
    if "negative_price_floor_usd_per_mwh" in scenario:
        floor = scenario["negative_price_floor_usd_per_mwh"]
        solve_overrides["negative_price_floor_usd_per_mwh"] = floor
        if floor is not None:
            da_price = np.maximum(da_price, float(floor))
            rt_price = np.maximum(rt_price, float(floor))
    selector = scenario.get("physical_ensemble_selector")
    if selector is not None:
        solve_overrides["physical_ensemble_selector"] = selector
        if selector != "scenario":
            quantile = {"p10": 0.10, "p90": 0.90}.get(str(selector))
            if quantile is None:
                raise ValueError(f"unknown physical ensemble selector {selector}")
            selected = np.quantile(
                physical.physical_multiplier_pool, quantile, axis=0
            )
            physical_multiplier = np.repeat(
                selected[None, :], bundle.reduced_count, axis=0
            )

    variant_physical = physical
    if "battery_duration_hours" in physical_transform:
        duration = float(physical_transform["battery_duration_hours"])
        solve_overrides["battery_duration_hours"] = duration
        variant_physical = replace(
            variant_physical,
            battery_energy_mwh=variant_physical.battery_power_mw * duration,
        )
    if "site_capacity_multiplier" in physical_transform:
        capacity_multiplier = float(physical_transform["site_capacity_multiplier"])
        hardware_multiplier = float(
            physical_transform.get("hardware_transfer_multiplier", 1.0)
        )
        solve_overrides.update(
            {
                "site_capacity_multiplier": capacity_multiplier,
                "hardware_transfer_multiplier": hardware_multiplier,
            }
        )
        variant_physical = replace(
            variant_physical,
            idle_mw=variant_physical.idle_mw * capacity_multiplier * hardware_multiplier,
            work_mw_per_unit=variant_physical.work_mw_per_unit * hardware_multiplier,
            capacity_work_units=variant_physical.capacity_work_units * capacity_multiplier,
            grid_import_limit_mw=variant_physical.grid_import_limit_mw
            * capacity_multiplier
            * hardware_multiplier,
            battery_power_mw=variant_physical.battery_power_mw * capacity_multiplier,
            battery_energy_mwh=variant_physical.battery_energy_mwh * capacity_multiplier,
        )
    if "support_distance_threshold" in physical_transform:
        if not support_distance_by_zone or set(support_distance_by_zone) != set(NYISO_ZONES):
            raise ValueError(
                "climate-support sensitivity requires verified support distance for every NYISO zone"
            )
        threshold = float(physical_transform["support_distance_threshold"])
        derating = float(
            physical_transform.get("out_of_support_capacity_derating", 0.0)
        )
        if not 0.0 <= derating <= 1.0:
            raise ValueError("out-of-support capacity derating must be between zero and one")
        factors = np.asarray(
            [
                1.0 if float(support_distance_by_zone[zone]) <= threshold else derating
                for zone in NYISO_ZONES
            ],
            dtype=float,
        )
        if not np.any(factors > 0):
            raise ValueError("support-distance threshold excludes every NYISO zone")
        solve_overrides.update(
            {
                "support_distance_threshold": threshold,
                "out_of_support_capacity_derating": derating,
                "support_distance_by_zone": {
                    zone: float(support_distance_by_zone[zone]) for zone in NYISO_ZONES
                },
            }
        )
        variant_physical = replace(
            variant_physical,
            idle_mw=variant_physical.idle_mw * factors,
            work_mw_per_unit=variant_physical.work_mw_per_unit * factors,
            capacity_work_units=variant_physical.capacity_work_units * factors,
            grid_import_limit_mw=variant_physical.grid_import_limit_mw * factors,
            battery_power_mw=variant_physical.battery_power_mw * factors,
            battery_energy_mwh=variant_physical.battery_energy_mwh * factors,
        )
        pv = pv * factors[None, :, None]

    transformed = replace(
        bundle,
        da_price=np.asarray(da_price, dtype=float),
        rt_price=np.asarray(rt_price, dtype=float),
        workload=np.asarray(workload, dtype=float),
        pv=np.asarray(pv, dtype=float),
        physical_multiplier=np.asarray(physical_multiplier, dtype=float),
    )
    transformed.validate()
    solve_overrides["sensitivity_variant"] = {
        key: variant.get(key)
        for key in ("dimension", "variant_id", "level", "value", "baseline")
    }
    return transformed, variant_physical, solve_overrides


def load_support_distances(
    root: Path,
    physical_result: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    def validated(raw: Mapping[str, Any]) -> dict[str, float] | None:
        try:
            values = {zone: float(raw[zone]) for zone in NYISO_ZONES}
        except (KeyError, TypeError, ValueError):
            return None
        if not all(np.isfinite(value) and value >= 0 for value in values.values()):
            return None
        return values

    explicit = settings.get("support_distance_by_zone")
    if isinstance(explicit, Mapping):
        values = validated(explicit)
        if values is None:
            raise ValueError(
                "explicit support_distance_by_zone must contain one finite nonnegative value per NYISO zone"
            )
        return values, {
            "source": "explicit_optimization_setting",
            "support_distance_by_zone": values,
        }

    frozen = physical_result.get("frozen_artifact", {})
    if isinstance(frozen, Mapping) and isinstance(
        frozen.get("support_distance_by_zone"), Mapping
    ):
        values = validated(frozen["support_distance_by_zone"])
        if values is not None:
            return values, {
                "source": "frozen_physical_artifact",
                "support_distance_by_zone": values,
            }

    outputs = physical_result.get("outputs", {})
    hashes = physical_result.get("output_hashes", {})
    raw = outputs.get("zone_twins") if isinstance(outputs, Mapping) else None
    zone_twin_reason = "zone_twins output is absent"
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            digest = _file_sha256(path)
            expected = hashes.get("zone_twins") if isinstance(hashes, Mapping) else None
            if expected and digest != expected:
                raise ValueError("zone_twins hash disagrees with frozen physical result")
            frame = pd.read_parquet(
                path, columns=["zone", "climate_support_distance"]
            )
            if frame["zone"].duplicated().any() or set(
                frame["zone"].astype(str)
            ) != set(NYISO_ZONES):
                raise ValueError(
                    "zone_twins must contain one climate-support row per NYISO zone"
                )
            values = validated(
                {
                    str(row.zone): row.climate_support_distance
                    for row in frame.itertuples(index=False)
                }
            )
            if values is not None:
                return values, {
                    "source": "hash_verified_zone_twins",
                    "path": str(path),
                    "sha256": digest,
                    "support_distance_by_zone": values,
                }
            zone_twin_reason = "zone_twins climate_support_distance values are null or nonfinite"
        else:
            zone_twin_reason = f"zone_twins not found: {path}"

    # The completed physical-v1 artifact predates the referenced NASA panel in
    # the pipeline and therefore truthfully stores null support distances.  The
    # sensitivity branch can reconstruct the exact diagnostic from its two
    # hash-addressed inputs without refitting or changing the frozen twin.
    zone_weather_path = Path(
        str(
            settings.get(
                "zone_weather_path", "data/processed/nyiso_zone_weather_15m.parquet"
            )
        )
    )
    if not zone_weather_path.is_absolute():
        zone_weather_path = root / zone_weather_path
    overlap_raw = outputs.get("overlap_primary") if isinstance(outputs, Mapping) else None
    split_raw = outputs.get("split_manifest") if isinstance(outputs, Mapping) else None
    if not overlap_raw or not split_raw or not zone_weather_path.is_file():
        return None, {
            "source": "missing",
            "reason": (
                f"{zone_twin_reason}; reconstruction inputs missing: "
                f"overlap_primary={bool(overlap_raw)}, split_manifest={bool(split_raw)}, "
                f"zone_weather={zone_weather_path.is_file()}"
            ),
        }

    overlap_path = Path(str(overlap_raw))
    split_path = Path(str(split_raw))
    if not overlap_path.is_absolute():
        overlap_path = root / overlap_path
    if not split_path.is_absolute():
        split_path = root / split_path
    for name, candidate in (("overlap_primary", overlap_path), ("split_manifest", split_path)):
        if not candidate.is_file():
            raise ValueError(f"support-distance reconstruction input is missing: {candidate}")
        expected = hashes.get(name) if isinstance(hashes, Mapping) else None
        observed = _file_sha256(candidate)
        if expected and observed != expected:
            raise ValueError(f"{name} hash disagrees with frozen physical result")

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    primary_split = split_payload.get("nlr300_srrl_primary", {})
    train_end = primary_split.get("train_end_exclusive")
    if not train_end:
        raise ValueError("physical split manifest has no primary train boundary")
    overlap_schema = pd.read_parquet(overlap_path).columns
    local_temperature_column = next(
        (
            column
            for column in ("srrl_air_temp_c", "outdoor_air_temp_c")
            if column in overlap_schema
        ),
        None,
    )
    weather_schema = pd.read_parquet(zone_weather_path).columns
    zone_column = next(
        (column for column in ("zone", "nyiso_zone", "location_domain") if column in weather_schema),
        None,
    )
    temperature_column = next(
        (
            column
            for column in (
                "air_temp_c",
                "temperature_c",
                "temperature_2m_c",
                "t2m_c",
                "temp_c",
                "T2M",
            )
            if column in weather_schema
        ),
        None,
    )
    if local_temperature_column is None or zone_column is None or temperature_column is None:
        raise ValueError(
            "support-distance reconstruction inputs lack local or zone temperature columns"
        )
    local_frame = pd.read_parquet(
        overlap_path, columns=["timestamp_utc", local_temperature_column]
    )
    timestamps = pd.to_datetime(local_frame["timestamp_utc"], utc=True)
    local = pd.to_numeric(
        local_frame.loc[timestamps < pd.Timestamp(train_end), local_temperature_column],
        errors="coerce",
    ).dropna()
    if local.empty:
        raise ValueError("physical training temperature support is empty")
    weather = pd.read_parquet(
        zone_weather_path, columns=[zone_column, temperature_column]
    )
    local_mean = float(local.mean())
    local_std = max(float(local.std(ddof=0)), 1e-6)
    local_min = float(local.min())
    local_max = float(local.max())
    values: dict[str, float] = {}
    out_of_support: dict[str, float] = {}
    zone_rows: dict[str, int] = {}
    for zone in NYISO_ZONES:
        zone_values = pd.to_numeric(
            weather.loc[weather[zone_column].astype(str).eq(zone), temperature_column],
            errors="coerce",
        ).dropna()
        if zone_values.empty:
            raise ValueError(f"zone weather contains no finite temperature rows for {zone}")
        values[zone] = abs(float(zone_values.mean()) - local_mean) / local_std
        out_of_support[zone] = float(
            ((zone_values < local_min) | (zone_values > local_max)).mean()
        )
        zone_rows[zone] = int(len(zone_values))
    return values, {
        "source": "reconstructed_from_hash_verified_physical_train_and_referenced_zone_weather",
        "formula": "absolute_zone_mean_minus_physical_train_mean_divided_by_physical_train_std",
        "support_distance_by_zone": values,
        "out_of_support_fraction_by_zone": out_of_support,
        "zone_row_count": zone_rows,
        "physical_train_row_count": int(len(local)),
        "physical_train_end_exclusive": str(train_end),
        "local_temperature_column": local_temperature_column,
        "zone_temperature_column": temperature_column,
        "overlap_primary_path": str(overlap_path),
        "overlap_primary_sha256": _file_sha256(overlap_path),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": _file_sha256(split_path),
        "zone_weather_path": str(zone_weather_path),
        "zone_weather_sha256": _file_sha256(zone_weather_path),
        "frozen_twin_support_distance_reason": zone_twin_reason,
    }


def variants_fingerprint(variants: list[Mapping[str, Any]]) -> str:
    raw = json.dumps(variants, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
