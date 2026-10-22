from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dcvpp.core import atomic_json, sha256_file, utc_now
from dcvpp.research.physical import (
    MEASUREMENT_RELATIONSHIP,
    SOURCE_RELATIONSHIP,
    _assign_physical_split,
    _atomic_parquet,
    _clean_facility_panel,
    _facility_design,
    _fit_selected_ridge,
    _physical_metric_row,
    _relative,
    _ridge_predict,
)


REQUIRED_BASELINES = (
    "uncalibrated_engineering_curve",
    "constant_pue_model",
    "static_linear_resource_to_power_model",
    "weather_free_model",
    "no_srrl_model",
    "calibrated_statistical_twin",
    "frozen_milp_surrogate",
)
REQUIRED_ABLATIONS = (
    "no_nlr_job_energy_labels",
    "no_nlr_facility_component_data",
    "constant_pue",
    "no_local_weather",
    "no_srrl_solar",
    "no_parameter_uncertainty",
    "no_climate_support_distance_flag",
    "central_nlr_transfer_without_uncertainty",
    "uncalibrated_engineering_twin",
)


def _base_row(
    study_group: str,
    variant: str,
    *,
    status: str = "EVALUATED",
    split: str | None = None,
    metric_scope: str = "facility_power",
) -> dict[str, Any]:
    return {
        "study_group": study_group,
        "variant": variant,
        "status": status,
        "split": split,
        "metric_scope": metric_scope,
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "evidence_role": "derived" if status == "EVALUATED" else "not_evaluable",
        "synthetic": False,
        "implementation_sha256": sha256_file(Path(__file__)),
    }


def _copy_existing_metrics(
    calibration: Mapping[str, Any],
    *,
    study_group: str,
    variant: str,
    model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in calibration["metrics"]:
        if metric["model"] != model:
            continue
        rows.append(
            {
                **_base_row(
                    study_group,
                    variant,
                    split=str(metric["split"]),
                ),
                **{
                    key: value
                    for key, value in metric.items()
                    if key
                    not in {
                        "model",
                        "source_relationship",
                        "measurement_relationship",
                        "evidence_role",
                    }
                },
                "source_model": model,
            }
        )
    if not rows:
        raise RuntimeError(f"missing frozen metric rows for {model}")
    return rows


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _no_srrl_design(frame: pd.DataFrame) -> np.ndarray:
    it = _numeric(frame, "it_power_kw")
    temperature = _numeric(frame, "outdoor_air_temp_c")
    humidity = _numeric(frame, "outdoor_air_humidity")
    hot = np.maximum(temperature - 18.0, 0.0)
    cold = np.maximum(18.0 - temperature, 0.0)
    return np.column_stack([it, hot, cold, humidity, it * hot / 1000.0])


def _no_srrl_solar_design(frame: pd.DataFrame) -> np.ndarray:
    full = _facility_design(frame)
    # Remove GHI while retaining IT, local SRRL temperature/humidity, and the
    # train-only fitted temperature interaction.
    return full[:, [0, 1, 2, 3, 5]]


def _fit_facility_variant(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    variant: str,
    design_builder: Callable[[pd.DataFrame], np.ndarray],
    study_group: str,
) -> list[dict[str, Any]]:
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    try:
        train_design = design_builder(train)
        validation_design = design_builder(validation)
    except KeyError as exc:
        return [
            {
                **_base_row(
                    study_group,
                    variant,
                    status="NOT_EVALUABLE",
                ),
                "reason_code": "REQUIRED_ABLATION_FEATURE_MISSING",
                "reason": str(exc),
            }
        ]
    if not np.isfinite(train_design).all() or not np.isfinite(validation_design).all():
        return [
            {
                **_base_row(
                    study_group,
                    variant,
                    status="NOT_EVALUABLE",
                ),
                "reason_code": "REQUIRED_ABLATION_FEATURE_NONFINITE",
                "reason": "train or validation design contains nonfinite referenced values",
            }
        ]
    scaler, model, selection = _fit_selected_ridge(
        train_design,
        train["facility_power_kw"].to_numpy(dtype=float),
        validation_design,
        validation["facility_power_kw"].to_numpy(dtype=float),
        alpha_values=[
            float(value)
            for value in cfg.get("facility_ridge_alpha", [0.01, 0.1, 1.0, 10.0])
        ],
    )
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "locked_test"):
        subset = frame.loc[frame["split"].eq(split)]
        design = design_builder(subset)
        it = subset["it_power_kw"].to_numpy(dtype=float)
        prediction = np.maximum(it, _ridge_predict(scaler, model, design))
        metric = _physical_metric_row(
            subset,
            prediction,
            prediction / it,
            model_name=variant,
            interpretation="primary_raw_ts_assumed_utc",
            interval_hours=float(cfg["interval_minutes"]) / 60.0,
        )
        rows.append(
            {
                **_base_row(study_group, variant, split=split),
                **{
                    key: value
                    for key, value in metric.items()
                    if key
                    not in {
                        "model",
                        "source_relationship",
                        "measurement_relationship",
                        "evidence_role",
                    }
                },
                "selected_alpha_on_validation": selection["selected_alpha"],
                "feature_count": design.shape[1],
            }
        )
    return rows


def _predict_from_parameters(
    frame: pd.DataFrame, parameters: Mapping[str, Any]
) -> np.ndarray:
    design = _facility_design(frame)
    mean = np.asarray(parameters["scaler_mean"], dtype=float)
    scale = np.asarray(parameters["scaler_scale"], dtype=float)
    coefficient = np.asarray(parameters["coefficient"], dtype=float)
    return (design - mean) / scale @ coefficient + float(parameters["intercept"])


def _no_component_rows(
    frame: pd.DataFrame, calibration: Mapping[str, Any], cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pue_bounds = tuple(float(value) for value in calibration["pue_bounds"])
    for split in ("train", "validation", "locked_test"):
        subset = frame.loc[frame["split"].eq(split)]
        it = subset["it_power_kw"].to_numpy(dtype=float)
        pue = np.clip(
            _predict_from_parameters(subset, calibration["pue_model"]),
            pue_bounds[0],
            pue_bounds[1],
        )
        prediction = np.maximum(it, pue * it)
        metric = _physical_metric_row(
            subset,
            prediction,
            pue,
            model_name="no_nlr_facility_component_data",
            interpretation="primary_raw_ts_assumed_utc",
            interval_hours=float(cfg["interval_minutes"]) / 60.0,
        )
        rows.append(
            {
                **_base_row(
                    "physical_transfer_ablation",
                    "no_nlr_facility_component_data",
                    split=split,
                ),
                **{
                    key: value
                    for key, value in metric.items()
                    if key
                    not in {
                        "model",
                        "source_relationship",
                        "measurement_relationship",
                        "evidence_role",
                    }
                },
                "ablation_semantics": (
                    "facility_power_reconstructed_from_calibrated_PUE_times_IT_without_component_sum_target"
                ),
            }
        )
    return rows


def _static_resource_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {
        "nlr295_active_nodes",
        "nlr295_active_jobs",
        "nlr295_reconstructed_node_power_kw",
    }
    if not required.issubset(frame.columns):
        return [
            {
                **_base_row(
                    "physical_baseline",
                    "static_linear_resource_to_power_model",
                    status="NOT_EVALUABLE",
                    metric_scope="resource_to_power",
                ),
                "reason_code": "RECONSTRUCTED_RESOURCE_POWER_COLUMNS_MISSING",
                "reason": ",".join(sorted(required - set(frame.columns))),
            }
        ]
    usable = frame.loc[
        frame["nlr295_active_nodes"].gt(0)
        & frame["nlr295_reconstructed_node_power_kw"].gt(0)
    ].copy()
    counts = usable["split"].value_counts()
    if any(int(counts.get(split, 0)) < 2 for split in ("train", "validation", "locked_test")):
        return [
            {
                **_base_row(
                    "physical_baseline",
                    "static_linear_resource_to_power_model",
                    status="NOT_EVALUABLE",
                    metric_scope="resource_to_power",
                ),
                "reason_code": "INSUFFICIENT_ACTIVE_MONITORING_OVERLAP_BY_SPLIT",
                "reason": json.dumps({key: int(value) for key, value in counts.items()}),
            }
        ]
    features = ["nlr295_active_nodes", "nlr295_active_jobs"]
    train = usable.loc[usable["split"].eq("train")]
    validation = usable.loc[usable["split"].eq("validation")]
    scaler, model, selection = _fit_selected_ridge(
        train[features].to_numpy(dtype=float),
        train["nlr295_reconstructed_node_power_kw"].to_numpy(dtype=float),
        validation[features].to_numpy(dtype=float),
        validation["nlr295_reconstructed_node_power_kw"].to_numpy(dtype=float),
        alpha_values=[0.01, 0.1, 1.0, 10.0],
    )
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "locked_test"):
        subset = usable.loc[usable["split"].eq(split)]
        actual = subset["nlr295_reconstructed_node_power_kw"].to_numpy(dtype=float)
        prediction = np.maximum(
            0.0,
            _ridge_predict(scaler, model, subset[features].to_numpy(dtype=float)),
        )
        error = prediction - actual
        rows.append(
            {
                **_base_row(
                    "physical_baseline",
                    "static_linear_resource_to_power_model",
                    split=split,
                    metric_scope="resource_to_power",
                ),
                "row_count": len(subset),
                "facility_mae_kw": float(np.mean(np.abs(error))),
                "facility_rmse_kw": float(np.sqrt(np.mean(np.square(error)))),
                "facility_bias_kw": float(np.mean(error)),
                "selected_alpha_on_validation": selection["selected_alpha"],
                "feature_names": json.dumps(features),
                "target": "nlr295_reconstructed_node_power_kw",
                "target_evidence": "publisher_monitoring_energy_uniformly_reconstructed_over_execution_interval",
            }
        )
    return rows


def _no_job_energy_label_rows(paths: Any, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = {
        "split",
        "node_energy_monitoring_wh",
        "cpu_energy_tdp_estimated_used_wh",
    }
    parquet = pq.ParquetFile(paths.nlr295_job_energy)
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        return [
            {
                **_base_row(
                    "physical_transfer_ablation",
                    "no_nlr_job_energy_labels",
                    status="NOT_EVALUABLE",
                    metric_scope="job_resource_to_energy",
                ),
                "reason_code": "INDEPENDENT_ENGINEERING_PROXY_FIELDS_MISSING",
                "reason": json.dumps(missing),
            }
        ]
    totals = {
        split: {
            "n": 0,
            "abs": 0.0,
            "sq": 0.0,
            "bias": 0.0,
            "actual": 0.0,
            "prediction": 0.0,
        }
        for split in ("train", "validation", "locked_test")
    }
    for batch in parquet.iter_batches(
        columns=sorted(required),
        batch_size=int(cfg.get("batch_size", 65_536)),
    ):
        frame = batch.to_pandas()
        actual = pd.to_numeric(
            frame["node_energy_monitoring_wh"], errors="coerce"
        )
        proxy = pd.to_numeric(
            frame["cpu_energy_tdp_estimated_used_wh"], errors="coerce"
        )
        valid = actual.gt(0) & proxy.ge(0) & actual.notna() & proxy.notna()
        for split in totals:
            selected = valid & frame["split"].eq(split)
            if not selected.any():
                continue
            y = actual.loc[selected].to_numpy(dtype=float)
            prediction = proxy.loc[selected].to_numpy(dtype=float)
            error = prediction - y
            total = totals[split]
            total["n"] += len(y)
            total["abs"] += float(np.abs(error).sum())
            total["sq"] += float(np.square(error).sum())
            total["bias"] += float(error.sum())
            total["actual"] += float(y.sum())
            total["prediction"] += float(prediction.sum())
    rows: list[dict[str, Any]] = []
    for split, total in totals.items():
        if not total["n"]:
            rows.append(
                {
                    **_base_row(
                        "physical_transfer_ablation",
                        "no_nlr_job_energy_labels",
                        status="NOT_EVALUABLE",
                        split=split,
                        metric_scope="job_resource_to_energy",
                    ),
                    "reason_code": "NO_MONITORED_EVALUATION_ROWS_WITH_TDP_PROXY",
                    "reason": (
                        "No rows jointly contain a publisher monitoring-derived node-energy "
                        "evaluation target and the separately published CPU-TDP estimate."
                    ),
                }
            )
            continue
        count = int(total["n"])
        rows.append(
            {
                **_base_row(
                    "physical_transfer_ablation",
                    "no_nlr_job_energy_labels",
                    split=split,
                    metric_scope="job_resource_to_energy",
                ),
                "row_count": count,
                "mae_wh": total["abs"] / count,
                "rmse_wh": float(np.sqrt(total["sq"] / count)),
                "bias_wh": total["bias"] / count,
                "energy_bias_fraction": (
                    total["prediction"] - total["actual"]
                )
                / max(total["actual"], 1e-12),
                "mean_actual_wh": total["actual"] / count,
                "mean_predicted_wh": total["prediction"] / count,
                "prediction_field": "cpu_energy_tdp_estimated_used_wh",
                "prediction_evidence_layer": "engineering_assumption",
                "prediction_record_type": "publisher_reported_cpu_tdp_estimate",
                "evaluation_target": "node_energy_monitoring_wh",
                "evaluation_target_evidence_layer": "publisher_monitoring_derived",
                "supervised_job_energy_labels_used_for_fit": False,
                "fitted_or_scaled_coefficients": False,
                "locked_target_used_for_evaluation_only": split == "locked_test",
                "interval_allocation_if_replayed": "uniform_over_execution_intervals",
                "gpu_energy_included": False,
                "gpu_exclusion_reason": (
                    "no separately published nominal GPU-TDP proxy field; monitoring-derived "
                    "GPU energy is not reused as an engineering input"
                ),
            }
        )
    return rows


def _transfer_ablation_rows(
    frozen: Mapping[str, Any], zone_twins: pd.DataFrame
) -> list[dict[str, Any]]:
    pool = np.asarray(frozen["physical_multiplier_pool"], dtype=float)
    rows: list[dict[str, Any]] = []
    for zone_index, zone in enumerate(frozen["zones"]):
        values = pool[:, zone_index]
        median = float(np.median(values))
        rows.append(
            {
                **_base_row(
                    "physical_transfer_ablation",
                    "no_parameter_uncertainty",
                    metric_scope="transfer_parameter_uncertainty",
                ),
                "zone": zone,
                "primary_multiplier_mean": float(values.mean()),
                "primary_multiplier_std": float(values.std(ddof=0)),
                "ablated_multiplier": median,
                "ablated_multiplier_std": 0.0,
                "ablation_semantics": "collapse_each_zone_empirical_transfer_pool_to_its_median",
            }
        )
        rows.append(
            {
                **_base_row(
                    "physical_transfer_ablation",
                    "central_nlr_transfer_without_uncertainty",
                    metric_scope="transfer_parameter_uncertainty",
                ),
                "zone": zone,
                "primary_multiplier_mean": float(values.mean()),
                "primary_multiplier_std": float(values.std(ddof=0)),
                "ablated_multiplier": 1.0,
                "ablated_multiplier_std": 0.0,
                "ablation_semantics": "central_calibrated_NLR_parameter_multiplier_one_for_every_zone",
            }
        )
    for row in zone_twins.itertuples(index=False):
        distance = getattr(row, "climate_support_distance", None)
        out_of_support = getattr(row, "out_of_support_fraction", None)
        climate_available = not (pd.isna(distance) or pd.isna(out_of_support))
        rows.append(
            {
                **_base_row(
                    "physical_transfer_ablation",
                    "no_climate_support_distance_flag",
                    status=("EVALUATED" if climate_available else "NOT_EVALUABLE"),
                    metric_scope="transfer_support_metadata",
                ),
                "zone": str(row.zone),
                "primary_climate_support_distance": (
                    float(distance) if climate_available else np.nan
                ),
                "primary_out_of_support_fraction": (
                    float(out_of_support) if climate_available else np.nan
                ),
                "ablated_climate_support_distance": np.nan,
                "ablated_out_of_support_fraction": np.nan,
                "maximum_physical_parameter_change": (
                    0.0 if climate_available else np.nan
                ),
                "ablation_semantics": "remove_support_flag_only_numeric_twin_parameters_unchanged",
                "reason_code": (
                    None
                    if climate_available
                    else "PRIMARY_CLIMATE_SUPPORT_NOT_COMPUTED"
                ),
                "reason": (
                    None
                    if climate_available
                    else "The primary transfer lacks referenced zone-weather support diagnostics."
                ),
            }
        )
    return rows


def evaluate_physical_baselines_and_ablations(
    root: Path,
    paths: Any,
    calibration: Mapping[str, Any],
    transfer: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    primary = calibration["primary"]
    frame, split_manifest = _assign_physical_split(
        _clean_facility_panel(paths.overlap_primary)
    )
    rows: list[dict[str, Any]] = []
    baseline_models = {
        "uncalibrated_engineering_curve": "uncalibrated_engineering_curve",
        "constant_pue_model": "constant_train_pue",
        "weather_free_model": "weather_free_static_linear",
        "calibrated_statistical_twin": "calibrated_statistical_twin",
        "frozen_milp_surrogate": "frozen_piecewise_linear_surrogate",
    }
    for variant, model in baseline_models.items():
        rows.extend(
            _copy_existing_metrics(
                primary,
                study_group="physical_baseline",
                variant=variant,
                model=model,
            )
        )
    rows.extend(_static_resource_rows(frame))
    rows.extend(
        _fit_facility_variant(
            frame,
            cfg,
            variant="no_srrl_model",
            design_builder=_no_srrl_design,
            study_group="physical_baseline",
        )
    )
    rows.extend(_no_job_energy_label_rows(paths, cfg))
    rows.extend(_no_component_rows(frame, primary, cfg))
    rows.extend(
        _copy_existing_metrics(
            primary,
            study_group="physical_transfer_ablation",
            variant="constant_pue",
            model="constant_train_pue",
        )
    )
    rows.extend(
        _copy_existing_metrics(
            primary,
            study_group="physical_transfer_ablation",
            variant="no_local_weather",
            model="weather_free_static_linear",
        )
    )
    rows.extend(
        _fit_facility_variant(
            frame,
            cfg,
            variant="no_srrl_solar",
            design_builder=_no_srrl_solar_design,
            study_group="physical_transfer_ablation",
        )
    )
    rows.extend(
        _copy_existing_metrics(
            primary,
            study_group="physical_transfer_ablation",
            variant="uncalibrated_engineering_twin",
            model="uncalibrated_engineering_curve",
        )
    )
    frozen = transfer["frozen_artifact"]
    zone_twins = pd.read_parquet(paths.zone_twins)
    rows.extend(_transfer_ablation_rows(frozen, zone_twins))

    table = pd.DataFrame(rows)
    covered = set(table["variant"].astype(str))
    required = set(REQUIRED_BASELINES) | set(REQUIRED_ABLATIONS)
    missing = sorted(required - covered)
    if missing:
        raise RuntimeError(f"physical baseline/ablation coverage missing: {missing}")
    invalid = table.loc[
        ~table["status"].isin(["EVALUATED", "NOT_EVALUABLE"]), "variant"
    ].tolist()
    if invalid:
        raise RuntimeError(f"invalid physical ablation status rows: {invalid}")
    not_evaluable = sorted(
        table.loc[table["status"].eq("NOT_EVALUABLE"), "variant"].unique().tolist()
    )
    for variant in not_evaluable:
        subset = table.loc[table["variant"].eq(variant)]
        if subset["reason_code"].isna().any() or subset["reason"].isna().any():
            raise RuntimeError(f"NOT_EVALUABLE row lacks reason: {variant}")
    _atomic_parquet(table, paths.physical_ablation_table)
    manifest = {
        "schema_version": 1,
        "status": (
            "COMPLETE_WITH_DECLARED_NOT_EVALUABLE"
            if not_evaluable
            else "COMPLETE"
        ),
        "generated_at": utc_now(),
        "required_baselines": list(REQUIRED_BASELINES),
        "required_ablations": list(REQUIRED_ABLATIONS),
        "covered_variants": sorted(covered),
        "not_evaluable_variants": not_evaluable,
        "not_evaluable_policy": "explicit_reason_required_no_placeholder_metrics",
        "fit_split": "train",
        "selection_split": "validation",
        "evaluation_split": "locked_test",
        "split_manifest": split_manifest,
        "row_count": len(table),
        "evaluated_row_count": int(table["status"].eq("EVALUATED").sum()),
        "not_evaluable_row_count": int(table["status"].eq("NOT_EVALUABLE").sum()),
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "synthetic": False,
        "inputs": {
            "overlap_primary": {
                "path": _relative(root, paths.overlap_primary),
                "sha256": sha256_file(paths.overlap_primary),
            },
            "job_metrics": {
                "path": _relative(root, paths.job_metrics),
                "sha256": sha256_file(paths.job_metrics),
            },
            "frozen_artifact": {
                "path": _relative(root, paths.frozen_artifact),
                "sha256": sha256_file(paths.frozen_artifact),
            },
            "zone_twins": {
                "path": _relative(root, paths.zone_twins),
                "sha256": sha256_file(paths.zone_twins),
            },
        },
        "output": {
            "path": _relative(root, paths.physical_ablation_table),
            "sha256": sha256_file(paths.physical_ablation_table),
        },
    }
    atomic_json(paths.physical_ablation_manifest, manifest)
    return manifest
