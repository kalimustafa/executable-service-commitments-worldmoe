from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dcvpp.core import atomic_json, sha256_json, utc_now
from dcvpp.data.gates import chronological_split


def _metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float | str]:
    return {
        "model": prefix,
        "facility_power_mae_kw": float(mean_absolute_error(y, pred)),
        "facility_power_rmse_kw": float(mean_squared_error(y, pred) ** 0.5),
        "daily_energy_bias_fraction": float((pred.sum() - y.sum()) / max(abs(y.sum()), 1e-9)),
        "peak_error_kw": float(pred.max() - y.max()),
        "ramp_mae_kw": float(np.mean(np.abs(np.diff(pred) - np.diff(y)))) if len(y) > 1 else 0.0,
    }


def calibrate_demo_twin(physical_path: Path, artifact_root: Path, seed: int) -> dict[str, Any]:
    frame = pd.read_parquet(physical_path)
    train, val, test = chronological_split(frame, "timestamp_utc")
    features = ["it_power_kw", "outside_air_temp_c", "solar_wm2"]
    model = LinearRegression().fit(train[features], train["facility_power_kw"])
    engineering_pred = 1.30 * test["it_power_kw"].to_numpy()
    calibrated_pred = model.predict(test[features])
    constant_pue = float(train["pue"].median()) * test["it_power_kw"].to_numpy()
    rows = [
        _metrics(test["facility_power_kw"].to_numpy(), engineering_pred, "uncalibrated_engineering"),
        _metrics(test["facility_power_kw"].to_numpy(), constant_pue, "constant_pue"),
        _metrics(test["facility_power_kw"].to_numpy(), calibrated_pred, "calibrated_linear_twin"),
    ]
    pue_pred = calibrated_pred / np.maximum(test["it_power_kw"].to_numpy(), 1e-6)
    rows[-1]["pue_mae"] = float(mean_absolute_error(test["pue"], pue_pred))
    rows[-1]["interval_coverage_90"] = float(
        np.mean(np.abs(test["facility_power_kw"].to_numpy() - calibrated_pred) <= 1.645 * np.std(train["facility_power_kw"] - model.predict(train[features])))
    )
    rows[-1]["balance_residual_max_kw"] = float(np.max(np.abs(test["facility_power_kw"] - (test["it_power_kw"] + test["cooling_kw"] + 55))))
    # A five-knot interpolation of the calibrated utilization-power relation is the demo MILP surrogate.
    knots = np.quantile(train["it_power_kw"], [0.0, 0.25, 0.50, 0.75, 1.0])
    knot_frame = pd.DataFrame({"it_power_kw": knots, "outside_air_temp_c": float(train["outside_air_temp_c"].median()), "solar_wm2": 0.0})
    knot_values = model.predict(knot_frame[features])
    surrogate = np.interp(test["it_power_kw"], knots, knot_values)
    surrogate_mae = float(mean_absolute_error(calibrated_pred, surrogate))
    rows.append({**_metrics(test["facility_power_kw"].to_numpy(), surrogate, "frozen_piecewise_milp_surrogate"), "surrogate_vs_calibrated_mae_kw": surrogate_mae})

    table_path = artifact_root / "tables/nlr_twin_validation.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(profile="demo", synthetic=True, evidence_role="derived").to_csv(table_path, index=False)
    params = {
        "profile": "demo",
        "synthetic": True,
        "calibration_id": f"demo-physical-{seed}",
        "fit_features": features,
        "intercept": float(model.intercept_),
        "coefficients": dict(zip(features, map(float, model.coef_), strict=True)),
        "surrogate_knots_it_kw": list(map(float, knots)),
        "surrogate_values_facility_kw": list(map(float, knot_values)),
        "uncertainty_residual_std_kw": float(np.std(val["facility_power_kw"] - model.predict(val[features]))),
        "split": {"train_rows": len(train), "validation_rows": len(val), "locked_test_rows": len(test)},
        "locked_test_evaluated_once": True,
        "claim_status": "measured_performance_not_paper_validation",
        "frozen_at": utc_now(),
    }
    params["parameter_hash"] = sha256_json(params)
    manifest_path = artifact_root / "manifests/physical_calibration_manifest.json"
    atomic_json(manifest_path, params)
    return {"manifest": manifest_path, "table": table_path, "parameters": params, "rows": rows}

