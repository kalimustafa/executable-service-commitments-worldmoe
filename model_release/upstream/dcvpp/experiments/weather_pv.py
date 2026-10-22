from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcvpp.experiments.forecast_common import atomic_parquet


WEATHER_PARAMETERS = (
    "T2M",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
)


PVLIB_ASSUMPTIONS = {
    "model": "pvlib fixed-tilt normalized DC availability",
    "dc_capacity_mw": 1.0,
    "surface_tilt": "site latitude",
    "surface_azimuth_degrees": 180.0,
    "gamma_pdc_per_degree_c": -0.0035,
    "reference_cell_temperature_c": 25.0,
    "temperature_model": "Faiman defaults",
    "loss_model": "none; normalized availability before inverter and balance-of-system losses",
    "capacity_factor_clip": [0.0, 1.0],
    "weather_resampling": "hourly NASA POWER value held constant within four 15-minute bins",
    "weather_operational_availability_lag_hours": 24,
}


def _hold_hourly_to_15m(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.sort_values(["zone", "timestamp_utc"], kind="stable").reset_index(drop=True)
    repeated = source.loc[source.index.repeat(4)].copy().reset_index(drop=True)
    repeated["source_hour_utc"] = repeated["timestamp_utc"]
    repeated["timestamp_utc"] = repeated["source_hour_utc"] + pd.to_timedelta(
        np.tile([0, 15, 30, 45], len(source)), unit="m"
    )
    repeated["deployable_available_at_utc"] = repeated["source_hour_utc"] + pd.Timedelta(
        hours=24
    )
    repeated["resampling_method"] = "hourly_value_held_constant"
    repeated["evidence_role"] = "reconstructed"
    repeated["source_relationship"] = "third_party_referenced_public_data"
    repeated["measurement_relationship"] = "not_measured_by_this_study"
    repeated["operational_weather_forecast_claimed"] = False
    return repeated


def _pv_for_site(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        import pvlib
    except ImportError as exc:
        raise RuntimeError("pvlib is required for the declared PV availability derivation") from exc
    latitude = float(frame["latitude"].iloc[0])
    longitude = float(frame["longitude"].iloc[0])
    timestamps = pd.DatetimeIndex(frame["timestamp_utc"])
    solar = pvlib.solarposition.get_solarposition(
        timestamps, latitude=latitude, longitude=longitude
    )
    irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=latitude,
        surface_azimuth=180.0,
        solar_zenith=solar["apparent_zenith"].to_numpy(),
        solar_azimuth=solar["azimuth"].to_numpy(),
        dni=frame["ALLSKY_SFC_SW_DNI"].to_numpy(dtype="float64"),
        ghi=frame["ALLSKY_SFC_SW_DWN"].to_numpy(dtype="float64"),
        dhi=frame["ALLSKY_SFC_SW_DIFF"].to_numpy(dtype="float64"),
    )
    poa = np.asarray(irradiance["poa_global"], dtype="float64")
    cell_temperature = np.asarray(
        pvlib.temperature.faiman(
            poa,
            frame["T2M"].to_numpy(dtype="float64"),
            frame["WS10M"].to_numpy(dtype="float64"),
        ),
        dtype="float64",
    )
    dc_mw = np.asarray(
        pvlib.pvsystem.pvwatts_dc(
            poa,
            cell_temperature,
            pdc0=1.0,
            gamma_pdc=-0.0035,
            temp_ref=25.0,
        ),
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "zone": frame["zone"].to_numpy(),
            "source_hour_utc": frame["source_hour_utc"].to_numpy(),
            "deployable_available_at_utc": frame["deployable_available_at_utc"].to_numpy(),
            "poa_global_w_m2": poa,
            "cell_temperature_c": cell_temperature,
            "pv_dc_availability_mw_per_mwdc": np.clip(dc_mw, 0.0, 1.0),
            "evidence_role": "derived",
            "source_relationship": (
                "counterfactual_derived_from_third_party_reconstructed_public_data"
            ),
            "measurement_relationship": "not_measured_by_this_study",
            "local_pv_measurement_claimed": False,
            "operational_pv_forecast_claimed": False,
        }
    )


def build_weather_and_pv_15m_panels(
    weather_rows: pd.DataFrame,
    repository_root: Path,
) -> dict[str, Any]:
    required = {"timestamp_utc", "zone", "latitude", "longitude", *WEATHER_PARAMETERS}
    missing = sorted(required - set(weather_rows.columns))
    if missing:
        raise ValueError(f"NASA POWER rows miss weather/PV fields: {missing}")
    weather = weather_rows[list(required)].copy()
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True, errors="raise")
    weather_15m = _hold_hourly_to_15m(weather)
    if (weather_15m["source_hour_utc"] >= weather_15m["deployable_available_at_utc"]).any():
        raise ValueError("weather availability lag is not strictly positive")
    weather_path = Path(repository_root) / "data/processed/nyiso_zone_weather_15m.parquet"
    weather_artifact = atomic_parquet(
        weather_15m,
        weather_path,
        metadata={
            "evidence_role": "reconstructed",
            "source": "NASA POWER referenced hourly reconstruction",
            "time_resolution": "15 minutes",
            "resampling_method": "hourly value held constant",
            "deployable_availability_lag_hours": 24,
            "operational_weather_forecast_claimed": False,
            "measurement_relationship": "not_measured_by_this_study",
        },
    )
    pv_frames = [
        _pv_for_site(group.reset_index(drop=True))
        for _, group in weather_15m.groupby("zone", sort=True)
    ]
    pv = pd.concat(pv_frames, ignore_index=True).sort_values(
        ["zone", "timestamp_utc"], kind="stable"
    )
    pv_path = Path(repository_root) / "data/processed/nyiso_zone_pv_15m.parquet"
    pv_artifact = atomic_parquet(
        pv,
        pv_path,
        metadata={
            "evidence_role": "derived",
            "source_relationship": (
                "counterfactual_derived_from_third_party_reconstructed_public_data"
            ),
            "measurement_relationship": "not_measured_by_this_study",
            "pvlib_assumptions": json.dumps(PVLIB_ASSUMPTIONS, sort_keys=True),
        },
    )
    return {
        "weather_artifact": weather_artifact,
        "pv_artifact": pv_artifact,
        "pvlib_assumptions": PVLIB_ASSUMPTIONS,
        "weather_rows": len(weather_15m),
        "pv_rows": len(pv),
        "future_realized_weather_used_in_deployable_features": False,
    }
