from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


POLICY_METRICS = (
    "expected_total_cost_usd",
    "cvar_total_cost_usd",
    "expected_sla_shortfall_work_units",
    "expected_grid_energy_mwh",
    "rebound_peak_mw",
    "maximum_zone_concentration_share",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _normal_two_sided_p(statistic: float) -> float:
    if not np.isfinite(statistic):
        return 1.0
    return float(math.erfc(abs(float(statistic)) / math.sqrt(2.0)))


def _moving_block_interval(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    block_length = max(1, min(int(block_length), len(values)))
    replicates = max(100, int(replicates))
    rng = np.random.default_rng(int(seed))
    block_count = int(math.ceil(len(values) / block_length))
    samples = np.empty(replicates, dtype=float)
    offsets = np.arange(block_length)
    for index in range(replicates):
        starts = rng.integers(0, len(values), size=block_count)
        positions = ((starts[:, None] + offsets[None, :]) % len(values)).ravel()
        samples[index] = float(values[positions[: len(values)]].mean())
    tail = (1.0 - confidence_level) / 2.0
    return float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))


def _newey_west_test(values: np.ndarray, lag: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), 1.0
    centered = values - values.mean()
    lag = max(0, min(int(lag), n - 1))
    long_run = float(np.dot(centered, centered) / n)
    for step in range(1, lag + 1):
        covariance = float(np.dot(centered[step:], centered[:-step]) / n)
        long_run += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / n)
    statistic = float(values.mean() / standard_error) if standard_error > 0 else 0.0
    return statistic, standard_error, _normal_two_sided_p(statistic)


def _holm(rows: list[dict[str, Any]], *, family: str) -> None:
    members = [
        (index, float(row["p_value_raw"]))
        for index, row in enumerate(rows)
        if row.get("test_family") == family
        and isinstance(row.get("p_value_raw"), (int, float))
        and np.isfinite(float(row["p_value_raw"]))
    ]
    members.sort(key=lambda item: item[1])
    previous = 0.0
    total = len(members)
    for rank, (index, p_value) in enumerate(members):
        adjusted = min(1.0, max(previous, (total - rank) * p_value))
        rows[index]["p_value_holm"] = adjusted
        previous = adjusted


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _paired_row(
    merged: pd.DataFrame,
    *,
    metric: str,
    comparison_id: str,
    candidate: str,
    baseline: str,
    block_length: int,
    replicates: int,
    confidence_level: float,
    run_id: str,
    family: str,
) -> dict[str, Any]:
    daily = (
        merged.assign(
            difference=pd.to_numeric(merged[f"{metric}_candidate"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce")
        )
        .groupby("operating_date", sort=True)["difference"]
        .mean()
        .dropna()
    )
    values = daily.to_numpy(dtype=float)
    lower, upper = _moving_block_interval(
        values,
        block_length=block_length,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=_seed(f"{run_id}|{comparison_id}|{metric}"),
    )
    statistic, standard_error, p_value = _newey_west_test(
        values, max(block_length - 1, 0)
    )
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "test_family": family,
        "analysis_role": (
            "confirmatory" if family.startswith("confirmatory") else "exploratory"
        ),
        "comparison_id": comparison_id,
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "paired_day_count": len(values),
        "paired_date_seed_count": len(merged),
        "mean_paired_difference": float(np.mean(values)) if len(values) else None,
        "effect_size_paired_dz": (
            float(np.mean(values) / standard_deviation)
            if standard_deviation > 0
            else 0.0
        ),
        "confidence_level": confidence_level,
        "block_length_days": block_length,
        "block_bootstrap_lower": lower,
        "block_bootstrap_upper": upper,
        "newey_west_standard_error": standard_error,
        "test_statistic": statistic,
        "p_value_raw": p_value,
        "p_value_holm": None,
        "null_hypothesis": "mean paired candidate-minus-baseline difference equals zero",
        "alternative_hypothesis": "mean paired candidate-minus-baseline difference is nonzero",
        "difference_direction": "candidate_minus_baseline; negative favors candidate for loss/cost/risk/constraint metrics",
        "inference_scope": "counterfactual_simulation_conditional_on_declared_inputs",
    }


def _policy_inference(
    rows: list[Mapping[str, Any]],
    *,
    block_length: int,
    replicates: int,
    confidence_level: float,
    run_id: str,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame = frame.loc[
        frame.get("phase", pd.Series(index=frame.index, dtype=object)).eq(
            "primary_96_day"
        )
        & frame.get(
            "sensitivity_dimension",
            pd.Series("reference", index=frame.index),
        ).eq("reference")
    ].copy()
    required = {"operating_date", "study_seed", "policy_id", "regime", "reduced_scenario_count", "constraint_variant"}
    if frame.empty or not required.issubset(frame.columns):
        return []

    comparisons: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = []
    common = {"regime": "independent", "reduced_scenario_count": 20, "constraint_variant": "base"}
    for policy in ("B1", "B2", "B3", "B4", "B5", "B6", "B8"):
        comparisons.append((f"{policy}_vs_B0", {"policy_id": policy, **common}, "B0", {"policy_id": "B0", **common}))
    for policy in ("B5", "B6"):
        comparisons.append((f"{policy}_vs_B4", {"policy_id": policy, **common}, "B4", {"policy_id": "B4", **common}))
    for baseline_policy in ("B1", "B2"):
        comparisons.append(
            (
                f"B6_vs_{baseline_policy}",
                {"policy_id": "B6", **common},
                baseline_policy,
                {"policy_id": baseline_policy, **common},
            )
        )
    for regime in ("kappa_0_3", "kappa_0_6"):
        criteria = {"regime": regime, "reduced_scenario_count": 20, "constraint_variant": "base"}
        comparisons.append((f"B7_vs_B6_{regime}", {"policy_id": "B7", **criteria}, "B6", {"policy_id": "B6", **criteria}))
        comparisons.append(
            (
                f"B6_{regime}_vs_nominal",
                {"policy_id": "B6", **criteria},
                "B6_nominal",
                {"policy_id": "B6", **common},
            )
        )
    comparisons.append(
        (
            "B6_H6_vs_B6_base",
            {"policy_id": "B6", "regime": "independent", "reduced_scenario_count": 20, "constraint_variant": "ramp_concentration"},
            "B6_base",
            {"policy_id": "B6", **common},
        )
    )

    def select(criteria: Mapping[str, Any]) -> pd.DataFrame:
        chosen = frame
        for key, value in criteria.items():
            chosen = chosen.loc[chosen[key].eq(value)]
        return chosen

    output: list[dict[str, Any]] = []
    keys = ["operating_date", "study_seed"]
    for comparison_id, candidate_criteria, baseline_name, baseline_criteria in comparisons:
        candidate = select(candidate_criteria)
        baseline = select(baseline_criteria)
        candidate_name = str(candidate_criteria["policy_id"])
        available_metrics = [
            metric
            for metric in POLICY_METRICS
            if metric in candidate.columns and metric in baseline.columns
        ]
        if not available_metrics:
            continue
        merged = candidate[keys + available_metrics].merge(
            baseline[keys + available_metrics],
            on=keys,
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        for metric in available_metrics:
            output.append(
                _paired_row(
                    merged,
                    metric=metric,
                    comparison_id=comparison_id,
                    candidate=candidate_name,
                    baseline=baseline_name,
                    block_length=block_length,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    run_id=run_id,
                    family="confirmatory_policy",
                )
            )
    _holm(output, family="confirmatory_policy")
    return output


def _sensitivity_inference(
    rows: list[Mapping[str, Any]],
    variants: list[Mapping[str, Any]],
    *,
    block_length: int,
    replicates: int,
    confidence_level: float,
    run_id: str,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty or "variant_id" not in frame:
        return []
    frame = frame.loc[frame["phase"].eq("primary_sensitivity")].copy()
    by_dimension: dict[str, list[Mapping[str, Any]]] = {}
    for variant in variants:
        by_dimension.setdefault(str(variant["dimension"]), []).append(variant)
    output: list[dict[str, Any]] = []
    for dimension, dimension_variants in by_dimension.items():
        baselines = [variant for variant in dimension_variants if variant.get("baseline") is True]
        if len(baselines) != 1:
            continue
        baseline_variant = baselines[0]
        baseline = frame.loc[frame["variant_id"].eq(baseline_variant["variant_id"])]
        for variant in dimension_variants:
            if variant.get("baseline") is True:
                continue
            candidate = frame.loc[frame["variant_id"].eq(variant["variant_id"])]
            available_metrics = [
                metric for metric in POLICY_METRICS if metric in candidate and metric in baseline
            ]
            if not available_metrics:
                continue
            merged = candidate[["operating_date", *available_metrics]].merge(
                baseline[["operating_date", *available_metrics]],
                on="operating_date",
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            for metric in available_metrics:
                record = _paired_row(
                    merged,
                    metric=metric,
                    comparison_id=f"{variant['variant_id']}_vs_{baseline_variant['variant_id']}",
                    candidate=str(variant["variant_id"]),
                    baseline=str(baseline_variant["variant_id"]),
                    block_length=block_length,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    run_id=run_id,
                    family="exploratory_sensitivity",
                )
                record["sensitivity_dimension"] = dimension
                output.append(record)
    _holm(output, family="exploratory_sensitivity")
    return output


def _forecast_dm(
    root: Path,
    forecast_result: Mapping[str, Any],
    *,
    horizon_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_paths = forecast_result.get("output_paths", [])
    output_paths = output_paths if isinstance(output_paths, list) else []
    prediction_paths = [
        Path(str(path))
        for path in output_paths
        if str(path).endswith("_dgpt_e_quantiles.parquet")
    ]
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for raw_path in prediction_paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        try:
            frame = pd.read_parquet(
                path,
                columns=[
                    "task",
                    "split",
                    "timestamp_utc",
                    "zone",
                    "observed_target",
                    "raw_q50",
                    "q50",
                ],
            )
            frame = frame.loc[frame["split"].eq("locked_test")].sort_values(
                ["timestamp_utc", "zone"]
            )
            frame["loss_differential"] = (
                (frame["observed_target"] - frame["raw_q50"]).abs()
                - (frame["observed_target"] - frame["q50"]).abs()
            )
            # Four zones share each 15-minute timestamp.  Average the paired
            # zone losses first so a 96-step Newey-West lag represents the
            # declared 24-hour forecast horizon rather than only 24 raw rows.
            differential = (
                frame.groupby("timestamp_utc", sort=True)["loss_differential"]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            lag = min(max(int(horizon_steps) - 1, 0), max(len(differential) - 1, 0))
            statistic, standard_error, p_value = _newey_west_test(differential, lag)
            standard_deviation = (
                float(np.std(differential, ddof=1)) if len(differential) > 1 else 0.0
            )
            rows.append(
                {
                    "test_family": "confirmatory_forecast_dm",
                    "analysis_role": "confirmatory_calibration_comparison",
                    "task": str(frame["task"].iloc[0]) if not frame.empty else path.stem,
                    "comparison": "raw_q50_vs_conformal_calibrated_q50_absolute_loss",
                    "observation_count": len(differential),
                    "raw_zone_interval_row_count": int(len(frame)),
                    "horizon_steps": int(horizon_steps),
                    "horizon_aware_lag": lag,
                    "mean_loss_difference_raw_minus_calibrated": (
                        float(np.mean(differential)) if len(differential) else None
                    ),
                    "newey_west_standard_error": standard_error,
                    "dm_style_statistic": statistic,
                    "p_value_raw": p_value,
                    "p_value_holm": None,
                    "effect_size_standardized_mean_difference": (
                        float(np.mean(differential) / standard_deviation)
                        if standard_deviation > 0
                        else 0.0
                    ),
                    "null_hypothesis": "mean raw-minus-calibrated absolute-loss differential equals zero",
                    "alternative_hypothesis": "mean raw-minus-calibrated absolute-loss differential is nonzero",
                    "difference_direction": "raw_minus_calibrated; positive favors conformal calibration",
                    "prediction_path": str(path),
                    "prediction_sha256": _sha256(path),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    _holm(rows, family="confirmatory_forecast_dm")
    return rows, failures


def build_forecast_decision_records(
    root: Path,
    branch_root: Path,
    forecast_result: Mapping[str, Any],
    optimization_rows: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Link observed forecast errors to paired B0-minus-B6 decision value.

    Every value is derived from the same operating date.  Prediction files are
    hash-verified against the forecasting run manifest, and optimization result
    hashes are retained per date/seed.  No decision-value proxy is synthesized.
    """

    failures: list[dict[str, Any]] = []
    output_records = forecast_result.get("output_paths", [])
    expected_hashes: dict[Path, str] = {}
    manifest_value = forecast_result.get("run_manifest")
    if manifest_value:
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("outputs", []):
                if not isinstance(item, Mapping) or not item.get("path"):
                    continue
                candidate = Path(str(item["path"]))
                if not candidate.is_absolute():
                    candidate = root / candidate
                expected_hashes[candidate.resolve()] = str(item.get("sha256", ""))
        except Exception as exc:
            failures.append(
                {
                    "stage": "forecast_manifest",
                    "path": str(manifest_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    prediction_paths: list[Path] = []
    for raw in output_records if isinstance(output_records, list) else []:
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        if path.name in {
            "nyiso_da_dgpt_e_quantiles.parquet",
            "nyiso_rt_dgpt_e_quantiles.parquet",
        }:
            prediction_paths.append(path.resolve())

    task_daily: list[pd.DataFrame] = []
    prediction_evidence: list[dict[str, Any]] = []
    for path in sorted(set(prediction_paths)):
        try:
            observed_hash = _sha256(path)
            expected_hash = expected_hashes.get(path.resolve())
            if expected_hash and observed_hash != expected_hash:
                raise ValueError("prediction hash disagrees with forecasting manifest")
            frame = pd.read_parquet(
                path,
                columns=[
                    "task",
                    "split",
                    "operating_date",
                    "observed_target",
                    "q05",
                    "q50",
                    "q95",
                ],
            )
            frame = frame.loc[frame["split"].eq("locked_test")].copy()
            numeric = ["observed_target", "q05", "q50", "q95"]
            frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
            frame = frame.dropna(subset=numeric)
            frame["absolute_point_error"] = (
                frame["observed_target"] - frame["q50"]
            ).abs()
            frame["outside_90"] = (
                (frame["observed_target"] < frame["q05"])
                | (frame["observed_target"] > frame["q95"])
            ).astype(float)
            task = str(frame["task"].iloc[0]) if not frame.empty else path.stem
            daily = frame.groupby("operating_date", sort=True).agg(
                point_error=("absolute_point_error", "mean"),
                outside_90_rate=("outside_90", "mean"),
                forecast_row_count=("absolute_point_error", "size"),
            )
            daily["tail_calibration_error"] = (
                daily["outside_90_rate"] - 0.10
            ).abs()
            daily["task"] = task
            task_daily.append(daily.reset_index())
            prediction_evidence.append(
                {"task": task, "path": str(path), "sha256": observed_hash}
            )
        except Exception as exc:
            failures.append(
                {
                    "stage": "forecast_prediction",
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if {str(frame["task"].iloc[0]) for frame in task_daily if not frame.empty} != {
        "nyiso_da",
        "nyiso_rt",
    }:
        failures.append(
            {
                "stage": "forecast_prediction",
                "error_type": "MissingTask",
                "error": "locked-test DA and RT quantile predictions are both required",
            }
        )
        return [], failures

    forecast_daily = pd.concat(task_daily, ignore_index=True)
    task_count = forecast_daily.groupby("operating_date")["task"].nunique()
    complete_dates = task_count.loc[task_count.eq(2)].index
    forecast_daily = forecast_daily.loc[
        forecast_daily["operating_date"].isin(complete_dates)
    ]
    combined = forecast_daily.groupby("operating_date", sort=True).agg(
        point_error=("point_error", "mean"),
        tail_calibration_error=("tail_calibration_error", "mean"),
        forecast_row_count=("forecast_row_count", "sum"),
    )

    scenario_path = branch_root / "scenario_diagnostics.json"
    try:
        scenario_hash = _sha256(scenario_path)
        scenario_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        scenario_rows = [
            item
            for item in scenario_payload
            if isinstance(item, Mapping)
            and item.get("regime") == "independent"
            and int(item.get("reduced_count", -1)) == 20
            and float(item.get("mixture_stress_mass", 0.0)) == 0.0
        ]
        quality_rows = []
        for item in scenario_rows:
            event = item.get("extreme_event", {})
            components = {
                "marginal_coverage_error": abs(
                    float(item["marginal_coverage_90"]) - 0.90
                ),
                "rank_correlation_error": float(
                    item["rank_correlation_mean_absolute_error"]
                ),
                "autocorrelation_error": float(
                    item["autocorrelation_mean_absolute_error"]
                ),
                "extreme_rate_error": float(event["absolute_rate_error"]),
                "extreme_duration_error_fraction": float(
                    event["absolute_duration_error_intervals"]
                )
                / 96.0,
                "standardized_reduction_distance": float(
                    item["full_vs_reduced_distribution_distance"]
                ),
            }
            quality_rows.append(
                {
                    "operating_date": str(item["operating_date"]),
                    "scenario_quality_error": float(np.mean(list(components.values()))),
                    "scenario_energy_score": float(item["energy_score"]),
                    "scenario_variogram_score": float(item["variogram_score"]),
                }
            )
        quality = pd.DataFrame(quality_rows)
        if quality.empty:
            raise ValueError("no independent K=20 scenario diagnostic rows")
        quality = quality.groupby("operating_date", sort=True).mean(numeric_only=True)
        combined = combined.join(quality, how="inner")
        scenario_evidence = {
            "path": str(scenario_path),
            "sha256": scenario_hash,
            "quality_definition": (
                "mean_of_abs_90coverage_error_rank_error_autocorrelation_error_"
                "extreme_rate_error_duration_error_over_96_and_standardized_reduction_distance"
            ),
        }
    except Exception as exc:
        failures.append(
            {
                "stage": "scenario_quality_linkage",
                "path": str(scenario_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return [], failures

    optimization = pd.DataFrame(list(optimization_rows))
    required = {
        "operating_date",
        "study_seed",
        "policy_id",
        "regime",
        "reduced_scenario_count",
        "constraint_variant",
        "expected_total_cost_usd",
        "checkpoint_key",
    }
    if optimization.empty or not required.issubset(optimization.columns):
        failures.append(
            {
                "stage": "optimization_linkage",
                "error_type": "MissingColumns",
                "error": f"missing columns: {sorted(required - set(optimization.columns))}",
            }
        )
        return [], failures
    reference = optimization.loc[
        optimization["regime"].eq("independent")
        & pd.to_numeric(
            optimization["reduced_scenario_count"], errors="coerce"
        ).eq(20)
        & optimization["constraint_variant"].eq("base")
        & optimization["policy_id"].isin(["B0", "B6"])
    ].copy()
    b0 = reference.loc[reference["policy_id"].eq("B0")]
    b6 = reference.loc[reference["policy_id"].eq("B6")]
    keys = ["operating_date", "study_seed"]
    pairs = b0.merge(
        b6,
        on=keys,
        suffixes=("_b0", "_b6"),
        validate="one_to_one",
    )
    pairs["decision_value_usd"] = pd.to_numeric(
        pairs["expected_total_cost_usd_b0"], errors="coerce"
    ) - pd.to_numeric(pairs["expected_total_cost_usd_b6"], errors="coerce")

    records: list[dict[str, Any]] = []
    result_root = branch_root / "results"
    for operating_date, group in pairs.groupby("operating_date", sort=True):
        if operating_date not in combined.index:
            continue
        hashes: list[dict[str, str]] = []
        for _, pair in group.iterrows():
            for suffix in ("b0", "b6"):
                key = str(pair[f"checkpoint_key_{suffix}"])
                result_path = result_root / str(operating_date) / f"{key}.json"
                evidence = {"checkpoint_key": key, "path": str(result_path)}
                if result_path.is_file():
                    evidence["sha256"] = _sha256(result_path)
                hashes.append(evidence)
        daily = combined.loc[operating_date]
        task_rows = forecast_daily.loc[
            forecast_daily["operating_date"].eq(operating_date)
        ]
        records.append(
            {
                "operating_date": str(operating_date),
                "point_error": float(daily["point_error"]),
                "tail_calibration_error": float(daily["tail_calibration_error"]),
                "scenario_quality_error": float(daily["scenario_quality_error"]),
                "scenario_energy_score": float(daily["scenario_energy_score"]),
                "scenario_variogram_score": float(
                    daily["scenario_variogram_score"]
                ),
                "decision_value_usd": float(group["decision_value_usd"].mean()),
                "decision_value_definition": "paired_B0_expected_total_cost_minus_B6_expected_total_cost",
                "paired_seed_count": int(group["study_seed"].nunique()),
                "paired_seeds": sorted(map(int, group["study_seed"].unique())),
                "forecast_task_count": int(task_rows["task"].nunique()),
                "forecast_row_count": int(daily["forecast_row_count"]),
                "task_point_error": {
                    str(row["task"]): float(row["point_error"])
                    for _, row in task_rows.iterrows()
                },
                "task_tail_calibration_error": {
                    str(row["task"]): float(row["tail_calibration_error"])
                    for _, row in task_rows.iterrows()
                },
                "forecast_prediction_evidence": prediction_evidence,
                "scenario_quality_evidence": scenario_evidence,
                "optimization_result_evidence": hashes,
                "evidence_role": "derived",
            }
        )
    return records, failures


def build_statistical_inference(
    root: Path,
    branch_root: Path,
    *,
    run_id: str,
    result_rows: list[Mapping[str, Any]],
    variants: list[Mapping[str, Any]],
    forecast_result: Mapping[str, Any],
    settings: Mapping[str, Any],
    expected_primary_days: int,
) -> dict[str, Any]:
    inference = settings.get("statistical_inference", {})
    inference = inference if isinstance(inference, Mapping) else {}
    block_length = int(inference.get("moving_block_length_days", 7))
    replicates = int(inference.get("bootstrap_replicates", 2000))
    confidence_level = float(inference.get("confidence_level", 0.95))
    if block_length < 1 or replicates < 100 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid statistical inference configuration")

    policy_rows = _policy_inference(
        result_rows,
        block_length=block_length,
        replicates=replicates,
        confidence_level=confidence_level,
        run_id=run_id,
    )
    sensitivity_rows = _sensitivity_inference(
        result_rows,
        variants,
        block_length=block_length,
        replicates=replicates,
        confidence_level=confidence_level,
        run_id=run_id,
    )
    forecast_rows, forecast_failures = _forecast_dm(
        root,
        forecast_result,
        horizon_steps=int(settings.get("horizon_steps", 96)),
    )
    decision_rows, decision_failures = build_forecast_decision_records(
        root, branch_root, forecast_result, result_rows
    )

    policy_path = _atomic_csv(branch_root / "paired_policy_inference.csv", policy_rows)
    sensitivity_path = _atomic_csv(
        branch_root / "sensitivity_effects.csv", sensitivity_rows
    )
    forecast_path = _atomic_csv(branch_root / "forecast_dm_tests.csv", forecast_rows)
    decision_table_rows = []
    for row in decision_rows:
        decision_table_rows.append(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    decision_path = _atomic_csv(
        branch_root / "forecast_decision_linkage.csv", decision_table_rows
    )
    policy_day_counts = {int(row.get("paired_day_count", 0)) for row in policy_rows}
    sensitivity_day_counts = {
        int(row.get("paired_day_count", 0)) for row in sensitivity_rows
    }
    complete = bool(
        policy_rows
        and sensitivity_rows
        and len(forecast_rows) >= 2
        and not forecast_failures
        and not decision_failures
        and policy_day_counts == {int(expected_primary_days)}
        and sensitivity_day_counts == {int(expected_primary_days)}
        and len(decision_rows) == int(expected_primary_days)
    )
    manifest_path = branch_root / "statistical_inference.json"
    payload = {
        "schema_version": "optimization_statistical_inference_v1",
        "run_id": run_id,
        "status": "COMPLETE" if complete else "PARTIAL",
        "completed": complete,
        "paired_daily_comparisons": True,
        "moving_block_bootstrap": {
            "block_length_days": block_length,
            "replicates": replicates,
            "confidence_level": confidence_level,
            "circular_blocks": True,
        },
        "dm_style": {
            "horizon_aware_newey_west": True,
            "comparison": "raw versus conformal-calibrated median absolute loss",
        },
        "holm_correction_applied": True,
        "effect_size": "paired_standardized_mean_difference_dz",
        "confirmatory_policy_row_count": len(policy_rows),
        "exploratory_sensitivity_row_count": len(sensitivity_rows),
        "forecast_dm_row_count": len(forecast_rows),
        "forecast_failures": forecast_failures,
        "forecast_decision_record_count": len(decision_rows),
        "forecast_decision_failures": decision_failures,
        "expected_primary_days": int(expected_primary_days),
        "inference_scope": (
            "Intervals quantify simulation uncertainty conditional on the frozen twin, "
            "transfer, workload, market, and coupling assumptions; they are not intervals "
            "for realized market savings."
        ),
        "outputs": {
            "paired_policy_inference": {
                "path": str(policy_path),
                "sha256": _sha256(policy_path),
            },
            "sensitivity_effects": {
                "path": str(sensitivity_path),
                "sha256": _sha256(sensitivity_path),
            },
            "forecast_dm_tests": {
                "path": str(forecast_path),
                "sha256": _sha256(forecast_path),
            },
            "forecast_decision_linkage": {
                "path": str(decision_path),
                "sha256": _sha256(decision_path),
            },
        },
    }
    _atomic_json(manifest_path, payload)
    payload["path"] = str(manifest_path)
    payload["sha256"] = _sha256(manifest_path)
    # Runtime-only records let the hypothesis aggregator consume exactly the
    # machine evidence that was written above without duplicating it in the
    # manifest.  The finalizer verifies the hashed CSV outputs independently.
    payload["policy_rows"] = policy_rows
    payload["forecast_decision_records"] = decision_rows
    return payload
