"""Derive registered three-seed R4 statistics and the five final artifacts.

All inputs are captured, validated day-level rows from :mod:`final_evaluation`.
No data bundle, checkpoint, or evaluator is opened here.  Consequently this
post-processing step can be resumed without repeating the formal temporal test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_io import atomic_json, atomic_parquet, atomic_text, read_json
from .config import FORMAL_SEEDS
from .final_evaluation import (
    DAILY_SCHEMA_VERSION,
    FinalProtocolError,
    load_captured_final_rows,
    mark_final_outputs_complete,
)
from .hashing import sha256_file
from .statistics import holm_adjust


RESULTS_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.final_three_seed_results.v1"
STATISTICS_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.final_statistics.v1"
METHOD_TABLE_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.final_method_comparison.v1"
DELTA_TABLE_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.paired_daily_delta.v1"

FINAL_RELATIVE_PATHS = (
    "reports/R4_FINAL_THREE_SEED_REPORT.md",
    "reports/R4_FINAL_THREE_SEED_RESULTS.json",
    "tables/r4_final_method_comparison.parquet",
    "tables/r4_paired_daily_deltas.parquet",
    "statistics/r4_bootstrap_confidence_intervals.json",
)

METHOD_COMPARISON_COLUMNS = (
    "schema_version",
    "method_id",
    "method_kind",
    "training_seed",
    "n_days",
    "origins_evaluated",
    "total_operating_cost_usd",
    "mean_daily_operating_cost_usd",
    "mean_weighted_particle_cvar95_usd",
    "total_sla_shortfall_raw",
    "mean_sla_shortfall_normalized",
    "independent_feasibility_fraction",
    "rejected_origins",
    "mean_corrective_intervention_rate",
    "mean_corrective_intervention_magnitude",
    "mean_daily_p95_corrective_intervention_magnitude",
    "mean_inference_latency_ms_b1",
    "freeze_sha256",
)

PAIRED_DELTA_COLUMNS = (
    "schema_version",
    "operating_date",
    "method_id",
    "benchmark_id",
    "training_seed",
    "metric",
    "method_value",
    "benchmark_value",
    "paired_difference",
    "inferential_role",
    "freeze_sha256",
)

PAIRED_METRICS = (
    "operating_cost_usd",
    "sla_shortfall_normalized",
    "weighted_particle_cvar95_usd",
)


def write_final_three_seed_outputs(freeze_manifest_path: str | Path) -> dict[str, Any]:
    """Write or verify the exact five registered final outputs.

    The hierarchical procedure is fixed at 10,000 replicates, outer training-
    seed resampling, and inner circular blocks of seven consecutive eligible
    dates.  Method-level cost comparisons other than the registered primary
    comparison receive Holm-adjusted one-sided p-values.
    """

    manifest_path = Path(freeze_manifest_path).resolve()
    root = manifest_path.parent.parent
    ledger_path = root / "final_temporal_test" / "execution_ledger.json"
    if ledger_path.is_file():
        ledger = read_json(ledger_path)
        if ledger.get("status") == "COMPLETE":
            _verify_completed_outputs(root, ledger)
            return read_json(root / FINAL_RELATIVE_PATHS[1])

    manifest, daily_rows = load_captured_final_rows(manifest_path)
    daily = pd.DataFrame.from_records(daily_rows)
    method_table = _method_comparison_rows(daily, manifest)
    delta_table = _paired_delta_rows(daily, manifest)
    statistics = _bootstrap_statistics(delta_table, daily, manifest)

    report_path = root / FINAL_RELATIVE_PATHS[0]
    results_path = root / FINAL_RELATIVE_PATHS[1]
    method_path = root / FINAL_RELATIVE_PATHS[2]
    delta_path = root / FINAL_RELATIVE_PATHS[3]
    statistics_path = root / FINAL_RELATIVE_PATHS[4]

    atomic_parquet(method_path, method_table)
    atomic_parquet(delta_path, delta_table)
    atomic_json(statistics_path, statistics)
    results = _results_payload(
        manifest,
        daily,
        method_table,
        delta_table,
        statistics,
        method_path=method_path,
        delta_path=delta_path,
        statistics_path=statistics_path,
    )
    atomic_json(results_path, results)
    atomic_text(report_path, _render_report(results, statistics))
    outputs = (report_path, results_path, method_path, delta_path, statistics_path)
    mark_final_outputs_complete(manifest_path, output_paths=outputs)
    return results


def _method_comparison_rows(
    daily: pd.DataFrame, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (method_id, training_seed), group in daily.groupby(
        ["method_id", "training_seed"], sort=True
    ):
        kinds = set(group["method_kind"])
        if len(kinds) != 1:
            raise FinalProtocolError(f"method kind is inconsistent for {method_id}")
        rows.append(
            {
                "schema_version": METHOD_TABLE_SCHEMA_VERSION,
                "method_id": str(method_id),
                "method_kind": next(iter(kinds)),
                "training_seed": int(training_seed),
                "n_days": int(len(group)),
                "origins_evaluated": int(group["origins_evaluated"].sum()),
                "total_operating_cost_usd": float(group["operating_cost_usd"].sum()),
                "mean_daily_operating_cost_usd": float(group["operating_cost_usd"].mean()),
                "mean_weighted_particle_cvar95_usd": float(group["weighted_particle_cvar95_usd"].mean()),
                "total_sla_shortfall_raw": float(group["sla_shortfall_raw"].sum()),
                "mean_sla_shortfall_normalized": float(group["sla_shortfall_normalized"].mean()),
                "independent_feasibility_fraction": float(group["independent_feasibility_fraction"].mean()),
                "rejected_origins": int(group["rejected_origins"].sum()),
                "mean_corrective_intervention_rate": float(group["corrective_intervention_rate"].mean()),
                "mean_corrective_intervention_magnitude": float(group["corrective_intervention_mean"].mean()),
                "mean_daily_p95_corrective_intervention_magnitude": float(group["corrective_intervention_p95"].mean()),
                "mean_inference_latency_ms_b1": float(group["inference_latency_ms_b1"].mean()),
                "freeze_sha256": manifest["freeze_sha256"],
            }
        )
    if any(tuple(row) != METHOD_COMPARISON_COLUMNS for row in rows):
        raise AssertionError("internal method-comparison schema drift")
    return rows


def _paired_delta_rows(
    daily: pd.DataFrame, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    benchmark_id = str(manifest["primary_benchmark_id"])
    benchmark = daily[(daily["method_id"] == benchmark_id) & (daily["training_seed"] == 0)]
    benchmark_by_date = benchmark.set_index("operating_date")
    dates = list(manifest["eligible_dates"])
    if list(benchmark_by_date.index) != dates:
        benchmark_by_date = benchmark_by_date.reindex(dates)
    if benchmark_by_date.isnull().any().any():
        raise FinalProtocolError("primary benchmark does not cover every eligible date")
    rows: list[dict[str, Any]] = []
    methods = [item for item in manifest["methods"] if item["kind"] != "benchmark"]
    for method in methods:
        for training_seed in FORMAL_SEEDS:
            selected = daily[
                (daily["method_id"] == method["method_id"])
                & (daily["training_seed"] == training_seed)
            ].set_index("operating_date").reindex(dates)
            if selected.isnull().any().any():
                raise FinalProtocolError(
                    f"method {method['method_id']} seed {training_seed} lacks paired dates"
                )
            for operating_date in dates:
                for metric in PAIRED_METRICS:
                    method_value = float(selected.at[operating_date, metric])
                    benchmark_value = float(benchmark_by_date.at[operating_date, metric])
                    role = (
                        "hierarchical_cost_inference"
                        if metric == "operating_cost_usd"
                        else "descriptive_particle_risk_no_statistical_ci"
                        if metric == "weighted_particle_cvar95_usd"
                        else "descriptive_service_guardrail"
                    )
                    rows.append(
                        {
                            "schema_version": DELTA_TABLE_SCHEMA_VERSION,
                            "operating_date": operating_date,
                            "method_id": method["method_id"],
                            "benchmark_id": benchmark_id,
                            "training_seed": int(training_seed),
                            "metric": metric,
                            "method_value": method_value,
                            "benchmark_value": benchmark_value,
                            "paired_difference": method_value - benchmark_value,
                            "inferential_role": role,
                            "freeze_sha256": manifest["freeze_sha256"],
                        }
                    )
    if any(tuple(row) != PAIRED_DELTA_COLUMNS for row in rows):
        raise AssertionError("internal paired-delta schema drift")
    return rows


def _bootstrap_statistics(
    delta_rows: Sequence[Mapping[str, Any]],
    daily: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    deltas = pd.DataFrame.from_records(delta_rows)
    dates = list(manifest["eligible_dates"])
    if len(dates) < 7:
        raise FinalProtocolError("registered 7-day bootstrap requires at least seven dates")
    primary = str(manifest["primary_method_id"])
    benchmark = str(manifest["primary_benchmark_id"])
    design = _bootstrap_design(len(dates), replicates=10_000, seed=20240813)
    comparisons: list[dict[str, Any]] = []
    nonbenchmark = [item["method_id"] for item in manifest["methods"] if item["kind"] != "benchmark"]
    benchmark_cost = daily[
        (daily["method_id"] == benchmark) & (daily["training_seed"] == 0)
    ]["operating_cost_usd"].mean()
    margin = 0.01 * abs(float(benchmark_cost))
    for method_id in nonbenchmark:
        selected = deltas[
            (deltas["method_id"] == method_id)
            & (deltas["metric"] == "operating_cost_usd")
        ]
        matrix = np.empty((len(FORMAL_SEEDS), len(dates)), dtype=np.float64)
        for seed_position, training_seed in enumerate(FORMAL_SEEDS):
            series = selected[selected["training_seed"] == training_seed].set_index("operating_date")["paired_difference"].reindex(dates)
            if series.isnull().any():
                raise FinalProtocolError(f"bootstrap matrix incomplete for {method_id}/{training_seed}")
            matrix[seed_position] = series.to_numpy(dtype=np.float64)
        inference = _apply_bootstrap_design(matrix, design)
        is_primary = method_id == primary
        decision = (
            "COST_SUPERIORITY"
            if inference["upper"] < 0.0
            else "COST_NONINFERIORITY"
            if inference["one_sided_upper"] < margin
            else "INSUFFICIENT_EVIDENCE"
        )
        comparisons.append(
            {
                "comparison_id": f"{method_id}_vs_{benchmark}_operating_cost",
                "method_id": method_id,
                "benchmark_id": benchmark,
                "metric": "operating_cost_usd",
                "is_primary": is_primary,
                "training_seeds": list(FORMAL_SEEDS),
                "n_dates_per_seed": len(dates),
                "mean_paired_difference_usd": inference["mean_difference"],
                "confidence_interval_95": [inference["lower"], inference["upper"]],
                "one_sided_upper_95": inference["one_sided_upper"],
                "unadjusted_one_sided_p": inference["one_sided_p"],
                "holm_adjusted_one_sided_p": None,
                "noninferiority_margin_usd": margin,
                "decision": decision,
            }
        )
    secondary_positions = [index for index, row in enumerate(comparisons) if not row["is_primary"]]
    if secondary_positions:
        adjusted = holm_adjust(
            [comparisons[index]["unadjusted_one_sided_p"] for index in secondary_positions]
        )
        for index, value in zip(secondary_positions, adjusted, strict=True):
            comparisons[index]["holm_adjusted_one_sided_p"] = float(value)
    primary_rows = [row for row in comparisons if row["is_primary"]]
    if len(primary_rows) != 1:
        raise FinalProtocolError("exactly one primary cost comparison is required")
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "status": "COMPLETE",
        "freeze_sha256": manifest["freeze_sha256"],
        "primary_unit": "day",
        "procedure": {
            "hierarchy": "training_seed_outer_date_circular_moving_block_inner",
            "training_seed_count": 3,
            "training_seeds": list(FORMAL_SEEDS),
            "block_days": 7,
            "replicates": 10_000,
            "confidence_level": 0.95,
            "bootstrap_seed": 20240813,
            "secondary_multiplicity": "Holm step-down",
            "one_sided_p_method": "null-centered hierarchical bootstrap",
        },
        "primary_comparison_id": primary_rows[0]["comparison_id"],
        "comparisons": comparisons,
        "weighted_particle_cvar95": {
            "role": "descriptive weighted 20-particle tail-risk metric",
            "is_statistical_confidence_interval": False,
            "included_in_bootstrap_confidence_intervals": False,
        },
    }


def _bootstrap_design(n_days: int, *, replicates: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_seeds = len(FORMAL_SEEDS)
    selected_seeds = rng.integers(0, n_seeds, size=(replicates, n_seeds), dtype=np.int8)
    block_count = int(np.ceil(n_days / 7))
    starts = rng.integers(0, n_days, size=(replicates, n_seeds, block_count), dtype=np.int32)
    offsets = np.arange(7, dtype=np.int32)
    date_indices = (starts[..., None] + offsets) % n_days
    date_indices = date_indices.reshape(replicates, n_seeds, -1)[..., :n_days]
    return {"selected_seeds": selected_seeds, "date_indices": date_indices}


def _apply_bootstrap_design(matrix: np.ndarray, design: Mapping[str, np.ndarray]) -> dict[str, float]:
    if matrix.shape[0] != len(FORMAL_SEEDS) or not np.isfinite(matrix).all():
        raise FinalProtocolError("hierarchical bootstrap requires finite three-seed rows")
    selected_seeds = design["selected_seeds"]
    date_indices = design["date_indices"]
    sampled = matrix[selected_seeds[..., None], date_indices].mean(axis=(1, 2))
    observed = float(matrix.mean())
    centered = sampled - sampled.mean()
    one_sided_p = float((1 + np.count_nonzero(centered <= observed)) / (sampled.size + 1))
    return {
        "mean_difference": observed,
        "lower": float(np.quantile(sampled, 0.025)),
        "upper": float(np.quantile(sampled, 0.975)),
        "one_sided_upper": float(np.quantile(sampled, 0.95)),
        "one_sided_p": one_sided_p,
    }


def _results_payload(
    manifest: Mapping[str, Any],
    daily: pd.DataFrame,
    method_table: Sequence[Mapping[str, Any]],
    delta_table: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    *,
    method_path: Path,
    delta_path: Path,
    statistics_path: Path,
) -> dict[str, Any]:
    primary = next(
        row for row in statistics["comparisons"] if row["is_primary"]
    )
    terminal_label = (
        "R4_FINAL_THREE_SEED_PASS"
        if primary["decision"] in {"COST_SUPERIORITY", "COST_NONINFERIORITY"}
        and float(daily["independent_feasibility_fraction"].min()) == 1.0
        and int(daily["rejected_origins"].sum()) == 0
        else "R4_FINAL_THREE_SEED_INSUFFICIENT_EVIDENCE"
    )
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "status": "COMPLETE",
        "terminal_label": terminal_label,
        "freeze_sha256": manifest["freeze_sha256"],
        "formal_selection_2024_opened": True,
        "locked_test_2025_opened": False,
        "eligible_dates": len(manifest["eligible_dates"]),
        "same_dates_for_all_methods": True,
        "exactly_once_evaluation": True,
        "training_seeds": list(FORMAL_SEEDS),
        "primary_method_id": manifest["primary_method_id"],
        "primary_benchmark_id": manifest["primary_benchmark_id"],
        "primary_cost_comparison": primary,
        "weighted_particle_cvar95_note": "particle CVaR95 is a risk metric, not a statistical confidence interval",
        "row_counts": {
            "daily_method": int(len(daily)),
            "method_comparison": int(len(method_table)),
            "paired_daily_deltas": int(len(delta_table)),
        },
        "derived_artifacts": {
            "method_comparison": {"path": str(method_path), "sha256": sha256_file(method_path)},
            "paired_daily_deltas": {"path": str(delta_path), "sha256": sha256_file(delta_path)},
            "bootstrap_confidence_intervals": {"path": str(statistics_path), "sha256": sha256_file(statistics_path)},
        },
    }


def _render_report(results: Mapping[str, Any], statistics: Mapping[str, Any]) -> str:
    primary = results["primary_cost_comparison"]
    lower, upper = primary["confidence_interval_95"]
    lines = [
        "# R4 Final Three-Seed Report",
        "",
        f"Terminal label: **{results['terminal_label']}**",
        "",
        f"Primary comparison: `{primary['method_id']}` versus `{primary['benchmark_id']}`.",
        f"Mean paired daily operating-cost difference: {primary['mean_paired_difference_usd']:.6f} USD.",
        f"Hierarchical 95% confidence interval: [{lower:.6f}, {upper:.6f}] USD.",
        f"Decision: **{primary['decision']}**.",
        "",
        "The primary statistical unit is one eligible day. The registered procedure resamples training seeds at the outer level and seven-day circular moving blocks within seed for 10,000 replicates. Secondary method comparisons use Holm correction.",
        "",
        "Weighted 20-particle CVaR95 is reported as an operational tail-risk quantity and is not interpreted as a statistical 95% confidence interval.",
        "",
        f"Eligible 2024 dates evaluated: {results['eligible_dates']}; every frozen method used the same dates and the formal evaluator was invoked exactly once.",
        "2025 remained closed.",
        "",
        "## Secondary cost comparisons",
        "",
    ]
    secondary = [row for row in statistics["comparisons"] if not row["is_primary"]]
    if not secondary:
        lines.append("No secondary frozen method comparison was registered.")
    else:
        for row in secondary:
            lines.append(
                f"- `{row['method_id']}` vs `{row['benchmark_id']}`: "
                f"Holm-adjusted p={row['holm_adjusted_one_sided_p']:.6g}; "
                f"decision={row['decision']}."
            )
    lines.append("")
    return "\n".join(lines)


def _verify_completed_outputs(root: Path, ledger: Mapping[str, Any]) -> None:
    records = ledger.get("outputs")
    if not isinstance(records, list) or len(records) != len(FINAL_RELATIVE_PATHS):
        raise FinalProtocolError("completed final ledger has an invalid output inventory")
    expected = {str((root / relative).resolve()) for relative in FINAL_RELATIVE_PATHS}
    actual = {str(Path(record["path"]).resolve()) for record in records}
    if actual != expected:
        raise FinalProtocolError("completed final output paths changed")
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise FinalProtocolError(f"completed final output changed: {path}")


__all__ = [
    "DELTA_TABLE_SCHEMA_VERSION",
    "FINAL_RELATIVE_PATHS",
    "METHOD_COMPARISON_COLUMNS",
    "METHOD_TABLE_SCHEMA_VERSION",
    "PAIRED_DELTA_COLUMNS",
    "RESULTS_SCHEMA_VERSION",
    "STATISTICS_SCHEMA_VERSION",
    "write_final_three_seed_outputs",
]
