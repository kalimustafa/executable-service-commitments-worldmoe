"""Measured, atomic Seed-1 reports for the formal R4 mainline."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifact_io import atomic_json, atomic_text
from .hashing import sha256_file
from .statistics import paired_daily_inference
from .trainer import PHASE_ORDER


SEED1_RESULTS_NAME = "R4_MAINLINE_SEED1_RESULTS.json"
SEED1_REPORT_NAME = "R4_MAINLINE_SEED1_REPORT.md"


def _phase_rows(
    state: Mapping[str, Any], phase_order: tuple[str, ...] = PHASE_ORDER
) -> list[dict[str, Any]]:
    seed1 = state.get("seed1", state)
    phases = seed1.get("phases", {}) if isinstance(seed1, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for phase in phase_order:
        item = phases.get(phase, {}) if isinstance(phases, Mapping) else {}
        item = item if isinstance(item, Mapping) else {}
        gate = item.get("capability_gate", {})
        gate = gate if isinstance(gate, Mapping) else {"status": str(gate)}
        rows.append(
            {
                "phase": phase,
                "status": str(item.get("status", "NOT_STARTED")),
                "capability_gate": str(gate.get("status", "NOT_RUN")),
                "evaluated_days": int(gate.get("evaluated_days", 0) or 0),
                "horizon": gate.get("horizon"),
                "metrics": dict(gate.get("metrics", {})) if isinstance(gate.get("metrics", {}), Mapping) else {},
                "checkpoint": item.get("checkpoint"),
                "selected_candidate_epoch": item.get("selected_candidate_epoch"),
            }
        )
    return rows


def _selected_a2_evidence(context: Any, state: Mapping[str, Any]) -> dict[str, Any] | None:
    is_distributional = (
        str(getattr(context, "architecture_version", ""))
        == "continuous_distributional_joint_moe_r4_v6"
    )
    seed1 = state.get("seed1", state)
    phases = seed1.get("phases", {}) if isinstance(seed1, Mapping) else {}
    a2 = phases.get("a2", {}) if isinstance(phases, Mapping) else {}
    if not isinstance(a2, Mapping) or not a2.get("checkpoint"):
        return None
    checkpoint = Path(str(a2["checkpoint"])).resolve()
    if not checkpoint.is_file():
        return None
    try:
        selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        selected = torch.load(checkpoint, map_location="cpu")
    if not isinstance(selected, Mapping):
        return None
    source_sha = str(selected.get("selected_from_sha256", ""))
    if len(source_sha) != 64:
        return None
    evidence_path = (
        Path(context.output_root)
        / f"seed={int(context.seeds[0])}"
        / "phase=a2"
        / "capability"
        / source_sha
        / "a2_daily.json"
    )
    if not evidence_path.is_file():
        return None
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if payload.get("complete") is not True or not isinstance(rows, list) or len(rows) != 179:
        return None
    dates = [str(row.get("date", ""))[:10] for row in rows]
    if len(set(dates)) != 179 or dates != sorted(dates):
        return None
    completed_origins = sum(int(row.get("completed_origins", 0)) for row in rows)
    validated_origins = sum(int(row.get("independently_validated_origins", 0)) for row in rows)
    costs = np.asarray([float(row["operating_cost_usd"]) for row in rows], dtype=float)
    benchmark = np.asarray([float(row["benchmark_operating_cost_usd"]) for row in rows], dtype=float)
    cvar = np.asarray([float(row["weighted_cvar95_usd"]) for row in rows], dtype=float)
    empirical_risk_samples: int | None = None
    if is_distributional:
        counts = {
            int(row.get("formal_empirical_risk_sample_count", -1)) for row in rows
        }
        measurements = {
            str(row.get("risk_measurement", "")) for row in rows
        }
        if counts != {512} or measurements != {
            "empirical_post_shield_complete_future_replay"
        }:
            # Neural risk-head outputs and small-N training proxies are never
            # admissible as formal V6 CVaR evidence.
            return None
        empirical_risk_samples = 512
    normalized_sla = np.asarray([float(row["normalized_sla_shortfall"]) for row in rows], dtype=float)
    raw_sla = np.asarray([float(row.get("sla_shortfall_raw_work_units", math.nan)) for row in rows], dtype=float)
    correction_rate = np.asarray([float(row["corrective_intervention_rate"]) for row in rows], dtype=float)
    correction_mean = np.asarray([float(row["corrective_intervention_mean"]) for row in rows], dtype=float)
    correction_p95 = np.asarray([float(row["corrective_intervention_p95"]) for row in rows], dtype=float)
    latency = np.asarray([float(row["inference_latency_ms"]) for row in rows], dtype=float)
    arrays = (costs, benchmark, cvar, normalized_sla, raw_sla, correction_rate, correction_mean, correction_p95, latency)
    if not all(np.isfinite(value).all() for value in arrays):
        return None
    paired = costs - benchmark
    inference = paired_daily_inference(
        paired, block_days=7, replicates=10_000, confidence=0.95, seed=20240813
    )
    categories: dict[str, int] = {}
    for row in rows:
        value = row.get("raw_action_violation_categories", {})
        if isinstance(value, Mapping):
            for name, count in value.items():
                categories[str(name)] = categories.get(str(name), 0) + int(count)
    model_total = float(costs.sum())
    benchmark_total = float(benchmark.sum())
    result = {
        "validation_dates": 179,
        "validation_origins": completed_origins,
        "independently_validated_origins": validated_origins,
        "complete_96_origin_validation_day": bool(
            completed_origins == 17_184 and validated_origins == 17_184
        ),
        "total_operating_cost_usd": model_total,
        "benchmark_total_operating_cost_usd": benchmark_total,
        "absolute_cost_difference_usd": model_total - benchmark_total,
        "cost_reduction_percentage": 100.0 * (benchmark_total - model_total) / max(abs(benchmark_total), 1.0e-12),
        "paired_daily_mean_difference_usd": float(paired.mean()),
        "daily_win_rate": float(np.mean(paired < 0.0)),
        "preliminary_7_day_block_ci_95": [inference.lower, inference.upper],
        "preliminary_7_day_block_one_sided_upper_95": inference.one_sided_upper,
        "sla_shortfall_raw_work_units_total": float(raw_sla.sum()),
        "sla_shortfall_normalized_daily_mean": float(normalized_sla.mean()),
        "independent_feasibility_fraction": validated_origins / max(completed_origins, 1),
        "rejected_origins": 17_184 - completed_origins,
        "corrective_intervention_rate": float(correction_rate.mean()),
        "corrective_intervention_mean": float(correction_mean.mean()),
        "corrective_intervention_p95_of_daily_p95": float(np.quantile(correction_p95, 0.95)),
        "raw_action_violation_categories": categories,
        "b1_inference_latency_ms_mean": float(latency.mean()),
        "selected_checkpoint_lineage": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "selected_from": selected.get("selected_from"),
            "selected_from_sha256": source_sha,
            "selected_epoch": selected.get("selected_epoch"),
        },
        "daily_evidence_path": str(evidence_path),
        "daily_evidence_sha256": sha256_file(evidence_path),
    }
    if is_distributional:
        result.update(
            {
                "empirical_post_shield_cvar95_usd_mean": float(cvar.mean()),
                "formal_empirical_risk_sample_count": empirical_risk_samples,
                "risk_measurement": "empirical_post_shield_complete_future_replay",
            }
        )
    else:
        result["weighted_20_particle_cvar95_usd_mean"] = float(cvar.mean())
    return result


def write_seed1_report(context: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    report_root = Path(context.output_root) / "reports"
    json_path = report_root / SEED1_RESULTS_NAME
    markdown_path = report_root / SEED1_REPORT_NAME
    rows = _phase_rows(
        state, tuple(getattr(context, "phase_order", PHASE_ORDER))
    )
    all_phases_pass = all(
        row["status"] == "COMPLETE" and row["capability_gate"] in {"PASS", "PASSED"}
        for row in rows
    )
    measured = _selected_a2_evidence(context, state) if all_phases_pass else None
    measured_complete = bool(
        measured
        and measured["validation_dates"] == 179
        and measured["validation_origins"] == 17_184
        and measured["independently_validated_origins"] == 17_184
        and measured["complete_96_origin_validation_day"]
    )
    any_failed = any(
        row["status"] == "FAILED" or row["capability_gate"] == "FAIL" for row in rows
    )
    is_distributional = (
        str(getattr(context, "architecture_version", ""))
        == "continuous_distributional_joint_moe_r4_v6"
    )
    pass_label = "R4_V6_MAINLINE_PASS" if is_distributional else "R4_MAINLINE_PASS"
    failed_label = "R4_V6_MAINLINE_FAILED" if is_distributional else "R4_MAINLINE_FAILED"
    incomplete_label = (
        "R4_V6_MAINLINE_NEEDS_IMPROVEMENT"
        if is_distributional
        else "R4_MAINLINE_NEEDS_IMPROVEMENT"
    )
    terminal_label = (
        pass_label
        if all_phases_pass and measured_complete
        else failed_label
        if any_failed
        else incomplete_label
    )
    payload = {
        "schema_version": (
            "dcvpp.e2e_moe_distributional_r4.mainline_seed1_results.v1"
            if is_distributional
            else "dcvpp.e2e_moe_hybrid_r4.mainline_seed1_results.v1"
        ),
        "status": "COMPLETE" if terminal_label == pass_label else "PARTIAL",
        "terminal_label": terminal_label,
        "lineage_id": context.lineage_id,
        "config_sha256": context.config_sha256,
        "experiment_id": getattr(
            context, "experiment_id", "e2e_moe_hybrid_r4_formal_v1"
        ),
        "architecture_version": getattr(
            context, "architecture_version", "hybrid_cfc_transformer_moe_r4"
        ),
        "phase_order": [row["phase"] for row in rows],
        "seed": int(context.seeds[0]),
        "capability_gates": {row["phase"]: row["capability_gate"] for row in rows},
        "phase_evidence": rows,
        "complete_validation_days": 0 if measured is None else measured["validation_dates"],
        "complete_validation_origins": 0 if measured is None else measured["validation_origins"],
        "complete_96_origin_validation_day": False if measured is None else measured["complete_96_origin_validation_day"],
        "mainline_metrics": measured,
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "scientific_completion": terminal_label == pass_label,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    atomic_json(json_path, payload)

    title = "# R4-V6 Distributional Mainline Seed-1 Report" if is_distributional else "# R4 Mainline Seed-1 Report"
    lines = [title, "", f"Terminal label: **{terminal_label}**", ""]
    if measured is None:
        lines.extend(
            [
                "The complete selected-A2 179-day evidence is not available; no mainline performance value is reported.",
                "",
            ]
        )
    else:
        ci = measured["preliminary_7_day_block_ci_95"]
        lines.extend(
            [
                f"- Total operating cost: {measured['total_operating_cost_usd']:.6f} USD; Price-aware EDF: {measured['benchmark_total_operating_cost_usd']:.6f} USD.",
                f"- Absolute cost difference: {measured['absolute_cost_difference_usd']:.6f} USD; cost reduction: {measured['cost_reduction_percentage']:.6f}%.",
                f"- Mean paired daily difference: {measured['paired_daily_mean_difference_usd']:.6f} USD; daily win rate: {measured['daily_win_rate']:.6f}.",
                f"- Preliminary 7-day moving-block 95% CI: [{ci[0]:.6f}, {ci[1]:.6f}] USD.",
                (
                    "- Empirical post-shield CVaR95: "
                    f"{measured['empirical_post_shield_cvar95_usd_mean']:.6f} USD "
                    f"from N={measured['formal_empirical_risk_sample_count']} complete-future replays "
                    "(risk metric, not a statistical CI)."
                    if is_distributional
                    else f"- Weighted 20-particle CVaR95: {measured['weighted_20_particle_cvar95_usd_mean']:.6f} USD (risk metric, not a statistical CI)."
                ),
                f"- SLA shortfall: {measured['sla_shortfall_raw_work_units_total']:.6f} raw work units; normalized daily mean {measured['sla_shortfall_normalized_daily_mean']:.9f}.",
                f"- Independent feasibility: {measured['independent_feasibility_fraction']:.9f}; rejected origins: {measured['rejected_origins']}.",
                f"- Corrective intervention: rate {measured['corrective_intervention_rate']:.9f}, mean {measured['corrective_intervention_mean']:.9f}, P95 {measured['corrective_intervention_p95_of_daily_p95']:.9f}.",
                f"- Raw-action violation categories: `{json.dumps(measured['raw_action_violation_categories'], sort_keys=True)}`.",
                f"- B1 deployment inference latency: {measured['b1_inference_latency_ms_mean']:.6f} ms.",
                f"- Selected checkpoint: `{measured['selected_checkpoint_lineage']['path']}`; SHA-256 `{measured['selected_checkpoint_lineage']['sha256']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "| Phase | Training status | Capability gate | Validation days | Horizon |",
            "|---|---:|---:|---:|---:|",
            *(f"| {row['phase']} | {row['status']} | {row['capability_gate']} | {row['evaluated_days']} | {row['horizon']} |" for row in rows),
            "",
            "2024 formal temporal-test payloads opened: **NO**",
            "",
            "2025 locked-test payloads opened: **NO**",
            "",
        ]
    )
    atomic_text(markdown_path, "\n".join(lines))
    return {**payload, "json_report": str(json_path), "markdown_report": str(markdown_path)}


__all__ = ["SEED1_REPORT_NAME", "SEED1_RESULTS_NAME", "write_seed1_report"]
