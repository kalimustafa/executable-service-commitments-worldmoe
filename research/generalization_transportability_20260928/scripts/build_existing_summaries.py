from __future__ import annotations

import csv
import json
import math
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
TASK = WORKSPACE / "generalization_transportability_20260928"
PACKAGE = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928"
SOURCE = PACKAGE / "source_data"
EXPERIMENTS = PACKAGE / "experiments"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def number(row: dict, *names: str, default: float = math.nan) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                result = float(value)
                if math.isfinite(result):
                    return result
            except (TypeError, ValueError):
                pass
    return default


def first(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def out_value(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.10g}" if math.isfinite(value) else "NA"
    return str(value)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return str(path)


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: out_value(row.get(key)) for key in columns})


NORM_COLUMNS = [
    "experiment_family", "condition", "market", "zone_scope", "workload_source",
    "workload_type", "service_window_h", "seed", "reference_cost_usd",
    "policy_cost_usd", "saving_usd", "saving_pct", "total_it_mwh",
    "flexible_it_mwh", "shifted_it_mwh", "saving_per_total_it_mwh",
    "saving_per_flexible_mwh", "saving_per_shifted_mwh", "certificate_usd",
    "capture_fraction", "constraint_retention", "late_work_it_mwh",
    "closing_it_mwh", "feasible", "source_file", "notes",
]

MATRIX_COLUMNS = [
    "experiment_family", "condition", "market", "zone_scope", "workload_source",
    "workload_type", "service_window_h", "utilization", "flexible_share",
    "pv_scale", "import_scale", "ramp_limit", "pipeline", "saving_usd",
    "saving_pct", "flexible_it_mwh", "shifted_it_mwh", "late_it_mwh",
    "closing_it_mwh", "cvar95_usd", "certificate_usd", "capture_fraction",
    "saving_per_flexible_mwh", "saving_per_shifted_mwh",
    "constraint_retention", "feasible", "source_file", "notes",
]

MANIFEST_COLUMNS = [
    "experiment_id", "family", "timestamp", "seed", "checkpoint", "zone",
    "workload", "utilization", "flexible_share", "service_window_h",
    "pv_scale", "import_scale", "ramp_limit", "reference_type", "status",
    "feasible", "runtime_seconds", "output_file", "notes",
]


def main() -> None:
    total_eagle = number(
        read_csv(EXPERIMENTS / "E03_heterogeneous_deadlines" / "mixed_deadline_results.csv")[0],
        "total_it_mwh",
        default=2981.607392515086,
    )
    windows = read_csv(SOURCE / "fig3_service_windows.csv")
    frontier = read_csv(EXPERIMENTS / "E04_frontier_nonpreemptive" / "frontier_results.csv")
    frontier_mapping = read_json(EXPERIMENTS / "E04_frontier_nonpreemptive" / "mapping.json")
    frontier_total = number(frontier_mapping, "job_energy_mwh", default=math.nan)
    physical = read_csv(EXPERIMENTS / "E06_physical_sensitivity" / "results.csv")
    mixed = read_csv(EXPERIMENTS / "E03_heterogeneous_deadlines" / "mixed_deadline_results.csv")
    grid = read_csv(SOURCE / "fig5_electrical_constraints.csv")
    events = read_csv(SOURCE / "fig6_event_influence.csv")
    exclusions = read_csv(SOURCE / "worldmoe_financial_exclusions.csv")
    information = read_csv(SOURCE / "fig4_information_gate.csv")
    signals = read_csv(SOURCE / "fig3_ablation.csv")
    g3_screen_path = TASK / "outputs/g3_utilization_flex/utilization_flex_results.csv"
    g3_screen = read_csv(g3_screen_path) if g3_screen_path.exists() else []
    g3_frontier_path = TASK / "outputs/g3_utilization_flex/feasibility_frontier.csv"
    g3_frontier = read_csv(g3_frontier_path) if g3_frontier_path.exists() else []

    norm: list[dict] = []
    matrix: list[dict] = []
    manifest: list[dict] = []

    def add_norm(
        family: str, condition: str, *, saving: float = math.nan,
        reference: float = math.nan, policy: float = math.nan,
        total: float = math.nan, flexible: float = math.nan,
        shifted: float = math.nan, certificate: float = math.nan,
        late: float = math.nan, closing: float = math.nan,
        service_window: float = math.nan, seed: str = "NA",
        feasible: str = "NA", constraint_retention: float = math.nan,
        source: Path, notes: str = "", market: str = "NYISO",
        zone_scope: str = "4 NYISO zones", workload_source: str = "Eagle",
        workload_type: str = "aggregate divisible workload",
    ) -> dict:
        pct = saving / reference * 100.0 if math.isfinite(saving) and math.isfinite(reference) and reference > 0 else math.nan
        per_total = saving / total if math.isfinite(saving) and math.isfinite(total) and total > 0 else math.nan
        per_flex = saving / flexible if math.isfinite(saving) and math.isfinite(flexible) and flexible > 0 else math.nan
        per_shifted = saving / shifted if math.isfinite(saving) and math.isfinite(shifted) and shifted > 1e-8 else math.nan
        capture = saving / certificate if math.isfinite(saving) and math.isfinite(certificate) and certificate > 0 else math.nan
        item = dict(
            experiment_family=family, condition=condition, market=market,
            zone_scope=zone_scope, workload_source=workload_source,
            workload_type=workload_type, service_window_h=service_window,
            seed=seed, reference_cost_usd=reference, policy_cost_usd=policy,
            saving_usd=saving, saving_pct=pct, total_it_mwh=total,
            flexible_it_mwh=flexible, shifted_it_mwh=shifted,
            saving_per_total_it_mwh=per_total,
            saving_per_flexible_mwh=per_flex,
            saving_per_shifted_mwh=per_shifted,
            certificate_usd=certificate, capture_fraction=capture,
            constraint_retention=constraint_retention,
            late_work_it_mwh=late, closing_it_mwh=closing,
            feasible=feasible, source_file=relative(source), notes=notes,
        )
        norm.append(item)
        return item

    def add_matrix(item: dict, **extra) -> None:
        row = {
            "experiment_family": item.get("experiment_family"),
            "condition": item.get("condition"),
            "market": item.get("market", "NYISO"),
            "zone_scope": item.get("zone_scope", "4 NYISO zones"),
            "workload_source": item.get("workload_source", "Eagle"),
            "workload_type": item.get("workload_type", "aggregate divisible workload"),
            "service_window_h": item.get("service_window_h"),
            "utilization": "NA",
            "flexible_share": "NA",
            "pv_scale": "NA",
            "import_scale": "NA",
            "ramp_limit": "NA",
            "pipeline": "WorldMoE",
            "saving_usd": item.get("saving_usd"),
            "saving_pct": item.get("saving_pct"),
            "flexible_it_mwh": item.get("flexible_it_mwh"),
            "shifted_it_mwh": item.get("shifted_it_mwh"),
            "late_it_mwh": item.get("late_work_it_mwh"),
            "closing_it_mwh": item.get("closing_it_mwh"),
            "cvar95_usd": "NA",
            "certificate_usd": item.get("certificate_usd"),
            "capture_fraction": item.get("capture_fraction"),
            "saving_per_flexible_mwh": item.get("saving_per_flexible_mwh"),
            "saving_per_shifted_mwh": item.get("saving_per_shifted_mwh"),
            "constraint_retention": item.get("constraint_retention"),
            "feasible": item.get("feasible"),
            "source_file": item.get("source_file"),
            "notes": item.get("notes"),
        }
        row.update(extra)
        matrix.append(row)

    # G4 and the main annual Eagle benchmark reuse the three frozen seeds.
    window_summary: dict[float, dict] = {}
    for row in windows:
        window = number(row, "window_h")
        seed = first(row, "seed", default="NA")
        saving = number(row, "saving_usd")
        reference = number(row, "reference_cost_usd")
        policy = number(row, "adjusted_cost_usd")
        shifted = number(row, "shifted_IT_mwh")
        ceiling = number(row, "certified_savings_ceiling_usd")
        late = number(row, "late_IT_mwh")
        closing = number(row, "closing_it_mwh")
        result = add_norm(
            "service_window", f"{window:g}h", saving=saving, reference=reference,
            policy=policy, total=total_eagle, flexible=total_eagle, shifted=shifted,
            certificate=ceiling, late=late, closing=closing, service_window=window,
            seed=seed, feasible="true" if late < 1e-8 and closing < 1e-8 else "false",
            source=SOURCE / "fig3_service_windows.csv",
            notes="Frozen 2025 retrospective replay; matched earliest-service reference.",
        )
        add_matrix(
            result, cvar95_usd=number(row, "cvar95_usd"),
            pipeline="WorldMoE", flexible_share=1.0,
        )
        window_summary.setdefault(window, {
            "savings": [], "references": [], "ceilings": [],
            "shifted": [], "late": [], "closing": [],
        })
        window_summary[window]["savings"].append(saving)
        window_summary[window]["references"].append(reference)
        window_summary[window]["ceilings"].append(ceiling)
        window_summary[window]["shifted"].append(shifted)
        window_summary[window]["late"].append(late)
        window_summary[window]["closing"].append(closing)
        manifest.append(dict(
            experiment_id=f"G4_existing_{int(window)}h_seed{seed}",
            family="G4_service_window", timestamp="2026-09-28",
            seed=seed, checkpoint=first(row, "forecast_checkpoint_sha256", default="CdfR2 frozen"),
            zone="all", workload="Eagle", utilization=0.0519,
            flexible_share=1.0, service_window_h=window, pv_scale=1.0,
            import_scale=1.0, ramp_limit="NA", reference_type="matched earliest service",
            status="REUSED_EXISTING", feasible="true" if late < 1e-8 and closing < 1e-8 else "false",
            runtime_seconds=number(row, "seconds"), output_file=relative(SOURCE / "fig3_service_windows.csv"),
            notes="Existing frozen-checkpoint annual replay; no refit.",
        ))

    # Add explicit pending rows for required windows not available in the current runner.
    service_rows = []
    for window in [8, 12, 16, 24, 48]:
        if window in window_summary:
            v = window_summary[window]
            mean_saving = sum(v["savings"]) / len(v["savings"])
            mean_reference = sum(v["references"]) / len(v["references"])
            ceiling = sum(v["ceilings"]) / len(v["ceilings"])
            mean_shifted = sum(v["shifted"]) / len(v["shifted"])
            mean_late = sum(v["late"]) / len(v["late"])
            mean_closing = sum(v["closing"]) / len(v["closing"])
            service_rows.append(dict(
                service_window_h=window, seed="three_seed",
                reference_cost_usd=mean_reference, policy_cost_usd=mean_reference - mean_saving,
                saving_usd=mean_saving, saving_pct=100.0 * mean_saving / mean_reference,
                flexible_it_mwh=total_eagle, shifted_it_mwh=mean_shifted,
                late_it_mwh=mean_late, closing_it_mwh=mean_closing, certificate_usd=ceiling,
                capture_fraction=mean_saving / ceiling if ceiling > 0 else "NA",
                saving_per_flexible_mwh=mean_saving / total_eagle,
                marginal_value_usd_per_added_hour="NA", feasible="true",
                status="REUSED_EXISTING", notes="Mean over three frozen seeds; matched reference.",
            ))
        else:
            service_rows.append(dict(
                service_window_h=window, seed="primary_only", reference_cost_usd="NA",
                policy_cost_usd="NA", saving_usd="NA", saving_pct="NA",
                flexible_it_mwh=total_eagle, shifted_it_mwh="NA", late_it_mwh="NA",
                closing_it_mwh="NA", certificate_usd="NA", capture_fraction="NA",
                saving_per_flexible_mwh="NA", marginal_value_usd_per_added_hour="NA",
                feasible="NA", status="NOT_RUN",
                notes="No current frozen replay/certificate for this window; preserve as pending.",
            ))
    valid_windows = [row for row in service_rows if row.get("status") == "REUSED_EXISTING"]
    ordered = sorted(valid_windows, key=lambda row: int(row["service_window_h"]))
    for i, item in enumerate(ordered):
        if i:
            prior = ordered[i - 1]
            item["marginal_value_usd_per_added_hour"] = (
                (float(item["saving_usd"]) - float(prior["saving_usd"]))
                / (int(item["service_window_h"]) - int(prior["service_window_h"]))
            )
    save_csv(
        TASK / "outputs/g4_service_window/service_window_results.csv",
        service_rows,
        ["service_window_h", "seed", "reference_cost_usd", "policy_cost_usd",
         "saving_usd", "saving_pct", "flexible_it_mwh", "shifted_it_mwh",
         "late_it_mwh", "closing_it_mwh", "certificate_usd", "capture_fraction",
         "saving_per_flexible_mwh", "marginal_value_usd_per_added_hour",
         "feasible", "status", "notes"],
    )

    # G2 compares different workload structures with normalized quantities.
    g2_rows = []
    eagle_window = window_summary[24]
    eagle_saving = sum(eagle_window["savings"]) / len(eagle_window["savings"])
    eagle_reference = sum(eagle_window["references"]) / len(eagle_window["references"])
    eagle24 = [r for r in windows if number(r, "window_h") == 24]
    eagle_shift = sum(number(r, "shifted_IT_mwh") for r in eagle24) / len(eagle24)
    eagle_late = sum(number(r, "late_IT_mwh") for r in eagle24) / len(eagle24)
    eagle_closing = sum(number(r, "closing_it_mwh") for r in eagle24) / len(eagle24)
    g2_rows.append(dict(
        workload="Eagle", resource_model="aggregate divisible workload",
        service_window_h=24, saving_usd=eagle_saving,
        saving_pct=100 * eagle_saving / eagle_reference,
        saving_per_it_mwh=eagle_saving / total_eagle,
        saving_per_shifted_mwh=eagle_saving / eagle_shift if eagle_shift > 1e-8 else "NA",
        total_it_mwh=total_eagle, shifted_it_mwh=eagle_shift,
        late_work_it_mwh=eagle_late, closing_work_it_mwh=eagle_closing,
        service_feasible="true" if eagle_late < 1e-8 and eagle_closing < 1e-8 else "false",
        source_file=relative(SOURCE / "fig3_service_windows.csv"),
        notes="Three-seed mean; dollars are not directly comparable to Frontier.",
    ))
    frontier_json_by_window = {}
    for row in frontier:
        window = number(row, "window_h")
        source_json = Path(first(row, "source_json"))
        summary = read_json(source_json) if source_json.exists() else {}
        saving = number(row, "saving_usd")
        late = number(row, "late_job_it_mwh")
        closing = number(row, "closing_job_it_mwh")
        shifted = number(summary, "shifted_IT_mwh", "shifted_it_mwh", "shifted_work_it_mwh")
        ref = number(summary, "reference_cost_usd")
        if not math.isfinite(ref):
            pct = number(row, "saving_pct")
            ref = saving / (pct / 100.0) if pct > 0 else math.nan
        policy = number(summary, "adjusted_cost_usd", "policy_cost_usd")
        if not math.isfinite(policy) and math.isfinite(ref):
            policy = ref - saving
        row_g2 = dict(
            workload="Frontier", resource_model="nonpreemptive jobs",
            service_window_h=window, saving_usd=saving,
            saving_pct=number(row, "saving_pct"),
            saving_per_it_mwh=saving / frontier_total if frontier_total > 0 else "NA",
            saving_per_shifted_mwh=saving / shifted if math.isfinite(shifted) and shifted > 1e-8 else "NA",
            total_it_mwh=frontier_total, shifted_it_mwh=shifted,
            late_work_it_mwh=late, closing_work_it_mwh=closing,
            service_feasible="true" if late < 1e-8 else "false",
            source_file=relative(source_json),
            notes="Nonpreemptive Frontier jobs; saving is against the same-job earliest reference. Dollar totals should not be compared directly with Eagle.",
        )
        g2_rows.append(row_g2)
        frontier_json_by_window[window] = summary
        item = add_norm(
            "cross_workload", f"Frontier {window:g}h nonpreemptive",
            saving=saving, reference=ref, policy=policy, total=frontier_total,
            flexible=frontier_total, shifted=shifted, late=late, closing=closing,
            service_window=window, feasible="true" if late < 1e-8 else "false",
            source=source_json, workload_source="Frontier",
            workload_type="nonpreemptive jobs",
            notes="Same-job earliest-service matched reference; not dollar-comparable with Eagle.",
        )
        add_matrix(
            item, pipeline="WorldMoE nonpreemptive",
            cvar95_usd=number(row, "cvar95_usd"),
            flexible_share=1.0,
        )
        manifest.append(dict(
            experiment_id=f"G2_Frontier_{int(window)}h_primary",
            family="G2_cross_workload", timestamp="2026-09-28",
            seed="20260920", checkpoint="CdfR2 frozen primary",
            zone="4 mapped sites", workload="Frontier nonpreemptive",
            utilization="NA", flexible_share=1.0, service_window_h=window,
            pv_scale=1.0, import_scale=1.0, ramp_limit="NA",
            reference_type="same-job earliest service", status="REUSED_EXISTING",
            feasible="true" if late < 1e-8 else "false",
            runtime_seconds=number(row, "execution_seconds"),
            output_file=relative(source_json),
            notes="Existing full-trace job replay; no Frontier tuning.",
        ))
    save_csv(
        TASK / "tables/TABLE_G2_CROSS_WORKLOAD.csv",
        g2_rows,
        ["workload", "resource_model", "service_window_h", "saving_usd",
         "saving_pct", "saving_per_it_mwh", "saving_per_shifted_mwh",
         "total_it_mwh", "shifted_it_mwh", "late_work_it_mwh",
         "closing_work_it_mwh", "service_feasible", "source_file", "notes"],
    )

    # G3's legacy all-flexible outcomes are retained as infeasible diagnostics.
    old_g3 = read_csv(EXPERIMENTS / "E05_workload_scaling" / "results.csv")
    old_feas = read_csv(EXPERIMENTS / "E05_workload_scaling" / "earliest_service_feasibility.csv")
    feas_by_util = {round(number(r, "target_utilization"), 4): r for r in old_feas}
    for row in old_g3:
        util = number(row, "target_utilization")
        result = add_norm(
            "utilization_flexibility_legacy_diagnostic", f"{util:.0%} utilization; all flexible",
            saving=number(row, "saving_usd"), reference=number(row, "reference_cost_usd"),
            policy=number(row, "policy_cost_usd"), total=number(row, "it_energy_mwh"),
            flexible=number(row, "it_energy_mwh"), shifted=number(row, "shifted_IT_mwh"),
            late=number(row, "late_IT_mwh"), closing=number(row, "closing_it_mwh"),
            service_window=24, feasible="false",
            source=Path(first(row, "source_json")),
            notes="Existing raw policy value is retained only as an infeasible diagnostic; the all-workload earliest-service screen fails.",
        )
        add_matrix(result, utilization=util, flexible_share=1.0,
                   cvar95_usd=number(row, "cvar95_usd"),
                   pipeline="WorldMoE (infeasible diagnostic)")
        manifest.append(dict(
            experiment_id=f"G3_existing_allflex_util{int(util*100)}",
            family="G3_utilization_flex", timestamp="2026-09-28",
            seed="20260920", checkpoint="CdfR2 frozen primary",
            zone="all", workload="Eagle scaled", utilization=util,
            flexible_share=1.0, service_window_h=24, pv_scale=1.0,
            import_scale=1.0, ramp_limit="NA", reference_type="matched earliest service",
            status="REUSED_INFEASIBLE_DIAGNOSTIC", feasible="false",
            runtime_seconds=number(row, "seconds"),
            output_file=relative(Path(first(row, "source_json"))),
            notes=f"Earliest-service lateness {number(feas_by_util.get(round(util, 4), {}), 'earliest_late_it_mwh'):.6g} MWh; raw policy economics excluded from delivered-value summaries.",
        ))

    # New rigid/flexible grid: preserve infeasible cells without economic values.
    g3_table = []
    for row in g3_screen:
        util = number(row, "utilization_target")
        share = number(row, "flexible_share")
        late = number(row, "late_it_mwh", "earliest_late_it_mwh")
        feasible = first(row, "structurally_feasible", default="NA")
        condition = f"{util:.3%} utilization / {share:.0%} flexible"
        item = add_norm(
            "utilization_flexibility", condition,
            saving=number(row, "annual_saving_usd"),
            reference=number(row, "earliest_reference_cost_usd"),
            policy=number(row, "policy_cost_usd"),
            total=number(row, "total_variable_it_mwh"),
            flexible=number(row, "flexible_it_mwh"),
            shifted=number(row, "shifted_it_mwh"),
            late=late, closing=number(row, "closing_it_mwh"),
            service_window=number(row, "service_window_h"),
            seed=first(row, "seed"), feasible=feasible,
            source=g3_screen_path,
            notes=first(row, "notes"),
        )
        add_matrix(
            item, utilization=util, flexible_share=share,
            pipeline="G3 structural screen; valuation not run",
            cvar95_usd=number(row, "cvar95_usd"),
        )
        g3_table.append({
            "utilization_target": util,
            "utilization_realized": number(row, "utilization_realized"),
            "flexible_share": share,
            "seed": first(row, "seed"),
            "seed_scope": first(row, "seed_scope"),
            "total_variable_it_mwh": number(row, "total_variable_it_mwh"),
            "rigid_it_mwh": number(row, "rigid_it_mwh"),
            "flexible_it_mwh": number(row, "flexible_it_mwh"),
            "earliest_late_it_mwh": number(row, "earliest_late_it_mwh"),
            "annual_saving_usd": number(row, "annual_saving_usd"),
            "saving_per_flexible_mwh": number(row, "saving_per_flexible_mwh"),
            "structurally_feasible": feasible,
            "policy_feasible": first(row, "policy_feasible", default="NA"),
            "status": "REUSED_EXISTING" if math.isfinite(number(row, "annual_saving_usd")) else (
                "STRUCTURALLY_INFEASIBLE" if feasible.lower() == "false" else "POLICY_REPLAY_PENDING"
            ),
            "notes": first(row, "notes"),
        })
        seed = first(row, "seed")
        checkpoint_path = (
            PACKAGE / "engine/checkpoints" /
            f"CdfR2_seed{seed}_epoch{ {'20260920': '06', '20260921': '02', '20260922': '00'}.get(seed, '06') }.pt"
        )
        status = "SKIPPED_STRUCTURALLY_INFEASIBLE" if feasible.lower() == "false" else (
            "REUSED_EXISTING" if math.isfinite(number(row, "annual_saving_usd")) else "FEASIBILITY_SCREEN_ONLY"
        )
        manifest.append(dict(
            experiment_id=first(row, "experiment_id"),
            family="G3_utilization_flex",
            timestamp="2026-09-28", seed=seed,
            checkpoint=relative(checkpoint_path) if checkpoint_path.exists() else "NA",
            zone="all", workload="Eagle rigid/flexible proportional split",
            utilization=util, flexible_share=share,
            service_window_h=number(row, "service_window_h"),
            pv_scale=1.0, import_scale=1.0, ramp_limit="NA",
            reference_type="condition-matched earliest service",
            status=status, feasible=feasible,
            runtime_seconds="NA", output_file=relative(g3_screen_path),
            notes=first(row, "notes"),
        ))

    for row in g3_frontier:
        frontier_late = number(row, "earliest_late_it_mwh")
        frontier_util = number(row, "utilization_tested")
        matrix.append(dict(
            experiment_family="G3_operating_envelope",
            condition=f"{frontier_util:.3%} utilization feasibility bracket",
            market="NYISO", zone_scope="4 NYISO zones", workload_source="Eagle",
            workload_type="aggregate divisible workload", service_window_h=24,
            utilization=frontier_util, flexible_share="NA", pv_scale=1.0,
            import_scale=1.0, ramp_limit="NA", pipeline="earliest-service screen",
            saving_usd="NA", saving_pct="NA", flexible_it_mwh="NA",
            shifted_it_mwh="NA", late_it_mwh=frontier_late,
            closing_it_mwh=number(row, "closing_it_mwh"), cvar95_usd="NA",
            certificate_usd="NA", capture_fraction="NA",
            saving_per_flexible_mwh="NA", saving_per_shifted_mwh="NA",
            constraint_retention="NA",
            feasible=first(row, "structurally_feasible"),
            source_file=relative(g3_frontier_path),
            notes="Eight-step deterministic bisection following the prespecified utilization grid.",
        ))

    # Physical PV and electrical-constraint cases use their own matched references.
    for row in physical:
        scale = number(row, "pv_fraction_of_nominal")
        item = add_norm(
            "physical_PV", f"PV {scale:.0%} of nominal",
            saving=number(row, "saving_usd"), reference=number(row, "reference_cost_usd"),
            policy=number(row, "policy_cost_usd"), total=total_eagle,
            flexible=total_eagle, shifted=number(row, "shifted_it_mwh"),
            late=number(row, "late_it_mwh"), closing=number(row, "closing_it_mwh"),
            service_window=24, feasible="true",
            source=EXPERIMENTS / "E06_physical_sensitivity" / "results.csv",
            notes="Primary-seed 2025 physical sensitivity; scenario-specific matched reference.",
        )
        add_matrix(item, pv_scale=scale, cvar95_usd=number(row, "cvar95_usd"))
    base24_mean = sum(window_summary[24]["savings"]) / len(window_summary[24]["savings"])
    for row in grid:
        saving = number(row, "saving_usd")
        cond = first(row, "scenario")
        family = "electrical_constraint"
        retention = saving / base24_mean if base24_mean > 0 else math.nan
        item = add_norm(
            family, cond, saving=saving, reference=number(row, "reference_cost_usd"),
            policy=number(row, "policy_cost_usd"), total=total_eagle,
            flexible=total_eagle, certificate=math.nan,
            late=number(row, "late_mean"), closing=number(row, "closing_max"),
            service_window=24, feasible="true" if number(row, "reference_late_max") < 1e-8 else "false",
            constraint_retention=retention,
            source=SOURCE / "fig5_electrical_constraints.csv",
            notes="Existing three-seed sensitivity summary; each condition uses its own reference. Shifted-work and certificate values are unavailable in this summary.",
        )
        import_scale = 0.75 if "cap75" in cond else (0.5 if "cap50" in cond else 1.0)
        ramp_limit = 0.25 if "ramp25" in cond else (0.125 if "ramp125" in cond else "NA")
        add_matrix(
            item, pipeline=first(row, "experiment", default="WorldMoE"),
            import_scale=import_scale, ramp_limit=ramp_limit,
            cvar95_usd=number(row, "cvar95_usd"),
        )

    # Existing mixed-deadline evidence is a separate service-contract condition.
    for row in mixed:
        pathway = first(row, "pathway")
        item = add_norm(
            "mixed_deadlines", pathway, saving=number(row, "annual_saving_usd"),
            reference=number(row, "annual_saving_usd") / (number(row, "saving_pct") / 100.0)
            if number(row, "saving_pct") > 0 else math.nan,
            total=number(row, "total_it_mwh"), flexible=number(row, "total_it_mwh"),
            shifted=number(row, "shifted_it_mwh"), late=number(row, "late_work_it_mwh"),
            closing=number(row, "closing_it_mwh"), service_window=24,
            seed=first(row, "seed"),
            feasible="true" if number(row, "late_work_it_mwh") < 1e-8 and number(row, "closing_it_mwh") < 1e-8 else "false",
            source=EXPERIMENTS / "E03_heterogeneous_deadlines" / "mixed_deadline_results.csv",
            notes=f"Existing deadline pathway {pathway}; full source includes per-deadline completion and lateness.",
        )
        add_matrix(item, pipeline=pathway, cvar95_usd=number(row, "cvar95_usd"),
                   flexible_share=1.0)

    # Event contrasts and financial exclusions are retained with explicit scope.
    for row in events:
        for condition, metric in [("extreme_days", "extreme_usd"), ("ordinary_days", "ordinary_usd")]:
            item = add_norm(
                "market_state_contrast", f"{first(row, 'contrast')} / {condition}",
                saving=number(row, metric), source=SOURCE / "fig6_event_influence.csv",
                notes="Paired model-vs-comparator event contrast; not absolute executable-flexibility savings.",
                workload_source="Eagle", workload_type="paired market-state contrast",
            )
            add_matrix(item, pipeline=first(row, "contrast"),
                       cvar95_usd="NA",
                       notes=f"{first(row, 'definition')} quantile={first(row, 'quantile')}; {first(row, 'extreme_days')} extreme days.")
    exclusion_candidates = []
    for row in exclusions:
        if int(number(row, "excluded_days", default=0)) in (1, 7):
            exclusion_candidates.append(row)
    selected_exclusions = []
    for days in (1, 7):
        candidates = [r for r in exclusion_candidates if int(number(r, "excluded_days")) == days]
        if candidates:
            selected_exclusions.append(min(candidates, key=lambda r: number(r, "retained_total_usd")))
    for row in selected_exclusions:
        item = add_norm(
            "financial_exclusion", f"exclude {first(row, 'excluded_days')} days",
            saving=number(row, "retained_total_usd"), source=SOURCE / "worldmoe_financial_exclusions.csv",
            notes=f"Retained paired contrast, dates {first(row, 'start_date')} to {first(row, 'end_date')}; not annual dispatch savings.",
            workload_type="paired financial-exclusion contrast",
        )
        add_matrix(item, pipeline=first(row, "contrast"))

    # Information restriction and frozen signal interventions.
    for row in information:
        for condition, col in [
            ("native", "native_usd"), ("post_only", "post_only_usd"),
            ("incremental_early_path", "early_path_usd"),
        ]:
            value = number(row, col)
            note = "Information-access pathway; post-only suppresses pre-publication features."
            if condition == "incremental_early_path":
                note = "Incremental native value lost under the pre-publication restriction; not whole-run savings."
            item = add_norm(
                "information_restriction", condition, saving=value,
                reference=555168.2050556969, total=total_eagle, flexible=total_eagle,
                late=number(row, "restricted_late_it_mwh"),
                closing=number(row, "restricted_closing_it_mwh"),
                service_window=24, seed=first(row, "seed"),
                feasible="true" if number(row, "restricted_late_it_mwh") < 1e-8 and number(row, "restricted_closing_it_mwh") < 1e-8 else "false",
                source=SOURCE / "fig4_information_gate.csv", notes=note,
            )
            add_matrix(item, pipeline=condition,
                       cvar95_usd=number(row, "restricted_cvar95_usd"))
    for row in signals:
        variant = first(row, "variant")
        item = add_norm(
            "decision_signal", variant, saving=number(row, "saving_usd"),
            reference=number(row, "reference_cost_usd"),
            policy=number(row, "adjusted_cost_usd"), total=total_eagle,
            flexible=total_eagle, shifted=number(row, "shifted_IT_mwh"),
            certificate=number(row, "certified_savings_ceiling_usd"),
            late=number(row, "late_IT_mwh"), closing=number(row, "closing_it_mwh"),
            service_window=number(row, "window_h"), seed=first(row, "seed"),
            feasible="true" if number(row, "late_IT_mwh") < 1e-8 and number(row, "closing_it_mwh") < 1e-8 else "false",
            source=SOURCE / "fig3_ablation.csv",
            notes="Frozen-weight decision-signal intervention under existing physical allocator; not a retrained module deletion.",
        )
        add_matrix(item, pipeline=f"WorldMoE signal: {variant}",
                   cvar95_usd=number(row, "cvar95_usd"))

    # Files for manuscript tables and reproducibility.
    save_csv(TASK / "outputs/g5_normalized/normalized_metrics.csv", norm, NORM_COLUMNS)
    save_csv(TASK / "tables/TABLE_G4_SERVICE_WINDOW.csv", service_rows,
             ["service_window_h", "seed", "reference_cost_usd", "policy_cost_usd",
              "saving_usd", "saving_pct", "flexible_it_mwh", "shifted_it_mwh",
              "late_it_mwh", "closing_it_mwh", "certificate_usd", "capture_fraction",
              "saving_per_flexible_mwh", "marginal_value_usd_per_added_hour",
              "feasible", "status", "notes"])
    save_csv(TASK / "tables/TABLE_G5_NORMALIZED_TRANSPORTABILITY.csv", norm, NORM_COLUMNS)
    save_csv(
        TASK / "tables/TABLE_G3_UTILIZATION_FLEX.csv", g3_table,
        ["utilization_target", "utilization_realized", "flexible_share", "seed",
         "seed_scope", "total_variable_it_mwh", "rigid_it_mwh", "flexible_it_mwh",
         "earliest_late_it_mwh", "annual_saving_usd", "saving_per_flexible_mwh",
         "structurally_feasible", "policy_feasible", "status", "notes"],
    )
    save_csv(TASK / "outputs/combined/transportability_robustness_matrix.csv", matrix, MATRIX_COLUMNS)
    save_csv(TASK / "experiment_manifest.csv", manifest, MANIFEST_COLUMNS)

    print(json.dumps({
        "normalized_rows": len(norm),
        "robustness_rows": len(matrix),
        "G2_rows": len(g2_rows),
        "G4_rows": len(service_rows),
        "manifest_rows": len(manifest),
        "outputs": [
            "outputs/g5_normalized/normalized_metrics.csv",
            "outputs/combined/transportability_robustness_matrix.csv",
            "tables/TABLE_G2_CROSS_WORKLOAD.csv",
            "tables/TABLE_G4_SERVICE_WINDOW.csv",
            "tables/TABLE_G5_NORMALIZED_TRANSPORTABILITY.csv",
            "experiment_manifest.csv",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
