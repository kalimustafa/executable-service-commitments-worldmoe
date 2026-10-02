"""Report saved matched W1-MPC suffix replays; never load or execute a model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import statistics

import numpy as np

from multiseed_common import REPO, now, write_csv, write_json
import report_multiseed_ablation as shared
from report_inference_ablation import panel_statistics


ROOT = REPO / "artifacts/r_ecmoe_mpc_benchmark_20260912"
RUN_IDS = ("legacy_seed20240813", "current_seed20240813", "current_seed20240829", "current_seed20240917")
POLICIES = {"full": "R-ECMoE", "w1_mpc": "W1-MPC"}
DESIGN = "matched_forecaster_independent_suffix_optimization_comparison"
WINDOWS = shared.WINDOWS
DAY_FIELDS = (*shared.DAY_FIELDS, "realized_operating_cost")
TIMING_METRICS = ("mean_latency_ms_batch1", "p50_latency_ms_batch1", "p95_latency_ms_batch1")
METRICS = {**shared.METRICS, "mean_realized_operating_cost": "平均原单位成本",
           "cvar95_realized_operating_cost": "原单位成本 CVaR95",
           "mean_latency_ms_batch1": "batch-1 端到端平均延迟（ms）",
           "p50_latency_ms_batch1": "batch-1 端到端中位延迟（ms）"}
METRICS["p95_latency_ms_batch1"] = "batch-1 端到端 p95（ms）"
require, finite, read_csv, read_json = shared.require, shared.finite, shared.read_csv, shared.read_json


def _optional(value, name):
    return None if value is None or str(value).strip().lower() in {"", "none", "null"} else finite(value, name)


def _key(row):
    return row["operating_date"], (int(row["stage_id"]), int(row["decision_origin"]), int(row["horizon"]))


def _path(value):
    return Path(value).resolve()


def _timing_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean_latency_ms_batch1": float(values.mean()), "p50_latency_ms_batch1": float(np.quantile(values, .5)),
            "p95_latency_ms_batch1": float(np.quantile(values, .95))}


def _timing_records(latency, summary_by_policy, run_id):
    result = {}
    for policy in POLICIES:
        entry = latency[policy]
        require(entry.get("latency_warmup") == 5 and entry.get("latency_repeats") == 30,
                f"Timing protocol is not five warmups and thirty measured repeats: {run_id}/{policy}")
        values = np.asarray([finite(value, "raw batch1 latency") for value in entry["raw_samples_ms"]])
        require(values.shape == (30,) and bool((values > 0).all()), f"Incomplete/invalid raw timing samples: {run_id}/{policy}")
        measured = _timing_summary(values)
        for name, value in measured.items():
            shared._same_number(entry[name], value, f"latency.json/{run_id}/{policy}/{name}")
            shared._same_number(summary_by_policy[policy][name], value, f"model_summary/{run_id}/{policy}/{name}")
        result[policy] = values
    require(len(latency["w1_mpc"].get("solver_raw_records", [])) == 30, f"Missing timing-case solver records: {run_id}")
    return result


def solver_summary(records, *, run_id, run_label, seed):
    optimal = [shared.flag(row["optimal"], "optimal") for row in records]
    incumbent = [shared.flag(row["has_incumbent"], "has_incumbent") for row in records]
    fallback = [shared.flag(row["fallback_used"], "fallback_used") for row in records]
    require(all(not good or (has and not used) for good, has, used in zip(optimal, incumbent, fallback)),
            f"An optimal record claims fallback/no incumbent: {run_id}")
    require(all(used == (not has) for used, has in zip(fallback, incumbent)), f"Fallback/solution-availability mismatch: {run_id}")
    solve = [finite(row["solve_seconds"], "solve seconds") for row in records]
    decision = [finite(row["decision_wall_seconds"], "MPC decision wall seconds") for row in records]
    require(all(value >= 0 for value in solve + decision), f"Negative solver timings: {run_id}")
    statuses = Counter((str(row.get("status_code", "")), str(row["status"])) for row in records)
    row = {"run_id": run_id, "run_label": run_label, "seed": seed, "record_scope": "formal_date_by_window_solves_only",
           "solver_calls": len(records), "optimal_count": sum(optimal), "optimal_rate": sum(optimal) / len(records),
           "incumbent_count": sum(incumbent), "incumbent_rate": sum(incumbent) / len(records),
           "nonoptimal_incumbent_count": sum(has and not good for has, good in zip(incumbent, optimal)),
           "fallback_count": sum(fallback), "fallback_rate": sum(fallback) / len(records),
           "status_counts_json": json.dumps([{"status_code": code, "status": text, "count": count}
                                              for (code, text), count in sorted(statuses.items())], ensure_ascii=False)}
    for source, prefix in (("relative_gap", "relative_gap"), ("objective", "forecast_plan_objective"),
                           ("objective_bound", "forecast_plan_objective_bound")):
        values = [_optional(record.get(source), source) for record in records]
        values = [value for value in values if value is not None]
        row[f"{prefix}_available_count"] = len(values)
        row[f"{prefix}_mean"] = statistics.mean(values) if values else None
        row[f"{prefix}_p95"] = float(np.quantile(values, .95)) if values else None
        row[f"{prefix}_max"] = max(values) if values else None
    for prefix, values in (("solve_seconds", solve), ("decision_wall_seconds", decision)):
        row[f"{prefix}_total"] = sum(values)
        row[f"{prefix}_mean"] = statistics.mean(values)
        row[f"{prefix}_p50"] = float(np.quantile(values, .5))
        row[f"{prefix}_p95"] = float(np.quantile(values, .95))
        row[f"{prefix}_max"] = max(values)
    row["timing_records_excluded_from_formal_solver_count"] = 30
    return row


def load_evidence(root, *, expected_days=362):
    root = _path(root)
    protocol = read_json(root / "protocol.json")
    require(protocol.get("comparison_design") == DESIGN, "Wrong benchmark comparison design")
    for field in ("no_added_terminal_soc_requirement", "matched_assembler_storage_energy_budgets",
                  "shared_constraint_assembler_and_realized_replay"):
        require(protocol.get(field) is True, f"Unmatched planning/replay contract: {field}")
    for field in ("continuous_closed_loop_claim", "training_performed", "checkpoint_selection_performed", "test_used_to_tune_mpc"):
        require(protocol.get(field) is False, f"Incorrect benchmark boundary: {field}")
    records = {entry["record"]["run_id"]: entry["record"] for entry in protocol.get("records", [])}
    require(set(records) == set(RUN_IDS), "Benchmark protocol does not contain the four training runs")
    panels, metrics, solvers, timings, sources = {}, [], [], {}, [str(root / "protocol.json")]
    common_dates, common_denominators, run_sources = None, None, []
    for run_id in RUN_IDS:
        folder = root / run_id
        metadata = read_json(folder / "metadata.json")
        record = metadata["full_model_record"]
        seed = int(record["seed"])
        require(metadata.get("status") == "completed" and metadata["run_id"] == run_id and int(metadata["seed"]) == seed,
                f"Benchmark run is incomplete: {run_id}")
        require(metadata.get("comparison_design") == DESIGN and record["run_id"] == run_id, f"Wrong run design/identity: {run_id}")
        for name in ("checkpoint", "w1_checkpoint"):
            require(_path(record[name]) == _path(records[run_id][name]), f"Changed frozen source {name}: {run_id}")
        for name in ("same_frozen_forecaster_within_run", "all_actions_fixed_before_realized_target_read"):
            require(metadata.get(name) is True, f"Missing matched input/outcome boundary: {run_id}/{name}")
        for name in ("continuous_closed_loop_claim", "training_performed", "checkpoint_selection_performed", "test_used_to_tune_mpc"):
            require(metadata.get(name) is False, f"Incorrect evaluation boundary: {run_id}/{name}")
        lineage = metadata.get("data_lineage", {})
        require(metadata.get("year", lineage.get("year")) == 2025
                and metadata.get("split", lineage.get("split")) == "locked_test", f"Not the 2025 locked test: {run_id}")
        decisions, daily, summaries, solver_records = (read_csv(folder / name) for name in
            ("per_decision_models.csv", "per_day_models.csv", "model_summary.csv", "solver_records.csv"))
        latency = read_json(folder / "latency.json")
        require(len(decisions) == expected_days * 6 * 2 and len(daily) == expected_days * 2 and len(summaries) == 2,
                f"Wrong replay coverage: {run_id}")
        require(len(solver_records) == expected_days * 6, f"Wrong formal solver coverage: {run_id}")
        require(int(metadata["observed_replay_rows"]) == len(decisions) and int(metadata["formal_mpc_solves"]) == len(solver_records),
                f"Metadata/table count mismatch: {run_id}")
        for rows in (decisions, daily, summaries):
            require(all(row["run_id"] == run_id and int(row["seed"]) == seed and row["variant"] in POLICIES
                        and row["policy_name"] == POLICIES[row["variant"]] and row["comparison_design"] == DESIGN
                        and not shared.flag(row["separately_retrained"], "separately_retrained") for row in rows),
                    f"Inconsistent policy identity or retraining claim: {run_id}")
        summary_by_policy = {row["variant"]: row for row in summaries}
        require(set(summary_by_policy) == set(POLICIES), f"Missing policy summary: {run_id}")
        timings[run_id] = _timing_records(latency, summary_by_policy, run_id)
        dates = sorted(metadata["dataset_dates"])
        require(len(dates) == len(set(dates)) == expected_days and all(date.fromisoformat(day).year == 2025 for day in dates),
                f"Invalid test date list: {run_id}")
        require(common_dates is None or dates == common_dates, "Four runs use different test dates")
        common_dates = dates
        expected_keys = {(day, window) for day in dates for window in WINDOWS}
        require({_key(row) for row in solver_records} == expected_keys, f"Duplicate/missing solver window: {run_id}")
        require(all(row["run_id"] == run_id and int(row["seed"]) == seed for row in solver_records), f"Foreign solver record: {run_id}")
        solvers.append(solver_summary(solver_records, run_id=run_id, run_label=record["run_label"], seed=seed))
        run_denominators = None
        for policy in POLICIES:
            selected = [row for row in decisions if row["variant"] == policy]
            lookup = {_key(row): row for row in selected}
            require(len(selected) == len(lookup) == expected_days * 6 and set(lookup) == expected_keys,
                    f"Duplicate/missing policy window: {run_id}/{policy}")
            day_list = [row for row in daily if row["variant"] == policy]
            day_map = {row["operating_date"]: row for row in day_list}
            require(len(day_list) == len(day_map) == expected_days and set(day_map) == set(dates), f"Unpaired daily dates: {run_id}/{policy}")
            matrix = lambda name: np.asarray([[finite(lookup[day, window][name], name) for window in WINDOWS] for day in dates])
            matrices = {name: matrix(name) for name in (*DAY_FIELDS, "hard_residual_max")}
            denominators = matrix("cost_normalization_denominator")
            require(run_denominators is None or np.array_equal(denominators, run_denominators), f"Policies have different cost normalization: {run_id}")
            run_denominators = denominators
            for index, day in enumerate(dates):
                saved = day_map[day]
                require(int(saved["decision_count"]) == 6, "Daily row does not average six windows")
                for name in DAY_FIELDS:
                    shared._same_number(saved[name], float(matrices[name][index].mean()), f"daily/{run_id}/{policy}/{day}/{name}")
                hard = float(matrices["hard_residual_max"][index].max())
                shared._same_number(saved["hard_residual_max"], hard, "daily hard residual")
                require(shared.flag(saved["hard_violation"], "daily hard flag") == (hard > shared.HARD_TOLERANCE), "Wrong daily hard flag")
            panel = {"dates": dates, "matrices": matrices,
                     "latency": _timing_summary(timings[run_id][policy])["p95_latency_ms_batch1"]}
            measured = {**panel_statistics(panel), **_timing_summary(timings[run_id][policy])}
            summary = summary_by_policy[policy]
            for name, value in measured.items():
                shared._same_number(summary[name], value, f"summary/{run_id}/{policy}/{name}")
            expected_checkpoint = record["checkpoint"] if policy == "full" else record["w1_checkpoint"]
            require(_path(summary["checkpoint"]) == _path(expected_checkpoint)
                    and _path(summary["w1_checkpoint"]) == _path(record["w1_checkpoint"]), f"Wrong forecast/controller lineage: {run_id}/{policy}")
            panels[run_id, policy] = panel
            metrics.append({"run_id": run_id, "run_label": record["run_label"], "seed": seed, "variant": policy,
                            "policy_name": POLICIES[policy], "comparison_design": DESIGN, "separately_retrained": False,
                            "days": expected_days, "replay_rows": expected_days * 6,
                            "checkpoint_epoch": int(summary["checkpoint_epoch"]), "checkpoint": summary["checkpoint"],
                            "w1_checkpoint": summary["w1_checkpoint"], **measured})
        require(common_denominators is None or np.array_equal(run_denominators, common_denominators), "Runs have different cost-normalization denominators")
        common_denominators = run_denominators
        sources.extend(str(folder / name) for name in ("metadata.json", "per_decision_models.csv", "per_day_models.csv",
                                                       "model_summary.csv", "solver_records.csv", "latency.json"))
        run_sources.append({"run_id": run_id, "seed": seed, "record": record, "metadata_path": str(folder / "metadata.json")})
    validation = {"scope": "coverage_lineage_and_recalculation_only_not_scientific_qualification", "training_runs": 4,
                  "unique_seed_count": len({row["seed"] for row in metrics}), "seeds": sorted({row["seed"] for row in metrics}),
                  "replay_rows": sum(row["replay_rows"] for row in metrics), "expected_replay_rows": 4 * 2 * expected_days * 6,
                  "solver_records": sum(row["solver_calls"] for row in solvers), "expected_solver_records": 4 * expected_days * 6,
                  "days_per_policy": expected_days, "windows_per_day": 6, "policies": POLICIES,
                  "comparison_design": DESIGN, "same_frozen_forecaster_within_run": True,
                  "continuous_closed_loop_claim": False, "matched_assembler_storage_energy_budgets": True,
                  "no_added_terminal_soc_requirement": True, "training_performed": False, "checkpoint_selection_performed": False,
                  "test_used_to_tune_mpc": False, "prior_2025_already_evaluated": protocol.get("prior_2025_already_evaluated"),
                  "timing_raw_samples_per_policy_per_run": 30, "timing_warmups_per_policy_per_run": 5,
                  "formal_solver_counts_exclude_timing_repeats": True, "source_paths": sources, "source_runs": run_sources,
                  "protocol": protocol, "multiple_comparison_adjustment": False}
    require(validation["unique_seed_count"] == 3, "Four-run/three-unique-seed scope changed")
    return panels, metrics, solvers, timings, validation


def paired_metrics(panels, timings, *, resamples=2000):
    date_indices = shared.circular_blocks(len(panels[RUN_IDS[0], "full"]["dates"]), resamples=resamples)
    repeat_indices = np.random.default_rng(shared.BOOTSTRAP_SEED).integers(0, 30, size=(resamples, 30))
    rows = []
    for run_id in RUN_IDS:
        estimates = {policy: {**panel_statistics(panels[run_id, policy]), **_timing_summary(timings[run_id][policy])} for policy in POLICIES}
        samples = {policy: panel_statistics(panels[run_id, policy], date_indices) for policy in POLICIES}
        for policy in POLICIES:
            values = timings[run_id][policy][repeat_indices]
            samples[policy].update(mean_latency_ms_batch1=values.mean(axis=1), p50_latency_ms_batch1=np.quantile(values, .5, axis=1),
                                   p95_latency_ms_batch1=np.quantile(values, .95, axis=1))
        for metric in METRICS:
            delta = samples["w1_mpc"][metric] - samples["full"][metric]
            low, high = map(float, np.quantile(delta, [.025, .975]))
            rows.append({"run_id": run_id, "seed": int(run_id[-8:]), "metric": metric,
                         "full_value": estimates["full"][metric], "mpc_value": estimates["w1_mpc"][metric],
                         "delta_mpc_minus_full": estimates["w1_mpc"][metric] - estimates["full"][metric],
                         "ci95_low": low, "ci95_high": high,
                         "interpretation": "full_lower" if low > 0 else "mpc_lower" if high < 0 else "interval_includes_zero",
                         "sampling_unit": "matched_batch1_repeat" if metric in TIMING_METRICS else "paired_test_date",
                         "paired_units": 30 if metric in TIMING_METRICS else len(panels[run_id, "full"]["dates"]),
                         "block_days": None if metric in TIMING_METRICS else 7,
                         "resamples": resamples, "bootstrap_seed": shared.BOOTSTRAP_SEED,
                         "method": "paired_repeat_bootstrap" if metric in TIMING_METRICS else "paired_7_day_circular_blocks_recompute_each_side",
                         "comparison_design": DESIGN})
    return rows


def aggregate_metrics(metrics):
    rows = []
    for policy in POLICIES:
        selected = [row for row in metrics if row["variant"] == policy]
        for metric in METRICS:
            values = [row[metric] for row in selected]
            rows.append({"variant": policy, "policy_name": POLICIES[policy], "metric": metric,
                         "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                         "available_runs": len(values), "unique_seeds": len({row["seed"] for row in selected}),
                         "aggregation_basis": "descriptive_four_runs_three_unique_seeds"})
    return rows


def render_report(metrics, aggregate, paired, solvers, validation):
    lookup = {(row["variant"], row["metric"]): row for row in aggregate}
    lines = ["# 同预测器 W1-MPC 与 R-ECMoE 的后缀窗口实测对照", "", "## 主要发现", ""]
    findings = []
    for metric in ("mean_normalized_operating_cost", "cvar95_normalized_operating_cost", "worst_origin_sla", "mean_deadline_expired_work"):
        values = [row for row in paired if row["metric"] == metric]
        counts = Counter(row["interpretation"] for row in values)
        findings.append(f"{METRICS[metric]}：{counts['full_lower']}/4个运行的区间支持R-ECMoE更低，"
                        f"{counts['mpc_lower']}/4支持W1-MPC更低，{counts['interval_includes_zero']}/4方向未确定")
    lines += ["；".join(findings) + "。这些是每运行未作多重比较校正的95%配对区间，不能只凭综合J分数宣称某控制器全面占优。", "",
              f"共{validation['replay_rows']:,}个已测决策窗口、{validation['solver_records']:,}次正式MPC求解；"
              "4个基础训练运行包含3个唯一seed。双方在每个运行内使用同一冻结W1预测器。", ""]
    def table(columns):
        result = ["| 控制策略 | " + " | ".join(METRICS[name] for name in columns) + " |", "|---|" + "---:|" * len(columns)]
        for policy in POLICIES:
            cells = [f"{shared.fmt(lookup[policy, name]['mean'])} ± {shared.fmt(lookup[policy, name]['sample_sd'])}" for name in columns]
            result.append(f"| {POLICIES[policy]} | " + " | ".join(cells) + " |")
        return result
    lines += ["## 成本与尾部风险", "", "下表为四运行描述性均值±样本标准差，不是四个独立seed的统计推断。", ""]
    lines += table(("mean_normalized_operating_cost", "cvar95_normalized_operating_cost", "mean_realized_operating_cost", "cvar95_realized_operating_cost"))
    lines += ["", "## 服务风险与物理约束", ""]
    lines += table(("mean_deadline_expired_work", "cvar95_deadline_expired_work", "worst_origin_sla", "max_hard_residual", "hard_violation_rate", "mean_unserved_energy"))
    lines += ["", "逾期工作量不是违反率或离散任务数；SLA是逾期占比与归一化未供电强度组成的严重度。"
              "成本与逾期CVaR95均从最高ceil(0.05×日期数)个日均值计算。失败回退的回放结果完整保留，不按求解状态删掉不利样本。", "",
              "## 逐运行配对差值", "", "Δ=MPC−R-ECMoE；正值代表R-ECMoE更低。", "",
              "| 运行 | 平均成本 Δ [95% CI] | 成本 CVaR95 Δ [95% CI] | 平均逾期 Δ [95% CI] | 最差 SLA Δ [95% CI] |",
              "|---|---:|---:|---:|---:|"]
    pairs = {(row["run_id"], row["metric"]): row for row in paired}
    run_labels = {row["run_id"]: row["run_label"] for row in metrics}
    for run_id in RUN_IDS:
        cells = []
        for metric in ("mean_normalized_operating_cost", "cvar95_normalized_operating_cost", "mean_deadline_expired_work", "worst_origin_sla"):
            row = pairs[run_id, metric]
            cells.append(f"{shared.fmt(row['delta_mpc_minus_full'])} [{shared.fmt(row['ci95_low'])}, {shared.fmt(row['ci95_high'])}]")
        lines.append(f"| {run_labels[run_id]} | " + " | ".join(cells) + " |")
    lines += ["", "日指标按7日循环移动块、2,000次、seed20240813重采样；每次双方使用相同日期并各自重算CVaR。"
              "SLA先求各窗口日期均值再取六窗口最大值。综合J的日期区间将已测延迟罚项视为固定值。", "",
              "## 端到端延迟与求解器耗时", "",
              "| 运行 | 策略 | batch-1平均 ms | 中位 ms | p95 ms | 测量次数 |", "|---|---|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['run_label']} | {row['policy_name']} | {row['mean_latency_ms_batch1']:.3f} | "
                     f"{row['p50_latency_ms_batch1']:.3f} | {row['p95_latency_ms_batch1']:.3f} | 30 |")
    lines += ["", "每策略5次预热、30次batch-1测量；端到端计时包含W1编码与控制器决策，排除真实目标回放。"
              "两策略按同一次重复配对，延迟区间使用30次重复的配对重采样，不使用日期块；"
              "它只反映登记的RT-origin0/H96计时案例，不代表362天所有情境的延迟分布。", "",
              "| 运行 | 正式求解数 | 记录为最优 | 有可用解 | 非最优可用解 | 回退数 | gap有效数 | gap p95 / max | solver平均 / p95秒 | 决策wall平均 / p95秒 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in solvers:
        lines.append(f"| {row['run_label']} | {row['solver_calls']} | {row['optimal_count']} | {row['incumbent_count']} | "
                     f"{row['nonoptimal_incumbent_count']} | {row['fallback_count']} | {row['relative_gap_available_count']} | "
                     f"{shared.fmt(row['relative_gap_p95'])} / {shared.fmt(row['relative_gap_max'])} | "
                     f"{row['solve_seconds_mean']:.4f} / {row['solve_seconds_p95']:.4f} | "
                     f"{row['decision_wall_seconds_mean']:.4f} / {row['decision_wall_seconds_p95']:.4f} |")
    lines += ["", "solver时间只覆盖求解器调用；decision wall包含建模、求解和公共动作装配但不包含W1编码，"
              "二者都不能替代上面的端到端batch-1时间。gap只对实际记录了界/解的样本统计，缺失不填零；"
              "有可用解不等于已证明最优。状态原文、计数、界与目标的有效样本数见solver_summary.csv；预测优化目标不是实测回放成本。", "",
              "## 比较定义与适用边界", "",
              "- W1-MPC是共享冻结W1确定性点预测的MILP规划器；不是随机或风险敏感MPC。求解时间、线程及gap上限使用冻结protocol.json配置，不根据2025结果重新选择。",
              "- 双方使用同一因果预测包、原动作空间、公共约束装配器与真实结果回放；储能总充放电能量预算匹配装配器，不给MPC额外终端SOC要求。",
              "- 每个登记起点独立优化并回放完整后缀。六个窗口存在重叠，没有在窗口之间传播执行状态；不能声称连续闭环MPC或年度调度，也不年化成本。",
              "- 2025仅用于此次评估，不用于训练、checkpoint选择或MPC参数调优；此前已多次分析过2025结果，不声称从未查看。",
              "- 四运行含重复seed20240813和不同W0训练预算；跨运行SD只是描述。科学阈值超标与求解回退如实记录，不通过删行制造优势。", "",
              "输出：run_metrics.csv、aggregate_metrics.csv、paired_metrics.csv、solver_summary.csv及validation.json。"
              "所有数值来自已保存结果和latency.json原始测量，本报告脚本不加载模型或执行求解。", ""]
    return "\n".join(lines)


def build_report(root=ROOT, *, expected_days=362, resamples=2000):
    root = _path(root)
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    try:
        panels, metrics, solvers, timings, validation = load_evidence(root, expected_days=expected_days)
        paired = paired_metrics(panels, timings, resamples=resamples)
        aggregate = aggregate_metrics(metrics)
        write_csv(output / "run_metrics.csv", metrics)
        write_csv(output / "aggregate_metrics.csv", aggregate)
        write_csv(output / "paired_metrics.csv", paired)
        write_csv(output / "solver_summary.csv", solvers)
        (output / "REPORT.md").write_text(render_report(metrics, aggregate, paired, solvers, validation), encoding="utf-8")
        validation.update(status="passed", created_at=now(), report_artifacts_complete=True,
                          bootstrap={"date_block_days": 7, "resamples": resamples, "seed": shared.BOOTSTRAP_SEED,
                                     "latency_resampling_unit": "thirty_paired_timing_repeats", "each_side_cvar_recomputed": True})
        write_json(output / "validation.json", validation)
        return validation
    except (shared.ReportEvidenceError, KeyError, TypeError, ValueError) as exc:
        write_json(output / "validation.json", {"status": "failed", "error": str(exc), "final_effects_reported": False,
                   "created_at": now(), "comparison_design": DESIGN})
        (output / "REPORT.md").write_text(f"# 同任务MPC实测报告\n\n正式结果尚不完整或记录检查未通过，未生成效应结论。\n\n{exc}\n", encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = build_report(args.root)
    print(json.dumps({"status": result["status"], "replay_rows": result["replay_rows"], "solver_records": result["solver_records"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
