"""Summarize native R inference versus SMPC H96 without running any model.

Each SMPC input is measured once; only the thirty repeated R measurements
receive per-input distribution summaries. No control-quality outcomes enter
the report, and short-horizon observations are not read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "artifacts/r_ecmoe_smpc96_timing_20260913"
EXPECTED_CASES = 6
OUTER_PARTS = ("mpc_encode_ms", "mpc_scenario_generation_ms",
               "mpc_optimization_pipeline_ms", "mpc_shared_assembly_ms")
STAGE_NAMES = ("build", "update", "solve", "extract")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite {name}")
    return result


def optional(value, name):
    return None if value is None else finite(value, name)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    return ordered[low] + (index - low) * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7), f"Saved timing mismatch: {name}")


def recorded_bool(value, name):
    require(isinstance(value, bool), f"Missing recorded boolean {name}")
    return value


def summarize_case(case):
    """Whitelist timing and solver-termination fields; never export outcomes."""
    case_id = case["case_id"]
    day = case.get("date", case_id[:10])
    require(int(case["origin"]) == 0 and int(case["horizon"]) == 96, f"Not a complete H96 origin-zero query: {case_id}")
    require(int(case.get("stage", 1)) == 1 and int(case.get("batch_size", 1)) == 1, "Queries must be RT batch one")
    if "realized_outcome_scoring_performed" in case:
        require(case["realized_outcome_scoring_performed"] is False, "Outcome scoring is outside this timing study")
    require(case["r_repeats"] == 30 and case["r_warmups"] == 5 and case["mpc_measured_calls"] == 1,
            f"Unexpected measurement counts: {case_id}")
    samples = [finite(value, "R complete-chain sample") for value in case["r_raw_samples_ms"]]
    require(len(samples) == 30 and min(samples) > 0, f"Missing R repetitions: {case_id}")
    r_summary = {"r_mean_ms": statistics.mean(samples), "r_median_ms": statistics.median(samples),
                 "r_p95_ms": percentile(samples, .95)}
    for name, value in r_summary.items():
        check_close(value, finite(case[name], name), f"{case_id}/{name}")
    stats = case["mpc_solverstats"]
    require(int(stats["scenario_count"]) == 20 and int(stats["origin"]) == 0 and int(stats["active_horizon"]) == 96,
            "Solver scenario count or active horizon differs from the timing contract")
    require(float(stats["target_relative_gap"]) == 0 and float(stats["target_absolute_gap"]) == 1e-10
            and float(stats["time_limit_seconds"]) == 600, "Changed solver tolerance or per-call budget")
    booleans = {name: recorded_bool(stats[name], name)
                for name in ("optimal", "accepted", "accepted_not_optimal", "fallback", "has_incumbent")}
    require(not booleans["optimal"] or (booleans["accepted"] and booleans["has_incumbent"] and not booleans["fallback"]),
            "Inconsistent optimal solver status")
    require(booleans["fallback"] == (not booleans["accepted"]), "Inconsistent fallback status")
    classification = "optimal" if booleans["optimal"] else "accepted_not_optimal" if booleans["accepted"] else "fallback_or_not_accepted"
    full = finite(case["mpc_full_chain_ms"], "SMPC single complete-chain time")
    producer_names = {"mpc_optimization_pipeline_ms": "mpc_optimize_wrapper_ms", "mpc_shared_assembly_ms": "mpc_assembly_ms"}
    outer = {name: finite(case.get(name, case.get(producer_names.get(name))), name) for name in OUTER_PARTS}
    stages = {f"mpc_{name}_ms": finite(stats[f"{name}_seconds"], name) * 1000 for name in STAGE_NAMES}
    require(full > 0 and all(value >= 0 for value in (*outer.values(), *stages.values())), "Invalid recorded duration")
    outer_remainder = full - sum(outer.values())
    optimizer_remainder = outer["mpc_optimization_pipeline_ms"] - sum(stages.values())
    require(outer_remainder >= -1 and optimizer_remainder >= -1, "Nested timing stages exceed their parent duration")
    if "effective_native_time_limit" in stats and "highs_runtime_before" in stats:
        budget = float(stats["effective_native_time_limit"]) - float(stats["highs_runtime_before"])
        check_close(budget, 600, "native per-call time allowance")
    else:
        budget = None
    scenario = case.get("scenario_diagnostics", {})
    if "path_count" in scenario:
        require(scenario["path_count"] == 20, "Scenario decoder did not produce twenty paths")
    if "all_decoded_paths_identical" in scenario:
        require(scenario["all_decoded_paths_identical"] is False, "Native scenario paths are all identical")
    probability = stats.get("scenario_probabilities")
    uniform = None
    if probability is not None:
        probability = [finite(x, "scenario probability") for x in probability]
        require(len(probability) == 20 and min(probability) >= 0 and math.isclose(sum(probability), 1, abs_tol=1e-7),
                "Invalid scenario probability weights")
        uniform = all(math.isclose(x, .05, abs_tol=1e-8) for x in probability)
    row = {"case_id": case_id, "date": day, "stage": 1, "origin": 0, "mpc_planning_horizon": 96, "batch_size": 1,
           "run_id": case["model_run_id"], "r_policy_name": "R-ECMoE", "mpc_policy_name": "W1-SMPC-20",
           "r_execution": "native dynamic routing and conditional tail; single neural decision",
           "r_forward_calls_per_timed_decision": 1,
           "r_warmups": 5, "r_repeats": 30, **r_summary, "r_min_ms": min(samples), "r_max_ms": max(samples),
           "mpc_measured_calls": 1, "mpc_full_chain_ms": full, **outer, **stages,
           "mpc_single_to_r_mean_ratio": full / r_summary["r_mean_ms"],
           "mpc_single_to_r_median_ratio": full / r_summary["r_median_ms"],
           "r_mean_relative_saving_percent_against_mpc_single": 100 * (full - r_summary["r_mean_ms"]) / full,
           "status": stats["status"], "solution_classification": classification, **booleans,
           "actual_gap": optional(stats.get("actual_gap"), "actual relative gap"),
           "target_relative_gap": 0.0, "target_absolute_gap": 1e-10, "time_limit_seconds_per_call": 600.0,
           "scenario_count": 20, "scenario_weights_uniform": uniform,
           "model_build_count": stats["model_build_count"], "pass_model_count": stats["pass_model_count"],
           "structure_rebuild_count": stats["structure_rebuild_count"], "solve_count": stats["solve_count"],
           "mpc_build_stage_executed": stages["mpc_build_ms"] > 0,
           "mpc_build_time_basis": "measured_build" if stages["mpc_build_ms"] > 0 else "not_executed_no_rebuild",
           "recorded_at": case.get("recorded_at", case.get("completed_at"))}
    details = dict(row, r_raw_samples_ms=samples,
                   scenario_source=scenario.get("scenario_source"), scenario_weighting_method=scenario.get("weighting_method"))
    alignment = {"case_id": case_id, "full_chain_minus_outer_parts_ms": outer_remainder,
                 "optimization_pipeline_minus_stages_ms": optimizer_remainder,
                 "native_per_call_budget_seconds": budget,
                 "controller_initialization_ms_outside_measured_call": optional(stats.get("controller_initialization_seconds"), "controller initialization")}
    if alignment["controller_initialization_ms_outside_measured_call"] is not None:
        alignment["controller_initialization_ms_outside_measured_call"] *= 1000
    return row, details, alignment


def render(rows, boundary):
    lines = ["# 六个固定输入：R原生神经决策与SMPC H96的完整计算链测速", "",
             "全部输入固定 RT origin=0、batch=1；SMPC的规划问题固定为H=96。R-ECMoE保留原生动态路由、条件尾部专家和每次一次神经决策，"
             "没有展开为96次在线求解或96次推理。每个输入的 R-ECMoE 先暖身5次、实测30次；"
             "W1-SMPC-20 每个输入只测1次。六个异质输入不视为同一输入的六次重复，不计算SMPC均值、p95或置信区间。", "",
             "|日期|R原生单次决策均值 ms|R中位数 ms|R p95 ms|SMPC H96单次完整链 ms|SMPC单次 / R平均|",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['date']}|{row['r_mean_ms']:.3f}|{row['r_median_ms']:.3f}|{row['r_p95_ms']:.3f}|{row['mpc_full_chain_ms']:.3f}|{row['mpc_single_to_r_mean_ratio']:.3f}|")
    lines += ["", "## SMPC完整链的实测分项", "",
              "|日期|W1编码 ms|20路径生成 ms|优化流程 ms|共享装配 ms|其中build ms|其中update ms|其中solve ms|其中extract ms|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row[name] for name in (*OUTER_PARTS, *(f"mpc_{name}_ms" for name in STAGE_NAMES))]
        lines.append("|" + row["date"] + "|" + "|".join(f"{value:.3f}" for value in values) + "|")
    lines += ["", "build/update/solve/extract是优化流程内部的子项，不能再次与完整链相加。build=0表示该次未执行重建，"
              "不表示存在零耗时的建模测量。每个输入的实际建模与求解次数保留于CSV/JSON。", "",
              "|日期|实际状态|实际相对gap|最优|接受非最优|回退|累计build / passModel / solve|",
              "|---|---|---:|---|---|---|---|"]
    for row in rows:
        gap = "未记录" if row["actual_gap"] is None else f"{row['actual_gap']:.9g}"
        lines.append(f"|{row['date']}|{row['status']}|{gap}|{row['optimal']}|{row['accepted_not_optimal']}|{row['fallback']}|"
                     f"{row['model_build_count']} / {row['pass_model_count']} / {row['solve_count']}|")
    lines += ["", "## 相同计算范围与解释", "",
              "- R-ECMoE完整链：准备好的输入包 → 每次重新进行冻结W1编码 → 保持动态路由与条件尾部的原生R神经决策（一次forward）→ 共享装配/首步提取。",
              "- W1-SMPC-20完整链：同一准备好的输入包 → 每次重新进行冻结W1编码 → 20条原生联合情景生成 → 建模或更新、风险MILP求解 → 共享装配/首步提取。",
              "- W0和W1是同一个世界模型的连续训练阶段；W1已继承W0参数，不应额外再加一次独立W0前向计算。模型加载、磁盘I/O、输入包构造与真实环境执行均在计时之外。",
              "- 相同的是输入和控制请求，不是内部计算结构。risk.horizon=96是RT起点全日任务的请求上下文，不能据此把R表述成96步在线优化器；R的原生计算路径保持不变。",
              "- 固定20个W1联合路径、原生均匀Monte Carlo权重、CVaR95与0.25权重。目标相对gap为0，600秒为每次求解上限；报告使用实际终止gap与耗时，不把上限当作延迟。",
              "- 此处SMPC的规划H全部为96。此前SMPC H96/H95查询与后来H15至4的短窗口调用具有不同有效优化规模，不能直接把约5秒与约1秒解释为同一任务的提速或退化；短窗口记录未进入本表。",
              "- 本报告只比较计算时间和求解终止情况；不报告成本、SLA或控制质量结果，不据此断言控制质量等效、全年闭环效果或MPC不适用。",
              "- 每个SMPC输入只有一次实测。与R均值的倍数仅描述该输入下的测量对比，不是SMPC平均加速比的估计。", "",
              "## 来源", "", "原始timing_cases.json、protocol.json与completed.json保留不变；来源与生成脚本SHA-256见timing_comparison.json。",
              "精确计时边界按原protocol记录保留于timing_comparison.json的timing_boundary字段。", ""]
    return "\n".join(lines)


def build(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    source_paths = [root / name for name in ("completed.json", "timing_cases.json", "protocol.json")]
    completed, saved_cases, protocol = [read_json(path) for path in source_paths]
    cases = saved_cases["cases"]
    require(completed.get("status") == saved_cases.get("status") == "completed"
            and completed.get("cases") == saved_cases.get("completed_cases") == EXPECTED_CASES
            and completed.get("mpc_calls") == 6 and completed.get("R_measured_calls") == 180
            and completed.get("H") == 96 and len(cases) == EXPECTED_CASES,
            "All six SMPC-H96 timing cases must be complete before reporting")
    require(completed["protocol_sha256"] == digest(root / "protocol.json"), "Frozen protocol hash differs from completion marker")
    required_protocol = {"batch_size": 1, "scenario_count": 20, "target_relative_gap": 0.,
                         "absolute_gap_tolerance": 1e-10, "time_limit_seconds_per_call": 600., "threads": 1,
                         "r_warmups_per_case": 5, "r_measured_repeats_per_case": 30, "mpc_measured_calls_per_case": 1,
                         "persistent_solver_across_dates": True, "solver_reset_between_dates": False,
                         "native_model_build_included_in_first_call": True, "realized_cost_or_SLA_evaluated": False,
                         "closed_loop_execution": False, "W0_is_inherited_by_W1_not_a_second_forward": True,
                         "short_H15_to4_results_allowed_in_primary_comparison": False,
                         "parameters_or_checkpoints_selected_from_test": False}
    for name, expected in required_protocol.items():
        require(protocol.get(name) == expected, f"Changed frozen timing protocol: {name}")
    require([(case["case_id"], case["date"]) for case in cases]
            == [(case["case_id"], case["date"]) for case in protocol["cases"]], "Measured dates do not match the fixed protocol")
    rows, details, alignments = [], [], []
    for index, case in enumerate(cases):
        row, detail, alignment = summarize_case(case)
        require(row["model_build_count"] == row["pass_model_count"] == 1 and row["structure_rebuild_count"] == 0
                and row["solve_count"] == index + 1, "Persistent solver call sequence differs from the protocol")
        require(row["mpc_build_stage_executed"] == (index == 0), "Only the initial input should include a model build")
        row["query_role"] = detail["query_role"] = "initial_model_build" if index == 0 else "persistent_update_for_new_date"
        rows.append(row)
        details.append(detail)
        alignments.append(alignment)
    require(len({row["case_id"] for row in rows}) == len({row["date"] for row in rows}) == EXPECTED_CASES,
            "Expected six distinct fixed input dates")
    require(len({row["run_id"] for row in rows}) == 1, "Timing cases must use one frozen model lineage")
    scope_path = root / "scope_clarification.json"
    scope_clarification = read_json(scope_path) if scope_path.exists() else None
    if scope_clarification is not None:
        source_paths.append(scope_path)
    boundary = {"included": protocol["timing_includes"], "excluded": protocol["timing_excludes"],
                "r_chain": "fresh W1 encoding -> native dynamic R routing/conditional tail, one neural decision -> shared assembly and first-action extraction",
                "mpc_chain": "fresh W1 encoding -> 20 joint paths -> H96 model build or persistent update and CVaR MILP solve -> shared assembly and first-action extraction",
                "same_input_and_control_request": True, "same_internal_computational_structure": False,
                "r_neural_decisions_per_measured_call": 1, "r_unrolled_to_96_inferences": False,
                "mpc_planning_horizon": 96, "separate_extra_W0_forward": False,
                "native_model_build_included_in_first_call": True, "persistent_solver_across_dates": True,
                "frozen_protocol_horizon_wording_interpretation": "The original predictive_horizon_identical_for_both_models field describes the same request context; it does not mean that R executes 96 optimization or forward steps."}
    result = {"schema": "dcvpp.smpc96.timing.report.v1", "status": "completed", "timing_only_completed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "case_count": EXPECTED_CASES,
              "cases": details, "mpc_planning_horizon": 96, "origin": 0, "batch_size": 1,
              "run_id": rows[0]["run_id"], "r_warmups_per_case": 5, "r_repeats_per_case": 30,
              "mpc_measured_calls_per_case": 1, "timing_boundary": boundary,
              "scope_clarification": scope_clarification,
              "statistical_scope": "six distinct fixed input requests; thirty native R single-decision repetitions and one SMPC H96 call per input",
              "mpc_summary_statistics_or_confidence_intervals_estimated": False,
              "short_horizon_measurements_included": False, "control_quality_outcomes_included": False,
              "outcome_quality_equivalence_established": False, "full_day_rolling_completed": False,
              "annual_performance_comparison_completed": False, "new_model_or_solver_execution_performed": False,
              "source_files": [{"path": str(path.resolve()), "sha256": digest(path)} for path in (*source_paths, Path(__file__))]}
    validation = {"status": "passed", "timing_only_completed": True, "cases": EXPECTED_CASES,
                  "mpc_planning_horizon96_for_all_inputs": True, "origin0_for_all_inputs": True,
                  "R_native_dynamic_routing_and_conditional_tail_preserved": True,
                  "R_neural_decisions_per_measured_call": 1, "R_unrolled_to_96_inferences": False,
                  "R_saved_statistics_match_thirty_samples": True, "mpc_measurements_per_case": 1,
                  "actual_solver_status_and_gap_preserved": True, "time_limit_is_not_measured_runtime": True,
                  "control_quality_outcomes_included": False, "short_horizon_measurements_included": False,
                  "timing_alignment": alignments, "new_model_or_solver_execution_performed": False}
    csv_path = root / "timing_comparison.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    write_json(root / "timing_comparison.json", result)
    (root / "REPORT.md").write_text(render(rows, boundary), encoding="utf-8")
    write_json(root / "validation.json", validation)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    result = build(args.root)
    print(json.dumps({"status": result["status"], "cases": result["case_count"], "mpc_planning_horizon": result["mpc_planning_horizon"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
