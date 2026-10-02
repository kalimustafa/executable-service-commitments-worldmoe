"""Report two saved SMPC-20 calls; no solver, model, inference or replay runs."""
from __future__ import annotations

import argparse
import _sha256
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "artifacts/r_ecmoe_smpc20_timing_20260912"
CASES = (("2025-01-01_RT_o0_h96", 0, 96), ("2025-01-01_RT_o1_h95", 1, 95))
TIMING_PARTS = ("mpc_encode_ms", "mpc_scenario_generation_ms", "mpc_optimization_pipeline_ms", "mpc_shared_assembly_ms")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return _sha256.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite(value, name):
    value = float(value)
    require(math.isfinite(value), f"Nonfinite {name}")
    return value


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = math.floor(index)
    return values[lower] + (index - lower) * (values[min(lower + 1, len(values) - 1)] - values[lower])


def optional(value, name):
    return None if value is None else finite(value, name)


def solution_status(stats):
    if stats["optimal"]:
        require(stats["accepted"] and stats["has_incumbent"] and not stats["fallback"], "Inconsistent optimal/incumbent flags")
        return "optimal"
    if stats["accepted"]:
        require(stats["accepted_not_optimal"] and stats["has_incumbent"] and not stats["fallback"], "Inconsistent accepted-not-optimal flags")
        return "accepted_not_optimal"
    return "fallback_or_not_accepted"


def build(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    completed_path, cases_path, protocol_path = (root / name for name in ("completed.json", "timing_cases.json", "protocol.json"))
    completed, cases, protocol = read_json(completed_path), read_json(cases_path), read_json(protocol_path)
    require(completed.get("timing_calls_completed") is True and completed.get("cases_completed") == 2 and len(cases) == 2,
            "Both recorded SMPC-20 calls must be complete before reporting")
    for name in ("full_day_rolling_completed", "annual_performance_comparison_completed", "controller_outcome_equivalence_established"):
        require(completed.get(name) is False, f"Unsupported full-performance claim: {name}")
    expected = {"scenario_count": 20, "cvar_alpha": .95, "expected_cost_weight": 1.0, "cvar_weight": .25,
                "target_relative_gap": 0.0, "absolute_gap_tolerance": 1e-10, "time_limit_seconds_per_call": 600.0,
                "r_warmups_per_case": 5, "r_measured_repeats_per_case": 30, "mpc_measured_calls_per_case": 1,
                "first_action_only": True, "complete_closed_loop_day_executed": False,
                "same_current_R_physical_and_cost_contract": True, "training_performed": False,
                "checkpoint_selection_performed": False, "test_used_for_parameter_tuning": False}
    for name, wanted in expected.items():
        require(protocol.get(name) == wanted, f"Recorded protocol differs at {name}")
    rows, case_details, alignment = [], [], []
    for case, (case_id, origin, horizon) in zip(cases, CASES, strict=True):
        require((case["case_id"], case["origin"], case["horizon"]) == (case_id, origin, horizon), "Unexpected measured query")
        require(case["model_run_id"] == "legacy_seed20240813" and case["r_repeats"] == 30 and case["r_warmups"] == 5
                and case["mpc_measured_calls"] == 1, "Incorrect run or repetition scope")
        require(case["full_day_rolling_completed"] is False, "A query is incorrectly represented as a full rolling day")
        samples = [finite(value, "R sample") for value in case["r_raw_samples_ms"]]
        require(len(samples) == 30 and min(samples) > 0, "R timing samples are incomplete")
        measured_r = {"r_mean_ms": statistics.mean(samples), "r_median_ms": statistics.median(samples),
                      "r_p95_ms": percentile(samples, .95)}
        for name, value in measured_r.items():
            require(math.isclose(value, finite(case[name], name), rel_tol=1e-12, abs_tol=1e-8), f"Saved R statistic mismatch: {case_id}/{name}")
        stats = case["mpc_solverstats"]
        scenario = case["scenario_diagnostics"]
        require(stats["scenario_count"] == scenario["path_count"] == 20, "Scenario count is not twenty")
        probabilities = [finite(value, "scenario probability") for value in stats["scenario_probabilities"]]
        require(len(probabilities) == 20 and all(value >= 0 for value in probabilities)
                and math.isclose(sum(probabilities), 1, abs_tol=1e-8), "Invalid scenario probability weights")
        uniform = all(math.isclose(value, 1 / 20, abs_tol=1e-10) for value in probabilities)
        require(scenario["all_decoded_paths_identical"] is False, "Twenty paths were recorded as identical")
        require(stats["origin"] == origin and stats["active_horizon"] == horizon and stats["first_action_only"] is True,
                "Solver query does not match recorded current-control scope")
        require(stats["target_relative_gap"] == 0 and stats["target_absolute_gap"] == 1e-10
                and stats["time_limit_seconds"] == 600, "Requested solver tolerance/budget changed")
        require(stats["model_build_count"] == 1 and stats["pass_model_count"] == 1
                and stats["structure_rebuild_count"] == 0 and stats["solve_count"] == origin + 1,
                "Recorded calls do not share one persistent optimization model")
        if origin == 1:
            require(stats["build_seconds"] == 0, "The second query was rebuilt rather than persistently updated")
        classification = solution_status(stats)
        if stats["accepted"]:
            require(case["mpc_requested_first_action"] is not None and case["mpc_assembled_first_action"] is not None,
                    "An accepted solver result omitted its first-control output")
        mpc_time = finite(case["mpc_full_chain_ms"], "SMPC single-call time")
        parts = {name: finite(case[name], name) for name in TIMING_PARTS}
        phase_ms = {f"mpc_{name}_ms": finite(stats[f"{name}_seconds"], name) * 1000 for name in ("build", "update", "solve", "extract")}
        require(mpc_time > 0 and all(value >= 0 for value in (*parts.values(), *phase_ms.values())), "Invalid stage timings")
        full_remainder = mpc_time - sum(parts.values())
        optimization_remainder = parts["mpc_optimization_pipeline_ms"] - sum(phase_ms.values())
        require(full_remainder >= -1.0 and optimization_remainder >= -1.0, "Nested timing stages exceed their parents beyond clock tolerance")
        per_call_budget = finite(stats["effective_native_time_limit"], "native deadline") - finite(stats["highs_runtime_before"], "native prior runtime")
        require(math.isclose(per_call_budget, 600, abs_tol=1e-8), "Cumulative native deadline does not provide a 600-second call budget")
        objective = optional(stats.get("objective"), "objective")
        expected_cost = optional(stats.get("expected_scenario_cost"), "expected scenario cost")
        cvar = optional(stats.get("scenario_cvar95"), "scenario CVaR")
        if objective is not None and expected_cost is not None and cvar is not None:
            require(math.isclose(objective, expected_cost + .25 * cvar, rel_tol=1e-8, abs_tol=1e-6), "Recorded objective is not E[cost]+.25*CVaR95")
        derived_without_build = mpc_time - phase_ms["mpc_build_ms"] if origin == 0 else None
        row = {"case_id": case_id, "date": "2025-01-01", "stage": 1, "origin": origin, "horizon": horizon,
               "batch_size": 1, "run_id": case["model_run_id"], "r_policy_name": "R-ECMoE", "mpc_policy_name": "W1-SMPC-20",
               "query_role": "initial_model_call" if origin == 0 else "persistent_origin_update_call",
               "r_warmups": 5, "r_repeats": 30, **measured_r, "r_min_ms": min(samples), "r_max_ms": max(samples),
               "mpc_measured_calls": 1, "mpc_full_chain_ms": mpc_time, **parts, **phase_ms,
               "mpc_build_stage_executed": origin == 0,
               "mpc_build_time_basis": "measured_initial_build" if origin == 0 else "not_executed_no_rebuild",
               "mpc_full_chain_minus_initial_build_ms_derived": derived_without_build,
               "derived_without_build_is_warm_measurement": False,
               "mpc_single_to_r_mean_ratio": mpc_time / measured_r["r_mean_ms"],
               "mpc_single_to_r_median_ratio": mpc_time / measured_r["r_median_ms"],
               "r_mean_relative_saving_percent_against_mpc_single": 100 * (mpc_time - measured_r["r_mean_ms"]) / mpc_time,
               "status": stats["status"], "solution_classification": classification, "optimal": stats["optimal"],
               "accepted": stats["accepted"], "accepted_not_optimal": stats["accepted_not_optimal"], "fallback": stats["fallback"],
               "has_incumbent": stats["has_incumbent"], "actual_gap": optional(stats.get("actual_gap"), "actual relative gap"),
               "target_relative_gap": stats["target_relative_gap"], "target_absolute_gap": stats["target_absolute_gap"],
               "objective": objective, "objective_bound": optional(stats.get("objective_bound"), "objective bound"),
               "time_limit_seconds_per_call": 600.0, "model_build_count": stats["model_build_count"],
               "pass_model_count": stats["pass_model_count"], "solve_count": stats["solve_count"],
               "scenario_count": 20, "scenario_weights_uniform": uniform, "scenario_source": scenario["scenario_source"],
               "scenario_weighting_method": scenario["weighting_method"], "full_day_rolling_completed": False}
        rows.append(row)
        case_details.append({**row, "r_raw_samples_ms": samples, "mpc_solverstats": stats, "scenario_diagnostics": scenario,
                             "r_first_action": case["r_first_action"], "mpc_requested_first_action": case["mpc_requested_first_action"],
                             "mpc_assembled_first_action": case["mpc_assembled_first_action"], "recorded_at": case["recorded_at"]})
        alignment.append({"case_id": case_id, "full_chain_minus_outer_parts_ms": full_remainder,
                          "optimization_pipeline_minus_build_update_solve_extract_ms": optimization_remainder,
                          "native_per_call_budget_seconds": per_call_budget,
                          "controller_initialization_ms_outside_measured_call": float(stats["controller_initialization_seconds"]) * 1000})
    result = {"schema": "dcvpp.smpc20.timing.report.v1", "status": "completed", "timing_only_completed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "primary_baseline": "W1-SMPC-20", "run_id": "legacy_seed20240813",
              "case_count": 2, "cases": case_details, "protocol": protocol,
              "mpc_summary_statistics_or_confidence_intervals_estimated": False,
              "full_day_rolling_completed": False, "annual_performance_comparison_completed": False,
              "controller_outcome_equivalence_established": False,
              "two_queries_use_historical_states_not_propagated_mpc_execution": True,
              "exact_original_R3_implementation_and_physics": False,
              "same_task_risk_scenario_adaptation": True, "exact_R3_reproduction": False,
              "scenario_weighting_method": "uniform_native_W1_monte_carlo",
              "full_closed_loop_completed": False, "outcome_quality_equivalent_established": False,
              "baseline_role": "legacy_20_scenario_CVaR_first_control_paradigm_adapted_to_current_R_task",
              "timing_boundary": {
                  "prepared_input": "WorldInputPacket constructed on the target device before the measured call",
                  "r_chain": "fresh frozen W1 encoding -> learned R decision/auxiliary outputs/shared assembly -> first-control extraction",
                  "mpc_chain": "fresh frozen W1 encoding -> 20-path joint scenario generation -> initial model build or persistent update -> CVaR MILP solve -> first-control extraction -> shared assembly",
                  "excluded": ["neural checkpoint loading", "disk I/O", "input batch construction", "realized-outcome scoring"],
                  "W1_inherits_W0_parameters": True, "separate_extra_W0_forward": False,
                  "cached_W1_encoding_assumed_zero_cost": False,
                  "scenario_neural_kernel_warmups_before_each_mpc_query": 2,
                  "first_case_build_subtraction_is_only_derived_not_measured_warm_runtime": True},
              "statistical_scope": "R has thirty repeated measurements per input; each SMPC input has one measured call, with no SMPC mean/p95/CI",
              "old_deterministic_838ms_primary_comparison_superseded": True,
              "training_performed": False, "checkpoint_selection_performed": False, "test_used_for_parameter_tuning": False,
              "prior_2025_already_evaluated": protocol.get("prior_2025_already_evaluated"),
              "source_files": [{"path": str(path), "sha256": digest(path)} for path in (cases_path, completed_path, protocol_path, Path(__file__))]}
    with (root / "timing_comparison.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "timing_comparison.json", result)
    (root / "REPORT.md").write_text(render(rows, protocol), encoding="utf-8")
    validation = {"status": "passed", "timing_only_completed": True, "cases": 2,
                  "scope": "existing_R_samples_and_nested_stage_timings_not_controller_quality_equivalence",
                  "R_saved_statistics_match_thirty_samples": True, "mpc_measurements_per_case": 1,
                  "same_persistent_model_model_build_count": 1, "same_persistent_model_pass_model_count": 1,
                  "second_case_build_seconds": 0.0, "second_case_solve_count": 2,
                  "scenario_count_per_case": 20, "scenario_weight_sums_checked": True,
                  "recorded_E_plus_0_25_CVaR95_objective_checked_when_available": True,
                  "actual_solver_status_and_gap_preserved": True, "target_gap_not_substituted_for_actual_gap": True,
                  "time_limit_is_not_measured_runtime": True, "timing_alignment": alignment,
                  "full_day_rolling_completed": False, "annual_performance_comparison_completed": False,
                  "controller_outcome_equivalence_established": False, "new_model_or_solver_execution_performed": False,
                  "output_files": [str(root / name) for name in ("REPORT.md", "timing_comparison.csv", "timing_comparison.json", "validation.json")]}
    write_json(root / "validation.json", validation)
    print(json.dumps({"status": "passed", "timing_only_completed": True,
                      "cases": [{key: row[key] for key in ("case_id", "r_mean_ms", "r_p95_ms", "mpc_full_chain_ms",
                                 "mpc_build_ms", "mpc_update_ms", "mpc_solve_ms", "actual_gap", "solution_classification",
                                 "mpc_single_to_r_mean_ratio")} for row in rows]}, ensure_ascii=False))
    return result, validation


def fmt(value, digits=3):
    return "未记录" if value is None else f"{value:.{digits}f}"


def render(rows, protocol):
    lines = ["# 20联合情景风险SMPC的两个查询计时", "",
             "本表报告旧Run 1、2025-01-01的两个历史状态输入查询。R每个输入预热5次后测30次；"
             "SMPC每个输入只执行并测量1次。", "",
             "| 输入 | 优化器调用类型 | R平均 ms | R中位 ms | R p95 ms | SMPC实测单次完整链 ms | SMPC单次/R均值 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        role = "首次模型初建" if row["origin"] == 0 else "同实例持久更新"
        lines.append(f"| RT o{row['origin']}/H{row['horizon']} | {role} | {fmt(row['r_mean_ms'])} | {fmt(row['r_median_ms'])} | "
                     f"{fmt(row['r_p95_ms'])} | {fmt(row['mpc_full_chain_ms'])} | {row['mpc_single_to_r_mean_ratio']:.4f} |")
    lines += ["", "比值只是该次SMPC记录相对R重复测量均值的描述性比较。**SMPC没有每case的均值、p95或置信区间**；"
              "也不把两个不同输入合并为SMPC延迟分布。计时比较不证明两控制器同等调度质量。", "",
              "## 计算链与分项", "",
              "两边均从预先构建并置于目标设备的WorldInputPacket开始，每次重新执行同一冻结W1编码。"
              "W1继承W0阶段训练的参数，不另外加算一次W0 forward；不把缓存W1输出当作零成本。", "",
              "- R：fresh W1编码 → R学习决策/辅助输出及公共动作装配 → 提取首步。",
              "- SMPC：fresh W1编码 → 20条联合情景生成 → 优化模型初建或持久更新 → E[cost]+0.25×CVaR95求解 → 首步提取 → 公共装配。",
              "- 两边排除神经网络权重加载、磁盘I/O、batch构建和真实结果评分；SMPC场景生成神经核预热2次不包含求解或新优化模型建立。", "",
              "| 输入 | W1编码 ms | 情景生成 ms | 初建 ms | 更新 ms | 纯solve ms | 首步提取 ms | 公共装配 ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        names = ("mpc_encode_ms", "mpc_scenario_generation_ms", "mpc_build_ms", "mpc_update_ms",
                 "mpc_solve_ms", "mpc_extract_ms", "mpc_shared_assembly_ms")
        values = ["未重建" if name == "mpc_build_ms" and not row["mpc_build_stage_executed"] else fmt(row[name]) for name in names]
        lines.append(f"| RT o{row['origin']}/H{row['horizon']} | " + " | ".join(values) + " |")
    lines += ["", f"首个查询的完整链减去本次初建记录，派生值为{rows[0]['mpc_full_chain_minus_initial_build_ms_derived']:.3f} ms。"
              "这不是另一次实测warm调用。第二个查询才是实际的持久更新调用：build=0、累计model_build=1、pass_model=1、solve_count=2。"
              "第二查询的起点和H已改变，不能等同于重复测试第一个输入。分项属于嵌套计时，优化pipeline不能再与其内部build/update/solve/extract重复相加。", "",
              "## 实际求解状态", "",
              "| 输入 | solver status | 接受 | 最优 | accepted_not_optimal | fallback | 目标gap | 实际gap | 目标/界 | 每调用预算 s |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| RT o{row['origin']}/H{row['horizon']} | {row['status']} | {row['accepted']} | {row['optimal']} | "
                     f"{row['accepted_not_optimal']} | {row['fallback']} | {row['target_relative_gap']:g} | "
                     f"{fmt(row['actual_gap'], 10)} | {fmt(row['objective'], 6)} / {fmt(row['objective_bound'], 6)} | 600 |")
    lines += ["", "目标相对gap=0、绝对gap容差1e-10不等于每次必然达到实际gap=0；表中保留原始实际状态和gap。"
              "若出现accepted_not_optimal，应按接受了非最优可用解报告。600秒是每调用求解上限，不是实测耗时；"
              "持久HiGHS的累计时间deadline已按原运行时间增加600秒设置，不把累计deadline当作第二次调用耗时。", "",
              "## 情景来源及证据范围", "",
              "20条情景来自冻结W1的原生联合潜空间Monte Carlo路径，实际权重均为1/20，包含随机RT、负荷、PV和PUE路径；"
              "不是单点预测复制，也不是旧R3场景head及其学习权重。CVaR置信水平0.95、权重0.25，首个请求控制施加跨情景非预见性约束。", "",
              "这是旧20情景＋风险目标＋首步控制范式适配当前R任务的基线，**不是原R3源码和物理设定的完全重现**。"
              "使用当前R的成本、截止期、储能预算与公共装配器，不额外施加终端SOC保留；grid按当前R动作契约派生。", "",
              "两个查询各自来自共同历史状态；第二个查询没有接收第一步SMPC实际执行后的SOC/queue。"
              "本次只记录首步输出及同一优化模型的origin更新，没有完成96步闭环或全年实验，也未证明完整闭环成本/风险质量等效。", "",
              "2024的10秒技术pilot不进入本表；正式两次查询使用600秒预算。2025仅用于这些补充评估查询，未训练、选模或调参，且此前已多次分析过2025结果。"
              "旧确定性838ms记录仍作为降级历史诊断保留，不并入本风险SMPC对照。", "",
              "原始timing_cases.json、completed.json和冻结protocol.json未改；来源哈希、原始R样本、SMPC实际solverstats与首步输出保存在timing_comparison.json。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    build(args.root)


if __name__ == "__main__":
    main()
