"""Report 18 completed short closed-loop sensitivity episodes from saved JSON.

No model, training runner or solver is imported or executed. Twelve consecutive
heterogeneous calls per episode are descriptive observations, not independent
replicates. Nondominated sets are empirical sets among the measured policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "artifacts/r_ecmoe_closedloop_sensitivity_20260913"
WEIGHTS = (0.0, 0.1, 1.0, 10.0)
FAMILIES = ("R-ECMoE", "W1-SMPC-20")
EXPECTED_EPISODES, EXPECTED_STEPS = 18, 12
SUM_FIELDS = ("raw_cost", "market_cost", "battery_degradation_cost", "energy_service_cost", "expiry_proxy_cost",
              "arrived_work", "served_work", "expiry_proxy_increment", "unserved_energy", "pv_curtailed_energy")
COHORT_FIELDS = ("cohort_matured_work", "cohort_ontime_completed_work", "cohort_censored_work",
                 "cohort_censored_remaining_work", "cohort_matured_late_work", "cohort_late_remaining_work",
                 "cohort_cumulative_late_service_work")
QUALITY_TIME = ("normalized_composite_cost", "on_time_workload_rate", "full_chain_mean_ms")
GRID_FRONT = ("normalized_composite_cost", "on_time_workload_rate", "local_ramp_mean", "positive_curvature_mean")


class IncompleteClosedLoopEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IncompleteClosedLoopEvidence(message)


def finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise IncompleteClosedLoopEvidence(f"Missing numeric {name}") from error
    require(math.isfinite(result), f"Nonfinite {name}")
    return result


def optional_finite(value, name):
    return None if value is None or value == "" else finite(value, name)


def flag(value, name):
    require(isinstance(value, bool), f"{name} must be a recorded boolean")
    return value


def same(a, b, label):
    require(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=2e-6, atol=2e-6),
            f"Saved summary/row mismatch: {label}")


def read_json(path, sources):
    path = Path(path).resolve()
    raw = path.read_bytes()
    sources[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def flatten(row):
    return {key: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flatten(row) for row in rows)
    temporary.replace(path)


def nondominated(rows, metrics, *, atol=1e-9, rtol=1e-9):
    """Return flags and dominators; a missing objective never implies dominance."""
    transformed = {}
    for row in rows:
        if all(row.get(metric) is not None and math.isfinite(float(row[metric])) for metric in metrics):
            transformed[row["policy_id"]] = np.array([
                -float(row[metric]) if metric == "on_time_workload_rate" else float(row[metric]) for metric in metrics])
    result = {}
    for row in rows:
        key = row["policy_id"]
        if key not in transformed:
            result[key] = {"nondominated": None, "dominated_by": [], "reason": "missing_declared_objective"}
            continue
        candidate = transformed[key]
        better = []
        for other, values in transformed.items():
            if other == key:
                continue
            tolerance = atol + rtol * np.maximum(1.0, np.maximum(np.abs(values), np.abs(candidate)))
            if np.all(values <= candidate + tolerance) and np.any(values < candidate - tolerance):
                better.append(other)
        result[key] = {"nondominated": not better, "dominated_by": better, "reason": None}
    return result


def percent_change(new, baseline):
    if baseline is None or baseline <= 0:
        return None
    return 100 * (new - baseline) / baseline


def load_evidence(root):
    root = Path(root).resolve()
    paths = sorted((root / "episodes").glob("*.json"))
    require(len(paths) == EXPECTED_EPISODES, f"Need all 18 completed episodes before reporting; found {len(paths)}")
    sources = {}
    protocol = read_json(root / "execution_protocol.json", sources)
    case_protocol = read_json(root / "case_protocol.json", sources)
    training = read_json(root / "training_protocol.json", sources)
    require(protocol["executed_steps_per_case"] == EXPECTED_STEPS, "Expected 12 executed decisions per case")
    require(protocol["policy_specific_executed_state_feedback"] is True, "Missing policy-specific closed-loop feedback contract")
    require(protocol["matched_initial_states"] is True and protocol["full_day_or_annual_closedloop"] is False,
            "Wrong initial-state or annual-extrapolation scope")
    require(protocol["test_used_for_fitting_or_checkpoint_or_weight_selection"] is False, "Test-based policy/weight selection is not this study")
    require(training["training_years"] == [2022, 2023] and training["parameters_fit_on_2025"] is False,
            "Short adaptations must use the frozen train-only protocol")
    require(tuple(float(x) for x in training["lambda_grid"]) == WEIGHTS and int(training["epochs_per_arm"]) == 4,
            "Prespecified four-arm/four-epoch training scope changed")
    cases = {case["date"]: case for case in protocol["cases"]}
    policies = {policy["policy_id"]: policy for policy in protocol["policies"]}
    require(len(cases) == 2 and {case["case_type"] for case in cases.values()} == {"rare", "routine"}, "Need one rare and one routine fixed case")
    require(len(policies) == 9, "Expected the untouched R plus four R and four SMPC grid weights")
    originals = [policy for policy in policies.values() if policy["family"] == "R-ECMoE" and policy["lambda_grid"] is None]
    require(len(originals) == 1, "The untouched R reference must be separate from the gamma=0 continuation")
    for family in FAMILIES:
        observed = sorted(float(p["lambda_grid"]) for p in policies.values() if p["family"] == family and p["lambda_grid"] is not None)
        require(observed == list(WEIGHTS), f"Wrong weights for {family}")
    summaries, step_rows, observed_keys = [], [], set()
    common_exogenous, common_boundary, common_matured, common_scales = {}, {}, {}, {}
    completed_times = []
    for path in paths:
        episode = read_json(path, sources)
        require(episode.get("completed") is True and episode.get("status") == "completed", f"Incomplete episode: {path.name}")
        case, policy_id = episode["case"], episode["policy_id"]
        day = case["date"]
        key = day, policy_id
        require(day in cases and policy_id in policies and key not in observed_keys, f"Foreign/duplicate episode {key}")
        observed_keys.add(key)
        declared_policy, declared_case = policies[policy_id], cases[day]
        require(episode["family"] == declared_policy["family"] and episode["lambda_grid"] == declared_policy["lambda_grid"], f"Wrong policy identity {key}")
        rows, saved = episode["rows"], episode["summary"]
        origins = list(range(int(declared_case["start_origin"]), int(declared_case["stop_origin_exclusive"])))
        require(len(rows) == len(origins) == EXPECTED_STEPS, f"Wrong executed horizon {key}")
        require([int(row["origin"]) for row in rows] == origins, f"Skipped or duplicate decisions {key}")
        require(saved["steps"] == EXPECTED_STEPS and int(saved["stop_origin_exclusive"]) == origins[-1] + 1, f"Summary is not the full episode {key}")
        daily_scale = finite(episode["train_daily_cost_scale"], "train-only daily normalization")
        require(daily_scale > 0, "Normalization denominator must be positive")
        denominator = daily_scale * EXPECTED_STEPS / 96
        if day in common_scales:
            same(daily_scale, common_scales[day], f"normalization within {day}")
        common_scales[day] = daily_scale
        times, solver_records = [], []
        for row in rows:
            require(row["policy_id"] == policy_id and row["operating_date"] == day and row["date"] == day, f"Mixed row identity {key}")
            require(row["family"] == episode["family"] and row["lambda_grid"] == episode["lambda_grid"], "Mixed row policy metadata")
            require(row["future_realizations_used"] is False, "Controller row reports future-realization use")
            require(int(row["prediction_horizon"]) == int(row["horizon"]) == min(16, 96 - int(row["origin"])), "Wrong clipped H16 planning horizon")
            same(row["train_daily_cost_scale"], daily_scale, "step normalization")
            raw = finite(row["raw_cost"], "executed raw composite cost")
            parts = sum(finite(row[name], name) for name in ("market_cost", "battery_degradation_cost", "energy_service_cost", "expiry_proxy_cost"))
            same(raw, parts, "base-cost components exclude gamma")
            latency = finite(row["full_chain_ms"], "complete-chain milliseconds")
            require(latency > 0, "Latency must be measured and positive")
            times.append(latency)
            arrivals = np.asarray(row["actual_arrivals"], dtype=float)
            require(arrivals.shape == (4,) and np.isfinite(arrivals).all(), "Actual arrivals must be four finite node quantities")
            exogenous_key = day, int(row["origin"])
            if exogenous_key in common_exogenous:
                same(arrivals, common_exogenous[exogenous_key], "common realized arrival sequence")
            common_exogenous[exogenous_key] = arrivals
            solver = row.get("solverstats")
            if episode["family"] == "W1-SMPC-20":
                require(isinstance(solver, dict), "SMPC decision lacks recorded solver diagnostics")
                require(int(solver["scenario_count"]) == 20 and float(solver["target_relative_gap"]) == 0, "Changed SMPC scenario/gap contract")
                require(float(solver["time_limit_seconds"]) == 600, "Changed per-call solver budget")
                for name in ("optimal", "accepted", "accepted_not_optimal", "fallback", "has_incumbent"):
                    flag(solver[name], name)
                require(not solver["optimal"] or (solver["accepted"] and not solver["fallback"]), "Contradictory solver status")
                require(solver["fallback"] == (not solver["accepted"]), "Fallback/accepted classification mismatch")
                solver_records.append(solver)
            else:
                require(solver is None, "Neural-policy row unexpectedly has solver statistics")
            flat = dict(row, episode_source=str(path), episode_normalization_denominator=denominator,
                        normalized_base_cost_contribution=raw / denominator)
            if solver is not None:
                for name in ("status", "actual_gap", "optimal", "accepted", "accepted_not_optimal", "fallback", "has_incumbent",
                             "solve_seconds", "update_seconds", "build_seconds", "model_build_count", "pass_model_count"):
                    flat["solver_" + name] = solver.get(name)
            step_rows.append(flat)
        for name in SUM_FIELDS:
            same(sum(finite(row[name], name) for row in rows), saved[name], name)
        last = rows[-1]
        for name in COHORT_FIELDS:
            same(last[name], saved[name], name)  # cumulative, never sum these over steps
        matured = finite(last["cohort_matured_work"], "matured original cohort work")
        ontime = finite(last["cohort_ontime_completed_work"], "on-time completed matured work")
        require(matured >= 0 and -1e-7 <= ontime <= matured + 1e-6, "Invalid cohort completion numerator/denominator")
        rate = ontime / matured if matured > 0 else None
        if rate is not None:
            same(rate, last["cohort_ontime_work_completion_ratio"], "true mature-cohort on-time ratio")
            same(rate, saved["cohort_ontime_work_completion_ratio"], "summary true on-time ratio")
        else:
            require(saved["cohort_ontime_work_completion_ratio"] is None, "No matured cohort cannot be reported as zero/one completion")
        if day in common_matured:
            same(matured, common_matured[day], "matched matured cohort denominator")
        common_matured[day] = matured
        boundary = np.asarray([rows[0]["previous_grid"], rows[0]["previous_previous_grid"], episode["final_state"]["initial_queue"]], dtype=float)
        if day in common_boundary:
            same(boundary, common_boundary[day], "matched boundary grid history and initial queue")
        common_boundary[day] = boundary
        same(last["soc_after"], saved["final_soc"], "terminal SOC")
        same(last["queue_after"], saved["final_queue"], "terminal queue")
        grid_means = {name: sum(finite(row[name], name) for row in rows) / EXPECTED_STEPS
                      for name in ("local_ramp", "recovery_curvature", "true_v_rebound")}
        for name, value in grid_means.items():
            same(value, saved["mean_" + name], name)
        statistics = {"date": day, "case_id": case["case_id"], "case_type": case["case_type"], "policy_id": policy_id,
                      "family": episode["family"], "lambda_grid": episode["lambda_grid"], "untouched_reference": episode["lambda_grid"] is None,
                      "steps": EXPECTED_STEPS, "start_origin": origins[0], "stop_origin_exclusive": origins[-1] + 1,
                      "planning_horizons": [int(row["prediction_horizon"]) for row in rows], "dt_hours": finite(last["dt_hours"], "dt"),
                      "train_daily_cost_scale": daily_scale, "episode_normalization_denominator": denominator,
                      "raw_composite_cost": finite(saved["raw_cost"], "raw episode cost"),
                      "normalized_composite_cost": finite(saved["raw_cost"], "raw episode cost") / denominator,
                      "market_cost_usd_total": finite(saved["market_cost"], "physical market settlement"),
                      "battery_degradation_cost_total": finite(saved["battery_degradation_cost"], "battery penalty"),
                      "energy_service_cost_total": finite(saved["energy_service_cost"], "energy service penalty"),
                      "expiry_proxy_cost_total": finite(saved["expiry_proxy_cost"], "expiry penalty"),
                      "on_time_workload_rate": rate, "cohort_matured_work": matured, "cohort_ontime_completed_work": ontime,
                      **{name: finite(last[name], name) for name in COHORT_FIELDS if name not in {"cohort_matured_work", "cohort_ontime_completed_work"}},
                      "arrived_work_total": finite(saved["arrived_work"], "arrivals"), "served_work_total": finite(saved["served_work"], "service"),
                      "expiry_proxy_increment_total": finite(saved["expiry_proxy_increment"], "expiry increments"),
                      "expiry_proxy_final": finite(saved["expiry_proxy_cumulative"], "final expiry proxy"),
                      "unserved_energy_total": finite(saved["unserved_energy"], "unserved energy"),
                      "pv_curtailed_energy_total": finite(saved["pv_curtailed_energy"], "PV curtailment"),
                      "terminal_soc": saved["final_soc"], "terminal_queue": saved["final_queue"],
                      "terminal_soc_total": sum(map(float, saved["final_soc"])), "terminal_queue_total": sum(map(float, saved["final_queue"])),
                      "local_ramp_mean": grid_means["local_ramp"], "positive_curvature_mean": grid_means["recovery_curvature"],
                      "true_v_rebound_mean": grid_means["true_v_rebound"],
                      "true_v_rebound_max": max(finite(row["true_v_rebound"], "V rebound") for row in rows),
                      "legacy_compound_sla_proxy_final": optional_finite(saved.get("legacy_compound_sla_proxy"), "legacy compound SLA proxy"),
                      "legacy_expiry_work_fraction_final": optional_finite(saved.get("legacy_expiry_work_fraction"), "legacy expiry fraction"),
                      "initial_backlog_age_assumption": saved.get("initial_backlog_age_assumption"),
                      "boundary_grid_source": saved.get("boundary_grid_source"),
                      "full_chain_mean_ms": float(np.mean(times)), "full_chain_p95_ms": float(np.quantile(times, .95)),
                      "full_chain_median_ms": float(np.median(times)), "full_chain_min_ms": min(times), "full_chain_max_ms": max(times),
                      "latency_observations": EXPECTED_STEPS, "latency_observations_are_independent": False,
                      "hard_violation_count": int(saved["hard_violation_count"]), "max_hard_residual": finite(saved["max_hard_residual"], "hard residual"),
                      "source_episode": str(path), "completed_at": episode.get("completed_at")}
        same(statistics["true_v_rebound_max"], saved["max_true_v_rebound"], "maximum true V rebound")
        statuses = Counter(str(row["status"]) for row in solver_records)
        gaps = [optional_finite(row.get("actual_gap"), "actual solver gap") for row in solver_records]
        gaps = [value for value in gaps if value is not None]
        statistics.update(solver_calls=len(solver_records), solver_status_counts=dict(statuses),
                          solver_optimal_calls=sum(row["optimal"] for row in solver_records),
                          solver_accepted_not_optimal_calls=sum(row["accepted_not_optimal"] for row in solver_records),
                          solver_fallback_calls=sum(row["fallback"] for row in solver_records),
                          solver_gap_available_count=len(gaps), solver_actual_gap_max=max(gaps) if gaps else None,
                          solver_actual_gap_mean=float(np.mean(gaps)) if gaps else None,
                          solver_solve_seconds_total=sum(finite(row["solve_seconds"], "solve time") for row in solver_records),
                          solver_build_seconds_total=sum(finite(row["build_seconds"], "build time") for row in solver_records),
                          solver_update_seconds_total=sum(finite(row["update_seconds"], "update time") for row in solver_records))
        summaries.append(statistics)
        completed_times.append(episode.get("completed_at"))
    require(observed_keys == {(day, policy) for day in cases for policy in policies}, "Missing declared policy-case combination")
    require(len(step_rows) == 216 and sum(row["solver_calls"] for row in summaries) == 96, "Expected 216 decisions and 96 SMPC solves")
    require(all(time is not None for time in completed_times), "Missing completion timestamps")
    return summaries, step_rows, {"execution_protocol": protocol, "case_protocol": case_protocol, "training_protocol": training,
                                 "source_files": sources, "latest_episode_completed_at": max(completed_times)}


def add_comparisons(episodes):
    paired = []
    for day in sorted({row["date"] for row in episodes}):
        selected = [row for row in episodes if row["date"] == day]
        fronts = {"quality_time": nondominated(selected, QUALITY_TIME), "grid": nondominated(selected, GRID_FRONT)}
        for row in selected:
            for name, values in fronts.items():
                row[name + "_nondominated"] = values[row["policy_id"]]["nondominated"]
                row[name + "_dominated_by"] = values[row["policy_id"]]["dominated_by"]
                row[name + "_front_exclusion_reason"] = values[row["policy_id"]]["reason"]
        for family in FAMILIES:
            arms = {float(row["lambda_grid"]): row for row in selected if row["family"] == family and row["lambda_grid"] is not None}
            reference = arms[0.0]
            for row in arms.values():
                for metric in ("normalized_composite_cost", "local_ramp_mean", "positive_curvature_mean", "true_v_rebound_mean", "true_v_rebound_max", "full_chain_mean_ms"):
                    row["delta_to_family_gamma0_" + metric] = row[metric] - reference[metric]
                row["delta_to_family_gamma0_on_time_percentage_points"] = (
                    100 * (row["on_time_workload_rate"] - reference["on_time_workload_rate"])
                    if row["on_time_workload_rate"] is not None and reference["on_time_workload_rate"] is not None else None)
            high = arms[10.0]
            paired.append({"date": day, "case_type": reference["case_type"], "family": family,
                           "baseline_policy_id": reference["policy_id"], "comparison_policy_id": high["policy_id"],
                           "cost_gamma0": reference["normalized_composite_cost"], "cost_gamma10": high["normalized_composite_cost"],
                           "cost_absolute_change": high["normalized_composite_cost"] - reference["normalized_composite_cost"],
                           "cost_percent_change": percent_change(high["normalized_composite_cost"], reference["normalized_composite_cost"]),
                           "cost_percent_unavailable_reason": None if reference["normalized_composite_cost"] > 0 else "nonpositive_gamma0_cost_denominator",
                           "on_time_gamma0": reference["on_time_workload_rate"], "on_time_gamma10": high["on_time_workload_rate"],
                           "on_time_percentage_point_change": high["delta_to_family_gamma0_on_time_percentage_points"],
                           "local_ramp_gamma0": reference["local_ramp_mean"], "local_ramp_gamma10": high["local_ramp_mean"],
                           "local_ramp_absolute_change": high["local_ramp_mean"] - reference["local_ramp_mean"],
                           "local_ramp_percent_change": percent_change(high["local_ramp_mean"], reference["local_ramp_mean"]),
                           "local_ramp_percent_unavailable_reason": None if reference["local_ramp_mean"] > 0 else "zero_gamma0_ramp_denominator",
                           "positive_curvature_change": high["positive_curvature_mean"] - reference["positive_curvature_mean"],
                           "mean_latency_ms_change": high["full_chain_mean_ms"] - reference["full_chain_mean_ms"],
                           "pairing_basis": "same case and matched within-family gamma0; descriptive single trajectories"})
    return paired


def _label(row):
    return "Original R" if row["lambda_grid"] is None else f"{'R' if row['family'] == 'R-ECMoE' else 'SMPC'} γ={row['lambda_grid']:g}"


def f(value, digits=5):
    return "n.a." if value is None else f"{float(value):.{digits}f}"


def render_report(episodes, paired, metadata):
    nondom = sum(row["quality_time_nondominated"] is True for row in episodes)
    gridfront = sum(row["grid_nondominated"] is True for row in episodes)
    lines = ["# 两个短闭环案例的成本 按时完成率与电网敏感性", "",
             f"已保存18条完整策略轨迹、216次真实首步执行；其中96次为SMPC求解。每个日期分别比较9个预定策略，成本—按时完成率—平均耗时三维集合中共有{nondom}个非支配策略点，"
             f"成本—按时完成率—局部爬坡—正曲率四维集合中共有{gridfront}个非支配策略点。计数按日期分别求得，不能拼成一个连续或全局Pareto前沿。", "",
             "原R保持原checkpoint不变；R的γ=0/.1/1/10均从同一个旧Run 1 R130额外训练固定4轮，γ=0是匹配的继续训练对照。SMPC的四个γ使用相同风险场景范式。"
             "下面完整列出所有18个结果，不按效果或求解状态筛掉策略。", "",
             "## γ10与各自γ0的差值", "",
             "|日期|模型家族|成本变化 %|按时完成率变化 百分点|局部爬坡变化 %|", "|---|---|---:|---:|---:|"]
    for row in paired:
        lines.append(f"|{row['date']}|{row['family']}|{f(row['cost_percent_change'],2)}|{f(row['on_time_percentage_point_change'],2)}|{f(row['local_ramp_percent_change'],2)}|")
    lines += ["", "成本和爬坡百分比为100×(γ10−γ0)/γ0；基准非正或为零时不输出误导性百分比，原值和绝对差仍保留于summary.json。"
              "这些是同一日期/同一家族的一次轨迹对比，不是跨随机种子的置信区间。", ""]
    for case_type in ("rare", "routine"):
        selected = sorted([row for row in episodes if row["case_type"] == case_type], key=lambda row: (not row["untouched_reference"], row["family"] == "W1-SMPC-20", row["lambda_grid"] or 0))
        day = selected[0]["date"]
        lines += [f"## {'稀有' if case_type == 'rare' else '常规'}日期 {day}", "",
                  "|策略|归一化基础成本|按时完成 %|平均 / p95 ms|局部爬坡|正曲率|最大V形反弹|3D非支配|4D非支配|",
                  "|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for row in selected:
            rate = None if row["on_time_workload_rate"] is None else 100 * row["on_time_workload_rate"]
            lines.append(f"|{_label(row)}|{f(row['normalized_composite_cost'])}|{f(rate,2)}|{f(row['full_chain_mean_ms'],1)} / {f(row['full_chain_p95_ms'],1)}|"
                         f"{f(row['local_ramp_mean'])}|{f(row['positive_curvature_mean'])}|{f(row['true_v_rebound_max'])}|"
                         f"{'是' if row['quality_time_nondominated'] else '否' if row['quality_time_nondominated'] is not None else 'n.a.'}|"
                         f"{'是' if row['grid_nondominated'] else '否' if row['grid_nondominated'] is not None else 'n.a.'}|")
        lines += ["", "|策略|成熟工作量|按时完成工作量|未成熟工作量 / 余量|末SOC MWh|末队列|新增expiry proxy|未供能量 MWh|市场结算 USD|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in selected:
            lines.append(f"|{_label(row)}|{f(row['cohort_matured_work'],3)}|{f(row['cohort_ontime_completed_work'],3)}|"
                         f"{f(row['cohort_censored_work'],3)} / {f(row['cohort_censored_remaining_work'],3)}|{f(row['terminal_soc_total'],4)}|"
                         f"{f(row['terminal_queue_total'],3)}|{f(row['expiry_proxy_increment_total'],3)}|{f(row['unserved_energy_total'],5)}|{f(row['market_cost_usd_total'],2)}|")
        lines += [""]
    lines += ["## 求解状态与耗时边界", "",
              "|日期|SMPC γ|调用数|最优|接受非最优|回退|gap有效数 / 最大实际gap|solve合计 s|build合计 s|update合计 s|状态原文与计数|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in episodes:
        if row["family"] != "W1-SMPC-20":
            continue
        lines.append(f"|{row['date']}|{row['lambda_grid']:g}|{row['solver_calls']}|{row['solver_optimal_calls']}|{row['solver_accepted_not_optimal_calls']}|"
                     f"{row['solver_fallback_calls']}|{row['solver_gap_available_count']} / {f(row['solver_actual_gap_max'],6)}|"
                     f"{f(row['solver_solve_seconds_total'],2)}|{f(row['solver_build_seconds_total'],2)}|{f(row['solver_update_seconds_total'],2)}|"
                     f"{json.dumps(row['solver_status_counts'], ensure_ascii=False)}|")
    lines += ["", "平均和p95仅描述每条episode中12个连续、状态和规划长度都变化的完整链调用；没有对这些异质调用做独立重采样或bootstrap。"
              "完整链包含fresh W1编码和控制器决策/情景生成求解、共享装配与首步提取，排除模型加载、磁盘I/O、输入准备及实际环境执行/记账。"
              "各SMPC episode第一次模型构建已包含在其调用中。600秒是每次求解上限，不能作为测得延迟；实际gap和回退保留，缺失gap不填零。", "",
              "## 指标与证据范围", "",
              "- 基础raw composite cost = 物理市场结算 + 电池退化罚项 + 弃光/未供电罚项 + 本步新增累计expiry proxy罚项，不含训练/优化的γ电网惩罚。"
              "归一化episode成本 = 12步raw cost之和 / (train_daily_D × 12/96)，不按每步规划H分别除分母。市场结算使用实际MW×小时×原始USD/MWh，单独列USD；混合composite罚项不自动称美元。",
              "- 真正按时完成率 = episode末已成熟cohort中在due之前完成的工作量 / 这些成熟cohort的原始工作量。"
              "这不是1−SLA proxy。未成熟cohort的原始工作量和剩余量单独列为右删失，不加入成熟率分母。未完成和迟到量、末队列与SOC均保留。",
              "- 初始pending工作量只在episode起点建立一次cohort；其真实年龄不可得，固定假设在start+deadline−1到期，此假设影响按时完成率解释。"
              "之后使用各政策实际执行状态反馈和FIFO分配，不在每个步重新重置初始队列期限。",
              "- local ramp为各节点abs(g_t−g_previous)/(import_limit+export_limit)的均值。positive curvature为max(G_t−2G_previous+G_previous2,0)/总grid range。"
              "另列真实三点V形进口回升：min(max(P_previous2−P_previous,0),max(P_t−P_previous,0))/总进口容量，P为各节点正进口之和。"
              "主表使用12步最大V形反弹，与峰值统计口径一致；均值另存CSV/JSON，并明确包含已知的前两期边界。三者不是电网福利或可靠性本身。",
              "- 每日期三维经验非支配集合最小化成本和平均完整链时间、最大化按时完成率；四维grid集合最小化成本/爬坡/正曲率、最大化按时完成率。"
              "仅在当日9个已测策略中按全部目标不差且至少一项严格更好判断，数值容差见summary.json；无连续前沿拟合、无全局最优或同等控制质量宣称。",
              "- 两天由固定价格规则后验选取：稀有日取稀有集合中最大RT尖峰，常规日取接近常规集合日均RT中位数的日期，并用相同UTC窗口。"
              "它们不是随机年度样本；12步真闭环也不是全天或全年闭环。四个R短继续训练分支共享一个父模型和训练seed，不能当成4个独立训练seed。",
              "- γ值和训练预算在2025执行前固定；2025用于案例选择与评估，不用于短适配参数拟合、checkpoint或γ筛选。此前查看过的2025结果和协议修订历史不被删除。", "",
              "## 图与可追溯数据", "", "![实测策略权衡](figures/Fig8_closedloop_tradeoffs.png)", "",
              "全部18条汇总见episode_summary.csv；216条首步记录及原始solverstats见step_records.csv；配对差、经验非支配关系、来源hash和定义见summary.json。", ""]
    return "\n".join(lines)


def figure8(root, episodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "axes.titleweight": "bold",
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.65, 5.25))
    fig.subplots_adjust(left=.11, right=.965, bottom=.18, top=.82, wspace=.34, hspace=.66)
    palettes = {"R-ECMoE": ["#93B1C8", "#729BB8", "#4E82A7", "#2B5E85"],
                "W1-SMPC-20": ["#E4C0A4", "#D6A078", "#C38453", "#A96A3B"]}
    omitted, log_axes = [], []
    for col, case_type in enumerate(("rare", "routine")):
        values = [row for row in episodes if row["case_type"] == case_type]
        values.sort(key=lambda row: (not row["untouched_reference"], row["family"] == "W1-SMPC-20", row["lambda_grid"] or 0))
        day = values[0]["date"]
        for grid in (False, True):
            ax = axes[int(grid), col]
            xy = [(row["local_ramp_mean"], None if row["on_time_workload_rate"] is None else 100 * row["on_time_workload_rate"])
                  if grid else (row["normalized_composite_cost"], row["full_chain_mean_ms"] / 1000) for row in values]
            finite_xy = [(x, y) for x, y in xy if y is not None]
            if not grid and min(y for _, y in finite_xy) > 0 and max(y for _, y in finite_xy) / min(y for _, y in finite_xy) > 20:
                ax.set_yscale("log")
                log_axes.append(f"{case_type}_latency")
            for row, (x, y) in zip(values, xy):
                if y is None:
                    omitted.append({"panel": f"{case_type}_grid", "policy_id": row["policy_id"], "reason": "undefined_mature_cohort_rate"})
                    continue
                if row["untouched_reference"]:
                    color, marker, size = "#505B66", "*", 75
                else:
                    color = palettes[row["family"]][WEIGHTS.index(float(row["lambda_grid"]))]
                    marker, size = ("o", 35) if row["family"] == "R-ECMoE" else ("s", 33)
                ax.scatter(x, y, color=color, marker=marker, s=size, edgecolors="white", linewidths=.45, zorder=3)
                if row["grid_nondominated" if grid else "quality_time_nondominated"] is True:
                    ax.scatter(x, y, facecolors="none", edgecolors="#29323A", marker="o", s=90, linewidths=.65, zorder=2)
            ax.margins(x=.20, y=.27)
            if grid and finite_xy and all(abs(y - 100) < 1e-6 for _, y in finite_xy):
                ax.set_ylim(98, 100.7)
                ax.set_yticks([98, 99, 100])
                ax.text(.5, .08, "All policies: 100% on-time completion", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=6, color="#626A72")
            ax.set_title(f"{'Rare' if case_type == 'rare' else 'Routine'} case  {day}")
            ax.set_xlabel("Mean normalized local ramp" if grid else "Normalized base composite cost")
            ax.set_ylabel("Mature-cohort on-time work (%)" if grid else "Mean complete-chain time (s)" + ("; log scale" if ax.get_yscale() == "log" else ""))
            ax.text(-.20, 1.10, "abcd"[int(grid) * 2 + col], transform=ax.transAxes, fontsize=9, weight="bold")
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            occupied = []
            for row, (x, y) in zip(values, xy):
                if y is None:
                    continue
                if row["family"] == "R-ECMoE" and not row["untouched_reference"]:
                    omitted.append({"panel": f"{case_type}_{'grid' if grid else 'latency'}", "policy_id": row["policy_id"],
                                    "reason": "near_coincident_R_points_labeled_as_group_full_values_in_table"})
                    continue
                label = "R: all γ + orig" if row["untouched_reference"] else f"{row['lambda_grid']:g}"
                chosen = None
                for dx, dy in [(5, 5), (5, -10), (-5, 5), (-5, -10), (7, 15), (-7, 15), (7, -20), (-7, -20), (20, 2), (-20, 2)]:
                    candidate = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                                            ha="left" if dx > 0 else "right", va="center", fontsize=6,
                                            color="#30343B", arrowprops={"arrowstyle": "-", "lw": .35, "color": "#7D8790"} if abs(dx) + abs(dy) > 15 else None,
                                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .15})
                    box = candidate.get_window_extent(renderer).expanded(1.05, 1.10)
                    limits = ax.get_window_extent(renderer)
                    if limits.x0 <= box.x0 and box.x1 <= limits.x1 and limits.y0 <= box.y0 and box.y1 <= limits.y1 and not any(box.overlaps(old) for old in occupied):
                        chosen = candidate
                        occupied.append(box)
                        break
                    candidate.remove()
                if chosen is None:
                    omitted.append({"panel": f"{case_type}_{'grid' if grid else 'latency'}", "policy_id": row["policy_id"], "reason": "dense_label_full_values_in_table"})
    fig.legend(handles=[Line2D([], [], color="#4E82A7", marker="o", ls="", label="R-ECMoE continuation"),
                        Line2D([], [], color="#C38453", marker="s", ls="", label="W1-SMPC-20"),
                        Line2D([], [], color="#505B66", marker="*", ls="", markersize=9, label="Original R (untouched)")],
               loc="upper center", bbox_to_anchor=(.53, .966), ncol=3, frameon=False, fontsize=6.8, handletextpad=.35, columnspacing=1.2)
    fig.text(.53, .895, "SMPC labels: γ = 0, 0.1, 1, 10; near-coincident R points are labeled together.", ha="center", fontsize=6.5)
    fig.text(.52, .066, "Outlines: empirical nondominated policies in cost / timeliness / time (a,b)", ha="center", fontsize=6.2)
    fig.text(.52, .038, "or cost / timeliness / local ramp / recovery curvature (c,d); no continuous front is fitted.", ha="center", fontsize=6.2)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    stem = folder / "Fig8_closedloop_tradeoffs"
    outputs = {}
    for suffix, options in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, facecolor="white", **options)
        outputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    return {"outputs": outputs, "dimensions_inches": [6.65, 5.25], "labels_omitted_where_dense": omitted, "log_latency_axes": log_axes,
            "visual_review": "requires_saved_PNG_inspection", "no_continuous_front_lines": True}


def build_report(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    episodes, steps, metadata = load_evidence(root)  # no outputs until all 18 exist
    comparisons = add_comparisons(episodes)
    episodes.sort(key=lambda row: (row["case_type"] != "rare", row["date"], not row["untouched_reference"], row["family"] == "W1-SMPC-20", row["lambda_grid"] or 0))
    figure = figure8(root, episodes)
    summary = {"schema": "dcvpp.closedloop-sensitivity.report.v1", "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
               "evidence_as_of": metadata["latest_episode_completed_at"], "complete_episodes": len(episodes), "executed_decisions": len(steps),
               "SMPC_solves": sum(row["solver_calls"] for row in episodes), "days": sorted({row["date"] for row in episodes}),
               "episodes": episodes, "gamma10_vs_gamma0": comparisons,
               "empirical_front_definitions": {"quality_time": {"metrics": QUALITY_TIME, "directions": ["min", "max", "min"]},
                                              "grid": {"metrics": GRID_FRONT, "directions": ["min", "max", "min", "min"]},
                                              "scope": "separate measured nine-policy set per date", "absolute_tolerance": 1e-9,
                                              "relative_tolerance": 1e-9, "continuous_or_global_front_claim": False},
               "definitions": {"normalized_composite_cost": "sum of twelve raw executed base costs / (train_daily_D * 12/96); gamma terms excluded",
                               "on_time_workload_rate": "final on-time completed original work in matured FIFO cohorts / their original work; not 1-SLA proxy",
                               "censoring": "unmatured cohort original work and remaining work reported separately; excluded from mature-cohort ratio",
                               "local_ramp_mean": "mean over executed steps and nodes of abs(g_t-g_previous)/(I+E)",
                               "positive_curvature_mean": "mean executed max(G_t-2G_previous+G_previous2,0)/sum(I+E)",
                               "true_v_rebound_mean": "mean min(max(P_previous2-P_previous,0),max(P_t-P_previous,0))/sum(I); P=sum positive node imports; includes known two-step boundary",
                               "true_v_rebound_max": "maximum of the same twelve observed three-point V-shaped rebound values; denominator sum(import limits), not import+export",
                               "latency": "descriptive mean/p95 of twelve consecutive heterogeneous complete-chain calls; no bootstrap or independent-repeat claim",
                               "market_cost_usd_total": "sum of actual raw USD/MWh prices times MW dispatch times hours; separate from mixed composite penalty units"},
               "scope": {"full_day_or_annual_closedloop": False, "posthoc_price_selected_cases": True, "random_annual_sample": False,
                         "independent_training_seeds": 1, "R_short_adaptation_arms": 4, "untouched_original_R_retained": True,
                         "control_quality_equivalence_assumed": False, "gamma_selected_using_test_outcomes": False,
                         "fallback_episodes_or_steps_removed": False, "new_model_or_solver_execution_by_report": False},
               "source_files": metadata["source_files"], "protocol_paths": {"execution": str(root / "execution_protocol.json"), "cases": str(root / "case_protocol.json"), "training": str(root / "training_protocol.json")},
               "figure": figure}
    write_csv(root / "episode_summary.csv", episodes)
    write_csv(root / "step_records.csv", steps)
    (root / "REPORT.md").write_text(render_report(episodes, comparisons, metadata), encoding="utf-8")
    write_json(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    result = build_report(args.root)
    print(json.dumps({"status": result["status"], "episodes": result["complete_episodes"], "steps": result["executed_decisions"],
                      "summary": str(args.root / "summary.json")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
