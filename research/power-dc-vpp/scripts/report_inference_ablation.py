"""Report fixed-checkpoint interventions without claiming retrained variants."""
from __future__ import annotations

import argparse
from datetime import date
import _sha256
import json
from pathlib import Path
import statistics

import numpy as np

from multiseed_common import ROOT, SEEDS, VARIANTS, W0_BUDGETS, now, write_csv, write_json
import report_multiseed_ablation as shared


DESIGN = "same_checkpoint_inference_intervention"
METRICS = {**shared.METRICS, "mean_realized_operating_cost": "平均原单位成本",
           "cvar95_realized_operating_cost": "原单位成本 CVaR95"}
DAY_FIELDS = (*shared.DAY_FIELDS, "realized_operating_cost")
require, read_json, read_csv = shared.require, shared.read_json, shared.read_csv
finite, within, flag = shared.finite, shared.within, shared.flag


def panel_statistics(panel, indices=None):
    result = shared.panel_statistics(panel, indices)
    daily_raw = panel["matrices"]["realized_operating_cost"].mean(axis=1)
    samples = daily_raw[None] if indices is None else daily_raw[indices]
    raw = {"mean_realized_operating_cost": samples.mean(axis=1),
           "cvar95_realized_operating_cost": shared.cvar95(samples, axis=1)}
    result.update({key: float(value[0]) for key, value in raw.items()} if indices is None else raw)
    return result


def paired_metrics(panels, seeds, *, resamples=shared.RESAMPLES):
    rows = []
    indices = shared.circular_blocks(len(panels[seeds[0], "full"]["dates"]), resamples=resamples)
    for seed in seeds:
        full, full_boot = panel_statistics(panels[seed, "full"]), panel_statistics(panels[seed, "full"], indices)
        for variant in VARIANTS:
            if variant == "full":
                continue
            other, other_boot = panel_statistics(panels[seed, variant]), panel_statistics(panels[seed, variant], indices)
            for metric in METRICS:
                if metric in full_boot:
                    distribution = other_boot[metric] - full_boot[metric]
                    low, high = map(float, np.quantile(distribution, [0.025, 0.975]))
                    conclusion = "full_lower" if low > 0 else "intervention_lower" if high < 0 else "interval_includes_zero"
                else:
                    low = high = shared.UNAVAILABLE
                    conclusion = "no_saved_timing_replicates"
                rows.append({"seed": seed, "variant": variant, "metric": metric,
                             "comparison_design": DESIGN, "separately_retrained": False,
                             "full_value": full[metric], "intervention_value": other[metric],
                             "delta_intervention_minus_full": other[metric] - full[metric],
                             "ci95_low": low, "ci95_high": high, "interpretation": conclusion,
                             "independent_trained_models_in_pair": 1,
                             "paired_dates": len(panels[seed, variant]["dates"]),
                             "block_days": shared.BLOCK_DAYS, "resamples": resamples if metric in full_boot else shared.UNAVAILABLE,
                             "bootstrap_seed": shared.BOOTSTRAP_SEED,
                             "method": "paired_circular_date_blocks_recompute_each_side" if metric in full_boot else "timing_ci_unavailable"})
    return rows


def aggregate_metrics(seed_metrics, seeds):
    lookup = {(row["seed"], row["variant"]): row for row in seed_metrics}
    rows = []
    for variant in VARIANTS:
        for metric in METRICS:
            values = [lookup[seed, variant][metric] for seed in seeds]
            deltas = [lookup[seed, variant][metric] - lookup[seed, "full"][metric] for seed in seeds]
            rows.append({"variant": variant, "label": VARIANTS[variant], "metric": metric,
                         "comparison_design": DESIGN, "separately_retrained": False,
                         "independent_training_seeds": len(seeds), "mean": statistics.mean(values),
                         "sample_sd": statistics.stdev(values) if len(seeds) > 1 else shared.UNAVAILABLE,
                         "min": min(values), "max": max(values), "paired_delta_mean": statistics.mean(deltas),
                         "paired_delta_sample_sd": statistics.stdev(deltas) if len(seeds) > 1 else shared.UNAVAILABLE})
    return rows


def _check_design(rows, seed, label):
    require(all(int(row["seed"]) == seed and row["variant"] in VARIANTS for row in rows),
            f"Foreign seed/intervention in {label}")
    require(all(row.get("comparison_design") == DESIGN and flag(row.get("separately_retrained"), label) is False for row in rows),
            f"Incorrect inference-intervention design in {label}")


def _seed_rows(folder, metadata, *, seed, root, expected_days):
    decisions = read_csv(folder / "per_decision_models.csv")
    daily = read_csv(folder / "per_day_models.csv")
    summaries = read_csv(folder / "model_summary.csv")
    paired_source = read_csv(folder / "paired_comparisons.csv")
    require(len(decisions) == expected_days * 6 * 7, f"Wrong inference replay count: {seed}")
    require(len(daily) == expected_days * 7 and len(summaries) == 7, f"Incomplete inference summaries: {seed}")
    require(bool(paired_source), f"Missing saved paired comparisons: {seed}")
    for label, rows in (("decision rows", decisions), ("daily rows", daily), ("summary", summaries), ("paired comparisons", paired_source)):
        _check_design(rows, seed, label)
    require({row["variant"] for row in paired_source} == set(VARIANTS) - {"full"}, f"Paired comparator coverage mismatch: {seed}")
    summary_by_variant = {row["variant"]: row for row in summaries}
    require(set(summary_by_variant) == set(VARIANTS), f"Duplicate/missing intervention summary: {seed}")
    full_record = metadata["full_model_record"]
    require(int(full_record["seed"]) == seed and full_record["variant"] == "full", f"Wrong full-model record: {seed}")
    checkpoint, w1_checkpoint = within(full_record["checkpoint"], root), within(full_record["w1_checkpoint"], root)
    day_lookup, decision_lookup = {}, {}
    for row in daily:
        key = row["variant"], row["operating_date"]
        require(key not in day_lookup and date.fromisoformat(key[1]).year == 2025, f"Duplicate/non-2025 daily row: {key}")
        day_lookup[key] = row
    dates = sorted(day for variant, day in day_lookup if variant == "full")
    require(len(dates) == expected_days and sorted(metadata["dataset_dates"]) == dates, f"Incomplete declared dates: {seed}")
    require(all({day for v, day in day_lookup if v == variant} == set(dates) for variant in VARIANTS), f"Unpaired dates: {seed}")
    for row in decisions:
        window = int(row["stage_id"]), int(row["decision_origin"]), int(row["horizon"])
        key = row["variant"], row["operating_date"], window
        require(key not in decision_lookup and window in shared.WINDOWS and row["operating_date"] in dates, f"Invalid/duplicate window: {key}")
        decision_lookup[key] = row
    panels, metrics = {}, []
    for variant in VARIANTS:
        summary = summary_by_variant[variant]
        require(within(summary["checkpoint"], root) == checkpoint and within(summary["w1_checkpoint"], root) == w1_checkpoint,
                f"Inference conditions do not share the exact full-model checkpoint: {seed}/{variant}")
        require(int(summary["checkpoint_epoch"]) > 0 and int(summary["days"]) == expected_days
                and int(summary["replay_rows"]) == expected_days * 6, f"Bad model epoch/coverage: {seed}/{variant}")
        matrices = {name: np.asarray([[finite(decision_lookup[variant, day, window][name], name) for window in shared.WINDOWS]
                                     for day in dates], dtype=np.float64)
                    for name in (*DAY_FIELDS, "hard_residual_max")}
        for index, day in enumerate(dates):
            saved = day_lookup[variant, day]
            for name in DAY_FIELDS:
                shared._same_number(saved[name], float(matrices[name][index].mean()), f"{seed}/{variant}/{day}/{name}")
            maximum = float(matrices["hard_residual_max"][index].max())
            shared._same_number(saved["hard_residual_max"], maximum, "daily hard residual")
            require(flag(saved["hard_violation"], "daily hard_violation") == (maximum > shared.HARD_TOLERANCE), "Incorrect daily hard flag")
            for window in shared.WINDOWS:
                row = decision_lookup[variant, day, window]
                require(flag(row["hard_violation"], "decision hard_violation") ==
                        (finite(row["hard_residual_max"], "hard residual") > shared.HARD_TOLERANCE), "Incorrect decision hard flag")
        latency = finite(summary["p95_latency_ms_batch1"], "batch1 p95 latency")
        require(latency >= 0, "Negative inference latency")
        panel = {"dates": dates, "matrices": matrices, "latency": latency}
        measured = panel_statistics(panel)
        for name, value in measured.items():
            if name in shared.METRICS or name in summary:
                shared._same_number(summary.get(name), value, f"{seed}/{variant}/{name}")
        panels[seed, variant] = panel
        metrics.append({"seed": seed, "variant": variant, "label": VARIANTS[variant],
                        "comparison_design": DESIGN, "separately_retrained": False,
                        "days": expected_days, "replay_rows": expected_days * 6,
                        "checkpoint_epoch": int(summary["checkpoint_epoch"]), "checkpoint": str(checkpoint),
                        "w1_checkpoint": str(w1_checkpoint), **measured,
                        "selection_policy": shared.SELECTION_POLICY,
                        "sla_reference_exceeded": measured["worst_origin_sla"] > shared.SLA_REFERENCE,
                        "hard_reference_exceeded": measured["max_hard_residual"] > shared.HARD_TOLERANCE,
                        "latency_reference_exceeded": measured["p95_latency_ms_batch1"] > shared.LATENCY_REFERENCE_MS})
    require(len({row["checkpoint_epoch"] for row in metrics}) == 1, f"Conditions claim different checkpoint epochs: {seed}")
    return panels, metrics, dates


def _same_record(left, right, root, label):
    require(int(left["seed"]) == int(right["seed"]) and left.get("variant", "full") == right.get("variant", "full") == "full",
            f"Wrong full-model seed/role: {label}")
    for name in ("checkpoint", "w1_checkpoint"):
        require(within(left[name], root) == within(right[name], root), f"Changed frozen {name}: {label}")


def _digest(path):
    # The builtin avoids the host's observed Torch/OpenSSL provider conflict.
    return _sha256.sha256(Path(path).read_bytes()).hexdigest()


def _protocol_versions(root, path, protocol):
    """Resolve retained evaluation hashes through explicit user amendments."""
    versions = {_digest(path): {"path": path, "protocol": protocol, "historical": False}}
    amendments = protocol.get("configuration_amendments", [])
    require(isinstance(amendments, list), "Configuration amendments must be a list")
    if protocol.get("future_seed_configurations_frozen_before_first_test") is not True:
        require(protocol.get("future_seed_configurations_frozen_before_first_test") is False
                and protocol.get("initial_configurations_frozen_before_first_test") is True and bool(amendments),
                "A changed final configuration requires initial-freeze and amendment evidence")
    current_hashes = {str(Path(name).resolve()): value for name, value in protocol["configuration_sha256"].items()}
    for amendment in amendments:
        require(amendment.get("user_authorized") is True and amendment.get("applied_before_target_training") is True
                and amendment.get("reason") == "explicit_user_instruction",
                "Configuration revision lacks explicit user/before-training authorization")
        require(int(amendment["seed"]) in SEEDS and amendment["phase"] == "w0",
                "Unregistered configuration-amendment target")
        prior_path = within(amendment["prior_protocol_path"], root)
        prior_digest = _digest(prior_path)
        require(prior_digest == amendment["prior_protocol_sha256"], "Archived pre-amendment protocol hash changed")
        prior = read_json(prior_path)
        prior_hashes = {str(Path(name).resolve()): value for name, value in prior.get("configuration_sha256", {}).items()}
        target = str(Path(amendment["configuration_path"]).resolve())
        require(prior_hashes.get(target) == amendment["before_sha256"]
                and current_hashes.get(target) == amendment["after_sha256"],
                "Configuration amendment does not match its recorded before/after protocol hashes")
        require(prior.get("comparison_design") == DESIGN and prior.get("separately_retrained") is False,
                "Archived protocol describes a different inference comparison")
        versions[prior_digest] = {"path": prior_path, "protocol": prior, "historical": True}
    return versions, amendments


def load_study(root, seeds, *, expected_days=362):
    root = Path(root).resolve()
    state = read_json(root / "inference_study_status.json")
    protocol_path = root / "inference_protocol.json"
    protocol = read_json(protocol_path)
    protocol_digest = _digest(protocol_path)
    require(protocol.get("comparison_design") == DESIGN and protocol.get("separately_retrained") is False,
            "Protocol must identify fixed-checkpoint inference interventions")
    require(tuple(protocol.get("seeds", ())) == tuple(SEEDS) and set(protocol.get("variants", ())) == set(VARIANTS),
            "Unregistered inference protocol seeds/conditions")
    for name, value in (("final_results_may_modify_selection", False),
                        ("later_seed_training_after_prior_seed_test", True), ("no_test_to_training_feedback", True)):
        require(protocol.get(name) is value, f"Missing/incorrect chronological protocol field: {name}")
    require(isinstance(protocol.get("configuration_sha256"), dict) and bool(protocol["configuration_sha256"]),
            "Protocol lacks the configuration freeze recorded before the first test")
    protocol_versions, amendments = _protocol_versions(root, protocol_path, protocol)
    budgets = protocol.get("w0_epoch_budgets", {})
    require(isinstance(budgets, dict) and all(str(seed) in budgets or seed in budgets for seed in SEEDS),
            "Protocol lacks per-seed W0 budgets")
    budgets = {str(seed): int(budgets.get(str(seed), budgets.get(seed))) for seed in SEEDS}
    planned = protocol.get("full_model_records", [])
    require(isinstance(planned, list) and len(planned) == len(SEEDS), "Protocol requires three full-model records")
    planned = {int(row["seed"]): row for row in planned}
    require(set(planned) == set(SEEDS), "Protocol full-model seed lineage is incomplete")
    jobs = state.get("jobs", [])
    expected_jobs = {(seed, phase, variant) for seed in SEEDS for phase, variant in (("w0", None), ("w1", None), ("r", "full"))}
    require(len(jobs) == 9 and {(int(row["seed"]), row["phase"], row.get("variant")) for row in jobs} == expected_jobs,
            "Inference study must contain nine base-training stages only")
    included_jobs = [row for row in jobs if int(row["seed"]) in seeds]
    require(all(row.get("status") == "completed" for row in included_jobs),
            "Requested seeds have not completed their W0/W1/full-R training")
    for job in included_jobs:
        within(job["run_root"], root)
    evaluations = {int(row["seed"]): row for row in state.get("evaluations", [])}
    require(set(evaluations) == set(SEEDS), "Evaluation schedule does not contain the three registered seeds")
    require(all(evaluations[seed].get("status") == "completed" for seed in seeds),
            "Requested seed evaluation is not complete")
    sources = [str(root / "inference_study_status.json"), *(str(row["path"]) for row in protocol_versions.values())]
    panels, metrics, metadata_records, common_dates = {}, [], [], None
    for seed in seeds:
        folder = root / "inference_evaluation_2025" / f"seed_{seed}"
        require(within(evaluations[seed]["output_root"], root) == folder, f"Wrong evaluation output directory: {seed}")
        metadata = read_json(folder / "metadata.json")
        require(metadata.get("status") == "completed" and int(metadata.get("seed", -1)) == seed,
                f"Incomplete/wrong-seed inference metadata: {seed}")
        require(metadata.get("year") == 2025 and metadata.get("split") == "locked_test", f"Wrong test split: {seed}")
        require(metadata.get("comparison_design") == DESIGN and metadata.get("separately_retrained") is False,
                f"Metadata describes a different scientific comparison: {seed}")
        require(set(metadata.get("variants", ())) == set(VARIANTS), f"Wrong inference condition registry: {seed}")
        require(metadata.get("seed_base_training_completed") is True
                and set(metadata.get("completed_base_stages", ())) == {"w0", "w1", "r_full"},
                f"Current seed was not frozen after all base stages completed: {seed}")
        for name in ("training_performed", "checkpoint_selection_performed", "final_results_may_modify_selection",
                     "other_seeds_training_required_before_this_evaluation"):
            require(metadata.get(name) is False, f"Incorrect inference protocol boundary: {seed}/{name}")
        recorded_protocol = protocol_versions.get(metadata["protocol_sha256"])
        require(recorded_protocol is not None, f"Evaluation protocol hash is neither current nor an authorized archive: {seed}")
        require(within(metadata["protocol_path"], root) in {protocol_path, recorded_protocol["path"]},
                f"Evaluation protocol source path is not registered: {seed}")
        historical_records = {int(row["seed"]): row for row in recorded_protocol["protocol"]["full_model_records"]}
        _same_record(historical_records[seed], metadata["full_model_record"], root, f"evaluation-time protocol {seed}")
        freeze_path = within(metadata["freeze_manifest_path"], root)
        require(metadata["freeze_manifest_sha256"] == _digest(freeze_path), f"Seed freeze manifest changed: {seed}")
        frozen = read_json(freeze_path)
        require(isinstance(frozen.get("full_model_record"), dict), f"Seed freeze must identify one trained full model: {seed}")
        _same_record(frozen["full_model_record"], metadata["full_model_record"], root, f"seed freeze {seed}")
        _same_record(planned[seed], metadata["full_model_record"], root, f"protocol {seed}")
        seed_panels, seed_metrics, dates = _seed_rows(folder, metadata, seed=seed, root=root, expected_days=expected_days)
        require(common_dates is None or dates == common_dates, "Training seeds do not share the same test dates")
        common_dates = dates
        panels.update(seed_panels)
        metrics.extend(seed_metrics)
        metadata_records.append({"seed": seed, "metadata_path": str(folder / "metadata.json"),
                                 "freeze_manifest_path": str(freeze_path), "freeze_manifest_sha256": metadata["freeze_manifest_sha256"],
                                 "full_model_record": metadata["full_model_record"], "seed_base_training_completed": True,
                                 "recorded_protocol_path": metadata["protocol_path"], "protocol_sha256": metadata["protocol_sha256"],
                                 "resolved_protocol_path": str(recorded_protocol["path"]),
                                 "protocol_is_historical": recorded_protocol["historical"]})
        sources.extend([str(freeze_path), *(str(folder / name) for name in ("metadata.json", "per_decision_models.csv",
                       "per_day_models.csv", "model_summary.csv", "paired_comparisons.csv"))])
    require(len({row["checkpoint"] for row in metrics}) == len(seeds), "Inference study must have exactly one distinct full R checkpoint per seed")
    require(len({row["w1_checkpoint"] for row in metrics}) == len(seeds), "Independent seeds must have distinct W1 checkpoints")
    cancelled = []
    legacy_path = root / "study_status.json"
    if legacy_path.exists():
        legacy = read_json(legacy_path)
        cancelled = [{name: row[name] for name in ("id", "seed", "phase", "variant", "status", "run_root", "elapsed_seconds") if name in row}
                     for row in legacy.get("jobs", []) if row.get("phase") == "r" and row.get("variant") != "full"
                     and str(row.get("status", "")).startswith("cancelled")]
        sources.append(str(legacy_path))
    validation = {
        "seeds": list(seeds), "report_scope": "single_seed" if len(seeds) == 1 else "three_seed_final",
        "comparison_design": DESIGN, "separately_retrained": False, "independent_trained_r_checkpoints": len(seeds),
        "training_stages_in_report": len(included_jobs), "planned_base_training_stages": 9,
        "conditions_per_seed": 7, "days_per_condition": expected_days, "windows_per_day": 6,
        "replay_rows": len(seeds) * 7 * expected_days * 6, "daily_rows": len(seeds) * 7 * expected_days,
        "dataset_dates": common_dates, "same_checkpoint_within_seed": True, "distinct_checkpoint_across_seeds": True,
        "coverage_and_saved_metrics_recomputed": True, "validation_scope": "coverage_lineage_numeric_consistency_only",
        "selection_policy": shared.SELECTION_POLICY, "scientific_qualification_required": False,
        "threshold_exceedances_block_report": False, "training_repeat_unit": "full_model_seed", "sd_ddof": 1,
        "protocol_path": str(protocol_path), "protocol_sha256": protocol_digest,
        "future_seed_configurations_frozen_before_first_test": protocol["future_seed_configurations_frozen_before_first_test"],
        "initial_configurations_frozen_before_first_test": protocol.get("initial_configurations_frozen_before_first_test",
                                                                       protocol["future_seed_configurations_frozen_before_first_test"]),
        "configuration_amendments": amendments, "no_test_to_training_feedback": True,
        "no_claim_final_configurations_unchanged_after_first_test": bool(amendments),
        "later_seed_training_after_prior_seed_test": True, "no_claim_all_seeds_trained_before_first_evaluation": True,
        "w0_short_epoch_budgets": budgets, "seed_and_budget_variation_not_separately_identified": True,
        "cancelled_independent_training_history_excluded_from_effects": cancelled,
        "seed_evidence": metadata_records, "source_paths": sources, "_included_jobs": included_jobs,
    }
    return state, panels, metrics, validation


def main_findings(validation, aggregate, paired):
    """Describe observed effect sizes and their paired support, never a preset win."""
    seeds = validation["seeds"]
    lookup = {(row["variant"], row["metric"]): row for row in aggregate}
    chosen = max((variant for variant in VARIANTS if variant != "full"),
                 key=lambda variant: abs(lookup[variant, "mean_normalized_operating_cost"]["paired_delta_mean"]))
    labels = {"mean_normalized_operating_cost": "平均成本", "cvar95_normalized_operating_cost": "成本 CVaR95",
              "mean_deadline_expired_work": "平均逾期工作量", "worst_origin_sla": "最差窗口 SLA 严重度"}
    changes = []
    for metric, label in labels.items():
        delta = lookup[chosen, metric]["paired_delta_mean"]
        baseline = lookup["full", metric]["mean"]
        if baseline:
            percent = delta / abs(baseline) * 100.0
            value = f"{percent:+.2f}%" if abs(percent) >= 0.01 else f"{percent:+.3g}%"
        else:
            value = f"变化 {delta:+.6g}（基线为零，未计算百分比）"
        changes.append(f"{label}{value}")
    sentences = [f"平均成本变化幅度最大的是「{VARIANTS[chosen]}」：相对完整条件，" + "、".join(changes) + "。"]
    by_pair = {(row["seed"], row["variant"], row["metric"]): row for row in paired}
    full_supported, intervention_supported, unresolved = [], [], []
    for metric, label in labels.items():
        intervals = [by_pair[seed, chosen, metric] for seed in seeds]
        if all(isinstance(row["ci95_low"], (int, float)) and row["ci95_low"] > 0 for row in intervals):
            full_supported.append(label)
        elif all(isinstance(row["ci95_high"], (int, float)) and row["ci95_high"] < 0 for row in intervals):
            intervention_supported.append(label)
        else:
            unresolved.append(label)
    qualifier = "配对95%区间" if len(seeds) == 1 else "三个 seed 各自的配对95%区间均"
    if full_supported:
        sentences.append(qualifier + "支持完整条件的" + "、".join(full_supported) + "更低。")
    if intervention_supported:
        sentences.append(qualifier + "支持该干预条件的" + "、".join(intervention_supported) + "更低。")
    if unresolved:
        sentences.append("、".join(unresolved) + ("的区间包含零，改善方向尚未确定。" if len(seeds) == 1 else
                                                "尚无三个 seed 一致的配对区间支持，不能概括为稳定改善。"))
    if len(seeds) == 1:
        sentences.append(f"证据仅来自 seed {seeds[0]} 的固定模型，不外推至其他 seed 或删除模块后重新训练的模型。")
    else:
        sentences.append("这些差异衡量三个固定模型的推理依赖性，不证明对应结构在重新训练后不可替代。")
    return ["## 主要发现", "", "".join(sentences), ""]


def render_report(validation, metrics, aggregate, paired):
    seeds, count = validation["seeds"], len(validation["seeds"])
    look = {(row["variant"], row["metric"]): row for row in aggregate}
    heading = f"Seed {seeds[0]} 固定模型推理消融报告" if count == 1 else "三种子固定模型推理消融报告"
    if validation["configuration_amendments"]:
        configuration_note = (
            "初始配置在首次本方案测试前冻结；其后发生明确用户批准、目标训练开始前生效的修订："
            + "、".join(f"seed {row['seed']} / {row['phase'].upper()}" for row in validation["configuration_amendments"])
            + "。该变更发生在首 seed 测试后，不能声称所有最终配置均预先冻结或全程无事后改动。"
            "协议仍记录禁止根据2025结果反馈调参；这一声明不抹去用户修订历史。旧评价保留当时的协议哈希，并通过原始归档快照核对。"
        )
    else:
        configuration_note = "后续 seed 配置在首次本方案测试前已冻结；已保存的协议记录禁止根据 2025 结果反馈调参。"
    budget_note = "、".join(f"{seed} 为{validation['w0_short_epoch_budgets'][str(seed)]}短轮" for seed in SEEDS)
    lines = [f"# {heading}", "", *main_findings(validation, aggregate, paired), "## 结果口径", "",
             f"本报告包含 **{count} 个独立训练的完整 R 模型**，每个 seed 在同一 checkpoint 上评估完整条件及六项推理干预。"
             f"共 {validation['replay_rows']:,} 个决策窗口；每条件覆盖 {validation['days_per_condition']} 天、每日六个窗口。", "",
             "比较设计为 same_checkpoint_inference_intervention，separately_retrained=False。"
             "结果说明固定已训练模型对这些机制的依赖性，不能证明删除模块并重新训练后仍无法替代。"
             "不会把七个推理条件写成七个独立训练模型。", "",
             ("本次只有一个训练 seed；跨 seed 标准差不可用。日期配对区间不代表训练随机性。" if count == 1 else
              "表中均值±样本标准差以三个独立训练的完整模型为单位，不将 1,086 个模型—日期观测视为独立训练重复。"), "",
             "## 方法与执行边界", "",
             "每个 seed 完成 W0→W1→完整 R 后即冻结权重并评估 2025，再继续下一 seed；"
             "不要求先训练全部 seed。" + configuration_note, "",
             "训练使用 2022–2023，最优节点按 2024 验证分数选择；validation_score_only 不把参考阈值超标当作资格拒绝。"
             "2025 此前已查看，本报告为补充测试，不能称为首次未见测试。", "",
             "W0 预算按当前协议读取：" + budget_note + "。"
             "因此跨 seed 离散程度同时包含随机初始化与已披露的预算差异，不将其全部归因于随机 seed。", "",
             "已取消的独立消融训练保留在历史状态和日志中，不进入本报告效果表。效率表仅统计本方案完整模型的基础训练。"]
    lines += shared.amendment_note(validation["policy_amendments"])
    lines += ["", "## 成本与风险结果", ""]
    columns = ("mean_normalized_operating_cost", "cvar95_normalized_operating_cost", "mean_deadline_expired_work",
               "cvar95_deadline_expired_work", "worst_origin_sla")
    def table(selected):
        result = ["| 推理条件 | " + " | ".join(METRICS[key] for key in selected) + " |", "|---|" + "---:|" * len(selected)]
        for variant, label in VARIANTS.items():
            cells = []
            for key in selected:
                row = look[variant, key]
                cells.append(shared.fmt(row["mean"]) + (f" ± {shared.fmt(row['sample_sd'])}" if count > 1 else ""))
            result.append(f"| {label} | " + " | ".join(cells) + " |")
        return result
    lines += table(columns)
    lines += ["", "## 原单位成本、约束与速度", ""]
    lines += table(("mean_realized_operating_cost", "cvar95_realized_operating_cost", "max_hard_residual", "hard_violation_rate",
                    "mean_unserved_energy", "p95_latency_ms_batch1"))
    lines += ["", "成本和逾期量 CVaR95 分别取各条件最高 ceil(0.05×日期数) 个日指标的均值；362天对应19天。"
              "原单位和归一化成本都是六个分别评估、部分重叠窗口的日均结果，不作全年连续运行成本解释。"
              "逾期工作量不是违反率或离散任务数；复合 SLA 严重度的 0.02 参考值不等于 2% 违反率。", "",
              "SLA、硬约束和延迟超标标记如实保留，均不阻塞本报告。passed 只表示记录覆盖、来源和数值一致，"
              "不表示模型满足全部科学阈值。成本、风险、综合决策分数及全部诊断列见 seed_metrics.csv。", "",
              "## 配对差值", "",
              f"Δ＝推理干预−完整条件；正值表示完整条件更低。每 seed 内采用7日循环日期块、"
              f"{validation['bootstrap']['resamples']:,} 次重采样、固定seed 20240813。"
              "每次双方使用相同日期索引，并分别重算 CVaR 和最差窗口 SLA 后相减。", "",
              "| Seed | 推理干预 | 平均成本 Δ [95% CI] | 成本 CVaR95 Δ [95% CI] | 最差 SLA Δ [95% CI] |",
              "|---:|---|---:|---:|---:|"]
    pairs = {(row["seed"], row["variant"], row["metric"]): row for row in paired}
    for seed in seeds:
        for variant in VARIANTS:
            if variant == "full":
                continue
            cells = []
            for key in ("mean_normalized_operating_cost", "cvar95_normalized_operating_cost", "worst_origin_sla"):
                row = pairs[seed, variant, key]
                cells.append(f"{shared.fmt(row['delta_intervention_minus_full'])} [{shared.fmt(row['ci95_low'])}, {shared.fmt(row['ci95_high'])}]")
            lines.append(f"| {seed} | {VARIANTS[variant]} | " + " | ".join(cells) + " |")
    lines += ["", "区间跨零表示方向证据不足，不证明等效；区间未作多重比较校正。"
              "模型的成本、服务逾期与尾部风险可能存在权衡，不凭单项平均成本宣布模块必要。"
              "延迟缺少保存的计时重复样本，不生成日期 bootstrap 延迟区间；综合分数区间将延迟罚项视为固定值。", "",
              "## 来源与效率记录", "",
              "checkpoint、W1来源及每seed覆盖见 seed_metrics.csv / report_validation.json。"
              "efficiency.csv 从实际 study_status 与 timing/progress 日志取值；主机墙钟不是隔离的GPU kernel耗时，缺项标unavailable。"
              "实际早期验证、保存次数和耗时不从最终配置倒推；取消的独立无协调器训练不会补充成一个推理条件的训练重复。", ""]
    return "\n".join(lines)


def build_report(root=ROOT, *, seed=None, expected_days=362, resamples=shared.RESAMPLES):
    root = Path(root).resolve()
    seeds = tuple(SEEDS) if seed is None else (int(seed),)
    require(all(value in SEEDS for value in seeds), "Unregistered training seed")
    output = root / "inference_report"
    if seed is not None:
        output /= f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        state, panels, metrics, validation = load_study(root, seeds, expected_days=expected_days)
        pairs = paired_metrics(panels, seeds, resamples=resamples)
        aggregate = aggregate_metrics(metrics, seeds)
        efficiency = shared.efficiency_rows(root, {"jobs": validation.pop("_included_jobs")})
        validation["policy_amendments"] = shared.policy_amendments(root)
        validation["bootstrap"] = {"block_days": 7, "resamples": resamples, "seed": shared.BOOTSTRAP_SEED,
                                   "paired_each_side_cvar_recomputed": True}
        validation["source_paths"].extend(row["path"] for row in validation["policy_amendments"]["records"] if row["status"] == "recorded")
        write_csv(output / "seed_metrics.csv", metrics)
        write_csv(output / "aggregate_metrics.csv", aggregate)
        write_csv(output / "paired_metrics.csv", pairs)
        write_csv(output / "efficiency.csv", efficiency)
        (output / "REPORT.md").write_text(render_report(validation, metrics, aggregate, pairs), encoding="utf-8")
        validation.update({"created_at": now(), "status": "passed", "report_artifacts_complete": True,
                           "output_files": [str(output / name) for name in ("REPORT.md", "seed_metrics.csv", "aggregate_metrics.csv",
                                                                          "paired_metrics.csv", "efficiency.csv", "report_validation.json")]})
        write_json(output / "report_validation.json", validation)
        return validation
    except (shared.ReportEvidenceError, KeyError, TypeError, ValueError) as exc:
        write_json(output / "report_validation.json", {"status": "failed", "error": str(exc), "created_at": now(),
                   "comparison_design": DESIGN, "separately_retrained": False, "final_effects_reported": False})
        (output / "REPORT.md").write_text(f"# 固定模型推理消融报告\n\n记录检查未完成，不输出效果结论。\n\n{exc}\n", encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    result = build_report(args.root, seed=args.seed)
    print(json.dumps({"status": result["status"], "seeds": result["seeds"], "replay_rows": result["replay_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
