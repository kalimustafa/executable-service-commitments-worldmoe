"""Report only the approved study's saved, paired 2025 evaluation evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

from multiseed_common import ROOT, SEEDS, VARIANTS, now, write_csv, write_json


WINDOWS = ((0, 0, 96), (1, 0, 96), (1, 4, 92), (1, 16, 80), (1, 48, 48), (1, 95, 1))
BOOTSTRAP_SEED, BLOCK_DAYS, RESAMPLES = 20240813, 7, 2000
HARD_TOLERANCE = 1e-4
SLA_REFERENCE = 0.02
LATENCY_REFERENCE_MS = 1900.0
SELECTION_POLICY = "validation_score_only"
AMENDMENT_FILES = (
    "selection_policy_amendment.json",
    "early_stop_policy_v2.json",
    "checkpoint_policy_amendment.json",
    "w0_budget_amendment.json",
)
UNAVAILABLE = "unavailable"
METRICS = {
    "mean_normalized_operating_cost": "平均归一化成本",
    "cvar95_normalized_operating_cost": "成本 CVaR95",
    "mean_deadline_expired_work": "平均逾期工作量",
    "cvar95_deadline_expired_work": "逾期工作量 CVaR95",
    "worst_origin_sla": "最差窗口复合 SLA 严重度",
    "mean_sla_deadline_severity": "平均复合 SLA 严重度",
    "mean_constraint_severity": "平均约束严重度",
    "max_hard_residual": "最大硬约束残差",
    "hard_violation_rate": "决策窗口硬约束超限率",
    "daily_hard_violation_rate": "日期硬约束超限率",
    "mean_unserved_energy": "平均未供电量",
    "p95_latency_ms_batch1": "batch-1 推理 p95 毫秒",
    "decision_value": "冻结定义综合决策分数",
}
DAY_FIELDS = ("normalized_operating_cost", "deadline_expired_work", "unserved_energy",
              "sla_deadline_severity", "constraint_severity")


class ReportEvidenceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ReportEvidenceError(message)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportEvidenceError(f"Cannot read JSON evidence: {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON evidence must be an object: {path}")
    return value


def read_csv(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))
    except OSError as exc:
        raise ReportEvidenceError(f"Cannot read CSV evidence: {path}: {exc}") from exc


def finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportEvidenceError(f"Missing/non-numeric {label}: {value!r}") from exc
    require(math.isfinite(result), f"Non-finite {label}")
    return result


def flag(value, label):
    if str(value).lower() in {"true", "1", "1.0"}:
        return True
    if str(value).lower() in {"false", "0", "0.0"}:
        return False
    raise ReportEvidenceError(f"Invalid Boolean {label}: {value!r}")


def within(path, root):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), f"Evidence points outside this study: {path}")
    return path


def cvar95(values, axis=-1):
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[axis]
    tail = max(1, math.ceil(0.05 * count))
    return np.take(np.partition(values, count - tail, axis=axis),
                   range(count - tail, count), axis=axis).mean(axis=axis)


def circular_blocks(days, *, resamples=RESAMPLES, block_days=BLOCK_DAYS, seed=BOOTSTRAP_SEED):
    require(days > 0 and resamples > 0 and block_days > 0, "Invalid bootstrap dimensions")
    generator = random.Random(seed)
    rows = []
    for _ in range(resamples):
        indices = []
        while len(indices) < days:
            start = generator.randrange(days)
            indices.extend((start + offset) % days for offset in range(block_days))
        rows.append(indices[:days])
    return np.asarray(rows, dtype=np.int64)


def panel_statistics(panel, indices=None):
    """Recompute metrics on whole paired dates, including each CVaR tail."""
    matrices = {name: values[None] if indices is None else values[indices]
                for name, values in panel["matrices"].items()}
    day = {name: values.mean(axis=2) for name, values in matrices.items()}
    hard = matrices["hard_residual_max"]
    cost, deadline = day["normalized_operating_cost"], day["deadline_expired_work"]
    result = {
        "mean_normalized_operating_cost": cost.mean(axis=1),
        "cvar95_normalized_operating_cost": cvar95(cost, axis=1),
        "mean_deadline_expired_work": deadline.mean(axis=1),
        "cvar95_deadline_expired_work": cvar95(deadline, axis=1),
        "worst_origin_sla": matrices["sla_deadline_severity"].mean(axis=1).max(axis=1),
        "mean_sla_deadline_severity": day["sla_deadline_severity"].mean(axis=1),
        "mean_constraint_severity": day["constraint_severity"].mean(axis=1),
        "max_hard_residual": hard.max(axis=(1, 2)),
        "hard_violation_rate": (hard > HARD_TOLERANCE).mean(axis=(1, 2)),
        "daily_hard_violation_rate": (hard.max(axis=2) > HARD_TOLERANCE).mean(axis=1),
        "mean_unserved_energy": day["unserved_energy"].mean(axis=1),
    }
    latency = panel["latency"]
    result["decision_value"] = (
        result["mean_normalized_operating_cost"] + 0.25 * result["cvar95_normalized_operating_cost"]
        + result["worst_origin_sla"] + result["mean_constraint_severity"]
        + max(0.0, latency / LATENCY_REFERENCE_MS - 1.0)
    )
    if indices is None:
        return {**{name: float(values[0]) for name, values in result.items()},
                "p95_latency_ms_batch1": latency}
    return result


def _same_number(saved, calculated, label):
    saved = finite(saved, label)
    require(math.isclose(saved, calculated, rel_tol=1e-6, abs_tol=1e-8),
            f"Saved statistic disagrees with its underlying rows: {label}: {saved} vs {calculated}")


def load_panels(root, *, expected_days=362):
    root = Path(root).resolve()
    state = read_json(root / "study_status.json")
    jobs = state.get("jobs", [])
    expected_jobs = {(seed, phase, variant) for seed in SEEDS
                     for phase, variant in [("w0", None), ("w1", None)] + [("r", v) for v in VARIANTS]}
    observed_jobs = [(int(job["seed"]), job["phase"], job.get("variant")) for job in jobs]
    require(len(jobs) == 27 and set(observed_jobs) == expected_jobs,
            "The study must contain exactly the registered 27 training stages")
    require(all(job.get("status") == "completed" for job in jobs),
            "All 27 training stages must be completed before reporting final effects")
    for job in jobs:
        within(job["run_root"], root)
    require(state.get("models_frozen") is True, "The study has not recorded the model freeze")
    require(state.get("evaluation_completed") is True, "The study has not completed 2025 evaluation")
    freeze = read_json(root / "freeze_manifest.json")
    frozen = {(int(row["seed"]), row["variant"]): row for row in freeze.get("models", [])}
    keys = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    require(len(freeze.get("models", [])) == 21 and set(frozen) == keys,
            "The freeze manifest must contain exactly 21 independently trained R models")
    require(freeze.get("all_27_training_stages_completed") is True
            and freeze.get("test_selection_feedback_allowed") is False, "Invalid model freeze contract")
    checkpoints = [within(row["checkpoint"], root) for row in frozen.values()]
    require(len(set(checkpoints)) == 21, "R variants/seeds must not reuse a trained checkpoint")
    worlds = []
    for seed in SEEDS:
        paths = {within(frozen[seed, variant]["w1_checkpoint"], root) for variant in VARIANTS}
        require(len(paths) == 1, f"R variants must share this seed's W1: {seed}")
        worlds.append(paths.pop())
    require(len(set(worlds)) == 3, "Independent seeds must not reuse one W1 checkpoint")

    panels, seed_metrics, source_paths = {}, [], []
    common_dates = None
    total_decisions = total_days = 0
    for seed in SEEDS:
        folder = root / "evaluation_2025" / f"seed_{seed}"
        metadata = read_json(folder / "metadata.json")
        require(str(metadata.get("status", "")).lower() == "completed", f"Incomplete evaluation: {seed}")
        require(int(metadata.get("year", -1)) == 2025 and metadata.get("split") == "locked_test",
                f"Evaluation must use the true 2025 locked_test split: {seed}")
        for key in ("training_performed", "checkpoint_selection_performed"):
            require(metadata.get(key) is False, f"Evaluation must explicitly forbid {key}: {seed}")
        decisions = read_csv(folder / "per_decision_models.csv")
        days = read_csv(folder / "per_day_models.csv")
        summaries = read_csv(folder / "model_summary.csv")
        source_paths.extend(str(folder / name) for name in
                            ("metadata.json", "per_decision_models.csv", "per_day_models.csv", "model_summary.csv"))
        require(len(decisions) == len(VARIANTS) * expected_days * len(WINDOWS), f"Wrong replay row count: {seed}")
        require(len(days) == len(VARIANTS) * expected_days, f"Wrong daily row count: {seed}")
        require(len(summaries) == len(VARIANTS), f"Wrong model summary count: {seed}")
        for rows in (decisions, days, summaries):
            require(all(int(row["seed"]) == seed and row["variant"] in VARIANTS for row in rows),
                    f"Rows include a foreign seed or mechanism: {seed}")
        summary_by_variant = {row["variant"]: row for row in summaries}
        require(set(summary_by_variant) == set(VARIANTS), f"Repeated/missing summary model: {seed}")
        day_by_variant = {variant: {} for variant in VARIANTS}
        for row in days:
            variant, operating = row["variant"], row["operating_date"]
            require(operating not in day_by_variant[variant], f"Duplicate daily key: {seed}/{variant}/{operating}")
            require(date.fromisoformat(operating).year == 2025, f"Non-2025 date: {operating}")
            day_by_variant[variant][operating] = row
        dates = sorted(day_by_variant["full"])
        require(len(dates) == expected_days, f"Incomplete unique date set: {seed}")
        require(all(set(rows) == set(dates) for rows in day_by_variant.values()), f"Unpaired mechanism dates: {seed}")
        if common_dates is None:
            common_dates = dates
        require(dates == common_dates, "Training seeds do not share the same test dates")
        declared_dates = metadata.get("dataset_dates", metadata.get("operating_dates"))
        require(isinstance(declared_dates, list) and sorted(declared_dates) == dates,
                f"Metadata does not identify the same complete test dates: {seed}")
        by_variant = {variant: {} for variant in VARIANTS}
        for row in decisions:
            variant, operating = row["variant"], row["operating_date"]
            window = (int(row["stage_id"]), int(row["decision_origin"]), int(row["horizon"]))
            key = (operating, window)
            require(operating in day_by_variant[variant] and window in WINDOWS, f"Unknown decision key: {seed}/{variant}/{key}")
            require(key not in by_variant[variant], f"Duplicate decision key: {seed}/{variant}/{key}")
            by_variant[variant][key] = row
        for variant in VARIANTS:
            summary = summary_by_variant[variant]
            record = frozen[seed, variant]
            require(within(summary["checkpoint"], root) == within(record["checkpoint"], root)
                    and within(summary["w1_checkpoint"], root) == within(record["w1_checkpoint"], root),
                    f"Summary/freeze checkpoint lineage mismatch: {seed}/{variant}")
            require(int(summary["checkpoint_epoch"]) > 0, f"Missing trained checkpoint epoch: {seed}/{variant}")
            require(int(summary["days"]) == expected_days and int(summary["replay_rows"]) == expected_days * 6,
                    f"Summary coverage mismatch: {seed}/{variant}")
            lookup = by_variant[variant]
            require(len(lookup) == expected_days * len(WINDOWS), f"Incomplete six-window panel: {seed}/{variant}")
            matrices = {name: np.asarray([[finite(lookup[day, window][name], name) for window in WINDOWS]
                                         for day in dates], dtype=np.float64)
                        for name in (*DAY_FIELDS, "hard_residual_max")}
            for index, day in enumerate(dates):
                row = day_by_variant[variant][day]
                for name in DAY_FIELDS:
                    _same_number(row[name], float(matrices[name][index].mean()), f"{seed}/{variant}/{day}/{name}")
                maximum = float(matrices["hard_residual_max"][index].max())
                _same_number(row["hard_residual_max"], maximum, f"{seed}/{variant}/{day}/hard residual")
                require(flag(row["hard_violation"], "daily hard_violation") == (maximum > HARD_TOLERANCE),
                        f"Daily hard-violation flag mismatch: {seed}/{variant}/{day}")
                for window in WINDOWS:
                    decision = lookup[day, window]
                    require(flag(decision["hard_violation"], "hard_violation")
                            == (finite(decision["hard_residual_max"], "hard residual") > HARD_TOLERANCE),
                            f"Decision hard-violation flag mismatch: {seed}/{variant}/{day}/{window}")
            latency = finite(summary["p95_latency_ms_batch1"], "batch-1 p95 latency")
            require(latency >= 0, "Inference latency cannot be negative")
            panel = {"dates": dates, "matrices": matrices, "latency": latency}
            measured = panel_statistics(panel)
            for name, calculated in measured.items():
                _same_number(summary.get(name), calculated, f"{seed}/{variant}/{name}")
            panels[seed, variant] = panel
            seed_metrics.append({"seed": seed, "variant": variant, "label": VARIANTS[variant],
                                 "days": expected_days, "replay_rows": expected_days * len(WINDOWS),
                                 "checkpoint_epoch": int(summary["checkpoint_epoch"]),
                                 "checkpoint": summary["checkpoint"], "w1_checkpoint": summary["w1_checkpoint"],
                                 **measured, "selection_policy": SELECTION_POLICY,
                                 "sla_reference_exceeded": measured["worst_origin_sla"] > SLA_REFERENCE,
                                 "hard_reference_exceeded": measured["max_hard_residual"] > HARD_TOLERANCE,
                                 "latency_reference_exceeded": measured["p95_latency_ms_batch1"] > LATENCY_REFERENCE_MS})
        total_decisions += len(decisions)
        total_days += len(days)
    require(total_decisions == len(SEEDS) * len(VARIANTS) * expected_days * len(WINDOWS), "Total replay coverage mismatch")
    validation = {
        "status": "passed", "independent_training_seeds": len(SEEDS), "seeds": list(SEEDS),
        "validation_scope": "coverage_lineage_and_numeric_consistency_only",
        "selection_policy": SELECTION_POLICY, "scientific_qualification_required": False,
        "threshold_exceedances_block_selection": False, "threshold_exceedances_block_report": False,
        "threshold_diagnostics": [
            {name: row[name] for name in ("seed", "variant", "sla_reference_exceeded",
                                         "hard_reference_exceeded", "latency_reference_exceeded")}
            for row in seed_metrics
        ],
        "models_per_seed": len(VARIANTS), "training_stages": len(jobs),
        "days_per_model": expected_days, "windows_per_day": len(WINDOWS),
        "replay_rows": total_decisions, "daily_rows": total_days, "dataset_dates": common_dates,
        "daily_and_summary_statistics_recomputed": True,
        "same_dates_all_models_and_seeds": True, "same_w1_within_seed": True,
        "distinct_w1_across_seeds": True, "distinct_r_checkpoints": True,
        "source_paths": [str(root / "study_status.json"), str(root / "freeze_manifest.json"), *source_paths],
    }
    return state, panels, seed_metrics, validation


def paired_metrics(panels, *, resamples=RESAMPLES):
    rows = []
    indices = circular_blocks(len(panels[SEEDS[0], "full"]["dates"]), resamples=resamples)
    for seed in SEEDS:
        full = panel_statistics(panels[seed, "full"])
        full_boot = panel_statistics(panels[seed, "full"], indices)
        for variant in VARIANTS:
            if variant == "full":
                continue
            other = panel_statistics(panels[seed, variant])
            other_boot = panel_statistics(panels[seed, variant], indices)
            for metric in METRICS:
                if metric in full_boot:
                    delta = other_boot[metric] - full_boot[metric]
                    low, high = (float(value) for value in np.quantile(delta, [0.025, 0.975]))
                    interpretation = ("full_lower" if low > 0 else "ablation_lower" if high < 0 else "interval_includes_zero")
                    method = "paired_7_day_circular_blocks_recompute_each_side"
                else:
                    low = high = UNAVAILABLE
                    interpretation = "no_saved_timing_replicates"
                    method = "no_date_bootstrap_for_batch1_latency"
                rows.append({"seed": seed, "variant": variant, "metric": metric,
                             "full_value": full[metric], "ablation_value": other[metric],
                             "delta_ablation_minus_full": other[metric] - full[metric],
                             "ci95_low": low, "ci95_high": high, "interpretation": interpretation,
                             "independent_training_replicates_in_row": 1,
                             "paired_dates": len(panels[seed, variant]["dates"]),
                             "block_days": BLOCK_DAYS, "resamples": resamples if metric in full_boot else UNAVAILABLE,
                             "bootstrap_seed": BOOTSTRAP_SEED, "method": method})
    return rows


def aggregate_metrics(seed_metrics):
    by_key = {(row["seed"], row["variant"]): row for row in seed_metrics}
    rows = []
    for variant in VARIANTS:
        for metric in METRICS:
            values = [by_key[seed, variant][metric] for seed in SEEDS]
            deltas = [by_key[seed, variant][metric] - by_key[seed, "full"][metric] for seed in SEEDS]
            rows.append({"variant": variant, "label": VARIANTS[variant], "metric": metric,
                         "independent_training_seeds": len(SEEDS), "mean": statistics.mean(values),
                         "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values),
                         "paired_delta_mean": statistics.mean(deltas),
                         "paired_delta_sample_sd": statistics.stdev(deltas),
                         "full_lower_point_estimate_seeds": sum(value > 0 for value in deltas),
                         "ablation_lower_point_estimate_seeds": sum(value < 0 for value in deltas)})
    return rows


def _jsonl(path):
    if not path.exists():
        return None
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return None


def efficiency_rows(root, state):
    """Use observed counters only; files/hardlinks are not save-time evidence."""
    rows = []
    for job in state.get("jobs", []):
        folder = within(job["run_root"], root)
        elapsed = job.get("elapsed_seconds", UNAVAILABLE)
        if elapsed != UNAVAILABLE:
            elapsed = finite(elapsed, "stage elapsed_seconds")
        epoch_seconds = validation_records = full_records = UNAVAILABLE
        source, count_basis = str(root / "study_status.json"), UNAVAILABLE
        if job["phase"] in {"w0", "w1"}:
            logged = _jsonl(folder / "train.jsonl")
            if logged is not None:
                source += ";" + str(folder / "train.jsonl")
                numeric = [row.get("seconds") for row in logged]
                if numeric and all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
                    epoch_seconds = sum(numeric)
                full_records = sum(isinstance(row.get("formal_selection"), dict) for row in logged)
                count_basis = "train.jsonl formal_selection non-null records; not inferred save events"
        else:
            path = folder / "validation.csv"
            if path.exists():
                logged = read_csv(path)
                source += ";" + str(path)
                validation_records = len(logged)
                full_records = sum(row.get("validation_scope") == "full" for row in logged)
                count_basis = "saved validation.csv rows"
        timing_candidates = []
        for name in ("timing.json", "progress.json"):
            path = folder / name
            if path.exists():
                payload = read_json(path)
                if isinstance(payload.get("timing"), dict):
                    timing_candidates.append((path.stat().st_mtime_ns, path, payload["timing"]))
        timing = {}
        if timing_candidates:
            _, timing_path, timing = max(timing_candidates, key=lambda item: item[0])
            source += ";" + str(timing_path)
        measured_timing = {}
        for name in ("train_step_seconds", "train_step_count", "validation_seconds", "validation_count",
                     "checkpoint_seconds", "checkpoint_write_count", "checkpoint_alias_seconds", "checkpoint_alias_count"):
            measured_timing[name] = finite(timing[name], name) if name in timing else UNAVAILABLE
            if measured_timing[name] != UNAVAILABLE:
                require(measured_timing[name] >= 0, f"Negative observed timing/counter: {folder}/{name}")
        rows.append({"seed": int(job["seed"]), "phase": job["phase"], "variant": job.get("variant") or "",
                     "status": job.get("status", UNAVAILABLE), "stage_wall_seconds": elapsed,
                     "logged_epoch_wall_seconds_includes_validation_and_save": epoch_seconds,
                     "validation_record_count": validation_records, "full_validation_record_count": full_records,
                     **measured_timing, "source": source, "count_basis": count_basis,
                     "timing_basis": "host wall-clock observation; no additional CUDA synchronization; not isolated kernel timing"})
    return rows


def policy_amendments(root):
    """Retain runtime amendments without retroactively rebuilding history."""
    records = []
    for name in AMENDMENT_FILES:
        path = Path(root) / name
        if not path.exists():
            records.append({"file": name, "path": str(path), "status": UNAVAILABLE})
            continue
        payload = read_json(path)
        # The original file retains every before/after hash. The report only
        # duplicates policy fields and actual pause/resume snapshots.
        compact = {key: value for key, value in payload.items() if key not in
                   {"configuration_changes", "source_sha256_after", "focused_checks"}}
        records.append({"file": name, "path": str(path), "status": "recorded", "record": compact})
    return {
        "records": records,
        "runtime_user_amendments_present": any(row["status"] == "recorded" for row in records),
        "historical_counts_reconstructed_from_current_config": False,
        "completed_w0_history_rewritten": False,
        "actual_epochs_counts_and_seconds_basis": "saved_stage_logs_and_timing_counters",
        "first_seed_did_not_follow_final_schedule_from_initialization": any(
            row["file"] == "early_stop_policy_v2.json" and row["status"] == "recorded"
            and isinstance(row["record"].get("current_w1_pause"), dict) for row in records
        ),
    }


def amendment_note(amendments):
    observed = {row["file"]: row for row in amendments["records"] if row["status"] == "recorded"}
    if not observed:
        return []
    sources = "、".join(f"[{name}]({Path(row['path']).as_posix()})" for name, row in observed.items())
    pieces = [f"运行中用户修订记录：{sources}。"]
    early = observed.get("early_stop_policy_v2.json", {}).get("record", {})
    schedules = []
    for phase in ("w1", "r"):
        entry = early.get(phase, {})
        if all(key in entry for key in ("first_formal_epoch", "interval", "patience")):
            schedules.append(f"{phase.upper()} 第{entry['first_formal_epoch']}轮起每{entry['interval']}轮验证，"
                             f"首次 best 建立后连续{entry['patience']}次未严格刷新即停")
    if schedules:
        pieces.append("后续日程为：" + "；".join(schedules) + "。取消额外早停轮次下限与最小提升比例。")
    if "checkpoint_policy_amendment.json" in observed:
        pieces.append("保存调整为 best_only：首次验证前无自动保存，只在严格刷新 best 时写一次；PAUSE/异常备份保留。")
    budget = observed.get("w0_budget_amendment.json", {}).get("record", {})
    if budget:
        pieces.append(f"用户在该阶段开始前将种子{budget.get('seed')}的W0预算改为{budget.get('max_epochs')}短轮；"
                      "因此跨种子差异同时包含该训练预算差异，不能解释为纯随机种子效应。")
    snapshot = early.get("current_w1_pause", {})
    if snapshot:
        pieces.append(f"首 seed 的 W1 修订时已暂停于第{snapshot.get('epoch', UNAVAILABLE)}轮、"
                      f"update {snapshot.get('update', UNAVAILABLE)}，已执行的早期验证与保存仍属于历史记录。")
    pieces.append("已完成的首 seed W0 历史不改；实际轮数、验证/保存次数及耗时以保存记录为准，不能按当前配置倒推。")
    return ["", "".join(pieces), ""]


def fmt(value):
    return f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def markdown_report(root, validation, aggregate, paired, efficiency):
    by_key = {(row["variant"], row["metric"]): row for row in aggregate}

    def table(metrics):
        output = ["| 配置 | " + " | ".join(METRICS[name] for name in metrics) + " |",
                  "|---|" + "---:|" * len(metrics)]
        for variant, label in VARIANTS.items():
            values = [f"{fmt(by_key[variant, name]['mean'])} ± {fmt(by_key[variant, name]['sample_sd'])}" for name in metrics]
            output.append(f"| {label} | " + " | ".join(values) + " |")
        return output

    lines = ["# 三种子独立重训练消融报告", "", "## 完成状态与结果口径", "",
             f"27 个训练阶段及 21 组模型评估已完成；记录覆盖 {validation['replay_rows']:,} 个决策窗口、"
             f"{validation['days_per_model']} 个共同测试日期。逐窗口、逐日和汇总指标一致性检查通过。", "",
             "本轮采用 **validation_score_only**：训练按预算或既定早停结束，保留最优验证分数节点，"
             "不再以科学资格阈值限制候选模型。报告的 passed 只表示证据覆盖与数值一致性检查通过，"
             "不表示所有科学或物理参考阈值均满足。", "",
             "以下均值 ± 样本标准差以 **3 个独立训练 seed** 为重复单位。成本、尾部损失、服务严重度与硬约束共同呈现，"
             "不依据单一成本指标删去负结果或判定机制必要性。", "",
             "## 实验设计与证据来源", "",
             "- Seed：20240813、20240829、20240917；每个 seed 新训练 W0→W1，并将该 seed 的冻结 W1 共享给七个独立训练的 R。",
             "- 七个 R 配置全部维持 Top-2。机制在训练、2024 验证选模与 2025 推理中持续施加。",
             "- 训练年为 2022–2023，选模年为 2024，测试年为 2025。只读取本轮 freeze_manifest 与保存的结果。",
             "- 2025 测试集此前已被查看；本轮属于补充测试，不能称为首次未见的最终外部验证，也不据此重新选参数或 seed。",
             "- 每日结果平均六个分别评估、部分重叠的窗口；不是连续闭环的全年调度，不将窗口成本年化。"]
    lines += amendment_note(validation["policy_amendments"])
    lines += ["", "## 成本、尾部与服务风险", ""]
    lines += table(["mean_normalized_operating_cost", "cvar95_normalized_operating_cost", "mean_deadline_expired_work",
                    "cvar95_deadline_expired_work", "worst_origin_sla"])
    lines += ["", "成本与逾期量 CVaR95 均取各模型最高 ceil(0.05×日期数) 个日指标的均值；"
              "362 天对应 19 天。逾期工作量是 FIFO 新增逾期量，不能解释为违反率或离散任务数。", "",
              "复合 SLA 严重度结合逾期工作量占比与归一化未供电强度；最差窗口指标先求各窗口的日期均值、再取最大。"
              "0.02 是保留用于描述结果的严重度参考阈值，不等于 2% 违反率，也不构成本轮选模资格条件。"
              "这些指标来自决策后的真实历史结果回放，不是风险预测头的输出。", "",
              "## 物理约束、速度与综合分数", ""]
    lines += table(["max_hard_residual", "hard_violation_rate", "mean_unserved_energy", "p95_latency_ms_batch1", "decision_value"])
    violations = [row for row in aggregate if row["metric"] == "hard_violation_rate" and row["max"] > 0]
    lines += ["", ("发现决策窗口硬约束超限：" + "、".join(row["label"] for row in violations) + "。超限结果如实保留，不阻塞选模或报告，也不因平均成本较低而隐去。"
                    if violations else "本轮保存的全部决策窗口均未超过 1e-4 硬约束残差阈值。"), "",
              "参考阈值超标的模型—seed 组合数："
              f"SLA {sum(row['sla_reference_exceeded'] for row in validation['threshold_diagnostics'])}/21；"
              f"硬约束 {sum(row['hard_reference_exceeded'] for row in validation['threshold_diagnostics'])}/21；"
              f"延迟 {sum(row['latency_reference_exceeded'] for row in validation['threshold_diagnostics'])}/21。"
              "逐 seed 标记见 seed_metrics.csv 和 report_validation.json；全部标记只作诊断。", "",
              "综合分数沿用冻结定义：平均归一化成本＋0.25×成本 CVaR95＋最差窗口 SLA＋平均约束严重度＋p95 超过 1900 ms 的相对超时罚项。"
              "测试分数仅用于报告，不重新选择 checkpoint。", "",
              "## 逐种子配对差值与不确定性", "",
              "Δ＝消融模型−完整模型；正值代表完整模型更低。下表每格是逐 seed 的点估计与 95% 区间。"
              f"每个 seed 内使用相同日期的 7 日循环移动块、{validation['bootstrap']['resamples']:,} 次重采样、seed 20240813。"
              "每次重采样分别重算双方 CVaR 及最差窗口 SLA，再相减；不是对差值直接计算 CVaR。", "",
              "| Seed | 消融 | 平均成本 Δ [95% CI] | 成本 CVaR95 Δ [95% CI] | 最差 SLA Δ [95% CI] |",
              "|---:|---|---:|---:|---:|"]
    by_pair = {(row["seed"], row["variant"], row["metric"]): row for row in paired}
    for seed in SEEDS:
        for variant, label in VARIANTS.items():
            if variant == "full":
                continue
            cells = []
            for metric in ("mean_normalized_operating_cost", "cvar95_normalized_operating_cost", "worst_origin_sla"):
                row = by_pair[seed, variant, metric]
                cells.append(f"{fmt(row['delta_ablation_minus_full'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]")
            lines.append(f"| {seed} | {label} | " + " | ".join(cells) + " |")
    lines += ["", "区间包含零表示方向证据不足，不证明两模型等效。日期块区间描述固定训练模型下的日期不确定性；"
              "训练不确定性由三 seed 的样本标准差单独呈现。区间未做多重比较校正。"
              "不能把 3×362＝1,086 当成独立训练重复。完整逐指标区间和跨 seed 配对均值±SD 分别见 paired_metrics.csv 与 aggregate_metrics.csv。", "",
              "batch-1 p95 延迟只有保存的汇总测量，没有计时重复样本，因此不伪造日期 bootstrap 延迟区间。"
              "综合分数区间将已测延迟罚项视为固定值，不包含计时测量误差。", "",
              "## 训练执行效率", "",
              "| 阶段 | 记录数 | 阶段墙钟秒 | 训练步骤秒 | 验证秒/调用数 | checkpoint 序列化秒/次数 | 别名处理秒/次数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for phase in ("w0", "w1", "r"):
        values = [row for row in efficiency if row["phase"] == phase]
        def total(name):
            actual = [row[name] for row in values]
            return fmt(sum(actual)) if actual and all(isinstance(value, (int, float)) for value in actual) else UNAVAILABLE
        lines.append(f"| {phase.upper()} | {len(values)} | {total('stage_wall_seconds')} | {total('train_step_seconds')} | "
                     f"{total('validation_seconds')} / {total('validation_count')} | "
                     f"{total('checkpoint_seconds')} / {total('checkpoint_write_count')} | "
                     f"{total('checkpoint_alias_seconds')} / {total('checkpoint_alias_count')} |")
    lines += ["", "阶段墙钟来自 study_status 的实际 elapsed_seconds；W0/W1 train.jsonl 的 seconds 是整轮耗时，"
              "包含验证与存盘，不能视为纯训练。分项来自最新 timing.json/progress.json 的实际累计计数；"
              "训练步骤可包含 batch 准备，验证计数包含实际调用的初态 reference 或失败尝试。"
              "计时使用主机墙钟且不额外同步 CUDA，不能称为独立 GPU kernel 耗时，分项间差额包括未分类开销。"
              "未持久记录的分项耗时和写盘次数统一标记 unavailable。"
              "文件数、hardlink 别名数或约 600 MiB/8 MiB 的文件体积不是耗时证据，也不据此编造加速比。", "",
              "## 适用边界与文件清单", "",
              "本报告不将同 seed 共享 W1 的七个 R 当作七次独立世界模型训练。移除路由先验仅移除 R logits 的先验项，仍保留路由连续性 KL；"
              "无决策尾部与尾部始终开启只改变 R 适配器，W1 尾专家保持原定义。", "",
              "输出：seed_metrics.csv（逐 seed 指标与 checkpoint 来源）、aggregate_metrics.csv（3 seed 均值±样本 SD）、"
              "paired_metrics.csv（逐 seed 配对差值及日期块区间）、efficiency.csv（实际日志效率字段）、report_validation.json（覆盖、一致性与来源）。", "",
              f"本轮冻结记录：{root / 'freeze_manifest.json'}", ""]
    return "\n".join(lines)


def build_report(root=ROOT, *, expected_days=362, resamples=RESAMPLES):
    root = Path(root).resolve()
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    try:
        state, panels, seeds, validation = load_panels(root, expected_days=expected_days)
        paired = paired_metrics(panels, resamples=resamples)
        aggregate = aggregate_metrics(seeds)
        efficiency = efficiency_rows(root, state)
        amendments = policy_amendments(root)
        validation["policy_amendments"] = amendments
        validation["source_paths"].extend(row["path"] for row in amendments["records"] if row["status"] == "recorded")
        write_csv(output / "seed_metrics.csv", seeds)
        write_csv(output / "aggregate_metrics.csv", aggregate)
        write_csv(output / "paired_metrics.csv", paired)
        write_csv(output / "efficiency.csv", efficiency)
        validation.update({"created_at": now(), "bootstrap": {"block_days": BLOCK_DAYS, "resamples": resamples,
                           "seed": BOOTSTRAP_SEED, "paired_each_side_cvar_recomputed": True},
                           "sd_ddof": 1, "training_repeat_unit": "seed", "test_feedback_used_for_selection": False,
                           "missing_efficiency_fields": sorted({name for row in efficiency for name in
                               ("train_step_seconds", "train_step_count", "validation_seconds", "validation_count",
                                "checkpoint_seconds", "checkpoint_write_count", "checkpoint_alias_seconds", "checkpoint_alias_count")
                               if row[name] == UNAVAILABLE})})
        (output / "REPORT.md").write_text(markdown_report(root, validation, aggregate, paired, efficiency), encoding="utf-8")
        validation["output_files"] = [str(output / name) for name in (
            "REPORT.md", "seed_metrics.csv", "aggregate_metrics.csv", "paired_metrics.csv",
            "efficiency.csv", "report_validation.json",
        )]
        validation["report_artifacts_complete"] = True
        # The completion record is written last, after all result tables and
        # the Markdown report were successfully written.
        write_json(output / "report_validation.json", validation)
        return validation
    except (ReportEvidenceError, KeyError, TypeError, ValueError) as exc:
        write_json(output / "report_validation.json", {"status": "failed", "created_at": now(), "error": str(exc),
                   "final_effects_reported": False, "study_root": str(root)})
        (output / "REPORT.md").write_text("# 三种子独立重训练消融报告\n\n证据检查未通过，未生成最终效应结论。\n\n"
                                           + str(exc) + "\n\n以 report_validation.json 的 failed 状态为准；先前 CSV 不代表本次验证通过。\n", encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = build_report(args.root)
    print(json.dumps({"status": result["status"], "replay_rows": result["replay_rows"],
                      "report": str(args.root.resolve() / "report/REPORT.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
