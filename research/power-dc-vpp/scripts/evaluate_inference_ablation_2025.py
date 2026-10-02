"""Evaluate seven inference interventions on one completed seed's frozen full R.

This supplemental study has previously seen 2025. Each seed is evaluated only
after its W0/W1/full-R training ends, under the fixed inference protocol; future
seeds are not required to finish first. No ablation condition is retrained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_multiseed_ablation_2025 as base
from multiseed_common import REPO, ROOT, VARIANTS, config_path, now, read_json, write_json

torch = base.torch
evaluation = base.evaluation
EXPECTED_DAYS = 362
DECISIONS = evaluation.EVALUATION_DECISIONS
DESIGN = "same_checkpoint_inference_intervention"
ROLE = "supplemental_multiseed_inference_ablation"
BOOTSTRAP_SEED = 20240813
BOOTSTRAP_BLOCK_DAYS = 7
NATIVE_NAMES = {name: "top2" if name == "full" else name for name in VARIANTS}


def registered_interventions() -> dict[str, Any]:
    registry = {variant.name: variant for variant in evaluation.REGISTERED_VARIANTS}
    result = {name: registry[native] for name, native in NATIVE_NAMES.items()}
    if any(variant.active_top_k != 2 or variant.safe_tensor for variant in result.values()):
        raise ValueError("Inference study permits only the seven registered Top-2 conditions")
    return result


def _base_record(root: Path, seed: int, protocol: Mapping[str, Any]) -> dict[str, Any]:
    supplied = protocol.get("full_model_records", protocol.get("models", []))
    if isinstance(supplied, Mapping):
        supplied = list(supplied.values())
    matches = [row for row in supplied if int(row.get("seed", -1)) == seed and row.get("variant", "full") == "full"]
    if len(matches) > 1:
        raise ValueError("Protocol has duplicate full-model records for this seed")
    if matches:
        record = dict(matches[0])
    else:
        config = config_path(seed, "r", "full").resolve()
        resolved = base._load_extended_yaml(config)
        record = {"seed": seed, "variant": "full", "config": str(config),
                  "checkpoint": str(root / "r" / f"multiseed-20260910-seed{seed}-full" / "checkpoints/best_decision_value.pt"),
                  "w1_checkpoint": resolved["source"]["w1_checkpoint"],
                  "world_config": resolved["source"]["world_config"]}
    record.update(seed=seed, variant="full")
    for key in ("config", "checkpoint", "w1_checkpoint", "world_config"):
        record[key] = str(base._path(record[key]))
    return record


def _completed_base_evidence(root: Path, seed: int, record: Mapping[str, Any]) -> dict[str, str]:
    evidence = {}
    state_path = root / "inference_study_status.json"
    study = read_json(state_path, {})
    if study:
        jobs = [job for job in study.get("jobs", []) if int(job.get("seed", -1)) == seed]
        completed = {("r_full" if job.get("phase") == "r" and job.get("variant") == "full"
                      else job.get("phase")) for job in jobs if job.get("status") == "completed"}
        if not {"w0", "w1", "r_full"}.issubset(completed):
            raise ValueError("The current seed's three base jobs must be completed")
        evidence["study_status"] = str(state_path)
    for phase in ("w0", "w1"):
        folder = root / "world_moe" / f"run=multiseed-20260910-seed{seed}" / f"phase={phase}"
        progress = read_json(folder / "progress.json", {})
        if str(progress.get("status", "")).upper() != "TRAINING_COMPLETE":
            raise ValueError(f"Seed {seed} {phase.upper()} has not completed training")
        state = read_json(folder / "state.json", {})
        checkpoint = base._path(state.get("formal_primary_best_checkpoint", folder / "checkpoints/formal_primary_best.pt"))
        if not checkpoint.is_file():
            raise ValueError(f"Seed {seed} {phase.upper()} has no selected checkpoint")
        if phase == "w1" and checkpoint != base._path(record["w1_checkpoint"]):
            raise ValueError("Selected W1 does not match the full-R source")
        evidence[f"{phase}_completion"] = str(folder / "progress.json")
        evidence[f"{phase}_checkpoint"] = str(checkpoint)
    summary_path = base._path(record["checkpoint"]).parent.parent / "summary.json"
    summary = read_json(summary_path, {})
    config = base._load_extended_yaml(base._path(record["config"]))
    if (summary.get("smoke") is not False or summary.get("stage") != "rb"
            or not (summary.get("stopped_early") is True
                    or int(summary.get("epoch", 0)) >= int(config["training"]["epochs"]))):
        raise ValueError("Full R must finish training before inference ablation")
    evidence["r_full_completion"] = str(summary_path)
    return evidence


def validate_protocol_amendments(protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Accept only a documented user-approved hash change before its training."""
    amendments = protocol.get("configuration_amendments", [])
    future_frozen = protocol.get("future_seed_configurations_frozen_before_first_test")
    initial_frozen = protocol.get("initial_configurations_frozen_before_first_test", future_frozen)
    if initial_frozen is not True:
        raise ValueError("The initial configuration freeze must remain documented")
    if future_frozen is True:
        if amendments:
            raise ValueError("Amended final configurations cannot be labeled frozen before first test")
    elif future_frozen is False and isinstance(amendments, list) and amendments:
        jobs = read_json(root / "inference_study_status.json", {}).get("jobs", [])
        for index, amendment in enumerate(amendments):
            if (amendment.get("user_authorized") is not True
                    or amendment.get("applied_before_target_training") is not True
                    or amendment.get("reason") != "explicit_user_instruction"):
                raise ValueError("Configuration amendment lacks explicit pre-training user authorization")
            prior_path = base._path(amendment["prior_protocol_path"])
            if base._sha256(prior_path) != amendment.get("prior_protocol_sha256"):
                raise ValueError("Configuration amendment prior protocol digest differs")
            prior = read_json(prior_path, {})
            if prior.get("configuration_amendments", []) != amendments[:index]:
                raise ValueError("Configuration amendment history is not an ordered chain")
            successor = (read_json(base._path(amendments[index + 1]["prior_protocol_path"]), {})
                         if index + 1 < len(amendments) else protocol)
            before = {str(base._path(path)): digest for path, digest in prior.get("configuration_sha256", {}).items()}
            after = {str(base._path(path)): digest for path, digest in successor.get("configuration_sha256", {}).items()}
            target = str(base._path(amendment["configuration_path"]))
            if before.get(target) != amendment.get("before_sha256"):
                raise ValueError("Configuration amendment before digest differs")
            expected_after = {**before, target: amendment.get("after_sha256")}
            if after != expected_after or not amendment.get("after_sha256"):
                raise ValueError("Unrecorded configuration hash drift in amended protocol")
            for key in ("seeds", "variants", "full_model_records", "comparison_design",
                        "separately_retrained", "final_results_may_modify_selection", "no_test_to_training_feedback"):
                if prior.get(key) != successor.get(key):
                    raise ValueError(f"Configuration amendment changed unrelated protocol field: {key}")
            target_jobs = [job for job in jobs if int(job.get("seed", -1)) == int(amendment["seed"])
                           and job.get("phase") == amendment["phase"]
                           and str(base._path(job.get("config", ""))) == target]
            if len(target_jobs) != 1:
                raise ValueError("Configuration amendment does not identify one registered training job")
            job = target_jobs[0]
            changed_at = datetime.fromisoformat(str(amendment["changed_at"]).replace("Z", "+00:00"))
            if job.get("started_at"):
                started_at = datetime.fromisoformat(str(job["started_at"]).replace("Z", "+00:00"))
                if changed_at > started_at:
                    raise ValueError("Configuration amendment was made after target training started")
            elif job.get("status") != "pending":
                raise ValueError("Cannot establish that the amended target was unstarted")
    else:
        raise ValueError("A revised final protocol requires explicit configuration amendment records")
    return {
        "initial_configurations_frozen_before_first_test": True,
        "future_seed_configurations_frozen_before_first_test": future_frozen,
        "configuration_amendments": amendments,
        "configuration_revision_disclosure": (
            "user_authorized_amendments_before_target_training" if amendments else "no_post_freeze_amendments"
        ),
    }


def _verify_completed_seed_inputs(existing: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    snapshots = {amendment["prior_protocol_sha256"]: base._path(amendment["prior_protocol_path"])
                 for amendment in protocol.get("configuration_amendments", [])}
    protocol_path = base._path(existing["protocol_path"])
    for path, expected in existing["sha256"].items():
        candidate = base._path(path)
        if candidate == protocol_path and base._sha256(candidate) != expected:
            candidate = snapshots.get(expected)
            if candidate is None:
                raise ValueError("Completed seed's historical protocol is not preserved")
        if base._sha256(candidate) != expected:
            raise ValueError(f"Completed seed's frozen input changed: {path}")


def freeze_seed(root: Path, seed: int, protocol_path: Path) -> tuple[dict[str, Any], Path]:
    protocol = read_json(protocol_path, {})
    if seed not in protocol.get("seeds", []):
        raise ValueError("Seed is not registered in the fixed inference protocol")
    if set(protocol.get("variants", [])) != set(VARIANTS):
        raise ValueError("Inference protocol must register exactly seven conditions")
    if (protocol.get("comparison_design") != DESIGN or protocol.get("separately_retrained") is not False
            or protocol.get("final_results_may_modify_selection") is not False
            or protocol.get("no_test_to_training_feedback") is not True):
        raise ValueError("Protocol must explicitly register inference intervention without test feedback")
    disclosure = validate_protocol_amendments(protocol, root)
    configuration_hashes = protocol.get("configuration_sha256", {})
    if not configuration_hashes:
        raise ValueError("Protocol must freeze the current and future base configurations")
    for path, expected in configuration_hashes.items():
        if base._sha256(base._path(path)) != expected:
            raise ValueError(f"A protocol-frozen training configuration changed: {path}")
    path = root / "inference_evaluation_2025" / f"seed_{seed}" / "freeze_manifest.json"
    existing = read_json(path)
    if existing is not None and read_json(path.parent / "metadata.json", {}).get("status") == "completed":
        _verify_completed_seed_inputs(existing, protocol)
        return existing, path
    record = _base_record(root, seed, protocol)
    evidence = _completed_base_evidence(root, seed, record)
    world_config = base._load_extended_yaml(base._path(record["world_config"]))
    normalizer = base._path(world_config["data"]["normalization"]["artifact_path"])
    paths = [base._path(record[key]) for key in ("checkpoint", "config", "w1_checkpoint", "world_config")]
    paths += [base._path(evidence["w0_checkpoint"]), protocol_path, normalizer]
    hashes = {str(path): base._sha256(path) for path in dict.fromkeys(paths)}
    manifest = {"schema": "dcvpp.inference-ablation.seed-freeze.v1", "seed": seed,
                "comparison_design": DESIGN, "separately_retrained": False,
                "full_model_record": record, "seed_base_training_completed": True,
                "completed_base_stages": ["w0", "w1", "r_full"], "completion_evidence": evidence,
                "protocol_path": str(protocol_path), "protocol_sha256": hashes[str(protocol_path)],
                "sha256": hashes, "final_results_may_modify_selection": False,
                "prior_2025_already_evaluated": True, **disclosure}
    if existing is not None and existing != manifest:
        raise ValueError("This seed's frozen inputs/protocol changed; preserve its original evidence")
    if existing is None:
        write_json(path, manifest)
    return manifest, path


def aggregate_inference(rows, records, latencies, *, expected_days=EXPECTED_DAYS):
    daily, summaries, dates = base.aggregate_models(rows, records, latencies, expected_days=expected_days)
    by_day = defaultdict(list)
    for row in rows:
        by_day[(row["variant"], row["operating_date"])].append(row)
    extra = ("terminal_unfinished_work", "terminal_soc_fraction", "tail_probability_brier",
             "tail_calibration_gap", "tail_probability_mean", "tail_event_rate",
             "grid_peak_import_ratio", "grid_peak_export_ratio", "grid_peak_ramp_ratio",
             "grid_rebound_ratio", "compute_concentration_hhi")
    for row in daily:
        selected = by_day[(row["variant"], row["operating_date"])]
        for key in extra:
            if all(key in item for item in selected):
                values = [float(item[key]) for item in selected]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Nonfinite inference diagnostic {key}")
                row[key] = sum(values) / len(values)
        row.update(comparison_design=DESIGN, separately_retrained=False)
    for row in summaries:
        selected = [item for item in daily if item["variant"] == row["variant"]]
        realized = [float(item["realized_operating_cost"]) for item in selected]
        row.update(comparison_design=DESIGN, separately_retrained=False,
                   mean_realized_operating_cost=sum(realized) / len(realized),
                   cvar95_realized_operating_cost=evaluation._cvar95(realized))
        for key in extra:
            if all(key in item for item in selected):
                row[f"mean_{key}"] = sum(float(item[key]) for item in selected) / len(selected)
    return daily, summaries, dates


def paired_comparisons(daily, decision_rows, *, seed: int, resamples: int = 2000, block_days: int | None = None):
    import numpy as np
    block_days = BOOTSTRAP_BLOCK_DAYS if block_days is None else block_days
    dates = sorted({row["operating_date"] for row in daily if row["variant"] == "full"})
    by_variant = {variant: {row["operating_date"]: row for row in daily if row["variant"] == variant}
                  for variant in VARIANTS}
    if any(set(values) != set(dates) for values in by_variant.values()):
        raise ValueError("Paired inference conditions have different dates")
    indices = np.asarray(evaluation.moving_block_bootstrap_indices(
        len(dates), block_days=block_days, resamples=resamples, seed=BOOTSTRAP_SEED), dtype=int)
    functions = {
        "mean_normalized_operating_cost": ("normalized_operating_cost", "mean"),
        "cvar95_normalized_operating_cost": ("normalized_operating_cost", "cvar"),
        "mean_deadline_expired_work": ("deadline_expired_work", "mean"),
        "cvar95_deadline_expired_work": ("deadline_expired_work", "cvar"),
        "mean_unserved_energy": ("unserved_energy", "mean"),
        "mean_sla_deadline_severity": ("sla_deadline_severity", "mean"),
        "mean_constraint_severity": ("constraint_severity", "mean"),
    }
    values = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for metric, (field, operator) in functions.items():
            array = np.asarray([float(by_variant[variant][day][field]) for day in dates])
            selected = array[indices]
            if operator == "mean":
                estimate, bootstrap = float(array.mean()), selected.mean(axis=1)
            else:
                count = max(1, math.ceil(0.05 * len(dates)))
                estimate = float(np.sort(array)[-count:].mean())
                bootstrap = np.sort(selected, axis=1)[:, -count:].mean(axis=1)
            values[variant][metric] = (estimate, bootstrap)
        lookup = {(row["operating_date"], int(row["stage_id"]), int(row["decision_origin"]), int(row["horizon"])):
                  float(row["sla_deadline_severity"]) for row in decision_rows if row["variant"] == variant}
        sla = np.asarray([[lookup[(day, *decision)] for decision in DECISIONS] for day in dates])
        values[variant]["worst_origin_sla"] = (float(sla.mean(axis=0).max()), sla[indices].mean(axis=1).max(axis=1))
    results = []
    for variant in VARIANTS:
        if variant == "full":
            continue
        for metric, (estimate, bootstrap) in values[variant].items():
            full, full_bootstrap = values["full"][metric]
            lower, upper = np.percentile(bootstrap - full_bootstrap, [2.5, 97.5])
            results.append({"seed": seed, "primary_variant": "full", "variant": variant,
                "metric": metric, "full_value": full, "variant_value": estimate,
                "difference": estimate - full,
                "relative_difference_percent": 100 * (estimate - full) / abs(full) if full else None,
                "ci95_low": float(lower), "ci95_high": float(upper),
                "effect_direction": "variant_minus_full", "block_days": block_days,
                "resamples": resamples, "bootstrap_seed": BOOTSTRAP_SEED,
                "comparison_design": DESIGN, "separately_retrained": False})
    return results


def intervention_latency(model, corpus, device, variant):
    batch = next(iter(corpus.batches(base.FINAL_SPLIT, 1, device=device, shuffle_seed=None,
        max_days=None, origin=0, require_h96=False, include_next_day=False)))
    packet = batch["world_input"]
    if packet.fast_values.shape[0] != 1:
        raise ValueError("Intervention latency requires batch 1")
    risk = base.RiskConditionPacket.defaults(1, device=device, horizon=96, stage=1)
    def forward():
        encoded = model.encode_expert_context(packet, da_anchor=batch["forecast_anchors"]["da"],
                                              da_residual_cap=batch["forecast_residual_caps"]["da"])
        return evaluation._execute_variant(model, encoded, risk, variant)
    samples = []
    with torch.inference_mode(), base._precision(device):
        for index in range(35):
            _, elapsed = base._timed(device, forward)
            if index >= 5:
                samples.append(elapsed)
    return {"p95_latency_ms_batch1": evaluation._percentile(samples, .95),
            "p50_latency_ms_batch1": evaluation._percentile(samples, .50),
            "mean_latency_ms_batch1": sum(samples) / len(samples),
            "latency_warmup": 5, "latency_repeats": 30,
            "latency_measurement": "batch1_h96_shared_full_weights_actual_intervention_no_replay"}


def evaluate_seed(manifest, manifest_path, output, *, device, batch_size=16):
    seed = int(manifest["seed"])
    expected_rows = EXPECTED_DAYS * len(DECISIONS) * len(VARIANTS)
    freeze_digest = base._sha256(manifest_path)
    launch = {"seed": seed, "freeze_manifest_sha256": freeze_digest,
              "evaluation_script_sha256": base._sha256(Path(__file__)), "batch_size": batch_size,
              "device": str(device), "expected_days": EXPECTED_DAYS, "expected_replay_rows": expected_rows}
    completed = read_json(output / "metadata.json", {})
    if completed.get("status") == "completed":
        if any(completed.get(key) != launch[key] for key in (
            "seed", "freeze_manifest_sha256", "expected_days", "expected_replay_rows"
        )):
            raise ValueError("Completed inference seed differs from the frozen launch")
        if completed.get("observed_replay_rows") != expected_rows or not all((output / name).is_file()
                for name in ("per_decision_models.csv", "per_day_models.csv", "model_summary.csv", "paired_comparisons.csv")):
            raise ValueError("Completed inference seed has incomplete evidence")
        return completed
    output.mkdir(parents=True, exist_ok=True)
    previous = read_json(output / "launch.json")
    if previous is not None and previous != launch:
        raise ValueError("Partial inference launch changed")
    write_json(output / "launch.json", launch)
    record = dict(manifest["full_model_record"])
    world_config, models = base._load_models([record], device)
    model = models["full"]
    if model.config.mechanism_variant != "full":
        raise ValueError("Inference interventions must start from the full model")
    interventions = registered_interventions()
    marker = {"schema": base.FROZEN_MARKER_SCHEMA, "status": "FROZEN_FOR_ONE_SHOT_FINAL_EVALUATION",
        "final_year": 2025, "best_checkpoint_frozen": True, "model_selection_frozen": True,
        "thresholds_frozen": True, "data_statistics_frozen": True, "development_evaluation_complete": True,
        "training_closed": True, "locked_final_opened": False, "one_shot_runs_allowed": 1,
        "final_results_may_modify_selection": False, "expected_evaluable_days": EXPECTED_DAYS,
        "best_checkpoint": record["checkpoint"], "run_id": record["checkpoint_run_id"],
        "one_shot_id": f"inference-ablation-20260911-seed{seed}", "evaluation_role": ROLE,
        "comparison_design": DESIGN, "separately_retrained": False,
        "prior_2025_already_evaluated": True, "freeze_manifest_sha256": freeze_digest}
    write_json(output / "frozen_evaluation_marker.json", marker)
    corpus = base.build_sealed_final_corpus(config=world_config, repository_root=REPO, frozen_evaluation_marker=marker)
    base._validate_sealed_corpus(corpus, marker)
    replay = base.SealedRealizedHistoricalReplay.from_model(model)
    latencies = read_json(output / "latency.json", {})
    for name, variant in interventions.items():
        if name not in latencies:
            latencies[name] = intervention_latency(model, corpus, device, variant)
            write_json(output / "latency.json", latencies)
    rows_by_variant = {name: base.read_partial(output / "partial" / f"{name}.jsonl") for name in interventions}
    keys = {name: {base._row_key(row) for row in rows} for name, rows in rows_by_variant.items()}
    reference_dates = None
    encode_calls = 0
    with torch.inference_mode():
        for stage, origin, horizon in DECISIONS:
            decision_dates = []
            for batch in corpus.batches(base.FINAL_SPLIT, batch_size, device=device, shuffle_seed=None,
                    max_days=None, origin=origin, require_h96=False, include_next_day=False):
                dates = tuple(str(day) for day in batch["operating_dates"])
                packet = batch["world_input"]
                if len(dates) != packet.fast_values.shape[0] or any(date.fromisoformat(day).year != 2025 for day in dates):
                    raise ValueError("Inference batch dates must be aligned 2025 samples")
                if not bool((packet.origin_index == origin).all()):
                    raise ValueError("Inference batch has the wrong causal origin")
                decision_dates.extend(dates)
                pending = [name for name in interventions if any((day, stage, origin, horizon) not in keys[name] for day in dates)]
                if not pending:
                    continue
                risk = base.RiskConditionPacket.defaults(len(dates), device=device, horizon=horizon, stage=stage)
                with base._precision(device):
                    expert_packet, encode_ms = base._timed(device, lambda: model.encode_expert_context(
                        packet, da_anchor=batch["forecast_anchors"]["da"], da_residual_cap=batch["forecast_residual_caps"]["da"]))
                    decisions = {name: base._timed(device, lambda name=name: evaluation._execute_variant(
                        model, expert_packet, risk, interventions[name])) for name in pending}
                # All outstanding intervention actions exist before realized targets enter replay.
                scale = float(batch["train_frozen_baseline_daily_cost"].detach().cpu())
                if not math.isclose(scale, record["train_frozen_cost_scale"], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("Frozen training cost scale changed")
                for name, (decision, decision_ms) in decisions.items():
                    realized = replay(decision, batch=batch, risk=risk)
                    emitted = evaluation._execution_rows(decision, realized, variant=interventions[name], dates=dates,
                        stage_id=stage, origin=origin, horizon=horizon, cost_scale=scale,
                        equivalent_latency_ms_per_sample=(encode_ms + decision_ms) / len(dates),
                        queue_event_target=batch.get("queue_risk_event_target"), queue_event_mask=batch.get("queue_risk_event_mask"))
                    sla = base._sla_deadline_severity(realized)[1].detach().cpu().tolist()
                    constraint = base._constraint_severity(realized)[1].detach().cpu().tolist()
                    new_rows = []
                    for index, row in enumerate(emitted):
                        row.update(seed=seed, variant=name, native_variant=interventions[name].name,
                            comparison_design=DESIGN, separately_retrained=False,
                            sla_deadline_severity=float(sla[index]), constraint_severity=float(constraint[index]),
                            hard_violation=float(row["hard_residual_max"]) > base.HARD_TOLERANCE)
                        if base._row_key(row) not in keys[name]:
                            new_rows.append(row)
                    base._append_batch(output / "partial" / f"{name}.jsonl", new_rows)
                    rows_by_variant[name].extend(new_rows)
                    keys[name].update(base._row_key(row) for row in new_rows)
                encode_calls += 1
                write_json(output / "progress.json", {"status": "running", "seed": seed,
                    "observed_replay_rows": sum(map(len, rows_by_variant.values())), "expected_replay_rows": expected_rows,
                    "stage_id": stage, "decision_origin": origin, "last_date": dates[-1], "updated_at": now()})
            if len(decision_dates) != EXPECTED_DAYS or len(set(decision_dates)) != EXPECTED_DAYS:
                raise ValueError("Inference window must cover exactly 362 unique dates")
            if reference_dates is not None and reference_dates != decision_dates:
                raise ValueError("Inference decision windows differ in dates")
            reference_dates = decision_dates
    rows = sorted((row for selected in rows_by_variant.values() for row in selected),
                  key=lambda row: (row["variant"], *base._row_key(row)))
    if len(rows) != expected_rows:
        raise ValueError("Incomplete inference replay")
    records = [{**record, "variant": name} for name in interventions]
    daily, summaries, dates = aggregate_inference(rows, records, latencies, expected_days=EXPECTED_DAYS)
    comparisons = paired_comparisons(daily, rows, seed=seed)
    for name, contents in (("per_decision_models.csv", rows), ("per_day_models.csv", daily),
                           ("model_summary.csv", summaries), ("paired_comparisons.csv", comparisons)):
        evaluation._atomic_csv(output / name, contents)
    metadata = {**launch, "status": "completed", "year": 2025, "split": base.FINAL_SPLIT,
        "comparison_design": DESIGN, "separately_retrained": False, "evaluation_role": ROLE,
        "prior_2025_already_evaluated": True, "dataset_dates": dates, "variants": list(interventions),
        "observed_replay_rows": len(rows), "observed_daily_rows": len(daily), "decisions_per_day": len(DECISIONS),
        "full_model_record": record, "models": records, "freeze_manifest_path": str(manifest_path),
        "protocol_path": manifest["protocol_path"], "protocol_sha256": manifest["protocol_sha256"],
        "seed_base_training_completed": True, "completed_base_stages": ["w0", "w1", "r_full"],
        "other_seeds_training_required_before_this_evaluation": False,
        "initial_configurations_frozen_before_first_test": manifest.get("initial_configurations_frozen_before_first_test", True),
        "future_seed_configurations_frozen_before_first_test": manifest.get("future_seed_configurations_frozen_before_first_test", True),
        "configuration_amendments": manifest.get("configuration_amendments", []),
        "configuration_revision_disclosure": manifest.get("configuration_revision_disclosure", "no_post_freeze_amendments"),
        "later_seed_training_after_prior_seed_test": True, "no_test_to_training_feedback": True,
        "training_performed": False, "checkpoint_selection_performed": False,
        "threshold_fitting_performed": False, "scaler_fitting_performed": False,
        "final_results_may_modify_selection": False, "locked_2025_opened": True,
        "all_actions_fixed_before_realized_target_read": True, "shared_checkpoint_for_all_variants": True,
        "w1_packet_reused_across_variants": True, "w1_encode_calls_this_invocation": encode_calls,
        "cost_normalization": "train_frozen_baseline_daily_cost * active_horizon / 96",
        "hard_violation_tolerance": base.HARD_TOLERANCE, "hard_violation_denominator": "all_decision_rows",
        "bootstrap": {"block_days": BOOTSTRAP_BLOCK_DAYS, "resamples": 2000, "seed": BOOTSTRAP_SEED, "effect_direction": "variant_minus_full"},
        "data_lineage": corpus.sealed_final_lineage, "completed_at": now()}
    write_json(output / "metadata.json", metadata)
    write_json(output / "completed.json", {"status": "completed", "seed": seed,
        "expected_replay_rows": expected_rows, "observed_replay_rows": len(rows),
        "comparison_design": DESIGN, "separately_retrained": False, "completed_at": now()})
    write_json(output / "progress.json", {"status": "completed", "seed": seed,
        "expected_replay_rows": expected_rows, "observed_replay_rows": len(rows),
        "updated_at": now()})
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    root = args.root.resolve()
    output = root / "inference_evaluation_2025" / f"seed_{args.seed}"
    try:
        manifest, path = freeze_seed(root, args.seed, root / "inference_protocol.json")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        evaluate_seed(manifest, path, output, device=device, batch_size=args.batch_size)
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "updated_at": now(), "partial_results_preserved": True})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
