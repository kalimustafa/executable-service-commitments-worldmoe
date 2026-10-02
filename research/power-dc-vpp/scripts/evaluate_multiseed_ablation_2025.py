"""Resume the frozen three-seed study's supplemental 2025 evaluation.

The 2025 year has previously been seen. This script never trains, selects a
checkpoint, or changes the historical one-shot markers/results. Each seed uses
one frozen W1 packet per batch and separately trained persistent R mechanisms.
"""
from __future__ import annotations

import argparse
import _sha256 as _sha256_builtin
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from multiseed_common import REPO, ROOT, SEEDS, VARIANTS, now, read_json, write_json

sys.path.insert(0, str(REPO / "src"))
# Keep the established Windows Arrow/Torch DLL import order.
import pyarrow.dataset  # noqa: E402,F401
import torch  # noqa: E402
from dcvpp.vpp_world_moe import r_ecmoe_evaluation as evaluation  # noqa: E402
from dcvpp.vpp_world_moe.config import WorldMoEConfig  # noqa: E402
from dcvpp.vpp_world_moe.r_ecmoe import RECMoE, RECMoEConfig, RiskConditionPacket  # noqa: E402
from dcvpp.vpp_world_moe.r_ecmoe_losses import _constraint_severity, _sla_deadline_severity  # noqa: E402
from dcvpp.vpp_world_moe.r_ecmoe_runtime import _validate_checkpoint_mechanism  # noqa: E402
from dcvpp.vpp_world_moe.r_ecmoe_sealed_2025 import (  # noqa: E402
    FINAL_SPLIT, FROZEN_MARKER_SCHEMA, SealedRealizedHistoricalReplay, _validate_sealed_corpus,
)
from dcvpp.vpp_world_moe.r_ecmoe_sealed_corpus import build_sealed_final_corpus  # noqa: E402
from run_r_ecmoe import _load_extended_yaml, _load_w1  # noqa: E402

EXPECTED_DAYS = 362
HARD_TOLERANCE = 1.0e-4
LATENCY_LIMIT_MS = 1900.0
DECISIONS = evaluation.EVALUATION_DECISIONS
ROLE = "supplemental_independently_retrained_multiseed_ablation"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        # Use CPython's SHA-256 implementation: lazily initializing OpenSSL
        # after Arrow/Torch DLLs is unsafe in the verified Windows environment.
        return hashlib.file_digest(source, _sha256_builtin.sha256).hexdigest()


def validate_manifest(
    manifest: Mapping[str, Any], *, seeds: Sequence[int] = SEEDS,
    variants: Sequence[str] = tuple(VARIANTS),
) -> list[dict[str, Any]]:
    if manifest.get("all_27_training_stages_completed") is not True:
        raise ValueError("All training stages must be completed before opening 2025")
    if manifest.get("test_selection_feedback_allowed") is not False:
        raise ValueError("The frozen study must prohibit test-to-selection feedback")
    records = manifest.get("models", [])
    observed = [(int(row["seed"]), str(row["variant"])) for row in records]
    expected = {(seed, variant) for seed in seeds for variant in variants}
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Freeze manifest must contain exactly one checkpoint per seed/variant")
    required = ("checkpoint", "config", "w1_checkpoint", "world_config")
    frozen_hashes: dict[Path, str] = {}
    for row in records:
        row_hashes = {_path(key): value for key, value in row["sha256"].items()}
        for key in required:
            path = _path(row[key])
            if path not in row_hashes:
                raise ValueError(f"Frozen manifest omitted {key} digest")
            digest = str(row_hashes[path])
            if path in frozen_hashes and frozen_hashes[path] != digest:
                raise ValueError(f"Conflicting frozen hashes: {path}")
            frozen_hashes[path] = digest
    for path, digest in frozen_hashes.items():
        if _sha256(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    for seed in seeds:
        selected = [row for row in records if int(row["seed"]) == seed]
        for key in ("w1_checkpoint", "world_config"):
            if len({_path(row[key]) for row in selected}) != 1:
                raise ValueError(f"Seed {seed} variants must share one frozen {key}")
    return [dict(row) for row in records]


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (str(row["operating_date"]), int(row["stage_id"]),
            int(row["decision_origin"]), int(row["horizon"]))


def read_partial(path: Path) -> list[dict[str, Any]]:
    """Recover complete atomic batch records; discard only a torn final append."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    chunks = raw.splitlines(keepends=True)
    result: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, chunk in enumerate(chunks):
        try:
            batch_rows = json.loads(chunk)
            if not isinstance(batch_rows, list):
                raise ValueError("Partial journal batch is not a row list")
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(chunks) - 1:
                raise ValueError(f"Corrupt nonterminal partial journal record: {path}")
            with path.open("r+b") as target:
                target.truncate(valid_bytes)
            break
        result.extend(batch_rows)
        valid_bytes += len(chunk)
    if len({_row_key(row) for row in result}) != len(result):
        raise ValueError(f"Duplicate keys in partial journal: {path}")
    if raw and not raw.endswith(b"\n") and valid_bytes == len(raw):
        with path.open("ab") as target:
            target.write(b"\n")
    return result


def _append_batch(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(list(rows), allow_nan=False) + "\n")
        target.flush()


def aggregate_models(
    rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
    latency: Mapping[str, Mapping[str, Any]], *, expected_days: int = EXPECTED_DAYS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    variants = [str(record["variant"]) for record in records]
    seed_values = {int(record["seed"]) for record in records}
    if len(seed_values) != 1:
        raise ValueError("Aggregation must contain exactly one seed")
    seed = next(iter(seed_values))
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["variant"] not in variants or int(row["seed"]) != seed:
            raise ValueError("Unknown seed or variant in replay output")
        day = str(row["operating_date"])
        if date.fromisoformat(day).year != 2025:
            raise ValueError("Replay includes a non-2025 date")
        grouped[(str(row["variant"]), day)].append(row)
    daily: list[dict[str, Any]] = []
    expected_decisions = set(DECISIONS)
    mean_fields = (
        "realized_operating_cost", "normalized_operating_cost", "deadline_expired_work",
        "unserved_energy", "sla_deadline_severity", "constraint_severity",
        "battery_throughput", "compute_energy", "migration_energy", "pv_curtailed_energy",
        "grid_import_energy", "grid_export_energy", "action_total_variation",
        "route_entropy", "route_gini", "tail_activation_rate",
    )
    for (variant, day), selected in sorted(grouped.items()):
        actual = [(int(row["stage_id"]), int(row["decision_origin"]), int(row["horizon"])) for row in selected]
        if len(actual) != len(expected_decisions) or set(actual) != expected_decisions:
            raise ValueError("A model/date does not contain all six unique decisions")
        row: dict[str, Any] = {
            "seed": seed, "variant": variant, "operating_date": day,
            "separately_retrained": True, "decision_count": len(selected),
        }
        for field in mean_fields:
            values = [float(item[field]) for item in selected]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite replay metric: {field}")
            row[field] = sum(values) / len(values)
        hard_values = [float(item["hard_residual_max"]) for item in selected]
        if not all(math.isfinite(value) for value in hard_values):
            raise ValueError("Nonfinite hard constraint residual")
        row["hard_residual_max"] = max(hard_values)
        row["hard_violation"] = row["hard_residual_max"] > HARD_TOLERANCE
        daily.append(row)
    reference_dates: list[str] | None = None
    summaries = []
    for record in records:
        variant = str(record["variant"])
        selected = [row for row in daily if row["variant"] == variant]
        dates = [str(row["operating_date"]) for row in selected]
        if len(dates) != expected_days or (reference_dates is not None and dates != reference_dates):
            raise ValueError("Model day sets do not match the frozen evaluation days")
        reference_dates = dates
        decisions = [row for row in rows if row["variant"] == variant]
        origin_sla = {
            f"s{stage}_o{origin}_h{horizon}": sum(
                float(row["sla_deadline_severity"]) for row in decisions
                if (int(row["stage_id"]), int(row["decision_origin"]), int(row["horizon"])) == (stage, origin, horizon)
            ) / expected_days for stage, origin, horizon in DECISIONS
        }
        cost = [float(row["normalized_operating_cost"]) for row in selected]
        deadline = [float(row["deadline_expired_work"]) for row in selected]
        summary = {
            "seed": seed, "variant": variant, "days": len(selected), "replay_rows": len(decisions),
            "mean_normalized_operating_cost": sum(cost) / len(cost),
            "cvar95_normalized_operating_cost": evaluation._cvar95(cost),
            "mean_deadline_expired_work": sum(deadline) / len(deadline),
            "cvar95_deadline_expired_work": evaluation._cvar95(deadline),
            "worst_origin_sla": max(origin_sla.values()),
            "mean_sla_deadline_severity": sum(float(row["sla_deadline_severity"]) for row in selected) / len(selected),
            "mean_constraint_severity": sum(float(row["constraint_severity"]) for row in selected) / len(selected),
            "max_hard_residual": max(float(row["hard_residual_max"]) for row in decisions),
            "hard_violation_rate": sum(float(row["hard_residual_max"]) > HARD_TOLERANCE for row in decisions) / len(decisions),
            "daily_hard_violation_rate": sum(bool(row["hard_violation"]) for row in selected) / len(selected),
            "mean_unserved_energy": sum(float(row["unserved_energy"]) for row in selected) / len(selected),
            **dict(latency[variant]),
            "checkpoint_epoch": int(record["checkpoint_epoch"]),
            "checkpoint": str(record["checkpoint"]), "w1_checkpoint": str(record["w1_checkpoint"]),
            "selection_policy": record.get("selection_policy", "legacy"),
            "validation_selection_thresholds_passed": record.get("selection_thresholds_passed"),
            "separately_retrained": True,
        }
        summary["decision_value"] = (
            summary["mean_normalized_operating_cost"] + 0.25 * summary["cvar95_normalized_operating_cost"]
            + summary["worst_origin_sla"] + summary["mean_constraint_severity"]
            + max(0.0, summary["p95_latency_ms_batch1"] / LATENCY_LIMIT_MS - 1.0)
        )
        if not all(math.isfinite(float(summary[field])) for field in (
            "p95_latency_ms_batch1", "p50_latency_ms_batch1", "mean_latency_ms_batch1", "decision_value"
        )):
            raise ValueError("Nonfinite latency or combined decision metric")
        summaries.append(summary)
    return daily, summaries, reference_dates or []


def _precision(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _timed(device: torch.device, function):
    evaluation._synchronize(device)
    start = time.perf_counter()
    value = function()
    evaluation._synchronize(device)
    return value, (time.perf_counter() - start) * 1000.0


def _latency(model: RECMoE, corpus: Any, device: torch.device) -> dict[str, Any]:
    batch = next(iter(corpus.batches(
        FINAL_SPLIT, 1, device=device, shuffle_seed=None, max_days=None,
        origin=0, require_h96=False, include_next_day=False,
    )))
    packet, anchor, cap = batch["world_input"], batch["forecast_anchors"]["da"], batch["forecast_residual_caps"]["da"]
    risk = RiskConditionPacket.defaults(1, device=device, horizon=96, stage=1)
    samples = []
    with torch.inference_mode(), _precision(device):
        for index in range(35):
            _, elapsed = _timed(device, lambda: model(
                packet, risk, da_anchor=anchor, da_residual_cap=cap, return_diagnostics=False,
            ))
            if index >= 5:
                samples.append(elapsed)
    return {"p95_latency_ms_batch1": evaluation._percentile(samples, 0.95),
            "p50_latency_ms_batch1": evaluation._percentile(samples, 0.50),
            "mean_latency_ms_batch1": sum(samples) / len(samples),
            "latency_warmup": 5, "latency_repeats": 30,
            "latency_measurement": "batch1_h96_direct_causal_forward_no_replay"}


def validate_selected_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require a finite formal choice without conflating eligibility with gates.

    The score-only exception is local to this explicitly approved study. The
    historical sealed evaluator and its eligibility contract remain unchanged.
    """
    metrics = payload.get("metrics", {})
    if (payload.get("schema") != "dcvpp.r_ecmoe.runner.v1"
            or payload.get("phase") != "r_ecmoe" or payload.get("locked_2025_opened") is not False
            or metrics.get("validation_scope") != "full"
            or metrics.get("outcome_basis") != "realized_historical_replay"
            or metrics.get("finite") is not True
            or metrics.get("selection_eligible") is not True
            or payload.get("stage") != "rb" or metrics.get("stage") != "rb"
            or metrics.get("rb_stage_ready_for_selection") is not True):
        raise ValueError("A frozen checkpoint requires finite, complete R-B validation evidence")
    finite_fields = (
        "decision_value", "realized_daily_mean_cost", "empirical_cvar95", "worst_origin_sla",
        "constraint_residual", "hard_residual_max", "latency_p95_ms_per_sample",
    )
    if any(isinstance(metrics.get(key), bool) or not isinstance(metrics.get(key), (int, float))
           or not math.isfinite(float(metrics[key])) for key in finite_fields):
        raise ValueError("Frozen validation scores must be finite")
    policy = str(payload.get("resolved_config", {}).get("selection_policy", "legacy"))
    if policy not in {"legacy", "validation_score_only"}:
        raise ValueError("Unknown frozen checkpoint selection policy")
    if policy == "validation_score_only":
        if payload.get("selection_policy") != policy or metrics.get("selection_policy") != policy:
            raise ValueError("Score-only selection must be explicitly recorded by training and validation")
    elif payload.get("selection_policy", "legacy") != policy or metrics.get("selection_policy", "legacy") != policy:
        raise ValueError("Conflicting frozen checkpoint selection policies")
    thresholds_passed = all(metrics.get(key) is True for key in (
        "finite", "hard_residual_passed", "sla_passed", "latency_passed"
    ))
    if "selection_thresholds_passed" in metrics and metrics["selection_thresholds_passed"] != thresholds_passed:
        raise ValueError("Diagnostic threshold status conflicts with recorded validation outcomes")
    if policy == "legacy" and not thresholds_passed:
        raise ValueError("Legacy checkpoints must satisfy their original selection thresholds")
    return {"selection_policy": policy, "selection_thresholds_passed": thresholds_passed}


def _load_models(records: Sequence[dict[str, Any]], device: torch.device):
    world_config = _load_extended_yaml(_path(records[0]["world_config"]))
    world, _ = _load_w1(WorldMoEConfig.from_mapping(world_config["model"]), _path(records[0]["w1_checkpoint"]))
    world.to(device).eval()
    models = {}
    for record in records:
        config = _load_extended_yaml(_path(record["config"]))
        if int(config["training"]["seed"]) != int(record["seed"]):
            raise ValueError("Frozen training configuration has the wrong seed")
        if any(_path(config["source"][key]) != _path(record[key]) for key in ("w1_checkpoint", "world_config")):
            raise ValueError("Frozen R configuration has different source checkpoints/configuration")
        rec_config = RECMoEConfig.from_mapping(config["model"])
        if rec_config.mechanism_variant != record["variant"] or rec_config.active_top_k != 2:
            raise ValueError("Frozen model configuration differs from the registered Top-2 mechanism")
        if rec_config.tune_w1_action_response:
            raise ValueError("Shared W1 evaluation requires frozen W1 action response")
        payload = torch.load(_path(record["checkpoint"]), map_location="cpu", weights_only=False)
        _validate_checkpoint_mechanism(payload, rec_config)
        if payload.get("model_config") != asdict(rec_config):
            raise ValueError("Resolved R hyperparameters differ from the frozen checkpoint configuration")
        selection = validate_selected_checkpoint(payload)
        if str(config["training"].get("selection_policy", "legacy")) != selection["selection_policy"]:
            raise ValueError("Frozen training configuration has a different selection policy")
        if int(payload["resolved_config"]["seed"]) != int(record["seed"]):
            raise ValueError("R checkpoint has the wrong training seed")
        if not payload.get("source_paths") or _path(payload["source_paths"][0]) != _path(record["w1_checkpoint"]):
            raise ValueError("R checkpoint source W1 differs from the frozen pairing")
        if payload.get("tuned_dynamics_state"):
            raise ValueError("An R model modified shared W1 parameters")
        model = RECMoE(world, rec_config).to(device)
        state = payload["r_state"]
        if any(str(key).startswith("world_model.") for key in state):
            raise ValueError("R state unexpectedly includes shared world parameters")
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys or any(not key.startswith("world_model.") for key in incompatible.missing_keys):
            raise ValueError("R checkpoint does not strictly match its persistent mechanism")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        record["checkpoint_epoch"] = int(payload["epoch"])
        record["checkpoint_run_id"] = str(payload["run_id"])
        record["train_frozen_cost_scale"] = float(payload["train_frozen_cost_scale"])
        record.update(selection)
        models[str(record["variant"])] = model
    return world_config, models


def evaluate_seed(records: Sequence[dict[str, Any]], output: Path, *, device: torch.device,
                  batch_size: int, manifest_digest: str) -> dict[str, Any]:
    seed = int(records[0]["seed"])
    expected_rows = EXPECTED_DAYS * len(DECISIONS) * len(records)
    completed = read_json(output / "metadata.json", {})
    if completed.get("status") == "completed":
        if (completed.get("freeze_manifest_sha256") != manifest_digest
                or completed.get("observed_replay_rows") != expected_rows
                or completed.get("evaluation_script_sha256") != _sha256(Path(__file__))):
            raise ValueError("Completed seed does not match the current frozen manifest")
        if not all((output / name).exists() for name in ("per_decision_models.csv", "per_day_models.csv", "model_summary.csv")):
            raise ValueError("Completed seed is missing its evidence tables")
        return completed
    output.mkdir(parents=True, exist_ok=True)
    launch = {"freeze_manifest_sha256": manifest_digest, "seed": seed,
              "evaluation_script_sha256": _sha256(Path(__file__)), "batch_size": batch_size,
              "device": str(device), "torch_version": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "expected_days": EXPECTED_DAYS, "expected_replay_rows": expected_rows}
    existing = read_json(output / "launch.json")
    if existing is not None and existing != launch:
        raise ValueError("Partial evaluation launch changed; preserve its evidence and use a new study")
    write_json(output / "launch.json", launch)
    world_config, models = _load_models(records, device)
    marker = {
        "schema": FROZEN_MARKER_SCHEMA, "status": "FROZEN_FOR_ONE_SHOT_FINAL_EVALUATION",
        "final_year": 2025, "best_checkpoint_frozen": True, "model_selection_frozen": True,
        "thresholds_frozen": True, "data_statistics_frozen": True,
        "development_evaluation_complete": True, "training_closed": True,
        "locked_final_opened": False, "one_shot_runs_allowed": 1,
        "final_results_may_modify_selection": False, "expected_evaluable_days": EXPECTED_DAYS,
        "best_checkpoint": str(records[0]["checkpoint"]), "run_id": records[0]["checkpoint_run_id"],
        "one_shot_id": f"multiseed-20260910-seed{seed}", "evaluation_role": ROLE,
        "prior_2025_already_evaluated": True, "freeze_manifest_sha256": manifest_digest,
        "models": [{key: row[key] for key in ("seed", "variant", "checkpoint", "w1_checkpoint")} for row in records],
    }
    write_json(output / "frozen_evaluation_marker.json", marker)
    corpus = build_sealed_final_corpus(config=world_config, repository_root=REPO, frozen_evaluation_marker=marker)
    _validate_sealed_corpus(corpus, marker)
    latencies = read_json(output / "latency.json", {})
    rows_by_variant = {variant: read_partial(output / "partial" / f"{variant}.jsonl") for variant in models}
    keys = {variant: {_row_key(row) for row in rows} for variant, rows in rows_by_variant.items()}
    replay = {variant: SealedRealizedHistoricalReplay.from_model(model) for variant, model in models.items()}
    definitions = {variant: evaluation.EvaluationVariant(
        name=variant, active_top_k=2, tail_mode="conditional", route_prior_beta=model.config.route_prior_beta,
        description=VARIANTS[variant] + "; independently retrained persistent mechanism",
    ) for variant, model in models.items()}
    for variant, model in models.items():
        if variant not in latencies:
            latencies[variant] = _latency(model, corpus, device)
            write_json(output / "latency.json", latencies)
    reference_dates = None
    encode_calls = 0
    shared_model = next(iter(models.values()))
    with torch.inference_mode():
        for stage, origin, horizon in DECISIONS:
            decision_dates = []
            for batch in corpus.batches(FINAL_SPLIT, batch_size, device=device, shuffle_seed=None,
                                       max_days=None, origin=origin, require_h96=False, include_next_day=False):
                dates = tuple(str(day) for day in batch["operating_dates"])
                packet = batch["world_input"]
                if len(dates) != int(packet.fast_values.shape[0]) or any(date.fromisoformat(day).year != 2025 for day in dates):
                    raise ValueError("Final batch dates are not aligned 2025 samples")
                if not bool((packet.origin_index == origin).all()):
                    raise ValueError("Final batch has the wrong causal origin")
                decision_dates.extend(dates)
                pending = [variant for variant in models if any((day, stage, origin, horizon) not in keys[variant] for day in dates)]
                if not pending:
                    continue
                anchor, cap = batch["forecast_anchors"]["da"], batch["forecast_residual_caps"]["da"]
                risk = RiskConditionPacket.defaults(len(dates), device=device, horizon=horizon, stage=stage)
                with _precision(device):
                    expert_packet, encode_ms = _timed(device, lambda: shared_model.encode_expert_context(
                        packet, da_anchor=anchor, da_residual_cap=cap,
                    ))
                    decisions = {}
                    for variant in pending:
                        model = models[variant]
                        decisions[variant] = _timed(device, lambda: evaluation._complete_output(
                            model, expert_packet, risk, *model.decide(expert_packet, risk),
                        ))
                # Every outstanding action is fixed before accessing realized targets.
                scale = float(batch["train_frozen_baseline_daily_cost"].detach().cpu())
                for record in records:
                    if not math.isclose(scale, record["train_frozen_cost_scale"], rel_tol=1.0e-6, abs_tol=1.0e-6):
                        raise ValueError("Test cost scale differs from the model's frozen train-only scale")
                for variant, (decision, decide_ms) in decisions.items():
                    realized = replay[variant](decision, batch=batch, risk=risk)
                    emitted = evaluation._execution_rows(
                        decision, realized, variant=definitions[variant], dates=dates,
                        stage_id=stage, origin=origin, horizon=horizon, cost_scale=scale,
                        equivalent_latency_ms_per_sample=(encode_ms + decide_ms) / len(dates),
                        queue_event_target=batch.get("queue_risk_event_target"), queue_event_mask=batch.get("queue_risk_event_mask"),
                    )
                    sla = _sla_deadline_severity(realized)[1].detach().cpu().tolist()
                    constraint = _constraint_severity(realized)[1].detach().cpu().tolist()
                    new_rows = []
                    for index, row in enumerate(emitted):
                        row.update(seed=seed, variant=variant, separately_retrained=True,
                                   comparison_design="independently_retrained_paired_seed",
                                   sla_deadline_severity=float(sla[index]), constraint_severity=float(constraint[index]),
                                   hard_violation=float(row["hard_residual_max"]) > HARD_TOLERANCE)
                        if _row_key(row) not in keys[variant]:
                            new_rows.append(row)
                    _append_batch(output / "partial" / f"{variant}.jsonl", new_rows)
                    rows_by_variant[variant].extend(new_rows)
                    keys[variant].update(_row_key(row) for row in new_rows)
                encode_calls += 1
                write_json(output / "progress.json", {"status": "running", "seed": seed,
                    "observed_replay_rows": sum(map(len, rows_by_variant.values())), "expected_replay_rows": expected_rows,
                    "stage_id": stage, "decision_origin": origin, "last_date": dates[-1], "updated_at": now()})
            if len(decision_dates) != EXPECTED_DAYS or len(set(decision_dates)) != EXPECTED_DAYS:
                raise ValueError("Decision window did not cover all 362 unique days")
            if reference_dates is not None and reference_dates != decision_dates:
                raise ValueError("Decision windows have different date sets")
            reference_dates = decision_dates
    rows = sorted((row for selected in rows_by_variant.values() for row in selected),
                  key=lambda row: (str(row["variant"]), *_row_key(row)))
    if len(rows) != expected_rows:
        raise ValueError("Incomplete seed replay")
    daily, summaries, dates = aggregate_models(rows, records, latencies, expected_days=EXPECTED_DAYS)
    evaluation._atomic_csv(output / "per_decision_models.csv", rows)
    evaluation._atomic_csv(output / "per_day_models.csv", daily)
    evaluation._atomic_csv(output / "model_summary.csv", summaries)
    for summary in summaries:
        write_json(output / "partial" / f"{summary['variant']}_completed.json", summary)
    metadata = {**launch, "status": "completed", "year": 2025, "split": FINAL_SPLIT,
                "evaluation_role": ROLE, "prior_2025_already_evaluated": True,
                "observed_replay_rows": len(rows), "observed_daily_rows": len(daily), "dataset_dates": dates,
                "decisions_per_day": len(DECISIONS), "models": records, "variants": list(models),
                "comparison_design": "independently_retrained_paired_seed",
                "w1_shared_within_seed": True, "w1_encode_calls_this_invocation": encode_calls,
                "all_actions_fixed_before_realized_target_read": True,
                "training_performed": False, "checkpoint_selection_performed": False,
                "threshold_fitting_performed": False, "scaler_fitting_performed": False,
                "final_results_may_modify_selection": False, "locked_2025_opened": True,
                "hard_violation_tolerance": HARD_TOLERANCE, "hard_violation_denominator": "all_decision_rows",
                "cost_normalization": "train_frozen_baseline_daily_cost * active_horizon / 96",
                "daily_aggregation": "mean_of_six_daily_equivalent_decision_costs",
                "sla_definition": "0.5 * (expired_work / available_work + normalized_unserved_power)",
                "worst_origin_sla_definition": "max_of_six_origins_mean_sla_over_362_days",
                "data_lineage": corpus.sealed_final_lineage, "completed_at": now()}
    write_json(output / "metadata.json", metadata)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    root = args.root.resolve()
    output = root / "evaluation_2025"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    manifest_path = root / "freeze_manifest.json"
    records = validate_manifest(read_json(manifest_path, {}))
    digest = _sha256(manifest_path)
    results = []
    try:
        for seed in SEEDS:
            selected = sorted([row for row in records if int(row["seed"]) == seed],
                              key=lambda row: list(VARIANTS).index(row["variant"]))
            results.append(evaluate_seed(selected, output / f"seed_{seed}", device=device,
                                         batch_size=args.batch_size, manifest_digest=digest))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dates = results[0]["dataset_dates"]
        if any(result["dataset_dates"] != dates for result in results):
            raise ValueError("Seeds do not share the exact paired evaluation dates")
        expected = len(SEEDS) * len(VARIANTS) * EXPECTED_DAYS * len(DECISIONS)
        observed = sum(result["observed_replay_rows"] for result in results)
        if observed != expected:
            raise ValueError("The complete study has an unexpected replay row count")
        write_json(output / "completed.json", {"status": "completed", "expected_replay_rows": expected,
            "observed_replay_rows": observed, "seeds": list(SEEDS), "variants": list(VARIANTS),
            "expected_days": EXPECTED_DAYS, "freeze_manifest_sha256": digest,
            "evaluation_role": ROLE, "prior_2025_already_evaluated": True,
            "training_performed": False, "checkpoint_selection_performed": False, "completed_at": now()})
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                            "updated_at": now(), "partial_results_preserved": True})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
