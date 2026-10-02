"""Train or run the direct solver-free R-ECMoE continuation phase."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pyarrow.dataset  # noqa: F401 - preload before CUDA on Windows
import torch
import yaml

from dcvpp.vpp_world_moe.config import WorldMoEConfig
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.model import (
    VPPWorldMoE,
    load_plannerless_released_w1_state,
)
from dcvpp.vpp_world_moe.r_ecmoe import RECMoE, RECMoEConfig, RiskConditionPacket
from dcvpp.vpp_world_moe.r_ecmoe_losses import RECMoELossWeights
from dcvpp.vpp_world_moe.r_ecmoe_evaluation import RECMoE2024Evaluator
from dcvpp.vpp_world_moe.r_ecmoe_runner import (
    TRAIN_DECISION_SCHEDULE,
    RECMoERunner,
    RECMoERunnerConfig,
)
from dcvpp.vpp_world_moe.r_ecmoe_runtime import RECMoEInferenceRuntime


def _compose_benchmark_suite(
    latency_batch1: Mapping[str, Any],
    throughput_batch: Mapping[str, Any],
) -> dict[str, Any]:
    if int(latency_batch1.get("batch_size", 0)) != 1:
        raise ValueError("latency profile must use batch_size=1")
    throughput_size = int(throughput_batch.get("batch_size", 0))
    if throughput_size < 1:
        raise ValueError("throughput profile must have a positive batch size")
    result = dict(latency_batch1)
    result.update(
        {
            "benchmark_suite": "batch1_latency_plus_batched_throughput",
            "throughput_batch_size": throughput_size,
            "throughput_windows_per_second": float(
                throughput_batch["throughput_windows_per_second"]
            ),
            "latency_batch1": dict(latency_batch1),
            "throughput_batch": dict(throughput_batch),
            "peak_memory_profiles_mib": {
                "latency_batch1": dict(latency_batch1["peak_memory_mib"]),
                "throughput_batch": dict(throughput_batch["peak_memory_mib"]),
            },
        }
    )
    return result


def _write_inference_latency(
    run_root: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    output = run_root / "inference_latency.json"
    payload = dict(result)
    payload["artifact_path"] = str(output.resolve())
    _atomic_text(
        output,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )
    return payload


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _load_extended_yaml(path: Path) -> dict[str, Any]:
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SystemExit(f"configuration must be a mapping: {source}")
    parent = raw.get("extends")
    if parent is None:
        return dict(raw)
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = source.parent / parent_path
    return _deep_merge(_load_extended_yaml(parent_path), raw)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_run_contract(
    run_root: Path,
    *,
    run_id: str,
    resolved_config: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    config_path: Path,
    world_config_path: Path,
    resume: bool,
) -> dict[str, Any]:
    config_output = run_root / "config_resolved.yaml"
    meta_output = run_root / "run_meta.json"
    config_value = dict(resolved_config)
    stable_meta = {
        "schema": "dcvpp.r_ecmoe.run-meta.v1",
        "phase": "r_ecmoe",
        "run_id": run_id,
        "profile": str(config_value.get("model", {}).get("profile", "")),
        "config_path": str(config_path.resolve()),
        "world_config_path": str(world_config_path.resolve()),
        "source_w1": dict(source_meta),
        "locked_2025_opened": False,
        "weather_coverage_gate": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
    }
    if "mechanism_variant" in config_value.get("model", {}):
        stable_meta["mechanism_variant"] = config_value["model"]["mechanism_variant"]
    if resume:
        if not config_output.is_file() or not meta_output.is_file():
            raise SystemExit(
                "resume requires existing config_resolved.yaml and run_meta.json"
            )
        existing_config = yaml.safe_load(config_output.read_text(encoding="utf-8"))
        existing_meta = json.loads(meta_output.read_text(encoding="utf-8"))
        comparable_meta = dict(existing_meta)
        comparable_meta.pop("created_at_utc", None)
        frozen_cost_scale = comparable_meta.pop(
            "train_frozen_baseline_daily_cost", None
        )
        if frozen_cost_scale is not None:
            if isinstance(frozen_cost_scale, bool):
                raise SystemExit(
                    "resume train_frozen_baseline_daily_cost must be finite and positive"
                )
            try:
                numeric_cost_scale = float(frozen_cost_scale)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    "resume train_frozen_baseline_daily_cost must be finite and positive"
                ) from exc
            if not math.isfinite(numeric_cost_scale) or numeric_cost_scale <= 0.0:
                raise SystemExit(
                    "resume train_frozen_baseline_daily_cost must be finite and positive"
                )
        if existing_config != config_value or comparable_meta != stable_meta:
            raise SystemExit("resume run contract differs from current inputs")
        return existing_meta
    if config_output.exists() or meta_output.exists():
        raise SystemExit("run contract already exists; use --mode resume")
    meta = {
        **stable_meta,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    _atomic_text(
        config_output,
        yaml.safe_dump(config_value, sort_keys=False, allow_unicode=True),
    )
    _atomic_text(
        meta_output,
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return meta


def _load_w1(
    world_config: WorldMoEConfig,
    checkpoint: Path,
) -> tuple[VPPWorldMoE, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SystemExit("W1 checkpoint payload is malformed")
    if str(payload.get("phase", "")).lower() != "w1":
        raise SystemExit("R-ECMoE requires a phase=w1 source checkpoint")
    if payload.get("locked_2025_opened") is not False:
        raise SystemExit("R-ECMoE source checkpoint opened the locked 2025 split")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise SystemExit("W1 checkpoint omitted model_state")
    resume_state = payload.get("resume_state", {})
    if not isinstance(resume_state, Mapping):
        resume_state = {}
    extra = payload.get("extra", {})
    if not isinstance(extra, Mapping):
        extra = {}
    model = VPPWorldMoE(world_config, include_legacy_planner=False)
    try:
        load_plannerless_released_w1_state(model, state)
    except RuntimeError as exc:
        raise SystemExit("W1 source does not load strictly") from exc
    return model, {
        "path": str(checkpoint),
        "phase": "w1",
        "epoch": int(resume_state.get("epoch", payload.get("epoch", -1))),
        "update": int(
            resume_state.get(
                "global_update",
                resume_state.get(
                    "update", payload.get("global_update", payload.get("update", -1))
                ),
            )
        ),
        "expert_assignment_schema": str(
            extra.get("expert_assignment_schema", "unknown")
        ),
        "locked_2025_opened": False,
        "strict_load": True,
        "hash_computed": False,
    }


def _runner_config(config: Mapping[str, Any]) -> RECMoERunnerConfig:
    training = config.get("training", {})
    loss = config.get("loss", {})
    selection = config.get("selection", {})
    outputs = config.get("outputs", {})
    if not all(isinstance(value, Mapping) for value in (training, loss, selection, outputs)):
        raise SystemExit("R-ECMoE training/loss/selection/outputs must be mappings")

    def optional_int(name: str) -> int | None:
        value = training.get(name)
        return None if value is None else int(value)

    def integer_schedule(
        name: str,
        default: tuple[int, ...],
        *,
        legacy_scalar: str | None = None,
    ) -> tuple[int, ...]:
        raw = training.get(name)
        if raw is None and legacy_scalar is not None and legacy_scalar in training:
            raw = (training[legacy_scalar],)
        if raw is None:
            return default
        if not isinstance(raw, (list, tuple)):
            raise SystemExit(f"training.{name} must be a list of integers")
        return tuple(int(value) for value in raw)

    def decision_schedule() -> tuple[tuple[int, int, int], ...]:
        raw = training.get("train_decision_schedule", TRAIN_DECISION_SCHEDULE)
        if not isinstance(raw, (list, tuple)):
            raise SystemExit(
                "training.train_decision_schedule must be a list of triples"
            )
        try:
            resolved = tuple(tuple(int(item) for item in value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "training.train_decision_schedule must contain integer triples"
            ) from exc
        if any(len(value) != 3 for value in resolved):
            raise SystemExit(
                "training.train_decision_schedule must contain integer triples"
            )
        return resolved  # type: ignore[return-value]

    loss_weights = RECMoELossWeights(
        expected_operating_cost=float(loss.get("expected_cost", 1.0)),
        upper_tail_cost=float(loss.get("tail_cost", 0.25)),
        quantile_pinball=float(loss.get("quantile", 0.25)),
        sla_deadline=float(loss.get("sla_deadline", 1.0)),
        constraint_residual=float(loss.get("constraint", 1.0)),
        route_continuation=float(loss.get("route_continuation", 0.02)),
        load_balance=float(loss.get("load_balance", 0.01)),
        tail_risk=float(loss.get("tail_expert", 0.05)),
        action_smoothness=float(loss.get("smoothness", 0.02)),
        safe_anchor=float(loss.get("behavior_anchor", 0.10)),
    )
    return RECMoERunnerConfig(
        epochs=int(training.get("epochs", training.get("maximum_epochs", 80))),
        ra_epochs=int(training.get("ra_epochs", training.get("stage_ra_epochs", 13))),
        batch_size=int(training.get("batch_size", 32)),
        validation_batch_size=int(training.get("validation_batch_size", 16)),
        learning_rate=float(
            training.get("learning_rate", training.get("base_learning_rate", 2.0e-4))
        ),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
        adamw_beta2=float(training.get("adamw_beta2", 0.95)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        warmup_updates=int(training.get("warmup_updates", 0)),
        cosine_decay_updates=int(training.get("cosine_decay_updates", 10_000)),
        minimum_learning_rate_ratio=float(
            training.get("minimum_learning_rate_ratio", 0.10)
        ),
        use_bf16=str(training.get("precision", "bf16")).lower() == "bf16",
        allow_tf32=bool(training.get("allow_tf32", True)),
        fused_adamw=bool(training.get("fused_adamw", True)),
        seed=int(training.get("seed", 20240813)),
        max_train_days=optional_int("max_train_days"),
        max_validation_days=optional_int("max_validation_days"),
        smoke_max_days=int(training.get("smoke_max_days", 4)),
        train_decision_schedule=decision_schedule(),
        validation_origins=integer_schedule(
            "validation_origins",
            (0, 4, 16, 48, 95),
            legacy_scalar="validation_origin",
        ),
        checkpoint_interval_updates=int(
            training.get("checkpoint_interval_updates", 100)
        ),
        first_full_validation_epoch=optional_int("first_full_validation_epoch"),
        selection_policy=str(training.get("selection_policy", "legacy")),
        enable_sentinel_validation=bool(training.get("enable_sentinel_validation", True)),
        save_latest_every_epoch=bool(training.get("save_latest_every_epoch", True)),
        resume_checkpoint_epochs=integer_schedule("resume_checkpoint_epochs", ()),
        early_stop_start_epoch=int(training.get("early_stop_start_epoch", 0)),
        coalesce_checkpoint_roles=bool(training.get("coalesce_checkpoint_roles", False)),
        save_only_on_improvement=bool(training.get("save_only_on_improvement", False)),
        save_on_exception=bool(training.get("save_on_exception", True)),
        train_log_interval_updates=int(
            training.get("train_log_interval_updates", 50)
        ),
        validation_interval_epochs=int(
            training.get("validation_interval_epochs", 2)
        ),
        sentinel_validation_days=int(
            training.get("sentinel_validation_days", 16)
        ),
        full_validation_interval_epochs=int(
            training.get("full_validation_interval_epochs", 10)
        ),
        early_stop_patience=int(training.get("early_stop_patience", 3)),
        early_stop_min_relative_improvement=float(
            training.get("early_stop_min_relative_improvement", 0.001)
        ),
        rolling_load_decay=float(training.get("rolling_load_decay", 0.90)),
        safe_anchor_decay_power=float(
            training.get("safe_anchor_decay_power", 1.0)
        ),
        decision_risk_weight=float(
            selection.get("decision_risk_weight", selection.get("risk_weight", 0.25))
        ),
        decision_sla_weight=float(
            selection.get("decision_sla_weight", selection.get("sla_penalty", 1.0))
        ),
        decision_constraint_weight=float(
            selection.get(
                "decision_constraint_weight",
                selection.get("constraint_penalty", 1.0),
            )
        ),
        decision_latency_penalty=float(
            selection.get("decision_latency_penalty", 1.0)
        ),
        hard_residual_tolerance=float(
            selection.get("hard_residual_tolerance", 1.0e-4)
        ),
        sla_tolerance=float(
            selection.get("sla_tolerance", selection.get("critical_queue_tolerance", 0.0))
        ),
        latency_limit_ms_per_sample=float(
            selection.get(
                "latency_limit_ms_per_sample",
                selection.get("latency_budget_ms", 1900.0),
            )
        ),
        selection_latency_warmup=int(
            selection.get("selection_latency_warmup", 5)
        ),
        selection_latency_repeats=int(
            selection.get("selection_latency_repeats", 30)
        ),
        save_best_speed_feasible=bool(
            outputs.get(
                "save_best_speed_feasible",
                outputs.get("keep_best_speed_feasible", True),
            )
        ),
        loss_weights=loss_weights,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/r_ecmoe_balanced.yaml")
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "train", "resume", "infer", "benchmark", "evaluate"),
        default="smoke",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id")
    parser.add_argument("--r-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--resume-smoke",
        action="store_true",
        help=(
            "with --mode resume, execute exactly one resumed optimizer update "
            "and one smoke validation batch per registered decision"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume_smoke and args.mode != "resume":
        raise SystemExit("--resume-smoke requires --mode resume")
    config_path = args.config.resolve()
    repository_root = config_path.parent.parent.resolve()
    config = _load_extended_yaml(config_path)
    if str(config.get("phase", "")).lower() != "r_ecmoe":
        raise SystemExit("configuration phase must be r_ecmoe")
    source = config.get("source")
    if not isinstance(source, Mapping) or source.get("locked_test_enabled") is not False:
        raise SystemExit("R-ECMoE development must keep locked test disabled")
    world_config_path = _resolve(repository_root, str(source["world_config"]))
    world_mapping = _load_extended_yaml(world_config_path)
    world_model_config = WorldMoEConfig.from_mapping(world_mapping["model"])
    rec_mapping = config.get("model")
    if not isinstance(rec_mapping, Mapping):
        raise SystemExit("R-ECMoE model configuration is required")
    rec_config = RECMoEConfig.from_mapping(dict(rec_mapping))
    checkpoint = _resolve(repository_root, str(source["w1_checkpoint"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    seed = int(config.get("training", {}).get("seed", 20240813))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    training_mode = args.mode in {"smoke", "train", "resume"}
    run_id = args.run_id
    if training_mode and run_id is None:
        run_id = (
            "r-ecmoe-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-seed{seed}"
        )
    output_mapping = config.get("outputs", {})
    output_root = _resolve(
        repository_root,
        str(output_mapping.get("root", "artifacts/r_ecmoe")),
    )
    resolved_runner_config = _runner_config(config)
    if training_mode:
        if run_id is None:  # guarded above; keeps the type contract explicit
            raise SystemExit("training requires a run id")
        world_model, source_meta = _load_w1(world_model_config, checkpoint)
        _write_run_contract(
            output_root / run_id,
            run_id=run_id,
            resolved_config=config,
            source_meta=source_meta,
            config_path=config_path,
            world_config_path=world_config_path,
            resume=args.mode == "resume",
        )
        model = RECMoE(world_model, rec_config)
        corpus = WorldMoECorpus(
            world_mapping,
            repository_root,
            weather_coverage_gate=False,
        )
        runner = RECMoERunner(
            model,
            corpus,
            resolved_runner_config,
            output_root,
            run_id,
            device,
            (checkpoint, config_path, world_config_path),
        )
        finish_observation = None
        if config.get("training", {}).get("first_full_validation_epoch") is not None:
            from observe_r_training import attach_progress
            finish_observation = attach_progress(runner)
        if args.mode == "smoke":
            result = runner.smoke()
        elif args.mode == "resume" and args.resume_smoke:
            result = runner.smoke(resume=True)
        else:
            try:
                result = runner.fit(resume=args.mode == "resume")
            except BaseException:
                if finish_observation is not None:
                    finish_observation(False)
                raise
            if finish_observation is not None:
                finish_observation(True)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    if args.r_checkpoint is not None:
        r_checkpoint = _resolve(repository_root, args.r_checkpoint)
    else:
        if run_id is None:
            raise SystemExit(
                "infer/benchmark requires --r-checkpoint or an existing --run-id"
            )
        r_checkpoint = output_root / run_id / "checkpoints" / "best_decision_value.pt"
    runtime = RECMoEInferenceRuntime.from_w1_checkpoint(
        world_model_config,
        rec_config,
        checkpoint,
        device,
        r_checkpoint_path=r_checkpoint,
    )
    corpus = WorldMoECorpus(
        world_mapping,
        repository_root,
        weather_coverage_gate=False,
    )
    if args.mode == "evaluate":
        evaluation = config.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            raise SystemExit("R-ECMoE evaluation configuration must be a mapping")
        evaluation_root = (
            output_root / run_id / "evaluation_2024"
            if run_id is not None
            else r_checkpoint.parent.parent / "evaluation_2024"
        )
        result = RECMoE2024Evaluator(
            runtime,
            corpus,
            evaluation_root,
            batch_size=int(
                evaluation.get(
                    "batch_size", resolved_runner_config.validation_batch_size
                )
            ),
            bootstrap_seed=int(evaluation.get("bootstrap_seed", 20240813)),
            bootstrap_resamples=int(evaluation.get("bootstrap_resamples", 2000)),
            block_days=int(evaluation.get("block_days", 7)),
        ).run()
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    def validation_batch(size: int) -> Mapping[str, Any]:
        return next(
            iter(
                corpus.batches(
                    "validation",
                    int(size),
                    device=device,
                    max_days=int(size),
                    origin=int(resolved_runner_config.validation_origins[0]),
                    require_h96=False,
                    include_next_day=False,
                )
            )
        )

    if args.mode == "benchmark":
        runtime_config = config.get("runtime", {})
        if not isinstance(runtime_config, Mapping):
            raise SystemExit("runtime configuration must be a mapping")
        runtime.configure_torch_compile(
            runtime_config.get("torch_compile", "off"),
            backend=str(runtime_config.get("compile_backend", "inductor")),
            mode=str(runtime_config.get("compile_mode", "reduce-overhead")),
        )
        warmup = (
            args.warmup
            if args.warmup is not None
            else int(runtime_config.get("warmup_repeats", 5))
        )
        repeats = (
            args.repeats
            if args.repeats is not None
            else int(runtime_config.get("measured_repeats", 30))
        )

        def execute_benchmark(size: int) -> dict[str, Any]:
            raw = validation_batch(size)
            item = raw["world_input"]
            item_risk = RiskConditionPacket.defaults(
                item.fast_values.shape[0],
                device=device,
                horizon=item.fast_values.shape[1],
                stage=1,
            )
            return runtime.benchmark(
                item,
                item_risk,
                {
                    "forecast_anchors": raw["forecast_anchors"],
                    "forecast_residual_caps": raw["forecast_residual_caps"],
                },
                warmup=warmup,
                repeats=repeats,
            )

        latency_result = execute_benchmark(1)
        configured_throughput = int(
            runtime_config.get(
                "throughput_batch_size",
                max(2, int(resolved_runner_config.validation_batch_size)),
            )
        )
        throughput_size = (
            int(args.batch_size) if int(args.batch_size) > 1 else configured_throughput
        )
        throughput_result = (
            latency_result
            if throughput_size == 1
            else execute_benchmark(throughput_size)
        )
        result = _compose_benchmark_suite(latency_result, throughput_result)
        benchmark_root = (
            output_root / run_id if run_id is not None else r_checkpoint.parent.parent
        )
        result = _write_inference_latency(benchmark_root, result)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    raw_batch = validation_batch(args.batch_size)
    packet = raw_batch["world_input"]
    risk = RiskConditionPacket.defaults(
        packet.fast_values.shape[0],
        device=device,
        horizon=packet.fast_values.shape[1],
        stage=1,
    )
    anchors = {
        "forecast_anchors": raw_batch["forecast_anchors"],
        "forecast_residual_caps": raw_batch["forecast_residual_caps"],
    }
    if args.mode == "infer":
        output = runtime.infer(packet, risk, anchors)
        result = {
            "schema": "dcvpp.r-ecmoe.inference-summary.v1",
            "phase": "r_ecmoe",
            "source": dict(runtime.checkpoint_load),
            "batch_size": int(packet.fast_values.shape[0]),
            "expected_cost": output.expected_cost.detach().float().cpu().tolist(),
            "upper_tail_cost": output.upper_tail_cost.detach().float().cpu().tolist(),
            "deadline_expired_total": output.deadline_expired_total.cpu().tolist(),
            "constraint_residual_max": output.constraint_residual_max.cpu().tolist(),
            "source_state_retained_internal_only": (
                output.source_state_retained_internal_only
            ),
        }
    else:  # pragma: no cover - parser choices and benchmark return guard this
        raise SystemExit(f"unsupported runtime mode: {args.mode}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
