"""One-shot, five-model final evaluation over one sealed 2025 corpus.

The suite is deliberately inference-only.  All five decisions for a batch are
materialised before :class:`SealedRealizedHistoricalReplay` is allowed to read
the realised target.  The primary Top-2 and the deterministic safe tensor also
share the exact same frozen W1 packet.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from .r_ecmoe import RECMoEOutput, RiskConditionPacket
from .r_ecmoe_baseline_evaluation import LoadedBaselineEvaluationRuntime
from .r_ecmoe_evaluation import (
    EVALUATION_DECISIONS,
    EvaluationVariant,
    _cvar95,
    _execute_variant,
    _execution_rows,
    _percentile,
    moving_block_bootstrap_indices,
)
from .r_ecmoe_runtime import RECMoEInferenceRuntime
from .r_ecmoe_sealed_2025 import (
    FINAL_SPLIT,
    FINAL_YEAR,
    RECMoESealed2025Error,
    SEALED_CORPUS_SCHEMA,
    SealedRealizedHistoricalReplay,
    _eligible_best_checkpoint,
    _exclusive_csv,
    _exclusive_json,
    _same_path,
    _utc_now,
    _validate_sealed_corpus,
    load_frozen_evaluation_marker,
)
from .r_ecmoe_top1_evaluation import TOP1_MODEL_ROLE


EXPECTED_FINAL_DAYS = 362
SUITE_MODEL_IDS = (
    "main_top2",
    "safe_tensor",
    "retrained_top1",
    "dense_direct",
    "independent_r_moe",
)
SUITE_META_SCHEMA = "dcvpp.r_ecmoe.sealed_2025_model_suite.v1"

_SUITE_OUTPUT_NAMES = (
    "suite_started.json",
    "suite_failed.json",
    "suite_completed.json",
    "suite_meta.json",
    "per_day_models.csv",
    "model_summary.csv",
    "paired_vs_main.csv",
)

_DAILY_MEAN_METRICS = (
    "active_horizon",
    "realized_operating_cost",
    "cost_normalization_denominator",
    "normalized_operating_cost",
    "deadline_expired_work",
    "unserved_energy",
    "action_total_variation",
    "battery_throughput",
    "compute_energy",
    "migration_energy",
    "pv_curtailed_energy",
    "grid_import_energy",
    "grid_export_energy",
    "grid_peak_import_ratio",
    "grid_peak_export_ratio",
    "grid_peak_ramp_ratio",
    "grid_rebound_ratio",
    "compute_concentration_hhi",
    "terminal_unfinished_work",
    "terminal_soc_fraction",
    "route_entropy",
    "route_gini",
    "tail_activation_rate",
    "tail_probability_mean",
    "tail_event_rate",
    "tail_probability_brier",
    "tail_calibration_gap",
    "equivalent_latency_ms_per_sample",
    *(f"route_load_{index}" for index in range(7)),
)


@dataclass(frozen=True)
class FrozenSuiteRuntimes:
    """Strictly loaded runtimes for the four trained checkpoints."""

    main: RECMoEInferenceRuntime
    top1: LoadedBaselineEvaluationRuntime
    dense: LoadedBaselineEvaluationRuntime
    independent: LoadedBaselineEvaluationRuntime


@dataclass(frozen=True)
class _ModelSpec:
    model_id: str
    display_name: str
    run_id: str
    checkpoint_path: Path
    source_w1_path: Path
    epoch: int
    model_class: str
    comparison_design: str
    separately_retrained: bool
    variant: EvaluationVariant


def _finite_positive(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RECMoESealed2025Error(f"{label} must be finite and positive")
    return float(value)


def _checkpoint_epoch(checkpoint: Mapping[str, Any], *, label: str) -> int:
    value = checkpoint.get("epoch")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RECMoESealed2025Error(f"{label} checkpoint lacks a completed epoch")
    return int(value)


def _suite_variants() -> dict[str, EvaluationVariant]:
    values = {
        "main_top2": EvaluationVariant(
            name="main_top2",
            active_top_k=2,
            tail_mode="conditional",
            route_prior_beta=1.0,
            description="frozen primary Top-2 conditional-tail model",
        ),
        "safe_tensor": EvaluationVariant(
            name="safe_tensor",
            active_top_k=None,
            tail_mode="off",
            route_prior_beta=0.0,
            safe_tensor=True,
            description="deterministic zero-residual neural safe tensor",
        ),
        "retrained_top1": EvaluationVariant(
            name="retrained_top1",
            active_top_k=1,
            tail_mode="conditional",
            route_prior_beta=1.0,
            description="separately trained Top-1 conditional-tail model",
        ),
        "dense_direct": EvaluationVariant(
            name="dense_direct",
            active_top_k=1,
            tail_mode="off",
            route_prior_beta=0.0,
            description="separately trained parameter-matched dense decision model",
        ),
        "independent_r_moe": EvaluationVariant(
            name="independent_r_moe",
            active_top_k=2,
            tail_mode="off",
            route_prior_beta=0.0,
            description="separately trained R-MoE with an independent router",
        ),
    }
    for value in values.values():
        value.validate()
    if tuple(values) != SUITE_MODEL_IDS:
        raise RECMoESealed2025Error("five-model suite registry drifted")
    return values


def _autocast(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return sum(float(row[name]) for row in rows) / float(len(rows))


def _aggregate_model_days(
    decision_rows: Sequence[Mapping[str, Any]],
    specs: Mapping[str, _ModelSpec],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in decision_rows:
        model_id = str(row.get("variant", ""))
        if model_id not in specs:
            raise RECMoESealed2025Error(f"unknown suite model row: {model_id}")
        grouped.setdefault((model_id, str(row["operating_date"])), []).append(row)
    output: list[dict[str, Any]] = []
    for model_id in SUITE_MODEL_IDS:
        spec = specs[model_id]
        dates = sorted(day for candidate, day in grouped if candidate == model_id)
        for operating_date in dates:
            values = grouped[(model_id, operating_date)]
            if len(values) != len(EVALUATION_DECISIONS):
                raise RECMoESealed2025Error(
                    f"{model_id}/{operating_date} does not contain six decisions"
                )
            row: dict[str, Any] = {
                "operating_date": operating_date,
                "model_id": model_id,
                "display_name": spec.display_name,
                "run_id": spec.run_id,
                "checkpoint": str(spec.checkpoint_path),
                "comparison_design": spec.comparison_design,
                "separately_retrained": spec.separately_retrained,
                "decision_count": len(values),
                "outcome_basis": "realized_historical_replay",
                "hard_residual_max": max(
                    float(value["hard_residual_max"]) for value in values
                ),
                "tail_calibration_target": str(values[0]["tail_calibration_target"]),
            }
            for metric in _DAILY_MEAN_METRICS:
                row[metric] = _mean(values, metric)
            output.append(row)
    return output


def _model_summaries(
    daily_rows: Sequence[Mapping[str, Any]],
    specs: Mapping[str, _ModelSpec],
    *,
    expected_days: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id in SUITE_MODEL_IDS:
        spec = specs[model_id]
        rows = [row for row in daily_rows if row["model_id"] == model_id]
        if len(rows) != expected_days:
            raise RECMoESealed2025Error(
                f"{model_id} daily result count differs from the frozen suite"
            )
        costs = [float(row["normalized_operating_cost"]) for row in rows]
        summary: dict[str, Any] = {
            "model_id": model_id,
            "display_name": spec.display_name,
            "run_id": spec.run_id,
            "checkpoint": str(spec.checkpoint_path),
            "source_w1_checkpoint": str(spec.source_w1_path),
            "checkpoint_epoch": spec.epoch,
            "model_class": spec.model_class,
            "comparison_design": spec.comparison_design,
            "separately_retrained": spec.separately_retrained,
            "days": len(rows),
            "decisions_per_day": len(EVALUATION_DECISIONS),
            "outcome_basis": "realized_historical_replay",
            "mean_realized_operating_cost": _mean(rows, "realized_operating_cost"),
            "mean_normalized_operating_cost": sum(costs) / len(costs),
            "median_normalized_operating_cost": _percentile(costs, 0.50),
            "cvar95_normalized_operating_cost": _cvar95(costs),
            "mean_deadline_expired_work": _mean(rows, "deadline_expired_work"),
            "mean_unserved_energy": _mean(rows, "unserved_energy"),
            "maximum_hard_residual": max(
                float(row["hard_residual_max"]) for row in rows
            ),
            "mean_action_total_variation": _mean(rows, "action_total_variation"),
            "mean_battery_throughput": _mean(rows, "battery_throughput"),
            "mean_compute_energy": _mean(rows, "compute_energy"),
            "mean_migration_energy": _mean(rows, "migration_energy"),
            "mean_pv_curtailed_energy": _mean(rows, "pv_curtailed_energy"),
            "mean_route_entropy": _mean(rows, "route_entropy"),
            "mean_tail_activation_rate": _mean(rows, "tail_activation_rate"),
            "latency_mean_ms_per_sample": _mean(
                rows, "equivalent_latency_ms_per_sample"
            ),
            "latency_p95_ms_per_sample": _percentile(
                [float(row["equivalent_latency_ms_per_sample"]) for row in rows],
                0.95,
            ),
        }
        for expert in range(7):
            summary[f"mean_route_load_{expert}"] = _mean(
                rows, f"route_load_{expert}"
            )
        output.append(summary)
    return output


def _paired_vs_main(
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    expected_days: int,
    hard_residual_tolerance: float,
    block_days: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    by_model = {
        model_id: {
            str(row["operating_date"]): row
            for row in daily_rows
            if row["model_id"] == model_id
        }
        for model_id in SUITE_MODEL_IDS
    }
    main = by_model["main_top2"]
    dates = sorted(main)
    if len(dates) != expected_days:
        raise RECMoESealed2025Error("main Top-2 day set is incomplete")
    samples = moving_block_bootstrap_indices(
        len(dates),
        block_days=block_days,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    output: list[dict[str, Any]] = []
    for model_id in SUITE_MODEL_IDS[1:]:
        comparator = by_model[model_id]
        if set(comparator) != set(main):
            raise RECMoESealed2025Error(
                f"{model_id} and main Top-2 do not share exact dates"
            )
        for operating_date in dates:
            if not math.isclose(
                float(main[operating_date]["cost_normalization_denominator"]),
                float(comparator[operating_date]["cost_normalization_denominator"]),
                rel_tol=1.0e-7,
                abs_tol=1.0e-7,
            ):
                raise RECMoESealed2025Error(
                    f"{model_id} and main Top-2 use different frozen cost scales"
                )
        deltas = [
            float(comparator[day]["normalized_operating_cost"])
            - float(main[day]["normalized_operating_cost"])
            for day in dates
        ]
        bootstrap = [
            sum(deltas[index] for index in sample) / float(len(sample))
            for sample in samples
        ]
        output.append(
            {
                "main_model": "main_top2",
                "comparator": model_id,
                "difference_direction": "comparator_minus_main",
                "outcome_basis": "realized_historical_replay",
                "days": len(dates),
                "mean_normalized_cost_delta": sum(deltas) / len(deltas),
                "cost_delta_ci95_lower": _percentile(bootstrap, 0.025),
                "cost_delta_ci95_upper": _percentile(bootstrap, 0.975),
                "comparator_daily_cost_win_rate": sum(value < 0.0 for value in deltas)
                / len(deltas),
                "deadline_delta_comparator_minus_main": sum(
                    float(comparator[day]["deadline_expired_work"])
                    - float(main[day]["deadline_expired_work"])
                    for day in dates
                )
                / len(dates),
                "unserved_delta_comparator_minus_main": sum(
                    float(comparator[day]["unserved_energy"])
                    - float(main[day]["unserved_energy"])
                    for day in dates
                )
                / len(dates),
                "main_hard_violation_rate": sum(
                    float(main[day]["hard_residual_max"]) > hard_residual_tolerance
                    for day in dates
                )
                / len(dates),
                "comparator_hard_violation_rate": sum(
                    float(comparator[day]["hard_residual_max"])
                    > hard_residual_tolerance
                    for day in dates
                )
                / len(dates),
                "bootstrap_block_days": block_days,
                "bootstrap_resamples": bootstrap_resamples,
                "bootstrap_seed": bootstrap_seed,
            }
        )
    return output


class RECMoESealed2025ModelSuiteEvaluator:
    """Evaluate the frozen five-model roster in one authorised corpus opening."""

    def __init__(
        self,
        runtimes: FrozenSuiteRuntimes,
        corpus_factory: Callable[[Mapping[str, Any]], Any],
        marker_path: str | Path,
        output_root: str | Path,
        *,
        batch_size: int = 16,
        bootstrap_seed: int = 20240813,
        bootstrap_resamples: int = 2000,
        block_days: int = 7,
    ) -> None:
        if not isinstance(runtimes, FrozenSuiteRuntimes):
            raise RECMoESealed2025Error("suite requires FrozenSuiteRuntimes")
        if not isinstance(runtimes.main, RECMoEInferenceRuntime):
            raise RECMoESealed2025Error("suite main runtime is invalid")
        for label, runtime in (
            ("top1", runtimes.top1),
            ("dense", runtimes.dense),
            ("independent", runtimes.independent),
        ):
            if not isinstance(runtime, LoadedBaselineEvaluationRuntime):
                raise RECMoESealed2025Error(f"suite {label} runtime is invalid")
        if runtimes.top1.baseline_type != TOP1_MODEL_ROLE:
            raise RECMoESealed2025Error("suite Top-1 runtime is not separately trained")
        if runtimes.dense.baseline_type != "dense_direct":
            raise RECMoESealed2025Error("suite dense runtime has the wrong architecture")
        if runtimes.independent.baseline_type != "independent_r_moe":
            raise RECMoESealed2025Error(
                "suite independent runtime has the wrong architecture"
            )
        if not callable(corpus_factory):
            raise RECMoESealed2025Error("suite sealed corpus factory must be callable")
        if batch_size < 1 or bootstrap_resamples < 1 or block_days < 1:
            raise RECMoESealed2025Error("suite batch/bootstrap settings must be positive")

        self.runtimes = runtimes
        self.device = runtimes.main.device
        if any(
            runtime.device != self.device
            for runtime in (runtimes.top1, runtimes.dense, runtimes.independent)
        ):
            raise RECMoESealed2025Error("all suite runtimes must share one device")
        if any(
            not _same_path(runtimes.main.checkpoint_path, runtime.w1_checkpoint_path)
            for runtime in (runtimes.top1, runtimes.dense, runtimes.independent)
        ):
            raise RECMoESealed2025Error(
                "all suite models must share the exact released W1 checkpoint"
            )

        self.main_checkpoint_path, self.main_checkpoint = _eligible_best_checkpoint(
            runtimes.main
        )
        self.marker_path = Path(marker_path).expanduser().resolve()
        self.marker = load_frozen_evaluation_marker(
            self.marker_path,
            best_checkpoint=self.main_checkpoint_path,
            checkpoint_run_id=str(self.main_checkpoint["run_id"]),
        )
        if int(self.marker["expected_evaluable_days"]) != EXPECTED_FINAL_DAYS:
            raise RECMoESealed2025Error(
                f"suite marker must freeze exactly {EXPECTED_FINAL_DAYS} evaluable days"
            )
        self.expected_days = EXPECTED_FINAL_DAYS
        self.cost_scale = _finite_positive(
            self.main_checkpoint.get("train_frozen_cost_scale"),
            label="main frozen cost scale",
        )
        for label, runtime in (
            ("Top-1", runtimes.top1),
            ("Dense Direct", runtimes.dense),
            ("Independent R-MoE", runtimes.independent),
        ):
            if not math.isclose(
                self.cost_scale, runtime.cost_scale, rel_tol=1.0e-7, abs_tol=1.0e-7
            ):
                raise RECMoESealed2025Error(
                    f"{label} frozen cost scale differs from main Top-2"
                )

        self.corpus_factory = corpus_factory
        self.output_root = Path(output_root).expanduser().resolve()
        self.batch_size = int(batch_size)
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.block_days = int(block_days)
        self.variants = _suite_variants()
        self.specs = self._build_specs()
        self.models = {
            "main_top2": runtimes.main.model,
            "safe_tensor": runtimes.main.model,
            "retrained_top1": runtimes.top1.model,
            "dense_direct": runtimes.dense.model,
            "independent_r_moe": runtimes.independent.model,
        }
        self.replays = {
            model_id: SealedRealizedHistoricalReplay.from_model(model)  # type: ignore[arg-type]
            for model_id, model in self.models.items()
        }
        for model in {id(value): value for value in self.models.values()}.values():
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def _build_specs(self) -> dict[str, _ModelSpec]:
        runtimes = self.runtimes
        main_run_id = str(self.main_checkpoint.get("run_id", ""))
        if not main_run_id:
            raise RECMoESealed2025Error("main checkpoint omitted run_id")
        separate_run_ids = (
            runtimes.top1.run_id,
            runtimes.dense.run_id,
            runtimes.independent.run_id,
        )
        if len(set((main_run_id, *separate_run_ids))) != 4:
            raise RECMoESealed2025Error(
                "separately trained suite checkpoints require distinct run ids"
            )
        return {
            "main_top2": _ModelSpec(
                "main_top2",
                "R-ECMoE Top-2",
                main_run_id,
                self.main_checkpoint_path,
                Path(runtimes.main.checkpoint_path).resolve(),
                _checkpoint_epoch(self.main_checkpoint, label="main"),
                type(runtimes.main.model).__name__,
                "frozen_suite_primary",
                True,
                self.variants["main_top2"],
            ),
            "safe_tensor": _ModelSpec(
                "safe_tensor",
                "Safe tensor",
                main_run_id,
                self.main_checkpoint_path,
                Path(runtimes.main.checkpoint_path).resolve(),
                _checkpoint_epoch(self.main_checkpoint, label="main"),
                "DeterministicZeroResidualSafeTensor",
                "same_main_checkpoint_intervention",
                False,
                self.variants["safe_tensor"],
            ),
            "retrained_top1": _ModelSpec(
                "retrained_top1",
                "Retrained R-ECMoE Top-1",
                runtimes.top1.run_id,
                runtimes.top1.checkpoint_path.resolve(),
                runtimes.top1.w1_checkpoint_path.resolve(),
                _checkpoint_epoch(runtimes.top1.checkpoint, label="Top-1"),
                runtimes.top1.model_class,
                "separately_retrained_architecture",
                True,
                self.variants["retrained_top1"],
            ),
            "dense_direct": _ModelSpec(
                "dense_direct",
                "Dense Direct R",
                runtimes.dense.run_id,
                runtimes.dense.checkpoint_path.resolve(),
                runtimes.dense.w1_checkpoint_path.resolve(),
                _checkpoint_epoch(runtimes.dense.checkpoint, label="Dense Direct"),
                runtimes.dense.model_class,
                "separately_retrained_architecture",
                True,
                self.variants["dense_direct"],
            ),
            "independent_r_moe": _ModelSpec(
                "independent_r_moe",
                "Independent R-MoE",
                runtimes.independent.run_id,
                runtimes.independent.checkpoint_path.resolve(),
                runtimes.independent.w1_checkpoint_path.resolve(),
                _checkpoint_epoch(
                    runtimes.independent.checkpoint, label="Independent R-MoE"
                ),
                runtimes.independent.model_class,
                "separately_retrained_architecture",
                True,
                self.variants["independent_r_moe"],
            ),
        }

    def _reserve_suite(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        existing = [
            name for name in _SUITE_OUTPUT_NAMES if (self.output_root / name).exists()
        ]
        if existing:
            raise RECMoESealed2025Error(
                "five-model suite was already started in this output root: "
                + ", ".join(existing)
            )
        _exclusive_json(
            self.output_root / "suite_started.json",
            {
                "schema": "dcvpp.r_ecmoe.sealed_2025_model_suite_started.v1",
                "status": "STARTED_AND_ONE_SHOT_SUITE_CONSUMED",
                "started_at_utc": _utc_now(),
                "one_shot_id": self.marker["one_shot_id"],
                "final_year": FINAL_YEAR,
                "expected_evaluable_days": self.expected_days,
                "model_ids": list(SUITE_MODEL_IDS),
                "main_checkpoint": str(self.main_checkpoint_path),
                "frozen_evaluation_marker": str(self.marker_path),
                "inference_only": True,
                "selection_performed": False,
                "results_may_modify_selection": False,
            },
        )

    def _batch_cost_scale(self, batch: Mapping[str, Any]) -> float:
        value = batch.get("train_frozen_baseline_daily_cost")
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise RECMoESealed2025Error("suite batch omitted train-only cost scale")
        numeric = float(value.detach().float().cpu())
        if not math.isclose(
            numeric, self.cost_scale, rel_tol=1.0e-7, abs_tol=1.0e-7
        ):
            raise RECMoESealed2025Error(
                "suite batch cost scale differs from all frozen checkpoints"
            )
        return numeric

    def _timed_main_packet(
        self, packet: Any, da_anchor: Tensor, da_cap: Tensor
    ) -> tuple[Any, float]:
        model = self.runtimes.main.model
        before = int(model.w1_backbone_calls)
        _synchronize(self.device)
        started = time.perf_counter()
        with _autocast(self.device):
            expert_packet = model.encode_expert_context(
                packet, da_anchor=da_anchor, da_residual_cap=da_cap
            )
        _synchronize(self.device)
        if int(model.w1_backbone_calls) - before != 1:
            raise RECMoESealed2025Error("main suite batch must encode W1 exactly once")
        return expert_packet, (time.perf_counter() - started) * 1000.0

    def _timed_variant(
        self, model: Any, packet: Any, risk: RiskConditionPacket, variant: EvaluationVariant
    ) -> tuple[RECMoEOutput, float]:
        _synchronize(self.device)
        started = time.perf_counter()
        with _autocast(self.device):
            output = _execute_variant(model, packet, risk, variant)
        _synchronize(self.device)
        return output, (time.perf_counter() - started) * 1000.0

    def _timed_separate_forward(
        self,
        runtime: LoadedBaselineEvaluationRuntime,
        packet: Any,
        risk: RiskConditionPacket,
        da_anchor: Tensor,
        da_cap: Tensor,
    ) -> tuple[RECMoEOutput, float]:
        model = runtime.model
        before = int(model.w1_backbone_calls)
        _synchronize(self.device)
        started = time.perf_counter()
        with _autocast(self.device):
            output = model(
                packet,
                risk,
                da_anchor=da_anchor,
                da_residual_cap=da_cap,
                return_diagnostics=True,
            )
        _synchronize(self.device)
        if int(model.w1_backbone_calls) - before != 1:
            raise RECMoESealed2025Error(
                f"{runtime.baseline_type} suite decision must encode W1 exactly once"
            )
        return output, (time.perf_counter() - started) * 1000.0

    def _run_inference(
        self, corpus: Any
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        decision_rows: list[dict[str, Any]] = []
        reference_dates: tuple[str, ...] | None = None
        counters = {
            "corpus_batch_visits": 0,
            "main_w1_encode_calls": 0,
            "main_top2_decision_calls": 0,
            "safe_tensor_decision_calls": 0,
            "retrained_top1_forward_calls": 0,
            "dense_direct_forward_calls": 0,
            "independent_r_moe_forward_calls": 0,
            "realized_replay_calls": 0,
            "batches_with_all_actions_fixed_before_target_read": 0,
        }
        with torch.inference_mode():
            for stage_id, origin, horizon in EVALUATION_DECISIONS:
                dates_for_decision: list[str] = []
                for batch in corpus.batches(
                    FINAL_SPLIT,
                    self.batch_size,
                    device=self.device,
                    shuffle_seed=None,
                    max_days=None,
                    origin=origin,
                    require_h96=False,
                    include_next_day=False,
                ):
                    counters["corpus_batch_visits"] += 1
                    dates = tuple(str(value) for value in batch.get("operating_dates", ()))
                    packet = batch.get("world_input")
                    if (
                        not dates
                        or packet is None
                        or len(dates) != int(packet.fast_values.shape[0])
                    ):
                        raise RECMoESealed2025Error(
                            "suite batch omitted dates or world_input"
                        )
                    try:
                        parsed = tuple(date.fromisoformat(value) for value in dates)
                    except ValueError as exc:
                        raise RECMoESealed2025Error(
                            "suite operating dates must use ISO format"
                        ) from exc
                    if any(value.year != FINAL_YEAR for value in parsed):
                        raise RECMoESealed2025Error(
                            "suite corpus emitted a non-final-year date"
                        )
                    if not bool((packet.origin_index == origin).all()):
                        raise RECMoESealed2025Error("suite packet origin mismatch")
                    dates_for_decision.extend(dates)
                    anchors = batch.get("forecast_anchors")
                    caps = batch.get("forecast_residual_caps")
                    da_anchor = anchors.get("da") if isinstance(anchors, Mapping) else None
                    da_cap = caps.get("da") if isinstance(caps, Mapping) else None
                    if not isinstance(da_anchor, Tensor) or not isinstance(da_cap, Tensor):
                        raise RECMoESealed2025Error(
                            "suite batch omitted frozen DA anchors"
                        )
                    risk = RiskConditionPacket.defaults(
                        len(dates), device=self.device, horizon=horizon, stage=stage_id
                    )

                    # Complete every action without reading any realised target.
                    main_packet, encode_ms = self._timed_main_packet(
                        packet, da_anchor, da_cap
                    )
                    counters["main_w1_encode_calls"] += 1
                    main, main_ms = self._timed_variant(
                        self.runtimes.main.model,
                        main_packet,
                        risk,
                        self.variants["main_top2"],
                    )
                    counters["main_top2_decision_calls"] += 1
                    safe, safe_ms = self._timed_variant(
                        self.runtimes.main.model,
                        main_packet,
                        risk,
                        self.variants["safe_tensor"],
                    )
                    counters["safe_tensor_decision_calls"] += 1
                    top1, top1_ms = self._timed_separate_forward(
                        self.runtimes.top1, packet, risk, da_anchor, da_cap
                    )
                    counters["retrained_top1_forward_calls"] += 1
                    dense, dense_ms = self._timed_separate_forward(
                        self.runtimes.dense, packet, risk, da_anchor, da_cap
                    )
                    counters["dense_direct_forward_calls"] += 1
                    independent, independent_ms = self._timed_separate_forward(
                        self.runtimes.independent, packet, risk, da_anchor, da_cap
                    )
                    counters["independent_r_moe_forward_calls"] += 1
                    decisions = {
                        "main_top2": (main, encode_ms + main_ms),
                        "safe_tensor": (safe, encode_ms + safe_ms),
                        "retrained_top1": (top1, top1_ms),
                        "dense_direct": (dense, dense_ms),
                        "independent_r_moe": (independent, independent_ms),
                    }
                    if tuple(decisions) != SUITE_MODEL_IDS:
                        raise RECMoESealed2025Error(
                            "suite action registry was incomplete before replay"
                        )
                    counters["batches_with_all_actions_fixed_before_target_read"] += 1
                    scale = self._batch_cost_scale(batch)

                    # This is the first target-reading boundary for the batch.
                    for model_id, (decision, elapsed_ms) in decisions.items():
                        realized = self.replays[model_id](
                            decision, batch=batch, risk=risk
                        )
                        counters["realized_replay_calls"] += 1
                        rows = _execution_rows(
                            decision,
                            realized,
                            variant=self.variants[model_id],
                            dates=dates,
                            stage_id=stage_id,
                            origin=origin,
                            horizon=horizon,
                            cost_scale=scale,
                            equivalent_latency_ms_per_sample=elapsed_ms / len(dates),
                        )
                        spec = self.specs[model_id]
                        for row in rows:
                            row["comparison_design"] = spec.comparison_design
                            row["separately_retrained"] = spec.separately_retrained
                        decision_rows.extend(rows)
                if (
                    len(dates_for_decision) != self.expected_days
                    or len(set(dates_for_decision)) != self.expected_days
                ):
                    raise RECMoESealed2025Error(
                        "suite did not consume the frozen 362-day set"
                    )
                ordered = tuple(dates_for_decision)
                if reference_dates is None:
                    reference_dates = ordered
                elif ordered != reference_dates:
                    raise RECMoESealed2025Error(
                        "suite decisions do not share identical ordered dates"
                    )
        return decision_rows, counters

    def _hard_residual_tolerance(self) -> float:
        resolved = self.main_checkpoint.get("resolved_config")
        if isinstance(resolved, Mapping):
            selection = resolved.get("selection")
            if isinstance(selection, Mapping):
                value = selection.get("hard_residual_tolerance")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            value = resolved.get("hard_residual_tolerance")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return 1.0e-4

    def run(self) -> Mapping[str, Any]:
        self._reserve_suite()
        try:
            # The suite constructs exactly one sealed corpus after reservation.
            corpus = self.corpus_factory(dict(self.marker))
            _validate_sealed_corpus(corpus, self.marker)
            decision_rows, counters = self._run_inference(corpus)
            daily_rows = _aggregate_model_days(decision_rows, self.specs)
            expected_rows = self.expected_days * len(SUITE_MODEL_IDS)
            if len(daily_rows) != expected_rows:
                raise RECMoESealed2025Error(
                    "suite daily aggregation omitted model-days"
                )
            summaries = _model_summaries(
                daily_rows, self.specs, expected_days=self.expected_days
            )
            paired = _paired_vs_main(
                daily_rows,
                expected_days=self.expected_days,
                hard_residual_tolerance=self._hard_residual_tolerance(),
                block_days=self.block_days,
                bootstrap_resamples=self.bootstrap_resamples,
                bootstrap_seed=self.bootstrap_seed,
            )
            _exclusive_csv(self.output_root / "per_day_models.csv", daily_rows)
            _exclusive_csv(self.output_root / "model_summary.csv", summaries)
            _exclusive_csv(self.output_root / "paired_vs_main.csv", paired)
            completed_at = _utc_now()
            registry = [
                {
                    "model_id": model_id,
                    "display_name": self.specs[model_id].display_name,
                    "run_id": self.specs[model_id].run_id,
                    "checkpoint": str(self.specs[model_id].checkpoint_path),
                    "source_w1_checkpoint": str(self.specs[model_id].source_w1_path),
                    "checkpoint_epoch": self.specs[model_id].epoch,
                    "model_class": self.specs[model_id].model_class,
                    "comparison_design": self.specs[model_id].comparison_design,
                    "separately_retrained": self.specs[model_id].separately_retrained,
                }
                for model_id in SUITE_MODEL_IDS
            ]
            meta = {
                "schema": SUITE_META_SCHEMA,
                "status": "COMPLETED",
                "completed_at_utc": completed_at,
                "one_shot_id": self.marker["one_shot_id"],
                "split": FINAL_SPLIT,
                "year": FINAL_YEAR,
                "expected_evaluable_days": self.expected_days,
                "observed_evaluable_days_per_model": self.expected_days,
                "model_count": len(SUITE_MODEL_IDS),
                "decisions_per_day": len(EVALUATION_DECISIONS),
                "model_registry": registry,
                "frozen_evaluation_marker": str(self.marker_path),
                "single_sealed_corpus_instance": True,
                "main_and_safe_share_w1_packet": True,
                "all_actions_fixed_before_realized_target_read": True,
                "outcome_basis": "realized_historical_replay",
                "cost_normalization": (
                    "train_frozen_baseline_daily_cost * active_horizon / 96"
                ),
                "inference_only": True,
                "optimizer_created": False,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "checkpoint_updated": False,
                "thresholds_updated": False,
                "data_statistics_updated": False,
                "checkpoint_hash_computed": False,
                "final_results_may_modify_selection": False,
                "repeat_or_overwrite_allowed": False,
                "external_solver": False,
                "candidate_search": False,
                "realized_replay_excluded_from_inference_latency": True,
                "bootstrap": {
                    "method": "circular_moving_block_over_paired_daily_differences",
                    "block_days": self.block_days,
                    "resamples": self.bootstrap_resamples,
                    "seed": self.bootstrap_seed,
                },
                "outputs": {
                    "per_day_models": str(self.output_root / "per_day_models.csv"),
                    "model_summary": str(self.output_root / "model_summary.csv"),
                    "paired_vs_main": str(self.output_root / "paired_vs_main.csv"),
                },
                **counters,
            }
            _exclusive_json(self.output_root / "suite_meta.json", meta)
            _exclusive_json(
                self.output_root / "suite_completed.json",
                {
                    "schema": "dcvpp.r_ecmoe.sealed_2025_model_suite_completed.v1",
                    "status": "COMPLETED_NO_RERUN_ALLOWED",
                    "completed_at_utc": completed_at,
                    "one_shot_id": self.marker["one_shot_id"],
                    "meta": str(self.output_root / "suite_meta.json"),
                },
            )
            return meta
        except BaseException as exc:
            failure = self.output_root / "suite_failed.json"
            if not failure.exists():
                _exclusive_json(
                    failure,
                    {
                        "schema": "dcvpp.r_ecmoe.sealed_2025_model_suite_failed.v1",
                        "status": "FAILED_ONE_SHOT_SUITE_CONSUMED_NO_RERUN_ALLOWED",
                        "failed_at_utc": _utc_now(),
                        "one_shot_id": self.marker["one_shot_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise


__all__ = [
    "EXPECTED_FINAL_DAYS",
    "FrozenSuiteRuntimes",
    "RECMoESealed2025ModelSuiteEvaluator",
    "SUITE_META_SCHEMA",
    "SUITE_MODEL_IDS",
]
