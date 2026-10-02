"""One-shot, inference-only final evaluation for a frozen R-ECMoE.

This module is intentionally separate from development training and the 2024
selector.  It cannot construct the development :class:`WorldMoECorpus` (that
class correctly rejects final-year records).  Instead, an explicitly supplied
sealed-corpus factory is invoked only after the frozen-evaluation marker and
``best_decision_value.pt`` have been validated and an exclusive one-shot
sentinel has been written.

No optimizer, checkpoint selection, threshold fitting, normalization fitting,
solver, or candidate search exists on this path.  The final corpus must emit
the same batch contract as development evaluation while declaring that its
statistics and thresholds remain frozen from development data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence, cast

import torch
from torch import Tensor

from .packets import NODES, WorldObservationPacket
from .r_ecmoe import (
    RECMoE,
    RECMoEOutput,
    RiskConditionPacket,
    W1ExpertPacket,
    differentiable_operating_cost,
)
from .r_ecmoe_evaluation import (
    EVALUATION_DECISIONS,
    HARD_RESIDUAL_EXCLUSIONS,
    REGISTERED_VARIANTS,
    EvaluationVariant,
    _autocast,
    _cvar95,
    _execute_variant,
    _execution_rows,
    _percentile,
    _synchronize,
)
from .r_ecmoe_replay import (
    RECMoEReplayError,
    RealizedReplayCostWeights,
    RealizedReplayTargets,
    _split_raw_structured_actions,
)
from .r_ecmoe_runtime import RECMoEInferenceRuntime


FINAL_YEAR = 2025
FINAL_SPLIT = "locked_test"
FROZEN_MARKER_SCHEMA = "dcvpp.r_ecmoe.frozen_evaluation_marker.v1"
SEALED_CORPUS_SCHEMA = "dcvpp.r_ecmoe.sealed_final_corpus.v1"
ONE_SHOT_META_SCHEMA = "dcvpp.r_ecmoe.sealed_2025_evaluation.v1"

_OUTPUT_NAMES = (
    "one_shot_started.json",
    "one_shot_failed.json",
    "one_shot_completed.json",
    "per_day_top2.csv",
    "top2_summary.json",
    "one_shot_meta.json",
)


class RECMoESealed2025Error(RuntimeError):
    """The final evaluation cannot preserve its frozen one-shot contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one JSON artifact without permitting an overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as sink:
            json.dump(_json_ready(payload), sink, indent=2, sort_keys=True)
            sink.write("\n")
    except BaseException:
        # ``fdopen`` owns the descriptor once it succeeds.  If it did not,
        # closing an already-closed descriptor is harmlessly guarded here.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _exclusive_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RECMoESealed2025Error(f"cannot write empty final table: {path.name}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise RECMoESealed2025Error(f"final CSV schema drifted: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_ready(rows))


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).expanduser().resolve())) == os.path.normcase(
        str(Path(right).expanduser().resolve())
    )


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise RECMoESealed2025Error(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RECMoESealed2025Error(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RECMoESealed2025Error(f"{label} must contain one JSON mapping")
    return payload


def load_frozen_evaluation_marker(
    marker_path: str | Path,
    *,
    best_checkpoint: str | Path,
    checkpoint_run_id: str,
) -> Mapping[str, Any]:
    """Validate the authorization record without touching any final data."""

    path = Path(marker_path).expanduser().resolve()
    payload = _load_json_mapping(path, label="frozen-evaluation marker")
    required = {
        "schema": FROZEN_MARKER_SCHEMA,
        "status": "FROZEN_FOR_ONE_SHOT_FINAL_EVALUATION",
        "final_year": FINAL_YEAR,
        "best_checkpoint_frozen": True,
        "model_selection_frozen": True,
        "thresholds_frozen": True,
        "data_statistics_frozen": True,
        "development_evaluation_complete": True,
        "training_closed": True,
        "locked_final_opened": False,
        "one_shot_runs_allowed": 1,
        "final_results_may_modify_selection": False,
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise RECMoESealed2025Error(
                f"frozen-evaluation marker field {name} must equal {expected!r}"
            )
    marker_checkpoint = payload.get("best_checkpoint")
    if not isinstance(marker_checkpoint, str) or not _same_path(
        marker_checkpoint, best_checkpoint
    ):
        raise RECMoESealed2025Error(
            "frozen-evaluation marker does not bind the supplied best checkpoint"
        )
    if str(payload.get("run_id", "")) != str(checkpoint_run_id):
        raise RECMoESealed2025Error(
            "frozen-evaluation marker run_id differs from the checkpoint"
        )
    one_shot_id = payload.get("one_shot_id")
    if not isinstance(one_shot_id, str) or not one_shot_id.strip():
        raise RECMoESealed2025Error("frozen-evaluation marker requires one_shot_id")
    expected_days = payload.get("expected_evaluable_days")
    if (
        isinstance(expected_days, bool)
        or not isinstance(expected_days, int)
        or not 1 <= expected_days <= 366
    ):
        raise RECMoESealed2025Error(
            "frozen-evaluation marker expected_evaluable_days must lie in [1,366]"
        )
    return dict(payload)


def _eligible_best_checkpoint(
    runtime: RECMoEInferenceRuntime,
) -> tuple[Path, Mapping[str, Any]]:
    value = runtime.checkpoint_load.get("r_checkpoint_path")
    if not value:
        raise RECMoESealed2025Error(
            "one-shot final evaluation requires an R best checkpoint"
        )
    path = Path(str(value)).expanduser().resolve()
    if path.name != "best_decision_value.pt" or not path.is_file():
        raise RECMoESealed2025Error(
            "one-shot final evaluation accepts only best_decision_value.pt"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RECMoESealed2025Error("R best checkpoint is malformed")
    required = {
        "schema": "dcvpp.r_ecmoe.runner.v1",
        "phase": "r_ecmoe",
        "locked_2025_opened": False,
        "external_solver": False,
        "candidate_search": False,
        "hash_computed": False,
    }
    if any(payload.get(name) != expected for name, expected in required.items()):
        raise RECMoESealed2025Error("R best checkpoint violates frozen lineage")
    metrics = payload.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("validation_scope") != "full"
        or metrics.get("selection_eligible") is not True
        or metrics.get("outcome_basis") != "realized_historical_replay"
    ):
        raise RECMoESealed2025Error(
            "R best must be an eligible full realized-replay validation checkpoint"
        )
    for name in ("empirical_cvar95", "worst_origin_sla"):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RECMoESealed2025Error(
                f"R best lacks finite frozen validation metric {name}"
            )
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RECMoESealed2025Error("R best checkpoint requires run_id")
    return path, payload


def _validate_sealed_corpus(corpus: Any, marker: Mapping[str, Any]) -> None:
    if not callable(getattr(corpus, "batches", None)):
        raise RECMoESealed2025Error("sealed corpus must expose batches")
    lineage = getattr(corpus, "sealed_final_lineage", None)
    if not isinstance(lineage, Mapping):
        raise RECMoESealed2025Error(
            "sealed corpus must expose sealed_final_lineage"
        )
    required = {
        "schema": SEALED_CORPUS_SCHEMA,
        "split": FINAL_SPLIT,
        "year": FINAL_YEAR,
        "opened_final_test": True,
        "normalization_fit": "train_2022_2023_only",
        "proper_score_thresholds_frozen": True,
        "fit_on_final_test": False,
        "selection_feedback_allowed": False,
    }
    if any(lineage.get(name) != expected for name, expected in required.items()):
        raise RECMoESealed2025Error("sealed corpus lineage violates final-test boundary")
    if int(lineage.get("evaluable_days", -1)) != int(
        marker["expected_evaluable_days"]
    ):
        raise RECMoESealed2025Error(
            "sealed corpus day count differs from the frozen marker"
        )


def _final_targets(
    batch: Mapping[str, Any],
    *,
    output: RECMoEOutput,
) -> RealizedReplayTargets:
    """Extract final outcomes using the development replay tensor contract."""

    batch_size, expected_time = output.raw_structured_actions.shape[:2]
    constraints = output.expert_packet.constraints
    raw_dates = batch.get("operating_dates")
    if not isinstance(raw_dates, (tuple, list)) or len(raw_dates) != batch_size:
        raise RECMoESealed2025Error("final batch requires one date per sample")
    dates = tuple(str(value) for value in raw_dates)
    try:
        parsed = tuple(date.fromisoformat(value) for value in dates)
    except ValueError as exc:
        raise RECMoESealed2025Error("final dates must use ISO YYYY-MM-DD") from exc
    if any(value.year != FINAL_YEAR for value in parsed):
        raise RECMoESealed2025Error("sealed evaluator received a non-final-year date")
    metadata = batch.get("target_metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("opened_final_test_2025") is not True
        or metadata.get("fit_statistics_on_final_test", False) is not False
        or metadata.get("updated_thresholds_on_final_test", False) is not False
    ):
        raise RECMoESealed2025Error("final target metadata violates frozen inference")

    raw_public = batch.get("stochastic_target")
    if raw_public is None:
        reconstruction = batch.get("reconstruction_targets")
        if isinstance(reconstruction, Mapping):
            raw_public = reconstruction.get("public_21")
    if not isinstance(raw_public, Tensor) or raw_public.shape != (
        batch_size,
        expected_time,
        21,
    ):
        raise RECMoESealed2025Error("final public target must be [B,T,21]")
    raw_masks = batch.get("quantile_masks")
    if not isinstance(raw_masks, Mapping):
        raise RECMoESealed2025Error("final replay requires quantile masks")
    family_shapes = {
        "price": (batch_size, expected_time, 12),
        "workload": (batch_size, expected_time, 1),
        "pv": (batch_size, expected_time, 4),
        "pue": (batch_size, expected_time, 4),
    }
    masks: list[Tensor] = []
    for family, shape in family_shapes.items():
        value = raw_masks.get(family)
        if not isinstance(value, Tensor) or value.shape != shape or value.dtype is not torch.bool:
            raise RECMoESealed2025Error(
                f"final {family} mask must be bool {shape}"
            )
        masks.append(value.detach().to(device=output.raw_structured_actions.device))
    public_mask = torch.cat(masks, dim=-1)
    public = raw_public.detach().to(
        device=output.raw_structured_actions.device, dtype=torch.float32
    )
    if not bool((public_mask & torch.isfinite(public)).all()):
        raise RECMoESealed2025Error("every final replay target must be valid and finite")

    da = public[..., 0:4]
    rt = public[..., 4:8]
    dart = da - rt
    workload = public[..., 12].clamp_min(0.0)
    pv = public[..., 13:17]
    pue = public[..., 17:21]
    capacity = constraints.compute_capacity.detach().to(
        device=public.device, dtype=torch.float32
    )
    if capacity.shape != (batch_size, NODES):
        raise RECMoESealed2025Error("final replay compute capacity must be [B,4]")
    denominator = capacity.sum(dim=-1, keepdim=True)
    if capacity.device.type == "cpu" and bool((denominator <= 0.0).any()):
        raise RECMoESealed2025Error("final replay compute capacity must be positive")
    arrivals = workload[..., None] * (
        capacity / denominator.clamp_min(1.0e-8)
    )[:, None]
    if public.device.type == "cpu" and (
        bool((pv < 0.0).any()) or bool((pue < 1.0).any())
    ):
        raise RECMoESealed2025Error("final PV/PUE violates physical support")
    closed_public = torch.cat((da, rt, dart, workload[..., None], pv, pue), dim=-1)
    return RealizedReplayTargets(
        public_21=closed_public,
        da=da,
        rt=rt,
        dart=dart,
        arrivals=arrivals,
        pv=pv,
        pue=pue,
        operating_dates=dates,
        # The shared target container is safe here; its Literal annotation is
        # narrower only because the development replay itself stays sealed.
        source_split=cast(Any, FINAL_SPLIT),
    )


@dataclass
class SealedRealizedHistoricalReplay:
    """Apply the same assembler and operating-cost metric after final decisions."""

    assembler: Any
    cost_weights: RealizedReplayCostWeights

    @classmethod
    def from_model(cls, model: RECMoE) -> "SealedRealizedHistoricalReplay":
        config = model.world_model.config
        return cls(
            assembler=model.assembler,
            cost_weights=RealizedReplayCostWeights(
                battery_degradation_weight=float(config.battery_degradation_weight),
                migration_weight=float(config.migration_weight),
                sla_weight=float(config.unserved_power_weight),
            ),
        )

    def __call__(
        self,
        output: RECMoEOutput,
        *,
        batch: Mapping[str, Any],
        risk: RiskConditionPacket,
    ) -> RECMoEOutput:
        if not isinstance(output, RECMoEOutput):
            raise RECMoEReplayError("final replay input must be RECMoEOutput")
        batch_size, time_steps = output.raw_structured_actions.shape[:2]
        risk.validate(batch_size, time_steps)
        targets = _final_targets(batch, output=output)
        realized_reference: WorldObservationPacket = targets.replace_reference(
            output.expert_packet.reference
        )
        replay_packet: W1ExpertPacket = replace(
            output.expert_packet, reference=realized_reference
        )
        raw = _split_raw_structured_actions(output.raw_structured_actions)
        assembled = self.assembler(raw, replay_packet, risk)
        operating_cost = differentiable_operating_cost(
            reference=realized_reference,
            actions=assembled.actions,
            day_ahead_commitment=output.day_ahead_commitment,
            constraints=replay_packet.constraints,
            decision_active_mask=assembled.decision_active_mask,
            deadline_expired_work=assembled.deadline_expired_work,
            battery_degradation_weight=self.cost_weights.battery_degradation_weight,
            migration_weight=self.cost_weights.migration_weight,
            sla_weight=self.cost_weights.sla_weight,
        )
        return replace(
            output,
            expert_packet=replay_packet,
            actions=assembled.actions,
            raw_structured_actions=assembled.raw_structured_actions,
            constraint_residuals=assembled.residuals,
            differentiable_operating_cost=operating_cost,
            decision_active_mask=assembled.decision_active_mask,
            deadline_expired_work=assembled.deadline_expired_work,
            diagnostics={
                **output.diagnostics,
                "outcome_basis": "realized_historical_replay",
                "replay_split": FINAL_SPLIT,
                "replay_opened_final_test_2025": 1,
            },
        )


def _top2_variant() -> EvaluationVariant:
    matches = tuple(value for value in REGISTERED_VARIANTS if value.name == "top2")
    if len(matches) != 1:
        raise RECMoESealed2025Error("registered evaluator must contain one Top-2 path")
    return matches[0]


def _aggregate_days(decision_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in decision_rows:
        if row.get("variant") != "top2":
            raise RECMoESealed2025Error("final evaluation accepts only frozen Top-2")
        grouped.setdefault(str(row["operating_date"]), []).append(row)
    metrics = (
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
        "route_entropy",
        "tail_activation_rate",
        "equivalent_latency_ms_per_sample",
        *(f"route_load_{index}" for index in range(7)),
    )
    rows: list[dict[str, Any]] = []
    for operating_date, values in sorted(grouped.items()):
        if len(values) != len(EVALUATION_DECISIONS):
            raise RECMoESealed2025Error(
                f"{operating_date} does not contain every registered decision"
            )
        row: dict[str, Any] = {
            "operating_date": operating_date,
            "variant": "top2",
            "decision_count": len(values),
            "outcome_basis": "realized_historical_replay",
            "hard_residual_max": max(
                float(value["hard_residual_max"]) for value in values
            ),
        }
        for metric in metrics:
            row[metric] = sum(float(value[metric]) for value in values) / len(values)
        rows.append(row)
    return rows


def _summary(daily_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not daily_rows:
        raise RECMoESealed2025Error("final daily results cannot be empty")
    costs = [float(value["normalized_operating_cost"]) for value in daily_rows]
    result: dict[str, Any] = {
        "schema": "dcvpp.r_ecmoe.sealed_2025_summary.v1",
        "variant": "top2",
        "days": len(daily_rows),
        "outcome_basis": "realized_historical_replay",
        "mean_realized_operating_cost": sum(
            float(value["realized_operating_cost"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_normalized_operating_cost": sum(costs) / len(costs),
        "median_normalized_operating_cost": _percentile(costs, 0.50),
        "cvar95_normalized_operating_cost": _cvar95(costs),
        "mean_deadline_expired_work": sum(
            float(value["deadline_expired_work"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_unserved_energy": sum(
            float(value["unserved_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "maximum_hard_residual": max(
            float(value["hard_residual_max"]) for value in daily_rows
        ),
        "mean_action_total_variation": sum(
            float(value["action_total_variation"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_battery_throughput": sum(
            float(value["battery_throughput"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_compute_energy": sum(
            float(value["compute_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_migration_energy": sum(
            float(value["migration_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_pv_curtailed_energy": sum(
            float(value["pv_curtailed_energy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_route_entropy": sum(
            float(value["route_entropy"]) for value in daily_rows
        )
        / len(daily_rows),
        "mean_tail_activation_rate": sum(
            float(value["tail_activation_rate"]) for value in daily_rows
        )
        / len(daily_rows),
        "latency_mean_ms_per_sample": sum(
            float(value["equivalent_latency_ms_per_sample"]) for value in daily_rows
        )
        / len(daily_rows),
        "latency_p95_ms_per_sample": _percentile(
            [
                float(value["equivalent_latency_ms_per_sample"])
                for value in daily_rows
            ],
            0.95,
        ),
    }
    for expert in range(7):
        result[f"mean_route_load_{expert}"] = sum(
            float(value[f"route_load_{expert}"]) for value in daily_rows
        ) / len(daily_rows)
    return result


class RECMoESealed2025Evaluator:
    """Consume one frozen checkpoint and one sealed final corpus exactly once."""

    def __init__(
        self,
        runtime: RECMoEInferenceRuntime,
        corpus_factory: Callable[[Mapping[str, Any]], Any],
        marker_path: str | Path,
        output_root: str | Path,
        *,
        batch_size: int = 16,
    ) -> None:
        if not isinstance(runtime, RECMoEInferenceRuntime):
            raise RECMoESealed2025Error("runtime must be RECMoEInferenceRuntime")
        if not callable(corpus_factory):
            raise RECMoESealed2025Error("sealed corpus factory must be callable")
        if batch_size < 1:
            raise RECMoESealed2025Error("final batch_size must be positive")
        self.runtime = runtime
        self.model = runtime.model
        self.corpus_factory = corpus_factory
        self.marker_path = Path(marker_path).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.batch_size = int(batch_size)
        self.r_checkpoint_path, self.r_checkpoint = _eligible_best_checkpoint(runtime)
        self.marker = load_frozen_evaluation_marker(
            self.marker_path,
            best_checkpoint=self.r_checkpoint_path,
            checkpoint_run_id=str(self.r_checkpoint["run_id"]),
        )
        self.expected_days = int(self.marker["expected_evaluable_days"])
        self.cost_scale: float | None = None

    def _reserve_one_shot(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        existing = [name for name in _OUTPUT_NAMES if (self.output_root / name).exists()]
        if existing:
            raise RECMoESealed2025Error(
                "one-shot final evaluation was already started or output exists: "
                + ", ".join(existing)
            )
        started = self.output_root / "one_shot_started.json"
        _exclusive_json(
            started,
            {
                "schema": "dcvpp.r_ecmoe.sealed_2025_started.v1",
                "status": "STARTED_AND_ONE_SHOT_CONSUMED",
                "started_at_utc": _utc_now(),
                "one_shot_run_index": 1,
                "one_shot_id": self.marker["one_shot_id"],
                "final_year": FINAL_YEAR,
                "best_checkpoint": str(self.r_checkpoint_path),
                "frozen_evaluation_marker": str(self.marker_path),
                "inference_only": True,
                "selection_performed": False,
                "results_may_modify_selection": False,
            },
        )
        return started

    def _cost_scale(self, batch: Mapping[str, Any]) -> float:
        if self.cost_scale is not None:
            return self.cost_scale
        value = batch.get("train_frozen_baseline_daily_cost")
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise RECMoESealed2025Error("final batch omitted train-only cost scale")
        numeric = float(value.detach().float().cpu())
        expected = self.r_checkpoint.get("train_frozen_cost_scale")
        if (
            not math.isfinite(numeric)
            or numeric <= 0.0
            or expected is None
            or not math.isclose(numeric, float(expected), rel_tol=1.0e-7, abs_tol=1.0e-7)
        ):
            raise RECMoESealed2025Error(
                "final cost normalization differs from frozen R training"
            )
        self.cost_scale = numeric
        return numeric

    def _timed_encode(
        self, packet: Any, da_anchor: Tensor, da_cap: Tensor
    ) -> tuple[W1ExpertPacket, float]:
        before = int(self.model.w1_backbone_calls)
        _synchronize(self.runtime.device)
        started = time.perf_counter()
        with _autocast(self.runtime):
            expert_packet = self.model.encode_expert_context(
                packet, da_anchor=da_anchor, da_residual_cap=da_cap
            )
        _synchronize(self.runtime.device)
        elapsed = (time.perf_counter() - started) * 1_000.0
        if int(self.model.w1_backbone_calls) - before != 1:
            raise RECMoESealed2025Error("each final batch must encode W1 exactly once")
        return expert_packet, elapsed

    def _timed_top2(
        self,
        packet: W1ExpertPacket,
        risk: RiskConditionPacket,
        variant: EvaluationVariant,
    ) -> tuple[RECMoEOutput, float]:
        _synchronize(self.runtime.device)
        started = time.perf_counter()
        with _autocast(self.runtime):
            output = _execute_variant(self.model, packet, risk, variant)
        _synchronize(self.runtime.device)
        return output, (time.perf_counter() - started) * 1_000.0

    def _run_inference(self, corpus: Any) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        variant = _top2_variant()
        replay = SealedRealizedHistoricalReplay.from_model(self.model)
        decision_rows: list[dict[str, Any]] = []
        reference_dates: tuple[str, ...] | None = None
        w1_encode_calls = 0
        top2_decision_calls = 0
        realized_replay_calls = 0
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with torch.inference_mode():
            for stage_id, origin, horizon in EVALUATION_DECISIONS:
                dates_for_decision: list[str] = []
                for batch in corpus.batches(
                    FINAL_SPLIT,
                    self.batch_size,
                    device=self.runtime.device,
                    shuffle_seed=None,
                    max_days=None,
                    origin=origin,
                    require_h96=False,
                    include_next_day=False,
                ):
                    dates = tuple(str(value) for value in batch.get("operating_dates", ()))
                    packet = batch.get("world_input")
                    if (
                        not dates
                        or packet is None
                        or len(dates) != int(packet.fast_values.shape[0])
                    ):
                        raise RECMoESealed2025Error(
                            "final batch omitted dates or world_input"
                        )
                    try:
                        parsed_dates = tuple(date.fromisoformat(value) for value in dates)
                    except ValueError as exc:
                        raise RECMoESealed2025Error(
                            "final operating dates must use ISO format"
                        ) from exc
                    if any(value.year != FINAL_YEAR for value in parsed_dates):
                        raise RECMoESealed2025Error(
                            "final corpus emitted a non-final-year date"
                        )
                    if not bool((packet.origin_index == origin).all()):
                        raise RECMoESealed2025Error("final packet origin mismatch")
                    dates_for_decision.extend(dates)
                    anchors = batch.get("forecast_anchors")
                    caps = batch.get("forecast_residual_caps")
                    da_anchor = anchors.get("da") if isinstance(anchors, Mapping) else None
                    da_cap = caps.get("da") if isinstance(caps, Mapping) else None
                    if not isinstance(da_anchor, Tensor) or not isinstance(da_cap, Tensor):
                        raise RECMoESealed2025Error("final batch omitted frozen DA anchors")
                    risk = RiskConditionPacket.defaults(
                        len(dates),
                        device=self.runtime.device,
                        horizon=horizon,
                        stage=stage_id,
                    )
                    expert_packet, encode_ms = self._timed_encode(
                        packet, da_anchor, da_cap
                    )
                    w1_encode_calls += 1
                    # The action is complete before any realized target enters replay.
                    decision, decision_ms = self._timed_top2(
                        expert_packet, risk, variant
                    )
                    top2_decision_calls += 1
                    scale = self._cost_scale(batch)
                    realized = replay(decision, batch=batch, risk=risk)
                    realized_replay_calls += 1
                    decision_rows.extend(
                        _execution_rows(
                            decision,
                            realized,
                            variant=variant,
                            dates=dates,
                            stage_id=stage_id,
                            origin=origin,
                            horizon=horizon,
                            cost_scale=scale,
                            equivalent_latency_ms_per_sample=(encode_ms + decision_ms)
                            / len(dates),
                        )
                    )
                if (
                    len(dates_for_decision) != self.expected_days
                    or len(set(dates_for_decision)) != self.expected_days
                ):
                    raise RECMoESealed2025Error(
                        "final evaluation did not consume the frozen expected day set"
                    )
                ordered = tuple(dates_for_decision)
                if reference_dates is None:
                    reference_dates = ordered
                elif ordered != reference_dates:
                    raise RECMoESealed2025Error(
                        "final decisions do not share identical ordered dates"
                    )
        return decision_rows, {
            "w1_encode_calls": w1_encode_calls,
            "top2_decision_calls": top2_decision_calls,
            "realized_replay_calls": realized_replay_calls,
        }

    def run(self) -> Mapping[str, Any]:
        self._reserve_one_shot()
        try:
            # This is the first point at which final-data construction is allowed.
            corpus = self.corpus_factory(dict(self.marker))
            _validate_sealed_corpus(corpus, self.marker)
            decision_rows, counters = self._run_inference(corpus)
            daily_rows = _aggregate_days(decision_rows)
            if len(daily_rows) != self.expected_days:
                raise RECMoESealed2025Error("final daily aggregation omitted days")
            summary = _summary(daily_rows)
            _exclusive_csv(self.output_root / "per_day_top2.csv", daily_rows)
            _exclusive_json(self.output_root / "top2_summary.json", summary)
            meta = {
                "schema": ONE_SHOT_META_SCHEMA,
                "status": "COMPLETED",
                "completed_at_utc": _utc_now(),
                "one_shot_run_index": 1,
                "one_shot_id": self.marker["one_shot_id"],
                "split": FINAL_SPLIT,
                "year": FINAL_YEAR,
                "expected_evaluable_days": self.expected_days,
                "observed_evaluable_days": len(daily_rows),
                "decisions_per_day": len(EVALUATION_DECISIONS),
                "variant": "top2",
                "best_checkpoint": str(self.r_checkpoint_path),
                "w1_checkpoint": str(self.runtime.checkpoint_path),
                "run_id": self.r_checkpoint["run_id"],
                "frozen_evaluation_marker": str(self.marker_path),
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
                "final_results_may_modify_selection": False,
                "repeat_or_overwrite_allowed": False,
                "external_solver": False,
                "candidate_search": False,
                "realized_replay_excluded_from_inference_latency": True,
                **counters,
                "summary": summary,
            }
            _exclusive_json(self.output_root / "one_shot_meta.json", meta)
            _exclusive_json(
                self.output_root / "one_shot_completed.json",
                {
                    "schema": "dcvpp.r_ecmoe.sealed_2025_completed.v1",
                    "status": "COMPLETED_NO_RERUN_ALLOWED",
                    "completed_at_utc": meta["completed_at_utc"],
                    "one_shot_id": self.marker["one_shot_id"],
                    "meta": str(self.output_root / "one_shot_meta.json"),
                },
            )
            return meta
        except BaseException as exc:
            failure = self.output_root / "one_shot_failed.json"
            if not failure.exists():
                _exclusive_json(
                    failure,
                    {
                        "schema": "dcvpp.r_ecmoe.sealed_2025_failed.v1",
                        "status": "FAILED_ONE_SHOT_CONSUMED_NO_RERUN_ALLOWED",
                        "failed_at_utc": _utc_now(),
                        "one_shot_id": self.marker["one_shot_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise


__all__ = [
    "FINAL_SPLIT",
    "FINAL_YEAR",
    "FROZEN_MARKER_SCHEMA",
    "ONE_SHOT_META_SCHEMA",
    "RECMoESealed2025Error",
    "RECMoESealed2025Evaluator",
    "SEALED_CORPUS_SCHEMA",
    "SealedRealizedHistoricalReplay",
    "load_frozen_evaluation_marker",
]
