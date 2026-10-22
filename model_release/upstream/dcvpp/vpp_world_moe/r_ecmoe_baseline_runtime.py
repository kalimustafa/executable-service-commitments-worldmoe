"""Inference-only performance benchmark for separately trained R baselines."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from .r_ecmoe import RiskConditionPacket
from .r_ecmoe_runtime import _ExecutedParameterTracker
from .r_ecmoe_baseline_evaluation import (
    LoadedBaselineEvaluationRuntime,
    load_baseline_evaluation_runtime,
)


BASELINE_BENCHMARK_SCHEMA = "dcvpp.r_ecmoe.baseline.inference_benchmark.v1"
BENCHMARK_WARMUP = 5
BENCHMARK_REPEATS = 30
LATENCY_BATCH_SIZE = 1
THROUGHPUT_BATCH_SIZE = 16


class BaselineBenchmarkError(RuntimeError):
    """A baseline cannot satisfy the fixed inference benchmark contract."""


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class BaselineBenchmarkRuntime:
    """Loaded eligible Dense/Independent model with no training surface."""

    def __init__(self, loaded: LoadedBaselineEvaluationRuntime | Any) -> None:
        if loaded.baseline_type not in {"dense_direct", "independent_r_moe"}:
            raise BaselineBenchmarkError("benchmark accepts only Dense or Independent R")
        model = loaded.model
        if (
            not bool(getattr(model, "external_solver_free", False))
            or not bool(getattr(model, "candidate_search_free", False))
        ):
            raise BaselineBenchmarkError("baseline benchmark prohibits solver/search")
        self.loaded = loaded
        self.model: nn.Module = model
        self.device = torch.device(loaded.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_eligible_best(
        cls, checkpoint: str | Path, *, device: str | torch.device
    ) -> "BaselineBenchmarkRuntime":
        # The shared loader rejects non-best, ineligible, same-checkpoint, and
        # non-realized-replay artifacts before model construction.
        return cls(load_baseline_evaluation_runtime(checkpoint, device=device))

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type in {"cuda", "cpu"},
        )

    def _forward(self, raw_batch: Mapping[str, Any]) -> Any:
        packet = raw_batch.get("world_input")
        anchors = raw_batch.get("forecast_anchors")
        caps = raw_batch.get("forecast_residual_caps")
        if packet is None or not isinstance(anchors, Mapping) or not isinstance(caps, Mapping):
            raise BaselineBenchmarkError("benchmark batch omitted packet/anchors")
        da = anchors.get("da")
        cap = caps.get("da")
        if not isinstance(da, Tensor) or not isinstance(cap, Tensor):
            raise BaselineBenchmarkError("benchmark batch omitted DA tensors")
        batch, horizon = packet.fast_values.shape[:2]
        if int(horizon) != 96:
            raise BaselineBenchmarkError("official baseline benchmark requires h96")
        risk = RiskConditionPacket.defaults(
            int(batch), device=self.device, horizon=96, stage=1
        )
        with torch.inference_mode(), self._autocast():
            return self.model(
                packet,
                risk,
                da_anchor=da,
                da_residual_cap=cap,
                return_diagnostics=False,
            )

    @staticmethod
    def _profile_dates(raw_batch: Mapping[str, Any], expected_batch: int) -> None:
        dates = tuple(str(value) for value in raw_batch.get("operating_dates", ()))
        packet = raw_batch.get("world_input")
        if packet is None or int(packet.fast_values.shape[0]) != expected_batch:
            raise BaselineBenchmarkError("benchmark batch size differs from contract")
        if len(dates) != expected_batch or any(not value.startswith("2024-") for value in dates):
            raise BaselineBenchmarkError("benchmark accepts only 2024 validation rows")

    def _parameter_counts(
        self,
        output: Any,
        *,
        executed_parameter_names: frozenset[str],
    ) -> dict[str, Any]:
        named = tuple(self.model.named_parameters())
        parameter_by_name = dict(named)
        executed_names = {
            name for name in executed_parameter_names if name in parameter_by_name
        }
        if not executed_names:
            raise BaselineBenchmarkError(
                "diagnostic forward traced no executed baseline parameters"
            )
        total = sum(parameter.numel() for _, parameter in named)
        world_total = sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("world_model.")
        )
        batch = int(output.expected_cost.shape[0])

        def active_names_for_sample(index: int) -> set[str]:
            inactive: list[str] = []
            for diagnostic in output.expert_packet.source_state.router_diagnostics:
                block = int(diagnostic.block_index)
                active = set(
                    int(item)
                    for item in diagnostic.topk_indices[index].detach().reshape(-1).cpu()
                )
                general = int(self.loaded.world_config.general_experts)
                for expert in range(general):
                    if expert not in active:
                        inactive.append(
                            f"world_model.backbone.blocks.{block}.moe.experts.{expert}."
                        )
                if not bool(diagnostic.tail_active_mask[index].any()):
                    inactive.append(
                        f"world_model.backbone.blocks.{block}.moe.tail_expert."
                    )
            if self.loaded.baseline_type == "independent_r_moe":
                active_r = set(
                    int(item)
                    for item in output.selected_expert_ids[index]
                    .detach()
                    .reshape(-1)
                    .cpu()
                )
                for expert in range(int(self.loaded.world_config.general_experts)):
                    if expert not in active_r:
                        inactive.append(
                            f"independent_adapter_bank.adapters.{expert}."
                        )
            return {
                name
                for name in executed_names
                if not any(name.startswith(prefix) for prefix in inactive)
            }

        per_sample_names = [active_names_for_sample(index) for index in range(batch)]
        per_sample = [
            sum(parameter_by_name[name].numel() for name in names)
            for names in per_sample_names
        ]
        active_union = sum(
            parameter_by_name[name].numel() for name in executed_names
        )
        return {
            "total_parameters": int(total),
            "w1_total_parameters": int(world_total),
            "r_total_parameters": int(total - world_total),
            "active_parameters_batch_union": int(active_union),
            "active_parameters_per_sample": per_sample,
            "active_parameters_per_sample_mean": sum(per_sample) / len(per_sample),
            "active_parameters_per_sample_min": min(per_sample),
            "active_parameters_per_sample_max": max(per_sample),
            "diagnostic_executed_parameter_tensors": int(len(executed_names)),
            "active_parameter_count_method": (
                "one_untimed_eager_diagnostic_forward_module_trace_with_"
                "per_sample_sparse_route_refinement"
            ),
            "active_parameter_semantics": (
                "traced_union_executed_over_one_sample_h96_window"
            ),
        }

    def benchmark_profile(
        self, raw_batch: Mapping[str, Any], *, expected_batch: int
    ) -> Mapping[str, Any]:
        self._profile_dates(raw_batch, expected_batch)
        calls_before = int(self.model.w1_backbone_calls)
        for _ in range(BENCHMARK_WARMUP):
            self._forward(raw_batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            base_allocated = torch.cuda.memory_allocated(self.device)
            base_reserved = torch.cuda.memory_reserved(self.device)
        else:
            base_allocated = base_reserved = 0
        latencies: list[float] = []
        output: Any | None = None
        for _ in range(BENCHMARK_REPEATS):
            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                start.record()
                output = self._forward(raw_batch)
                finish.record()
                finish.synchronize()
                latencies.append(float(start.elapsed_time(finish)))
            else:
                started = time.perf_counter()
                output = self._forward(raw_batch)
                latencies.append((time.perf_counter() - started) * 1_000.0)
        if output is None:
            raise BaselineBenchmarkError("benchmark produced no output")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            timing_method = "cuda_event"
        else:
            peak_allocated = peak_reserved = 0
            timing_method = "cpu_perf_counter"
        timed_calls_after = int(self.model.w1_backbone_calls)
        expected_calls = BENCHMARK_WARMUP + BENCHMARK_REPEATS
        if timed_calls_after - calls_before != expected_calls:
            raise BaselineBenchmarkError("each inference must call W1 exactly once")
        del output
        tracker = _ExecutedParameterTracker(self.model)
        diagnostic_before = int(self.model.w1_backbone_calls)
        try:
            tracker.install()
            diagnostic_output = self._forward(raw_batch)
        finally:
            tracker.remove()
        calls_after = int(self.model.w1_backbone_calls)
        if calls_after - diagnostic_before != 1:
            raise BaselineBenchmarkError(
                "parameter diagnostic must call W1 exactly once"
            )
        parameters = self._parameter_counts(
            diagnostic_output,
            executed_parameter_names=tracker.executed_parameter_names,
        )
        del diagnostic_output
        mean = sum(latencies) / len(latencies)
        return {
            "batch_size": expected_batch,
            "horizon": 96,
            "warmup": BENCHMARK_WARMUP,
            "repeats": BENCHMARK_REPEATS,
            "timing_method": timing_method,
            "full_inference_latency_ms": _latency(latencies),
            "throughput_windows_per_second": expected_batch * 1_000.0 / mean,
            "peak_memory_mib": {
                "base_allocated": base_allocated / 1024**2,
                "peak_allocated": peak_allocated / 1024**2,
                "incremental_peak_allocated": (peak_allocated - base_allocated)
                / 1024**2,
                "base_reserved": base_reserved / 1024**2,
                "peak_reserved": peak_reserved / 1024**2,
            },
            "parameters": parameters,
            "w1_backbone_calls": {
                "timed_delta": timed_calls_after - calls_before,
                "diagnostic": 1,
                "delta": calls_after - calls_before,
                "per_full_inference": 1,
            },
            "diagnostic_forwards_added": 1,
            "diagnostic_forward_included_in_latency": False,
            "diagnostic_forward_included_in_peak_memory": False,
            "candidate_search": False,
            "external_solver": False,
        }

    def benchmark_suite(
        self,
        batch1: Mapping[str, Any],
        batch16: Mapping[str, Any],
    ) -> dict[str, Any]:
        latency = self.benchmark_profile(batch1, expected_batch=LATENCY_BATCH_SIZE)
        throughput = self.benchmark_profile(
            batch16, expected_batch=THROUGHPUT_BATCH_SIZE
        )
        return {
            "schema": BASELINE_BENCHMARK_SCHEMA,
            "baseline_type": self.loaded.baseline_type,
            "model_class": self.loaded.model_class,
            "best_checkpoint": str(self.loaded.checkpoint_path),
            "run_id": self.loaded.run_id,
            "device": str(self.device),
            "autocast_dtype": "bfloat16",
            "latency_batch1": latency,
            "throughput_batch": throughput,
            "full_inference_latency_ms": latency["full_inference_latency_ms"],
            "throughput_windows_per_second": throughput[
                "throughput_windows_per_second"
            ],
            "parameters": latency["parameters"],
            "one_w1_call_per_inference": True,
            "locked_2025_opened": False,
            "training_performed": False,
            "candidate_search": False,
            "external_solver": False,
        }

    def write_official(self, result: Mapping[str, Any]) -> Path:
        output = self.loaded.run_root / "inference_latency.json"
        _atomic_json(output, result)
        return output


def load_baseline_benchmark_runtime(
    checkpoint: str | Path, *, device: str | torch.device
) -> BaselineBenchmarkRuntime:
    return BaselineBenchmarkRuntime.from_eligible_best(checkpoint, device=device)


__all__ = [
    "BASELINE_BENCHMARK_SCHEMA",
    "BENCHMARK_REPEATS",
    "BENCHMARK_WARMUP",
    "BaselineBenchmarkError",
    "BaselineBenchmarkRuntime",
    "load_baseline_benchmark_runtime",
]
