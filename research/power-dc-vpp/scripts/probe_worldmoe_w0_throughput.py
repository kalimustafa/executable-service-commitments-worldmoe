"""Read-only CUDA throughput probe for the NASA VPP-WorldMoE W0 phase.

This script never invokes :class:`WorldMoERunner`, never creates checkpoints,
and constructs the corpus with ``materialize=False``.  It measures real W0
forward/loss, backward/gradient clipping, and AdamW updates on development data.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# On Windows, eagerly load Arrow's dataset DLLs before CUDA/PyTorch.  Lazy
# loading them after Torch has populated the DLL search path can access-violate.
import pyarrow.dataset  # noqa: F401
import torch

from dcvpp.vpp_world_moe.config import WorldMoEConfig, load_world_moe_config
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.losses import PhaseLoss, phase_loss
from dcvpp.vpp_world_moe.model import VPPWorldMoE
from dcvpp.vpp_world_moe.runner import phase_spec_from_config
from dcvpp.vpp_world_moe.trainer import WorldModelTrainer


MIB = 1024**2
W0_UNUSED_TARGET_KEYS = (
    "target_packet",
    "next_day_packet",
    "future_input_packet",
    "next_day_batch",
    "target_batch",
)


def strip_w0_unused_targets(batch: Any) -> Any:
    """Remove next-day packets that W0 does not supervise or consume."""

    if not isinstance(batch, Mapping):
        return batch
    return {
        key: value for key, value in batch.items() if key not in W0_UNUSED_TARGET_KEYS
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def safe_results(
    results: Sequence[Mapping[str, Any]], *, minimum_free_mib: float
) -> list[Mapping[str, Any]]:
    """Return successful candidates satisfying total-device headroom."""

    return [
        result
        for result in results
        if result.get("status") == "ok"
        and float(result.get("gpu", {}).get("minimum_free_mib", -1.0))
        >= minimum_free_mib
    ]


def select_recommendations(
    results: Sequence[Mapping[str, Any]], *, minimum_free_mib: float
) -> dict[str, Any]:
    """Select both throughput-optimal and largest-safe measured candidates."""

    safe = safe_results(results, minimum_free_mib=minimum_free_mib)
    if not safe:
        return {
            "minimum_free_mib": minimum_free_mib,
            "throughput_best_case": None,
            "largest_safe_case": None,
        }
    throughput_best = max(
        safe, key=lambda item: float(item.get("samples_per_second_end_to_end", 0.0))
    )
    largest_safe = max(
        safe,
        key=lambda item: (
            int(item.get("batch_size", 0)),
            float(item.get("samples_per_second_end_to_end", 0.0)),
        ),
    )
    return {
        "minimum_free_mib": minimum_free_mib,
        "throughput_best_case": throughput_best.get("case_id"),
        "largest_safe_case": largest_safe.get("case_id"),
    }


def optimizer_update_budget(
    *,
    train_days: int,
    batch_sizes: Sequence[int],
    original_batch_size: int,
    minimum_updates: int,
    maximum_updates: int,
    patience_updates: int,
) -> dict[str, Any]:
    """Show how a batch-size change alters the registered update budget."""

    original_updates = math.ceil(train_days / original_batch_size)
    return {
        "train_days": train_days,
        "original_batch_size": original_batch_size,
        "original_updates_per_full_epoch": original_updates,
        "registered_update_targets": {
            "minimum": minimum_updates,
            "maximum": maximum_updates,
            "patience": patience_updates,
        },
        "candidates": {
            str(batch_size): {
                "updates_per_full_epoch": math.ceil(train_days / batch_size),
                "full_epochs_to_preserve_update_targets": {
                    "minimum": math.ceil(
                        minimum_updates / math.ceil(train_days / batch_size)
                    ),
                    "maximum": math.ceil(
                        maximum_updates / math.ceil(train_days / batch_size)
                    ),
                    "patience": math.ceil(
                        patience_updates / math.ceil(train_days / batch_size)
                    ),
                },
            }
            for batch_size in batch_sizes
        },
        "warning": (
            "changing batch size while retaining epoch counters changes the number "
            "of optimizer updates; the micro-epoch schedule should preserve these "
            "registered update targets explicitly"
        ),
    }


class NvidiaSmiSampler:
    """Low-rate total-device utilization/headroom sampler."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = float(interval_seconds)
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _sample() -> tuple[float, float, float] | None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            row = completed.stdout.strip().splitlines()[0]
            utilization, used, free = (float(item.strip()) for item in row.split(","))
            return utilization, used, free
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def __enter__(self) -> "NvidiaSmiSampler":
        def worker() -> None:
            while not self._stop.is_set():
                sample = self._sample()
                if sample is not None:
                    self.samples.append(sample)
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        final = self._sample()
        if final is not None:
            self.samples.append(final)

    def summary(self) -> dict[str, Any]:
        utilization = [sample[0] for sample in self.samples]
        used = [sample[1] for sample in self.samples]
        free = [sample[2] for sample in self.samples]
        return {
            "sample_count": len(self.samples),
            "utilization_mean_percent": _mean(utilization),
            "utilization_p50_percent": _percentile(utilization, 0.50),
            "utilization_p95_percent": _percentile(utilization, 0.95),
            "utilization_max_percent": max(utilization, default=0.0),
            "peak_total_used_mib": max(used, default=0.0),
            "minimum_free_mib": min(free, default=-1.0),
        }


def _synchronize() -> None:
    torch.cuda.synchronize()


def _timed_w0_step(
    trainer: WorldModelTrainer, batch: Mapping[str, Any], *, seed: int
) -> dict[str, float]:
    if trainer.optimizer is None:
        raise RuntimeError("W0 probe unexpectedly has no optimizer")

    _synchronize()
    started = time.perf_counter()
    trainer.optimizer.zero_grad(set_to_none=True)
    _synchronize()
    zero_grad_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    output = trainer.forward(batch, seed=seed)
    loss = phase_loss(
        "w0",
        output,
        batch,
        weights=trainer.weights,
        cvar_alpha=trainer.spec.cvar_alpha,
    )
    if not isinstance(loss, PhaseLoss) or not bool(torch.isfinite(loss.total.detach())):
        raise RuntimeError("W0 probe produced a non-finite loss")
    _synchronize()
    forward_loss_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    loss.total.backward()
    trainer.firewall.assert_after_backward()
    active = trainer.groups.active_parameters(trainer.spec.active_groups)
    torch.nn.utils.clip_grad_norm_(
        active, max_norm=trainer.spec.gradient_clip_norm, error_if_nonfinite=True
    )
    _synchronize()
    backward_clip_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    trainer.optimizer.step()
    trainer.firewall.assert_after_backward()
    _synchronize()
    optimizer_step_ms = (time.perf_counter() - started) * 1000.0
    loss_value = float(loss.total.detach().float().cpu())
    del output, loss
    return {
        "zero_grad_ms": zero_grad_ms,
        "forward_loss_ms": forward_loss_ms,
        "backward_clip_ms": backward_clip_ms,
        "optimizer_step_ms": optimizer_step_ms,
        "loss": loss_value,
    }


def _cleanup_cuda(*values: Any) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_case(
    *,
    config: Mapping[str, Any],
    corpus: WorldMoECorpus,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
    checkpointing: bool,
    tf32: bool,
    matmul_precision: str,
    drop_unused_target: bool,
    seed: int,
    sampler_interval_seconds: float,
) -> dict[str, Any]:
    case_id = (
        f"b{batch_size}-gc{int(checkpointing)}-tf32{int(tf32)}-"
        f"{matmul_precision}-drop{int(drop_unused_target)}"
    )
    result: dict[str, Any] = {
        "case_id": case_id,
        "status": "running",
        "batch_size": int(batch_size),
        "warmup_steps": int(warmup_steps),
        "measured_steps": int(measured_steps),
        "gradient_checkpointing": bool(checkpointing),
        "tf32": bool(tf32),
        "matmul_precision": matmul_precision,
        "drop_unused_w0_target": bool(drop_unused_target),
    }
    model = trainer = iterator = batch = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision(matmul_precision)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model_mapping = dict(config["model"])
        model_mapping["gradient_checkpointing"] = bool(checkpointing)
        model_config = WorldMoEConfig.from_mapping(model_mapping)
        model = VPPWorldMoE(model_config).to("cuda")
        spec = dataclasses.replace(
            phase_spec_from_config(config, "w0"), batch_size=int(batch_size)
        )
        trainer = WorldModelTrainer(model, spec, device="cuda")
        iterator = iter(
            corpus.batches(
                "train",
                batch_size,
                device="cuda",
                shuffle_seed=seed,
                max_days=batch_size * (warmup_steps + measured_steps),
                include_next_day=not drop_unused_target,
                require_h96=False,
            )
        )

        for index in range(warmup_steps):
            batch = next(iterator)
            if drop_unused_target:
                batch = strip_w0_unused_targets(batch)
            _timed_w0_step(trainer, batch, seed=seed + index)
            del batch
            batch = None

        _synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings: list[dict[str, float]] = []
        data_prep_ms: list[float] = []
        with NvidiaSmiSampler(sampler_interval_seconds) as sampler:
            for index in range(measured_steps):
                started = time.perf_counter()
                batch = next(iterator)
                if drop_unused_target:
                    batch = strip_w0_unused_targets(batch)
                _synchronize()
                data_prep_ms.append((time.perf_counter() - started) * 1000.0)
                timings.append(
                    _timed_w0_step(
                        trainer,
                        batch,
                        seed=seed + warmup_steps + index,
                    )
                )
                del batch
                batch = None

        component_means = {
            key: _mean([timing[key] for timing in timings])
            for key in (
                "zero_grad_ms",
                "forward_loss_ms",
                "backward_clip_ms",
                "optimizer_step_ms",
            )
        }
        compute_ms = sum(
            timing[key]
            for timing in timings
            for key in (
                "zero_grad_ms",
                "forward_loss_ms",
                "backward_clip_ms",
                "optimizer_step_ms",
            )
        )
        prep_ms = sum(data_prep_ms)
        samples = batch_size * measured_steps
        result.update(
            {
                "status": "ok",
                "data_prep_ms_mean": _mean(data_prep_ms),
                "timing_ms_mean": component_means,
                "compute_ms_total": compute_ms,
                "end_to_end_ms_total": compute_ms + prep_ms,
                "samples_per_second_compute": samples * 1000.0 / compute_ms,
                "samples_per_second_end_to_end": samples
                * 1000.0
                / (compute_ms + prep_ms),
                "loss_mean": _mean([timing["loss"] for timing in timings]),
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                "gpu": sampler.summary(),
            }
        )
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = (
            isinstance(exc, torch.OutOfMemoryError)
            or "out of memory" in str(exc).lower()
        )
        if not is_oom:
            raise
        result.update(
            {
                "status": "oom",
                "error": str(exc).splitlines()[0][:500],
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
            }
        )
    finally:
        del batch, iterator, trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _tree_transform(
    value: Any,
    tensor_transform: Callable[[torch.Tensor], torch.Tensor],
    memo: dict[int, Any],
) -> Any:
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        result = tensor_transform(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: _tree_transform(
                getattr(value, field.name), tensor_transform, memo
            )
            for field in dataclasses.fields(value)
        }
        result = dataclasses.replace(value, **changes)
    elif isinstance(value, Mapping):
        result = {
            key: _tree_transform(item, tensor_transform, memo)
            for key, item in value.items()
        }
    elif isinstance(value, tuple):
        result = tuple(_tree_transform(item, tensor_transform, memo) for item in value)
    elif isinstance(value, list):
        result = [_tree_transform(item, tensor_transform, memo) for item in value]
    else:
        result = value
    memo[identity] = result
    return result


def _unique_tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            _unique_tensor_bytes(getattr(value, field.name), visited)
            for field in dataclasses.fields(value)
        )
    if isinstance(value, Mapping):
        return sum(_unique_tensor_bytes(item, visited) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_unique_tensor_bytes(item, visited) for item in value)
    return 0


def _transfer_trials(
    batch: Any, *, non_blocking: bool, warmup_steps: int, measured_steps: int
) -> list[float]:
    values: list[float] = []
    for index in range(warmup_steps + measured_steps):
        started = time.perf_counter()
        copied = _tree_transform(
            batch,
            lambda tensor: tensor.to("cuda", non_blocking=non_blocking),
            {},
        )
        _synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmup_steps:
            values.append(elapsed)
        del copied
    torch.cuda.empty_cache()
    return values


def benchmark_pin_and_cache(
    *,
    corpus: WorldMoECorpus,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
    seed: int,
    drop_unused_target: bool,
) -> dict[str, Any]:
    """Measure CPU batch formation and pageable versus cached-pinned H2D copy."""

    started = time.perf_counter()
    cpu_batch = next(
        iter(
            corpus.batches(
                "train",
                batch_size,
                device="cpu",
                shuffle_seed=seed,
                max_days=batch_size,
                include_next_day=False,
                require_h96=False,
            )
        )
    )
    if drop_unused_target:
        cpu_batch = strip_w0_unused_targets(cpu_batch)
    cpu_prepare_ms = (time.perf_counter() - started) * 1000.0
    tensor_mib = _unique_tensor_bytes(cpu_batch) / MIB

    pageable = _transfer_trials(
        cpu_batch,
        non_blocking=False,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    started = time.perf_counter()
    pinned_batch = _tree_transform(cpu_batch, lambda tensor: tensor.pin_memory(), {})
    pin_setup_ms = (time.perf_counter() - started) * 1000.0
    pinned = _transfer_trials(
        pinned_batch,
        non_blocking=True,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    pageable_mean = _mean(pageable)
    pinned_mean = _mean(pinned)
    result = {
        "batch_size": batch_size,
        "drop_unused_w0_target": drop_unused_target,
        "cpu_batch_tensor_mib": tensor_mib,
        "cpu_prepare_ms_once": cpu_prepare_ms,
        "cached_batch_lookup_ms": 0.0,
        "pin_setup_ms_once": pin_setup_ms,
        "pageable_h2d_ms_mean": pageable_mean,
        "pinned_nonblocking_h2d_ms_mean": pinned_mean,
        "pinned_h2d_speedup": pageable_mean / pinned_mean if pinned_mean > 0 else 0.0,
        "note": (
            "cached lookup is descriptive only: reusing the same batch is invalid for "
            "training; the corpus already caches normalized day records in RAM"
        ),
    }
    del pinned_batch, cpu_batch
    _cleanup_cuda()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/vpp_world_moe_R5_v1.yaml")
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(8, 12, 16))
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20240813)
    parser.add_argument("--minimum-free-mib", type=float, default=1024.0)
    parser.add_argument("--sampler-interval-seconds", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the W0 throughput probe")
    if args.warmup_steps < 1 or args.measured_steps < 3:
        raise SystemExit("probe requires at least one warmup and three measured steps")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise SystemExit("all batch sizes must be positive")

    config, repository_root, _ = load_world_moe_config(args.config)
    corpus_started = time.perf_counter()
    corpus = WorldMoECorpus(
        config,
        repository_root,
        weather_coverage_gate=True,
        materialize=False,
    )
    corpus_seconds = time.perf_counter() - corpus_started
    print(
        json.dumps(
            {
                "event": "corpus_ready",
                "seconds": corpus_seconds,
                "weather_coverage": corpus.weather_coverage,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    baseline = NvidiaSmiSampler._sample()
    original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_tf32_cudnn = torch.backends.cudnn.allow_tf32
    original_precision = torch.get_float32_matmul_precision()

    results: list[dict[str, Any]] = []
    try:
        # Current runtime: checkpointing, strict FP32 matmul, and the unnecessary
        # W0 next-day packet.  This establishes a directly comparable baseline.
        for batch_size in args.batch_sizes:
            print(
                json.dumps({"event": "case_start", "case": f"legacy-b{batch_size}"}),
                flush=True,
            )
            result = benchmark_case(
                config=config,
                corpus=corpus,
                batch_size=batch_size,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                checkpointing=True,
                tf32=False,
                matmul_precision="highest",
                drop_unused_target=False,
                seed=args.seed,
                sampler_interval_seconds=args.sampler_interval_seconds,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        # Minimal optimized path: retain checkpointing, enable high/TF32, and
        # avoid target-state encoding that W0's registered loss never consumes.
        for batch_size in args.batch_sizes:
            print(
                json.dumps({"event": "case_start", "case": f"optimized-b{batch_size}"}),
                flush=True,
            )
            result = benchmark_case(
                config=config,
                corpus=corpus,
                batch_size=batch_size,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                checkpointing=True,
                tf32=True,
                matmul_precision="high",
                drop_unused_target=True,
                seed=args.seed,
                sampler_interval_seconds=args.sampler_interval_seconds,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        optimized = [
            result
            for result in results
            if result["tf32"] is True and result["drop_unused_w0_target"] is True
        ]
        safe_optimized = safe_results(optimized, minimum_free_mib=args.minimum_free_mib)
        candidate_batch = (
            max(int(result["batch_size"]) for result in safe_optimized)
            if safe_optimized
            else min(args.batch_sizes)
        )

        # At the best safe batch, isolate TF32/high from target stripping and
        # then test whether disabling activation checkpointing is actually safe.
        for checkpointing, tf32, precision in (
            (True, False, "highest"),
            (False, False, "highest"),
            (False, True, "high"),
        ):
            case_id = (
                f"b{candidate_batch}-gc{int(checkpointing)}-tf32{int(tf32)}-"
                f"{precision}-drop1"
            )
            if any(result["case_id"] == case_id for result in results):
                continue
            print(json.dumps({"event": "case_start", "case": case_id}), flush=True)
            result = benchmark_case(
                config=config,
                corpus=corpus,
                batch_size=candidate_batch,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                checkpointing=checkpointing,
                tf32=tf32,
                matmul_precision=precision,
                drop_unused_target=True,
                seed=args.seed,
                sampler_interval_seconds=args.sampler_interval_seconds,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        pin_cache = benchmark_pin_and_cache(
            corpus=corpus,
            batch_size=candidate_batch,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            seed=args.seed,
            drop_unused_target=True,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_tf32_cudnn
        torch.set_float32_matmul_precision(original_precision)
        _cleanup_cuda()

    payload = {
        "schema": "dcvpp.vpp-world-moe.w0-throughput-probe.v1",
        "formal_training_started": False,
        "runner_invoked": False,
        "checkpoint_written": False,
        "development_data_only": True,
        "sealed_2025_opened": False,
        "config": str(args.config.resolve()),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_mib": torch.cuda.get_device_properties(0).total_memory / MIB,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "bf16": torch.cuda.is_bf16_supported(),
            "baseline_utilization_percent": None if baseline is None else baseline[0],
            "baseline_used_mib": None if baseline is None else baseline[1],
            "baseline_free_mib": None if baseline is None else baseline[2],
        },
        "corpus_construction_seconds": corpus_seconds,
        "weather_coverage": corpus.weather_coverage,
        "probe": {
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "minimum_free_mib": args.minimum_free_mib,
        },
        "results": results,
        "recommendations": select_recommendations(
            results, minimum_free_mib=args.minimum_free_mib
        ),
        "pin_and_cache": pin_cache,
        "optimizer_update_budget": optimizer_update_budget(
            train_days=int(
                config["training"]["phases"]["w0"]["target_update_budgets"][
                    "train_examples"
                ]
            ),
            batch_sizes=args.batch_sizes,
            original_batch_size=int(
                config["training"]["phases"]["w0"]["target_update_budgets"][
                    "reference_batch_size"
                ]
            ),
            minimum_updates=int(
                config["training"]["phases"]["w0"]["target_update_budgets"]["minimum"]
            ),
            maximum_updates=int(
                config["training"]["phases"]["w0"]["target_update_budgets"]["maximum"]
            ),
            patience_updates=int(
                config["training"]["phases"]["w0"]["target_update_budgets"][
                    "patience_equivalent"
                ]
            ),
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
