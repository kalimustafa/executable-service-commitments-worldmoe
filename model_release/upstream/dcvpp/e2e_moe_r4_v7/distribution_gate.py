"""Formal D0 capability materialization and PASS-only checkpoint selection.

This module deliberately keeps the paper gate separate from the optimization
loss.  A D0 checkpoint is compared day-by-day with the frozen train-residual
baseline on an explicit development-validation panel.  The default remains the
legacy 179-day 2023-H2 panel.  Selection uses the same registered one-sided
7-day circular moving-block bootstrap as the older R4 Distribution experiment;
point MAE is neither computed nor consulted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from dcvpp.e2e_moe_r4.hashing import sha256_file
from dcvpp.e2e_moe_r4.losses import (
    empirical_coverage,
    weighted_energy_score,
    weighted_interval_score,
    weighted_marginal_crps,
    weighted_quantile,
    weighted_variogram_score,
)
from dcvpp.e2e_moe_r4.statistics import paired_daily_inference
from dcvpp.e2e_moe_r4.training_data import PreloadedR4Corpus

from .data import FrozenResidualAnchor, issue_center_from_batch


GATE_SCHEMA = "dcvpp.r4.v7.d0-distribution-capability-gate.v2"
SELECTION_SCHEMA = "dcvpp.r4.v7.d0-distribution-checkpoint-selection.v1"
CANDIDATE_REGISTRY_SCHEMA = "dcvpp.r4.v7.d0-candidate-registry.v1"
BASELINE_NAME = "frozen_training_residual_sampling_variable_n"
EXPECTED_DAYS = 179
DEFAULT_VALIDATION_START = date(2023, 7, 1)
DEFAULT_VALIDATION_END = date(2023, 12, 31)
FORMAL_SAMPLES = 256
BLOCK_DAYS = 7
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
NONINFERIORITY_MARGIN = 0.01
IDENTITY_TOLERANCE = 1.0e-6


class D0DistributionGateError(RuntimeError):
    """Raised when formal D0 evidence or checkpoint lineage is incomplete."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _finite_vector(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    try:
        result = np.asarray([row[field] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise D0DistributionGateError(f"invalid daily field {field!r}") from exc
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise D0DistributionGateError(
            f"daily field {field!r} must contain one finite scalar per day"
        )
    return result


def _parse_validation_date(value: str | date, *, name: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise D0DistributionGateError(f"{name} must be an ISO date") from exc


def _validation_contract(
    *,
    validation_start: str | date,
    validation_end: str | date,
    expected_days: int,
) -> dict[str, Any]:
    start = _parse_validation_date(validation_start, name="validation_start")
    end = _parse_validation_date(validation_end, name="validation_end")
    days = int(expected_days)
    if start > end:
        raise D0DistributionGateError("validation_start must not exceed validation_end")
    if days < 1:
        raise D0DistributionGateError("expected_days must be positive")
    locked_start = date(2025, 1, 1)
    locked_end = date(2025, 12, 31)
    if start <= locked_end and end >= locked_start:
        raise D0DistributionGateError(
            "locked 2025 cannot enter D0 development validation"
        )
    return {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "expected_days": days,
    }


def _ordered_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_days: int,
    validation_start: str | date,
    validation_end: str | date,
) -> tuple[dict[str, Any], ...]:
    contract = _validation_contract(
        validation_start=validation_start,
        validation_end=validation_end,
        expected_days=expected_days,
    )
    start = date.fromisoformat(str(contract["requested_start"]))
    end = date.fromisoformat(str(contract["requested_end"]))
    copied = tuple(dict(row) for row in rows)
    if len(copied) != int(contract["expected_days"]):
        raise D0DistributionGateError(
            f"D0 gate requires exactly {expected_days} validation days; got {len(copied)}"
        )
    parsed: list[date] = []
    for row in copied:
        try:
            current = date.fromisoformat(str(row["date"])[:10])
        except (KeyError, ValueError) as exc:
            raise D0DistributionGateError("every D0 row requires one ISO date") from exc
        if current.year == 2025:
            raise D0DistributionGateError(
                "locked 2025 cannot enter D0 development validation"
            )
        if not start <= current <= end:
            raise D0DistributionGateError(
                "D0 checkpoint selection date "
                f"{current} falls outside {start} through {end}"
            )
        parsed.append(current)
    if len(set(parsed)) != len(parsed):
        raise D0DistributionGateError("D0 validation dates must be unique")
    order = np.argsort(np.asarray(parsed, dtype="datetime64[D]"), kind="stable")
    return tuple(copied[int(index)] for index in order)


def _inference(
    values: np.ndarray,
    *,
    seed: int,
    block_days: int,
    replications: int,
    confidence: float,
) -> dict[str, float | int]:
    result = paired_daily_inference(
        values,
        block_days=int(block_days),
        replicates=int(replications),
        confidence=float(confidence),
        seed=int(seed),
    )
    return result.as_dict()


def evaluate_d0_distribution_gate(
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = EXPECTED_DAYS,
    validation_start: str | date = DEFAULT_VALIDATION_START,
    validation_end: str | date = DEFAULT_VALIDATION_END,
    formal_samples: int = FORMAL_SAMPLES,
    block_days: int = BLOCK_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 20240813,
    noninferiority_margin: float = NONINFERIORITY_MARGIN,
    identity_tolerance: float = IDENTITY_TOLERANCE,
) -> dict[str, Any]:
    """Apply the paper Distribution gate to already-materialized daily rows.

    Negative paired differences favour V7.  Secondary proper scores are
    tested after subtracting one percent of the paired baseline loss, exactly
    matching the old R4 non-inferiority construction.
    """

    contract = _validation_contract(
        validation_start=validation_start,
        validation_end=validation_end,
        expected_days=expected_days,
    )
    rows = _ordered_rows(
        daily_rows,
        expected_days=int(expected_days),
        validation_start=validation_start,
        validation_end=validation_end,
    )
    reasons: list[str] = []
    dates = tuple(str(row["date"])[:10] for row in rows)
    validation_window = {
        **contract,
        "actual_start": dates[0],
        "actual_end": dates[-1],
        "actual_days": len(rows),
    }
    if {str(row.get("baseline_name", "")) for row in rows} != {BASELINE_NAME}:
        reasons.append("formal frozen train-residual baseline was not used on every day")
    if any(int(row.get("formal_probability_sample_count", -1)) != int(formal_samples) for row in rows):
        reasons.append(f"formal probability comparison did not use N={formal_samples}")
    if any(row.get("locked_test_2025_opened") is not False for row in rows):
        reasons.append("locked 2025 status is not closed")
    if any(row.get("all_values_finite") is not True for row in rows):
        reasons.append("one or more D0 daily rows contain non-finite evidence")

    metric_vectors: dict[str, dict[str, np.ndarray]] = {}
    for metric in (
        "energy_score",
        "marginal_crps",
        "interval_score",
        "interval_score_90",
        "interval_score_95",
        "tail_score",
        "dependence_score",
    ):
        metric_vectors[metric] = {
            "model": _finite_vector(rows, f"model_{metric}"),
            "baseline": _finite_vector(rows, f"baseline_{metric}"),
        }

    energy_delta = (
        metric_vectors["energy_score"]["model"]
        - metric_vectors["energy_score"]["baseline"]
    )
    energy = _inference(
        energy_delta,
        seed=seed,
        block_days=block_days,
        replications=replications,
        confidence=confidence,
    )
    if float(energy["one_sided_upper"]) >= 0.0:
        reasons.append("energy_score_paired_one_sided_upper_not_strictly_below_zero")

    noninferiority: dict[str, dict[str, float | int]] = {}
    secondary_names = (
        "marginal_crps",
        "interval_score_90",
        "interval_score_95",
        "tail_score",
        "dependence_score",
    )
    for offset, metric in enumerate(secondary_names, start=1):
        model_values = metric_vectors[metric]["model"]
        baseline_values = metric_vectors[metric]["baseline"]
        adjusted = (
            model_values
            - baseline_values
            - float(noninferiority_margin) * np.abs(baseline_values)
        )
        inference = _inference(
            adjusted,
            seed=seed + offset,
            block_days=block_days,
            replications=replications,
            confidence=confidence,
        )
        noninferiority[metric] = inference
        if float(inference["one_sided_upper"]) > 0.0:
            reasons.append(f"{metric}_exceeds_1pct_paired_noninferiority_margin")

    coverage90 = float(_finite_vector(rows, "model_coverage_90").mean())
    coverage95 = float(_finite_vector(rows, "model_coverage_95").mean())
    coverage90_error = abs(coverage90 - 0.90)
    coverage95_error = abs(coverage95 - 0.95)
    if coverage90_error > 0.03:
        reasons.append("coverage90_absolute_error_exceeds_0.03")
    if coverage95_error > 0.03:
        reasons.append("coverage95_absolute_error_exceeds_0.03")
    model_identity = float(_finite_vector(rows, "model_identity_residual").max())
    baseline_identity = float(
        _finite_vector(rows, "baseline_identity_residual").max()
    )
    identity = max(model_identity, baseline_identity)
    if identity > float(identity_tolerance):
        reasons.append("da_rt_dart_identity_exceeds_1e-6")

    means: dict[str, float] = {}
    ratios: dict[str, float] = {}
    for metric, values in metric_vectors.items():
        model_mean = float(values["model"].mean())
        baseline_mean = float(values["baseline"].mean())
        means[f"model_{metric}"] = model_mean
        means[f"baseline_{metric}"] = baseline_mean
        ratios[f"{metric}_ratio"] = (
            model_mean / baseline_mean if baseline_mean > 0.0 else math.inf
        )
    metrics: dict[str, Any] = {
        **means,
        **ratios,
        "energy_score_delta_inference": energy,
        "energy_score_delta_one_sided_upper": float(energy["one_sided_upper"]),
        "secondary_noninferiority_inference": noninferiority,
        "marginal_crps_noninferiority_one_sided_upper": float(
            noninferiority["marginal_crps"]["one_sided_upper"]
        ),
        "interval_score_noninferiority_one_sided_upper": float(
            max(
                noninferiority["interval_score_90"]["one_sided_upper"],
                noninferiority["interval_score_95"]["one_sided_upper"],
            )
        ),
        "tail_score_noninferiority_one_sided_upper": float(
            noninferiority["tail_score"]["one_sided_upper"]
        ),
        "dependence_score_noninferiority_one_sided_upper": float(
            noninferiority["dependence_score"]["one_sided_upper"]
        ),
        "model_coverage_90": coverage90,
        "model_coverage_95": coverage95,
        "absolute_coverage_error_90": coverage90_error,
        "absolute_coverage_error_95": coverage95_error,
        "identity_residual_max_abs": identity,
        "formal_probability_sample_count": int(formal_samples),
        "validation_days": len(rows),
    }
    return {
        "schema": GATE_SCHEMA,
        "phase": "d0",
        "status": "PASS" if not reasons else "FAIL",
        "passed": not reasons,
        "dates": dates,
        "validation_window": validation_window,
        "metrics": metrics,
        "reasons": tuple(reasons),
        "bootstrap": {
            "method": "paired_circular_moving_block",
            "block_days": int(block_days),
            "replications": int(replications),
            "confidence": float(confidence),
            "seed": int(seed),
        },
        "noninferiority_margin_fraction": float(noninferiority_margin),
        "baseline_name": BASELINE_NAME,
        "mae_used": False,
        "locked_test_2025_opened": False,
    }


def _registered_indices(
    dates: Sequence[str], sample_count: int, seed: int, *, pool_size: int
) -> Tensor:
    if int(sample_count) < 1:
        raise D0DistributionGateError("formal sample_count must be positive")
    if int(pool_size) < int(sample_count):
        raise D0DistributionGateError(
            "frozen residual pool is smaller than the formal sample count: "
            f"{pool_size} < {sample_count}"
        )
    rows: list[Tensor] = []
    for operating_date in dates:
        digest = hashlib.sha256(
            f"r4-v7-d0|{int(seed)}|{operating_date}|common-formal-paths".encode()
        ).digest()
        path_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        generator = torch.Generator(device="cpu").manual_seed(path_seed)
        rows.append(
            torch.randperm(int(pool_size), generator=generator)[: int(sample_count)]
        )
    return torch.stack(rows)


def _close_identity(samples: Tensor) -> Tensor:
    da = samples[..., 0:4]
    rt = samples[..., 4:8]
    return torch.cat((da, rt, da - rt, samples[..., 12:]), dim=-1)


def _tail_score(samples: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    values: list[Tensor] = []
    for quantile in (0.05, 0.95):
        forecast = weighted_quantile(samples, weights, quantile)
        residual = target - forecast
        values.append(
            torch.maximum(
                quantile * residual, (quantile - 1.0) * residual
            ).flatten(1).mean(1)
        )
    return torch.stack(values, dim=1).mean(1)


def _formal_metric_vectors(
    samples: Tensor, weights: Tensor, target: Tensor
) -> dict[str, Tensor]:
    coverage90, _ = empirical_coverage(samples, weights, target, 0.90)
    coverage95, _ = empirical_coverage(samples, weights, target, 0.95)
    interval90 = weighted_interval_score(samples, weights, target, 0.90)
    interval95 = weighted_interval_score(samples, weights, target, 0.95)
    return {
        "energy_score": weighted_energy_score(samples, weights, target),
        "marginal_crps": weighted_marginal_crps(samples, weights, target),
        "interval_score": 0.5 * (interval90 + interval95),
        "interval_score_90": interval90,
        "interval_score_95": interval95,
        "tail_score": _tail_score(samples, weights, target),
        # Formal evaluation uses the exact registered score, never the stable
        # differentiable continuation used by training.
        "dependence_score": weighted_variogram_score(
            samples, weights, target, lags=(1, 4, 16), power=0.5
        ),
        "coverage_90": coverage90,
        "coverage_95": coverage95,
    }


def materialize_d0_distribution_candidate(
    *,
    model: nn.Module,
    corpus: PreloadedR4Corpus,
    anchor: FrozenResidualAnchor,
    checkpoint: str | Path,
    candidate_cycle: int,
    output_path: str | Path,
    device: torch.device,
    seed: int = 20240813,
    batch_size: int = 32,
    formal_samples: int = FORMAL_SAMPLES,
    replications: int = BOOTSTRAP_REPLICATIONS,
    baseline_sha256: str | None = None,
    validation_start: str | date = DEFAULT_VALIDATION_START,
    validation_end: str | date = DEFAULT_VALIDATION_END,
    expected_days: int = EXPECTED_DAYS,
) -> dict[str, Any]:
    """Materialize the N=256 paired daily panel for one immutable candidate."""

    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise D0DistributionGateError(f"D0 candidate checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise D0DistributionGateError("D0 candidate checkpoint must be a mapping")
    if payload.get("phase") != "d0" or int(payload.get("seed", -1)) != int(seed):
        raise D0DistributionGateError("D0 candidate phase/seed lineage is invalid")
    if int(payload.get("cycles_completed", -1)) != int(candidate_cycle):
        raise D0DistributionGateError("D0 candidate registry/cycle mismatch")
    if payload.get("locked_2025_opened") is not False:
        raise D0DistributionGateError("D0 candidate reports that locked 2025 was opened")
    model.load_state_dict(payload["model_state"], strict=True)
    checkpoint_sha = sha256_file(source)
    baseline_sha = baseline_sha256 or sha256_file(anchor.source_path)
    residuals = anchor.residuals.to(device=device, dtype=torch.float32)
    residual_pool_size = int(anchor.residuals.shape[0])
    channel_scale = anchor.channel_scale.to(device=device, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in corpus.batches(
                "validation", int(batch_size), device=device, origin=0, shuffle_seed=None
            ):
                dates = tuple(str(value)[:10] for value in batch["operating_dates"])
                indices_cpu = _registered_indices(
                    dates,
                    int(formal_samples),
                    int(seed),
                    pool_size=residual_pool_size,
                )
                indices = indices_cpu.to(device=device)
                output = model.forward_distribution(
                    batch,
                    sample_count=int(formal_samples),
                    base_indices=indices,
                )
                distribution = output.get("distribution")
                if not isinstance(distribution, Mapping):
                    raise D0DistributionGateError("V7 Distribution output is malformed")
                samples = torch.as_tensor(
                    distribution.get("samples"), device=device, dtype=torch.float32
                )
                weights = torch.as_tensor(
                    distribution.get("weights"), device=device, dtype=torch.float32
                )
                expected = (
                    len(dates), int(formal_samples), 96, 21
                )
                if samples.shape != expected or weights.shape != expected[:2]:
                    raise D0DistributionGateError(
                        f"formal V7 samples/weights must be {expected} and {expected[:2]}"
                    )
                if not bool(torch.equal(distribution["base_indices"], indices)):
                    raise D0DistributionGateError("model did not use registered common base paths")
                if not bool(torch.isfinite(samples).all() and torch.isfinite(weights).all()):
                    raise D0DistributionGateError("formal V7 samples/weights are non-finite")
                mass = weights.sum(1)
                if bool((weights < 0).any()) or not bool(
                    torch.allclose(mass, torch.ones_like(mass), rtol=0.0, atol=1.0e-6)
                ):
                    raise D0DistributionGateError("formal V7 weights are invalid")
                target = torch.as_tensor(
                    batch["scenario_target_zonal"], device=device, dtype=torch.float32
                )
                baseline_center = issue_center_from_batch(batch, anchor).to(
                    device=device, dtype=torch.float32
                )
                baseline = _close_identity(
                    baseline_center[:, None] + residuals[indices]
                )
                model_identity = (
                    samples[..., 0:4] - samples[..., 4:8] - samples[..., 8:12]
                ).abs().flatten(1).amax(1)
                baseline_identity = (
                    baseline[..., 0:4] - baseline[..., 4:8] - baseline[..., 8:12]
                ).abs().flatten(1).amax(1)
                scale = channel_scale.view(1, 1, 1, 21)
                normalized_target = target / channel_scale.view(1, 1, 21)
                normalized_model = samples / scale
                normalized_baseline = baseline / scale
                baseline_weights = torch.full_like(
                    weights, 1.0 / float(formal_samples)
                )
                model_metrics = _formal_metric_vectors(
                    normalized_model, weights, normalized_target
                )
                baseline_metrics = _formal_metric_vectors(
                    normalized_baseline, baseline_weights, normalized_target
                )
                for local, operating_date in enumerate(dates):
                    row: dict[str, Any] = {
                        "date": operating_date,
                        "baseline_name": BASELINE_NAME,
                        "formal_probability_sample_count": int(formal_samples),
                        "common_registered_base_paths": True,
                        "model_identity_residual": float(model_identity[local].cpu()),
                        "baseline_identity_residual": float(
                            baseline_identity[local].cpu()
                        ),
                        "locked_test_2025_opened": False,
                    }
                    for name, values in model_metrics.items():
                        row[f"model_{name}"] = float(values[local].cpu())
                    for name in (
                        "energy_score",
                        "marginal_crps",
                        "interval_score",
                        "interval_score_90",
                        "interval_score_95",
                        "tail_score",
                        "dependence_score",
                    ):
                        row[f"baseline_{name}"] = float(
                            baseline_metrics[name][local].cpu()
                        )
                    numeric = [
                        value
                        for value in row.values()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    ]
                    row["all_values_finite"] = bool(
                        np.isfinite(np.asarray(numeric, dtype=np.float64)).all()
                    )
                    rows.append(row)
    finally:
        model.train(was_training)

    gate = evaluate_d0_distribution_gate(
        rows,
        expected_days=int(expected_days),
        validation_start=validation_start,
        validation_end=validation_end,
        formal_samples=int(formal_samples),
        replications=int(replications),
        seed=int(seed),
    )
    artifact = {
        "schema": GATE_SCHEMA,
        "phase": "d0",
        "candidate_cycle": int(candidate_cycle),
        "checkpoint": str(source),
        "checkpoint_sha256": checkpoint_sha,
        "baseline": str(anchor.source_path),
        "baseline_sha256": baseline_sha,
        "validation_window": dict(gate["validation_window"]),
        "validation_days": len(rows),
        "formal_probability_sample_count": int(formal_samples),
        "metric_space": "frozen_train_residual_channel_normalized_joint_space",
        "formal_score_definitions": {
            "energy_score": "weighted_joint_trajectory_energy_score",
            "marginal_crps": "weighted_empirical_marginal_crps",
            "interval_score": "mean_of_exact_90_and_95_central_interval_scores",
            "interval_score_90": "exact_90_percent_central_interval_score",
            "interval_score_95": "exact_95_percent_central_interval_score",
            "tail_score": "mean_5_and_95_percent_pinball_loss",
            "dependence_score": "exact_temporal_variogram_lags_1_4_16_power_0.5",
        },
        "baseline_fairness": "same_dates_same_N256_same_base_indices_same_score_definitions",
        "daily_rows": rows,
        "gate": gate,
        "mae_used": False,
        "locked_test_2025_opened": False,
    }
    destination = Path(output_path).resolve()
    _atomic_json(destination, artifact)
    return artifact


def register_d0_candidate(
    registry_path: str | Path,
    *,
    cycle: int,
    checkpoint: str | Path,
    cheap_validation: float,
    cheap_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Append one immutable cheap-improvement candidate without re-hashing it."""

    path = Path(registry_path).resolve()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"schema": CANDIDATE_REGISTRY_SCHEMA, "phase": "d0", "candidates": []}
    if payload.get("schema") != CANDIDATE_REGISTRY_SCHEMA:
        raise D0DistributionGateError("D0 candidate registry schema is invalid")
    candidates = list(payload.get("candidates", ()))
    if any(int(item.get("cycle", -1)) == int(cycle) for item in candidates):
        return payload
    candidates.append(
        {
            "cycle": int(cycle),
            "checkpoint": str(Path(checkpoint).resolve()),
            "cheap_validation": float(cheap_validation),
            "cheap_metrics": {str(key): float(value) for key, value in cheap_metrics.items()},
            "gate_status": "PENDING",
        }
    )
    payload["candidates"] = sorted(candidates, key=lambda item: int(item["cycle"]))
    _atomic_json(path, payload)
    return payload


def _checkpoint_candidate(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("phase") != "d0":
        raise D0DistributionGateError(f"not a D0 checkpoint: {path}")
    return {
        "cycle": int(payload.get("cycles_completed", -1)),
        "checkpoint": str(path.resolve()),
        "cheap_validation": float(payload.get("cheap_validation_loss", math.inf)),
        "cheap_metrics": dict(payload.get("cheap_validation_metrics", {})),
        "gate_status": "PENDING",
    }


def discover_d0_candidates(phase_root: str | Path) -> list[dict[str, Any]]:
    root = Path(phase_root).resolve()
    registry_path = root / "capability" / "d0_candidates.json"
    candidates: list[dict[str, Any]] = []
    if registry_path.is_file():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if payload.get("schema") != CANDIDATE_REGISTRY_SCHEMA:
            raise D0DistributionGateError("D0 candidate registry schema is invalid")
        candidates.extend(dict(item) for item in payload.get("candidates", ()))
    for path in (
        root / "checkpoints" / "best_cheap.pt",
        root / "checkpoints" / "latest.pt",
    ):
        if path.is_file():
            candidates.append(_checkpoint_candidate(path))
    by_cycle: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        cycle = int(candidate.get("cycle", -1))
        source = Path(str(candidate.get("checkpoint", ""))).resolve()
        if cycle < 1 or not source.is_file():
            continue
        previous = by_cycle.get(cycle)
        if previous is None or float(candidate.get("cheap_validation", math.inf)) < float(
            previous.get("cheap_validation", math.inf)
        ):
            by_cycle[cycle] = candidate
    return sorted(by_cycle.values(), key=lambda item: int(item["cycle"]))


Materializer = Callable[..., Mapping[str, Any]]


def _matches_validation_contract(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> bool:
    window = payload.get("validation_window")
    if not isinstance(window, Mapping):
        return False
    return all(window.get(name) == contract[name] for name in contract)


def _bounded_pareto_candidates(
    candidates: Sequence[Mapping[str, Any]], *, top_k: int
) -> list[dict[str, Any]]:
    """Keep a cheap-metric Pareto frontier, then fill to the bounded top-k."""

    copied = [dict(item) for item in candidates]
    metric_names = (
        "raw_energy_score",
        "raw_marginal_crps",
        "raw_coverage95_error",
        "raw_tail",
        "raw_dependence",
    )

    def cheap_rank(item: Mapping[str, Any]) -> tuple[float, int]:
        return (
            float(item.get("cheap_validation", math.inf)),
            int(item.get("cycle", 10**9)),
        )

    vectors: dict[int, tuple[float, ...]] = {}
    for index, item in enumerate(copied):
        metrics = item.get("cheap_metrics")
        if not isinstance(metrics, Mapping):
            continue
        try:
            vector = tuple(float(metrics[name]) for name in metric_names)
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in vector):
            vectors[index] = vector
    frontier: list[int] = []
    for index, vector in vectors.items():
        dominated = any(
            other_index != index
            and all(left <= right for left, right in zip(other, vector, strict=True))
            and any(left < right for left, right in zip(other, vector, strict=True))
            for other_index, other in vectors.items()
        )
        if not dominated:
            frontier.append(index)
    selected_indices = sorted(frontier, key=lambda index: cheap_rank(copied[index]))[
        : int(top_k)
    ]
    for index in sorted(range(len(copied)), key=lambda value: cheap_rank(copied[value])):
        if len(selected_indices) >= int(top_k):
            break
        if index not in selected_indices:
            selected_indices.append(index)
    return [copied[index] for index in selected_indices]


def gate_and_select_d0_distribution(
    *,
    model: nn.Module,
    corpus: PreloadedR4Corpus,
    anchor: FrozenResidualAnchor,
    phase_root: str | Path,
    device: torch.device,
    seed: int,
    top_k: int = 5,
    candidate_records: Sequence[Mapping[str, Any]] | None = None,
    materializer: Materializer = materialize_d0_distribution_candidate,
    validation_start: str | date = DEFAULT_VALIDATION_START,
    validation_end: str | date = DEFAULT_VALIDATION_END,
    expected_days: int = EXPECTED_DAYS,
) -> dict[str, Any]:
    """Full-gate at most five candidates and copy only a passing winner."""

    root = Path(phase_root).resolve()
    validation_contract = _validation_contract(
        validation_start=validation_start,
        validation_end=validation_end,
        expected_days=expected_days,
    )
    selection_path = root / "capability" / "d0_distribution_selection.json"
    if selection_path.is_file():
        prior = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = prior.get("selected_checkpoint")
        if (
            prior.get("schema") == SELECTION_SCHEMA
            and prior.get("gate_schema") == GATE_SCHEMA
            and prior.get("status") == "PASS"
            and _matches_validation_contract(prior, validation_contract)
            and selected
            and Path(str(selected)).is_file()
        ):
            return prior

    candidates = [
        dict(item)
        for item in (
            discover_d0_candidates(root) if candidate_records is None else candidate_records
        )
    ]
    if not candidates:
        raise D0DistributionGateError("D0 produced no checkpoint candidate")
    bounded = _bounded_pareto_candidates(candidates, top_k=int(top_k))
    baseline_sha = sha256_file(anchor.source_path)
    evaluated: list[dict[str, Any]] = []
    for candidate in bounded:
        cycle = int(candidate["cycle"])
        checkpoint = Path(str(candidate["checkpoint"])).resolve()
        gate_path = root / "capability" / "candidates" / f"cycle={cycle:04d}.json"
        artifact: Mapping[str, Any]
        if gate_path.is_file():
            existing = json.loads(gate_path.read_text(encoding="utf-8"))
            if (
                existing.get("schema") == GATE_SCHEMA
                and
                existing.get("checkpoint") == str(checkpoint)
                and existing.get("checkpoint_sha256") == sha256_file(checkpoint)
                and isinstance(existing.get("gate"), Mapping)
                and _matches_validation_contract(existing, validation_contract)
            ):
                artifact = existing
            else:
                artifact = materializer(
                    model=model,
                    corpus=corpus,
                    anchor=anchor,
                    checkpoint=checkpoint,
                    candidate_cycle=cycle,
                    output_path=gate_path,
                    device=device,
                    seed=seed,
                    baseline_sha256=baseline_sha,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    expected_days=int(expected_days),
                )
        else:
            artifact = materializer(
                model=model,
                corpus=corpus,
                anchor=anchor,
                checkpoint=checkpoint,
                candidate_cycle=cycle,
                output_path=gate_path,
                device=device,
                seed=seed,
                baseline_sha256=baseline_sha,
                validation_start=validation_start,
                validation_end=validation_end,
                expected_days=int(expected_days),
            )
        gate = artifact.get("gate")
        if not isinstance(gate, Mapping):
            raise D0DistributionGateError("candidate materializer omitted formal gate")
        reported_candidate = dict(candidate)
        cheap_value = float(reported_candidate.get("cheap_validation", math.inf))
        if not math.isfinite(cheap_value):
            # Latest checkpoints retained for recovery may not have their own
            # cheap score.  JSON evidence records that absence as null rather
            # than emitting a non-standard Infinity token.
            reported_candidate["cheap_validation"] = None
        evaluated.append(
            {
                **reported_candidate,
                "gate_status": str(gate.get("status", "FAIL")),
                "gate_artifact": str(gate_path),
                "checkpoint_sha256": str(artifact.get("checkpoint_sha256", "")),
                "gate": dict(gate),
            }
        )

    passing = [item for item in evaluated if item["gate_status"] == "PASS"]

    def formal_rank(item: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
        metrics = item["gate"]["metrics"]
        return (
            float(metrics["energy_score_delta_one_sided_upper"]),
            float(metrics["model_marginal_crps"]),
            float(metrics["absolute_coverage_error_95"]),
            float(metrics["tail_score_noninferiority_one_sided_upper"]),
            int(item["cycle"]),
        )

    selected_checkpoint: str | None = None
    selected_cycle: int | None = None
    selected_gate_artifact: str | None = None
    selected_sha: str | None = None
    if passing:
        selected = min(passing, key=formal_rank)
        source = Path(str(selected["checkpoint"])).resolve()
        destination = root / "checkpoints" / "d0_selected_best.pt"
        _atomic_copy(source, destination)
        selected_checkpoint = str(destination)
        selected_cycle = int(selected["cycle"])
        selected_gate_artifact = str(selected["gate_artifact"])
        selected_sha = sha256_file(destination)
    evidence_window: dict[str, Any] = {
        **validation_contract,
        "actual_start": None,
        "actual_end": None,
        "actual_days": None,
    }
    if evaluated:
        candidate_window = evaluated[0]["gate"].get("validation_window")
        if isinstance(candidate_window, Mapping):
            evidence_window.update(dict(candidate_window))
    artifact = {
        "schema": SELECTION_SCHEMA,
        "gate_schema": GATE_SCHEMA,
        "phase": "d0",
        "seed": int(seed),
        "validation_window": evidence_window,
        "validation_days": evidence_window["actual_days"],
        "status": "PASS" if passing else "FAIL",
        "passed": bool(passing),
        "candidate_limit": int(top_k),
        "candidate_count_evaluated": len(evaluated),
        "candidate_prefilter": (
            "cheap_metric_pareto_frontier_then_lowest_cheap_validation_fill"
        ),
        "candidates": evaluated,
        "selection_rule": [
            "PASS_only",
            "lowest_energy_score_delta_one_sided_upper",
            "lowest_model_marginal_crps",
            "lowest_absolute_coverage_error_95",
            "lowest_tail_score_noninferiority_one_sided_upper",
            "earliest_cycle",
        ],
        "selected_cycle": selected_cycle,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": selected_sha,
        "selected_gate_artifact": selected_gate_artifact,
        "p0_release": bool(passing),
        "mae_used": False,
        "locked_test_2025_opened": False,
        "locked_2025_opened": False,
    }
    _atomic_json(selection_path, artifact)
    return artifact


def require_d0_selected_checkpoint(
    phase_root: str | Path, *, seed: int
) -> tuple[Path, dict[str, Any]]:
    """Fail closed unless D0 has a machine-readable PASS selection artifact."""

    root = Path(phase_root).resolve()
    selection_path = root / "capability" / "d0_distribution_selection.json"
    if not selection_path.is_file():
        raise D0DistributionGateError("P0 is blocked: D0 formal gate has not run")
    artifact = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != SELECTION_SCHEMA
        or artifact.get("gate_schema") != GATE_SCHEMA
        or artifact.get("phase") != "d0"
        or int(artifact.get("seed", -1)) != int(seed)
        or artifact.get("status") != "PASS"
        or artifact.get("p0_release") is not True
        or artifact.get("mae_used") is not False
        or artifact.get("locked_test_2025_opened") is not False
        or artifact.get("locked_2025_opened") is not False
    ):
        raise D0DistributionGateError("P0 is blocked: D0 formal gate did not pass")
    selected = Path(str(artifact.get("selected_checkpoint", ""))).resolve()
    if not selected.is_file():
        raise D0DistributionGateError("P0 is blocked: selected D0 checkpoint is missing")
    expected_sha = str(artifact.get("selected_checkpoint_sha256", ""))
    if sha256_file(selected) != expected_sha:
        raise D0DistributionGateError("P0 is blocked: selected D0 checkpoint changed")
    return selected, artifact


__all__ = [
    "BASELINE_NAME",
    "BOOTSTRAP_REPLICATIONS",
    "D0DistributionGateError",
    "DEFAULT_VALIDATION_END",
    "DEFAULT_VALIDATION_START",
    "EXPECTED_DAYS",
    "FORMAL_SAMPLES",
    "GATE_SCHEMA",
    "SELECTION_SCHEMA",
    "discover_d0_candidates",
    "evaluate_d0_distribution_gate",
    "gate_and_select_d0_distribution",
    "materialize_d0_distribution_candidate",
    "register_d0_candidate",
    "require_d0_selected_checkpoint",
]
