from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from dcvpp.core import sha256_file
from dcvpp.experiments.forecast_calibration import quantile_metrics


QUANTILE_FIELDS = (
    "crps_quantile_approximation",
    "mean_pinball",
    "quantile_crossing_rate",
    "coverage_80",
    "coverage_90",
    "absolute_calibration_error",
    "mean_width_80",
    "mean_width_90",
)


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _tail_metrics(
    truth: np.ndarray, prediction: np.ndarray, thresholds: Iterable[float]
) -> dict[str, Any]:
    ordered = np.sort(np.asarray(prediction, dtype="float64"), axis=-1)
    upper = ordered[..., 4]
    threshold = np.asarray(tuple(thresholds), dtype="float64")[None, :, None]
    observed = np.asarray(truth) >= threshold
    predicted = upper >= threshold
    tp = int(np.sum(observed & predicted))
    fp = int(np.sum(~observed & predicted))
    fn = int(np.sum(observed & ~predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miss_rate": None if recall is None else 1.0 - recall,
    }


def _close(left: Any, right: Any, *, tolerance: float = 1e-5) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def audit_seed_round(
    repository_root: Path,
    *,
    run_id: str,
    round_index: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    run_root = root / "artifacts/real_forecasting_extensions" / f"run_id={run_id}"
    round_path = run_root / "manifests" / f"seed_round_{round_index}_{seed}.json"
    round_record = json.loads(round_path.read_text(encoding="utf-8"))
    if round_record.get("status") != "COMPLETE":
        raise ValueError("seed round is not COMPLETE")
    cells = list(round_record.get("cells", []))
    if len(cells) != int(round_record.get("expected_cell_count", -1)):
        raise ValueError("seed round cell count is incomplete")

    probability_rows: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    metric_recalculation_failures: list[str] = []
    round_manifest_metric_hash_mismatches: list[dict[str, str]] = []
    prediction_nonfinite_cells: list[str] = []
    checkpoint_nonfinite_cells: list[str] = []

    for cell in cells:
        task = str(cell["task"])
        variant = str(cell["variant"])
        identity = f"{task}/{variant}/seed_{seed}"
        metrics_path = Path(cell["metrics_artifact"]["path"])
        if not metrics_path.is_absolute():
            metrics_path = root / metrics_path
        current_metric_sha256 = sha256_file(metrics_path)
        if current_metric_sha256 != cell["metrics_artifact"]["sha256"]:
            round_manifest_metric_hash_mismatches.append(
                {
                    "cell": identity,
                    "round_manifest_sha256": str(cell["metrics_artifact"]["sha256"]),
                    "current_sha256": current_metric_sha256,
                }
            )
        metrics_record = json.loads(metrics_path.read_text(encoding="utf-8"))
        prediction_path = Path(metrics_record["prediction_artifact"]["path"])
        if not prediction_path.is_absolute():
            prediction_path = root / prediction_path
        if sha256_file(prediction_path) != metrics_record["prediction_artifact"]["sha256"]:
            raise ValueError(f"prediction hash mismatch: {identity}")
        with np.load(prediction_path, allow_pickle=False) as saved_prediction:
            arrays = {key: saved_prediction[key] for key in saved_prediction.files}
        predictions_finite = all(bool(np.isfinite(value).all()) for value in arrays.values())
        if not predictions_finite:
            prediction_nonfinite_cells.append(identity)

        if task in {"da_lbmp", "rt_lbmp"}:
            for split, prefix in (("validation", "validation"), ("locked_test", "locked_test")):
                prediction = arrays[f"{prefix}_prediction"]
                target = arrays[f"{prefix}_target"]
                recalculated = quantile_metrics(
                    target,
                    prediction,
                    quantiles=np.asarray([0.05, 0.10, 0.50, 0.90, 0.95]),
                )
                reported = metrics_record["metrics"][split]
                consistent = all(_close(recalculated[key], reported[key]) for key in QUANTILE_FIELDS)
                tail = _tail_metrics(target, prediction, reported["tail"]["thresholds_by_zone"])
                tail_consistent = all(
                    _close(tail[key], reported["tail"].get(key))
                    for key in ("precision", "recall", "f1")
                )
                if not consistent or not tail_consistent:
                    metric_recalculation_failures.append(f"{identity}/{split}")
                probability_rows.append(
                    {
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "split": split,
                        **{key: recalculated[key] for key in QUANTILE_FIELDS},
                        **tail,
                        "q99_thresholds_by_zone": json.dumps(
                            reported["tail"]["thresholds_by_zone"], separators=(",", ":")
                        ),
                        "metric_recalculation_consistent": consistent and tail_consistent,
                        "prediction_arrays_finite": predictions_finite,
                    }
                )
        else:
            for split, prefix in (("validation", "validation"), ("locked_test", "locked_test")):
                count_recalculated = quantile_metrics(
                    arrays[f"{prefix}_count_target"],
                    arrays[f"{prefix}_count_prediction"],
                    quantiles=np.asarray([0.05, 0.10, 0.50, 0.90, 0.95]),
                )
                reported_split = metrics_record["metrics"][split]
                count_reported = reported_split["count_quantiles"]
                count_consistent = all(
                    _close(count_recalculated[key], count_reported[key])
                    for key in QUANTILE_FIELDS
                )
                volume_recalculated: dict[str, Any] | None = None
                volume_consistent: bool | None = None
                if f"{prefix}_volume_prediction" in arrays:
                    volume_recalculated = quantile_metrics(
                        arrays[f"{prefix}_volume_target"],
                        arrays[f"{prefix}_volume_prediction"],
                        quantiles=np.asarray([0.05, 0.10, 0.50, 0.90, 0.95]),
                    )
                    volume_reported = reported_split["volume_quantiles"]
                    volume_consistent = all(
                        _close(volume_recalculated[key], volume_reported[key])
                        for key in QUANTILE_FIELDS
                    )
                if not count_consistent or volume_consistent is False:
                    metric_recalculation_failures.append(f"{identity}/{split}")
                distribution = reported_split["count_distribution"]
                workload_rows.append(
                    {
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "split": split,
                        "count_distribution": distribution.get("distribution"),
                        "negative_binomial_nll": distribution.get("negative_binomial_nll"),
                        "negative_binomial_deviance": distribution.get(
                            "negative_binomial_deviance"
                        ),
                        "poisson_nll": distribution.get("poisson_nll"),
                        **{f"count_{key}": count_recalculated[key] for key in QUANTILE_FIELDS},
                        **(
                            {f"volume_{key}": volume_recalculated[key] for key in QUANTILE_FIELDS}
                            if volume_recalculated is not None
                            else {f"volume_{key}": None for key in QUANTILE_FIELDS}
                        ),
                        "volume_status": (
                            "AVAILABLE" if volume_recalculated is not None else "DISABLED_BY_ABLATION"
                        ),
                        "metric_recalculation_consistent": count_consistent
                        and volume_consistent is not False,
                        "prediction_arrays_finite": predictions_finite,
                    }
                )

        checkpoint_path = Path(metrics_record["checkpoint"]["path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        checkpoint_hash_match = sha256_file(checkpoint_path) == metrics_record["checkpoint"]["sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state_finite = _all_finite(checkpoint.get("model_state", {}))
        best_state_finite = _all_finite(checkpoint.get("best_model_state", {}))
        optimizer_state_finite = _all_finite(checkpoint.get("optimizer_state", {}))
        if not (model_state_finite and best_state_finite and optimizer_state_finite):
            checkpoint_nonfinite_cells.append(identity)
        epoch = int(checkpoint.get("epoch", 0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        health_rows.append(
            {
                "task": task,
                "variant": variant,
                "seed": seed,
                "completed": bool(checkpoint.get("completed")),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "early_stopped": bool(checkpoint.get("completed")) and epoch < 100,
                "early_stop_patience_consistent": epoch == 100 or epoch - best_epoch >= 10,
                "checkpoint_hash_match": checkpoint_hash_match,
                "model_state_finite": model_state_finite,
                "best_state_finite": best_state_finite,
                "optimizer_state_finite": optimizer_state_finite,
                "prediction_arrays_finite": predictions_finite,
                "gradient_norm_history_status": "NOT_RECORDED_IN_ROUND_1",
                "abnormal_gradient_status": "NOT_EVALUABLE_FROM_SAVED_CHECKPOINT",
                "runtime_seconds_status": "NOT_RECORDED_IN_ROUND_1",
                "peak_gpu_memory_status": "NOT_RECORDED_IN_ROUND_1",
                "checkpoint_bytes": checkpoint_path.stat().st_size,
            }
        )
        del checkpoint

    audit_root = run_root / "audits" / f"seed_round_{round_index}_{seed}"
    probability_path = audit_root / "probability_calibration_tail.csv"
    workload_path = audit_root / "workload_uncertainty.csv"
    health_path = audit_root / "model_health.csv"
    _atomic_csv(probability_path, probability_rows)
    _atomic_csv(workload_path, workload_rows)
    _atomic_csv(health_path, health_rows)
    outputs = {
        "probability_calibration_tail": {
            "path": probability_path.relative_to(root).as_posix(),
            "sha256": sha256_file(probability_path),
            "row_count": len(probability_rows),
        },
        "workload_uncertainty": {
            "path": workload_path.relative_to(root).as_posix(),
            "sha256": sha256_file(workload_path),
            "row_count": len(workload_rows),
        },
        "model_health": {
            "path": health_path.relative_to(root).as_posix(),
            "sha256": sha256_file(health_path),
            "row_count": len(health_rows),
        },
    }
    completed = not (
        metric_recalculation_failures
        or prediction_nonfinite_cells
        or checkpoint_nonfinite_cells
        or any(not bool(row["checkpoint_hash_match"]) for row in health_rows)
        or any(not bool(row["completed"]) for row in health_rows)
    )
    manifest: dict[str, Any] = {
        "schema": "forecast_extension_seed_round_audit_v1",
        "status": "COMPLETE_WITH_DISCLOSED_LIMITATIONS" if completed else "FAILED",
        "completed": completed,
        "run_id": run_id,
        "round_index": round_index,
        "seed": seed,
        "round_manifest": {
            "path": round_path.relative_to(root).as_posix(),
            "sha256": sha256_file(round_path),
            "cell_count": len(cells),
        },
        "counts": {
            "cells": len(cells),
            "energy_metric_rows": len(probability_rows),
            "workload_metric_rows": len(workload_rows),
            "health_rows": len(health_rows),
            "metric_recalculation_failures": len(metric_recalculation_failures),
            "prediction_nonfinite_cells": len(prediction_nonfinite_cells),
            "checkpoint_nonfinite_cells": len(checkpoint_nonfinite_cells),
            "round_manifest_metric_hash_mismatches": len(
                round_manifest_metric_hash_mismatches
            ),
        },
        "failures": {
            "metric_recalculation": metric_recalculation_failures,
            "prediction_nonfinite": prediction_nonfinite_cells,
            "checkpoint_nonfinite": checkpoint_nonfinite_cells,
            "round_manifest_metric_hash_mismatches": round_manifest_metric_hash_mismatches,
        },
        "limitations": [
            "Round 1 did not persist per-step gradient norms; abnormal-gradient history is not retrospectively inferable.",
            "Round 1 did not persist per-model wall time or peak CUDA allocation; these fields are not fabricated.",
            "The post-round CUDA-graph benchmark is runtime evidence for future rounds, not a relabeling of Round-1 utilization.",
            "A stopped resume pass may have deterministically reserialized some metric JSON files; every stale round-manifest hash is retained and disclosed rather than overwritten.",
        ],
        "outputs": outputs,
    }
    manifest_path = audit_root / "audit_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.relative_to(root).as_posix()
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-id", default="paper-forecast-extensions-v1")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260809)
    arguments = parser.parse_args()
    result = audit_seed_round(
        arguments.root,
        run_id=arguments.run_id,
        round_index=arguments.round,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
