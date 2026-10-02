"""Live, checkpoint-bound capability evidence for formal R4 phases.

The materializers in this module run only after an immutable phase checkpoint
exists.  They measure raw predictions on all 179 validation dates, persist one
row per date, and then delegate acceptance to :mod:`capability_evaluation`.
Training loss is never used as capability evidence.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .artifact_io import atomic_json
from .capability_evaluation import build_capability_evaluator
from .hashing import sha256_file, sha256_json
from .training_data import PreloadedR4Corpus


HEADS = ("dart", "da", "rt", "workload", "pv", "pue")


def _target(record: Mapping[str, Any], head: str) -> Tensor:
    targets = record.get("targets")
    if not isinstance(targets, Mapping) or not isinstance(targets.get(head), Tensor):
        raise ValueError(f"R4 record is missing tensor target {head!r}")
    return targets[head].detach().cpu().float()


def _daily_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.detach().cpu().float()
    target = target.detach().cpu().float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )
    error = prediction - target
    scale = float(target.abs().mean().clamp_min(1.0e-6))
    mae = float(error.abs().mean())
    rmse = float(error.square().mean().sqrt())
    return {"mae": mae, "rmse": rmse, "nmae": mae / scale, "nrmse": rmse / scale}


def _issue_time(record: Mapping[str, Any]) -> datetime:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("R4 record is missing issue-time provenance")
    value = str(provenance.get("issue_time_utc", "")).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("issue_time_utc must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _source_completed_before_issue(
    source: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    operating = date.fromisoformat(str(source["operating_date"]))
    completed = datetime.combine(
        operating + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    return completed <= _issue_time(current)


def _baseline_definitions(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Tensor]]], Mapping[str, Any]]:
    """Freeze causal persistence and same-weekday seasonal-naive predictions."""

    ordered = sorted(records, key=lambda row: str(row["operating_date"]))
    by_date = {str(row["operating_date"]): row for row in ordered}
    dates = list(by_date)
    predictions: dict[str, dict[str, dict[str, Tensor]]] = {
        "causal_persistence": {},
        "seasonal_naive": {},
    }
    history: list[Mapping[str, Any]] = []
    for day_text in dates:
        row = by_date[day_text]
        safe = [item for item in history if _source_completed_before_issue(item, row)]
        if not safe:
            history.append(row)
            continue
        weekday = date.fromisoformat(day_text).weekday()
        persistence_source = safe[-1]
        same_weekday = [
            item for item in safe
            if date.fromisoformat(str(item["operating_date"])).weekday() == weekday
        ]
        seasonal_source = same_weekday[-1] if same_weekday else persistence_source
        for name, source in (
            ("causal_persistence", persistence_source),
            ("seasonal_naive", seasonal_source),
        ):
            predictions[name][day_text] = {
                head: _target(source, head) for head in HEADS
            }
        history.append(row)
    definitions = {
        "schema_version": "dcvpp.r4.representation_baselines.v1",
        "fit_split": "train",
        "causal_persistence": (
            "latest eligible operating date fully completed no later than issue_time_utc"
        ),
        "seasonal_naive": (
            "latest same-weekday eligible operating date fully completed no later than issue_time_utc"
        ),
        "undefined_warmup": "excluded from train-only baseline selection",
        "heads": list(HEADS),
    }
    return predictions, definitions


def _select_representation_baseline(
    train_records: Sequence[Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]]:
    predictions, definitions = _baseline_definitions(train_records)
    scores: dict[str, float] = {}
    # Exclude the first warm-up date, whose causal predecessor is undefined.
    for name, values in predictions.items():
        daily: list[float] = []
        for row in sorted(train_records, key=lambda item: str(item["operating_date"])):
            day_text = str(row["operating_date"])
            if day_text not in values:
                continue
            target = _target(row, "dart")
            prediction = values[day_text]["dart"]
            daily.append(_daily_metrics(prediction, target)["nmae"])
        if not daily:
            raise ValueError(f"no causal train dates exist for baseline {name}")
        scores[name] = sum(daily) / len(daily)
    selected = min(scores, key=lambda name: (scores[name], name))
    payload = {**definitions, "selection_metric": "train_mean_dart_nmae", "scores": scores,
               "selected": selected}
    return selected, payload


def _validation_baselines(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Tensor]]]:
    """Return head-wise baselines while allowing only earlier realized dates."""

    ordered = sorted(
        [*train_records, *validation_records], key=lambda row: str(row["operating_date"])
    )
    history: list[Mapping[str, Any]] = []
    result: dict[str, dict[str, Mapping[str, Tensor]]] = {}
    validation_dates = {str(row["operating_date"]) for row in validation_records}
    for row in ordered:
        day_text = str(row["operating_date"])
        weekday = date.fromisoformat(day_text).weekday()
        if day_text in validation_dates:
            safe = [item for item in history if _source_completed_before_issue(item, row)]
            if not safe:
                raise ValueError("validation baseline has no earlier eligible day")
            persistence = safe[-1]
            same_weekday = [
                item for item in safe
                if date.fromisoformat(str(item["operating_date"])).weekday() == weekday
            ]
            seasonal = same_weekday[-1] if same_weekday else persistence
            result[day_text] = {
                "causal_persistence": {head: _target(persistence, head) for head in HEADS},
                "seasonal_naive": {head: _target(seasonal, head) for head in HEADS},
            }
        history.append(row)
    return result


class LiveRepresentationCapabilityEvaluator:
    """Materialize and validate raw-unit representation evidence on demand."""

    def __init__(
        self,
        corpus: PreloadedR4Corpus,
        phase_directory: str | Path,
        *,
        seed: int,
        batch_size: int = 179,
    ) -> None:
        self.corpus = corpus
        self.phase_directory = Path(phase_directory)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        if not 1 <= self.batch_size <= 179:
            raise ValueError("representation capability batch_size must be 1..179")
        self.selected_baseline, self.baseline_payload = _select_representation_baseline(
            corpus.records("train")
        )
        baseline_path = self.phase_directory.parent.parent / "data" / "representation_baselines.json"
        atomic_json(baseline_path, dict(self.baseline_payload))
        self.baseline_sha256 = sha256_file(baseline_path)
        self.definition_hashes = {
            name: sha256_json({"name": name, **dict(self.baseline_payload)})
            for name in ("causal_persistence", "seasonal_naive")
        }
        self.validation_baselines = _validation_baselines(
            corpus.records("train"), corpus.records("validation")
        )
        self.validation_records = {
            str(row["operating_date"]): row
            for row in corpus.records("validation")
        }

    def __call__(
        self,
        *,
        phase: str,
        horizon: int | None,
        checkpoint: str | None = None,
        checkpoint_path: str | None = None,
        epoch: int | None = None,
        model: nn.Module,
        **_: Any,
    ) -> Mapping[str, Any]:
        # The trainer supplies K=96 for every 179-day phase capability gate,
        # even though representation itself is not a rolling decision phase.
        if phase != "representation" or horizon not in (None, 96):
            raise ValueError("representation capability evaluator received wrong phase/horizon")
        source = Path(checkpoint_path or checkpoint or "").resolve()
        if not source.is_file() or epoch is None:
            raise ValueError("representation gate requires an immutable checkpoint and epoch")
        checkpoint_sha256 = sha256_file(source)
        evidence_path = (
            self.phase_directory / "capability" / checkpoint_sha256 /
            "representation_daily_v2.json"
        )
        device = next(model.parameters()).device
        rows: list[dict[str, Any]] = []
        # Older test/evidence adapters constructed the evaluator without
        # calling __init__.  Retain their exact B=1 behavior while real formal
        # construction uses the explicit default B=179 above.
        evaluation_batch_size = int(getattr(self, "batch_size", 1))
        model.eval()
        with torch.inference_mode():
            for batch in self.corpus.batches(
                "validation", evaluation_batch_size, device=device, origin=0
            ):
                output = model(batch)
                forecasts = output.get("forecasts") if isinstance(output, Mapping) else None
                if not isinstance(forecasts, Mapping):
                    raise ValueError("R4 model did not return named forecasts")
                target_counterfactual = {
                    head: torch.as_tensor(value).clone()
                    for head, value in batch["targets"].items()
                }
                target_counterfactual["da"] = target_counterfactual["da"] + 37.0
                target_counterfactual["dart"] = target_counterfactual["dart"] - 19.0
                target_counterfactual["rt"] = (
                    target_counterfactual["da"] - target_counterfactual["dart"]
                )
                target_counterfactual["workload"] = (
                    target_counterfactual["workload"] + 3.0
                ).clamp_min(1.0e-6)
                target_counterfactual["pv"] = (
                    target_counterfactual["pv"] + 1.0
                ).clamp_min(1.0e-6)
                target_counterfactual["pue"] = (
                    target_counterfactual["pue"] + 0.05
                ).clamp_min(1.0e-6)
                counterfactual_batch = dict(batch)
                counterfactual_batch["targets"] = target_counterfactual
                if "scenario_target_zonal" in batch:
                    counterfactual_batch["scenario_target_zonal"] = torch.cat(
                        (
                            target_counterfactual["da"],
                            target_counterfactual["rt"],
                            target_counterfactual["dart"],
                            target_counterfactual["workload"],
                            target_counterfactual["pv"],
                            target_counterfactual["pue"],
                        ),
                        dim=-1,
                    )
                counterfactual_output = model(counterfactual_batch)
                counterfactual_forecasts = (
                    counterfactual_output.get("forecasts")
                    if isinstance(counterfactual_output, Mapping)
                    else None
                )
                if not isinstance(counterfactual_forecasts, Mapping):
                    raise ValueError("R4 model did not return counterfactual forecasts")
                leakage_residual = torch.zeros(
                    len(batch["operating_dates"]), device=device
                )
                for head in HEADS:
                    left = torch.as_tensor(forecasts[head], device=device)
                    right = torch.as_tensor(counterfactual_forecasts[head], device=device)
                    if left.shape != right.shape:
                        raise ValueError(
                            f"representation counterfactual shape changed for {head}"
                        )
                    leakage_residual = torch.maximum(
                        leakage_residual, (left - right).abs().flatten(1).amax(1)
                    )
                for index, raw_date in enumerate(batch["operating_dates"]):
                    day_text = str(raw_date)
                    validation_records = getattr(self, "validation_records", None)
                    if isinstance(validation_records, Mapping):
                        record = validation_records[day_text]
                    else:
                        record = {
                            "targets": {
                                head: value[index].detach().cpu()
                                for head, value in batch["targets"].items()
                            }
                        }
                    baselines = self.validation_baselines[day_text][
                        self.selected_baseline
                    ]
                    row: dict[str, Any] = {
                        "date": day_text,
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_epoch": int(epoch),
                        "baseline_name": self.selected_baseline,
                        "baseline_sha256": self.baseline_sha256,
                        "causal_persistence_sha256": self.definition_hashes[
                            "causal_persistence"
                        ],
                        "seasonal_naive_sha256": self.definition_hashes[
                            "seasonal_naive"
                        ],
                        "causal_no_future_leakage_passed": bool(
                            float(leakage_residual[index]) <= 1.0e-6
                        ),
                        "future_leakage_audit_status": (
                            "PASSED"
                            if float(leakage_residual[index]) <= 1.0e-6
                            else "FAILED"
                        ),
                        "future_leakage_forecast_invariance_max_abs": float(
                            leakage_residual[index]
                        ),
                    }
                    for head in HEADS:
                        target = _target(record, head)
                        prediction = forecasts[head][index].detach().cpu()
                        baseline = baselines[head]
                        for name, value in _daily_metrics(prediction, target).items():
                            row[f"model_{head}_{name}"] = value
                        for name, value in _daily_metrics(baseline, target).items():
                            row[f"baseline_{head}_{name}"] = value
                    identity = (
                        forecasts["da"][index]
                        - forecasts["rt"][index]
                        - forecasts["dart"][index]
                    )
                    row["da_rt_dart_identity_max_abs"] = float(
                        identity.abs().max().cpu()
                    )
                    row["evidence_sha256"] = sha256_json(
                        {
                            key: value
                            for key, value in row.items()
                            if key != "evidence_sha256"
                        }
                    )
                    rows.append(row)
        payload = {
            "schema_version": "dcvpp.r4.representation_capability_daily.v2",
            "phase": phase,
            "epoch": int(epoch),
            "checkpoint": str(source),
            "checkpoint_sha256": checkpoint_sha256,
            "baseline": self.selected_baseline,
            "leakage_audit_days": len(rows),
            "all_validation_dates_leakage_audited": bool(
                len(rows) == 179
                and all(row["causal_no_future_leakage_passed"] for row in rows)
            ),
            "future_leakage_audit_status": (
                "PASSED"
                if len(rows) == 179
                and all(row["causal_no_future_leakage_passed"] for row in rows)
                else "FAILED"
            ),
            "evaluation_batch_size": evaluation_batch_size,
            "rows": rows,
        }
        atomic_json(evidence_path, payload)
        evaluator = build_capability_evaluator(
            phase, rows, baseline=self.selected_baseline, seed=self.seed
        )
        return evaluator(phase=phase, horizon=horizon)


__all__ = ["LiveRepresentationCapabilityEvaluator"]
