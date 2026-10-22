"""Checkpoint-bound R0 Representation capability materialization.

This module evaluates exactly the 364 eligible 2024 development-validation
days.  It deliberately keeps the paper's original Representation comparison:
the stronger of causal persistence and same-weekday seasonal naive is selected
using train-only mean DART NMAE, and every output is then tested with paired
daily NMAE and a one-sided 95% seven-day moving-block bootstrap.

MAE and RMSE are retained as descriptive raw-unit evidence only.  They are
never supplied to the scientific gate.  The locked 2025 final test is neither
constructed nor read here.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from dcvpp.e2e_moe_r4.capability_materialization import (
    _daily_metrics as _paper_daily_metrics,
    _select_representation_baseline as _paper_select_baseline,
    _validation_baselines as _paper_validation_baselines,
)
from dcvpp.e2e_moe_r4.hashing import sha256_file, sha256_json

from .config import ARCHITECTURE_VERSION, load_v8_config
from .corpus import PreloadedV8Corpus, V8ResidualAnchor, load_v8_residual_anchor
from .gates import evaluate_representation_gate
from .model import TriPlaneCausalWorldModelV8


R0_EVIDENCE_SCHEMA = "dcvpp.r4.v8.r0-representation-daily-evidence.v1"
R0_GATE_SCHEMA = "dcvpp.r4.v8.r0-representation-selection.v1"
EXPECTED_TRAIN_DAYS = 722
EXPECTED_VALIDATION_DAYS = 364
VALIDATION_START = "2024-01-01"
VALIDATION_END = "2024-12-31"
DEFAULT_BATCH_SIZE = 181
BOOTSTRAP_REPLICATIONS = 10_000
HEADS = ("dart", "da", "rt", "workload", "pv", "pue")
HEAD_SLICES = {
    "da": slice(0, 4),
    "rt": slice(4, 8),
    "dart": slice(8, 12),
    "workload": slice(12, 13),
    "pv": slice(13, 17),
    "pue": slice(17, 21),
}


class R0EvaluationError(RuntimeError):
    """R0 evidence cannot be materialized without weakening its lineage."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _synchronize_phase_state(
    phase_directory: Path,
    *,
    gate_path: Path,
    passed: bool,
    selected_checkpoint: str | None,
) -> None:
    """Attach the completed capability decision without replacing train state."""

    state_path = phase_directory / "state.json"
    if not state_path.is_file():
        return
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R0EvaluationError(f"cannot update malformed R0 state: {state_path}") from exc
    if not isinstance(raw, Mapping):
        raise R0EvaluationError("R0 state.json must contain a JSON object")
    status = "PASS" if passed else "FAIL"
    validation = raw.get("validation")
    synchronized_validation = {
        **(dict(validation) if isinstance(validation, Mapping) else {}),
        "formal_364_day_gate_executed": True,
        "formal_364_day_gate_status": status,
        "formal_364_day_gate_days": EXPECTED_VALIDATION_DAYS,
    }
    updated = {
        **dict(raw),
        "status": "CAPABILITY_GATE_PASSED" if passed else "CAPABILITY_GATE_FAILED",
        "capability_gate_executed": True,
        "capability_gate_status": status,
        "capability_gate": str(gate_path),
        "next_phase_released": bool(passed),
        "selected_checkpoint": selected_checkpoint if passed else None,
    }
    updated["validation"] = synchronized_validation
    _atomic_json(state_path, updated)


def _date_text(record: Mapping[str, Any]) -> str:
    value = str(record.get("operating_date", ""))[:10]
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise R0EvaluationError("corpus record has no valid operating_date") from exc
    return value


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_days: int,
) -> tuple[Mapping[str, Any], ...]:
    ordered = tuple(sorted(records, key=_date_text))
    if len(ordered) != expected_days:
        raise R0EvaluationError(
            f"R0 {split} requires exactly {expected_days} records; got {len(ordered)}"
        )
    dates = tuple(_date_text(record) for record in ordered)
    if len(set(dates)) != expected_days:
        raise R0EvaluationError(f"R0 {split} dates must be unique")
    years = {date.fromisoformat(value).year for value in dates}
    expected_years = {2022, 2023} if split == "train" else {2024}
    if not years.issubset(expected_years) or (split == "validation" and years != {2024}):
        raise R0EvaluationError(
            f"R0 {split} may use only {sorted(expected_years)}; got {sorted(years)}"
        )
    if any(value.startswith("2025-") for value in dates):
        raise R0EvaluationError("locked 2025 cannot enter R0 materialization")
    return ordered


def _load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    seed: int,
) -> tuple[Path, Mapping[str, Any], str]:
    source = Path(checkpoint_path).resolve()
    if source.name != "best.pt" or not source.is_file():
        raise R0EvaluationError(f"R0 requires an existing best.pt checkpoint: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise R0EvaluationError("R0 best checkpoint is not a mapping")
    if (
        payload.get("architecture_version") != ARCHITECTURE_VERSION
        or str(payload.get("phase", "")).lower() != "r0"
        or int(payload.get("seed", -1)) != int(seed)
        or payload.get("locked_2025_opened") is not False
    ):
        raise R0EvaluationError("R0 best checkpoint lineage is incompatible")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise R0EvaluationError("R0 best checkpoint has no model_state")
    model.load_state_dict(state, strict=True)
    return source, payload, sha256_file(source)


def _point_forecast(output: Any) -> Tensor:
    if not isinstance(output, Mapping):
        raise R0EvaluationError("forward_representation must return a mapping")
    packet = output.get("representation")
    forecast = (
        packet.get("point_forecast")
        if isinstance(packet, Mapping)
        else getattr(packet, "point_forecast", None)
    )
    if not isinstance(forecast, Tensor) or forecast.ndim != 3 or forecast.shape[1:] != (96, 21):
        shape = None if not isinstance(forecast, Tensor) else tuple(forecast.shape)
        raise R0EvaluationError(
            f"Representation point_forecast must be [batch,96,21]; got {shape}"
        )
    if not bool(torch.isfinite(forecast).all()):
        raise R0EvaluationError("Representation point_forecast contains non-finite values")
    return forecast


def _target_from_record(record: Mapping[str, Any], head: str) -> Tensor:
    targets = record.get("targets")
    value = targets.get(head) if isinstance(targets, Mapping) else None
    if not isinstance(value, Tensor):
        raise R0EvaluationError(f"record is missing targets.{head}")
    return value.detach().cpu().float()


def _autocast_disabled(device: torch.device) -> Any:
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def materialize_r0_checkpoint(
    *,
    model: nn.Module,
    corpus: Any,
    checkpoint_path: str | Path,
    phase_root: str | Path,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable R0 best checkpoint and atomically publish its gate.

    ``corpus`` is expected to be a :class:`PreloadedV8Corpus`; the structural
    protocol checks intentionally remain duck-typed so deterministic 364-day
    test fixtures can exercise the complete materializer.
    """

    size = int(batch_size)
    if not 1 <= size <= EXPECTED_VALIDATION_DAYS:
        raise R0EvaluationError("R0 evaluation batch_size must lie in [1,364]")
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_records = _validate_records(
        tuple(corpus.records("train")), split="train", expected_days=EXPECTED_TRAIN_DAYS
    )
    validation_records = _validate_records(
        tuple(corpus.records("validation")),
        split="validation",
        expected_days=EXPECTED_VALIDATION_DAYS,
    )
    validation_by_date = {_date_text(record): record for record in validation_records}

    source, checkpoint, checkpoint_sha256 = _load_checkpoint(model, checkpoint_path, seed=seed)
    model.to(device=target_device, dtype=torch.float32)
    model.eval()
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise R0EvaluationError("R0 neural evaluation must use FP32 parameters")

    # These are the exact old-paper helpers: selection is global and based
    # only on train mean DART NMAE; validation baselines use only completed
    # history available before each issue time.
    selected_baseline, baseline_selection = _paper_select_baseline(train_records)
    all_validation_baselines = _paper_validation_baselines(
        train_records, validation_records
    )
    baseline_sha256 = sha256_json(dict(baseline_selection))
    baseline_definition_hashes = {
        name: sha256_json({"name": name, **dict(baseline_selection)})
        for name in ("causal_persistence", "seasonal_naive")
    }

    rows: list[dict[str, Any]] = []
    batch_sizes: list[int] = []
    with torch.inference_mode(), _autocast_disabled(target_device):
        for batch in corpus.batches(
            "validation", size, device=target_device, shuffle_seed=None, origin=0
        ):
            dates = tuple(str(value)[:10] for value in batch.get("operating_dates", ()))
            if not dates:
                raise R0EvaluationError("validation batch omitted operating_dates")
            batch_sizes.append(len(dates))
            output = model.forward_representation(
                batch, origin=0, scientific_score_dtype=torch.float32
            )
            forecasts = _point_forecast(output).detach().cpu().float()
            if forecasts.shape[0] != len(dates):
                raise R0EvaluationError("forecast batch axis does not match operating_dates")
            for index, day_text in enumerate(dates):
                record = validation_by_date.get(day_text)
                if record is None:
                    raise R0EvaluationError(f"batch returned an unregistered validation day: {day_text}")
                baselines = all_validation_baselines[day_text][selected_baseline]
                public_prediction = forecasts[index]
                row: dict[str, Any] = {
                    "date": day_text,
                    "checkpoint": str(source),
                    "checkpoint_id": checkpoint_sha256,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_cycle": checkpoint.get("cycles_completed"),
                    "best_cycle": checkpoint.get("best_cycle"),
                    "baseline_name": selected_baseline,
                    "baseline_sha256": baseline_sha256,
                    "causal_persistence_sha256": baseline_definition_hashes[
                        "causal_persistence"
                    ],
                    "seasonal_naive_sha256": baseline_definition_hashes[
                        "seasonal_naive"
                    ],
                    "future_leakage_counterfactual_status": "NOT_RUN",
                    "future_leakage_audit_status": "NOT_RUN",
                    "locked_2025_opened": False,
                }
                for head in HEADS:
                    target = _target_from_record(record, head)
                    prediction = public_prediction[..., HEAD_SLICES[head]]
                    baseline = torch.as_tensor(baselines[head]).detach().cpu().float()
                    model_metrics = _paper_daily_metrics(prediction, target)
                    baseline_metrics = _paper_daily_metrics(baseline, target)
                    for metric, value in model_metrics.items():
                        row[f"model_{head}_{metric}"] = float(value)
                    for metric, value in baseline_metrics.items():
                        row[f"baseline_{head}_{metric}"] = float(value)
                    # The V8 gate deliberately receives only NMAE.  Raw-unit
                    # MAE/RMSE and descriptive NRMSE stay in evidence fields.
                    row[f"model_{head}_loss"] = float(model_metrics["nmae"])
                    row[f"baseline_{head}_loss"] = float(baseline_metrics["nmae"])
                predicted_identity = (
                    public_prediction[..., HEAD_SLICES["da"]]
                    - public_prediction[..., HEAD_SLICES["rt"]]
                    - public_prediction[..., HEAD_SLICES["dart"]]
                )
                target_identity = (
                    _target_from_record(record, "da")
                    - _target_from_record(record, "rt")
                    - _target_from_record(record, "dart")
                )
                row["da_rt_dart_identity_max_abs"] = float(
                    predicted_identity.abs().max().item()
                )
                row["target_da_rt_dart_identity_max_abs"] = float(
                    target_identity.abs().max().item()
                )
                row["evidence_sha256"] = sha256_json(row)
                rows.append(row)

    ordered_rows = sorted(rows, key=lambda row: str(row["date"]))
    if (
        len(ordered_rows) != EXPECTED_VALIDATION_DAYS
        or len({str(row["date"]) for row in ordered_rows}) != EXPECTED_VALIDATION_DAYS
        or tuple(str(row["date"]) for row in ordered_rows)
        != tuple(_date_text(record) for record in validation_records)
    ):
        raise R0EvaluationError("R0 inference did not cover each validation day exactly once")

    phase_directory = Path(phase_root).resolve()
    evidence_path = phase_directory / "capability" / "daily_evidence.json"
    evidence_payload = {
        "schema": R0_EVIDENCE_SCHEMA,
        "phase": "r0",
        "seed": int(seed),
        "checkpoint": str(source),
        "checkpoint_id": checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "validation_days": len(ordered_rows),
        "evaluation_batch_size": size,
        "observed_batch_sizes": batch_sizes,
        "evaluation_dtype": "float32",
        "metric_contract": {
            "descriptive": ["raw_unit_mae", "raw_unit_rmse", "daily_nmae", "daily_nrmse"],
            "gate_loss": "daily_nmae_only",
            "daily_normalizer": "target_abs_mean_clamped_at_1e-6",
            "mae_used_by_gate": False,
        },
        "baseline_selection": dict(baseline_selection),
        "baseline_sha256": baseline_sha256,
        "future_leakage_counterfactual_status": "NOT_RUN",
        "future_leakage_counterfactual_reason": (
            "target-perturbation counterfactual intentionally not executed"
        ),
        "locked_2025_opened": False,
        "rows": ordered_rows,
    }
    _atomic_json(evidence_path, evidence_payload)

    scientific = evaluate_representation_gate(
        ordered_rows,
        outputs=HEADS,
        expected_days=EXPECTED_VALIDATION_DAYS,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        block_days=7,
        replications=BOOTSTRAP_REPLICATIONS,
        confidence=0.95,
        seed=int(seed),
    )
    scientific_payload = scientific.as_dict()
    passed = bool(scientific.passed)
    gate_path = phase_directory / "gate.json"
    gate_payload = {
        "schema": R0_GATE_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "phase": "r0",
        "seed": int(seed),
        "selected_checkpoint": str(source) if passed else None,
        "checkpoint": str(source),
        "checkpoint_id": checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "next_phase": "d0",
        "next_phase_released": passed,
        "daily_evidence": str(evidence_path),
        "daily_evidence_sha256": sha256_file(evidence_path),
        "future_leakage_counterfactual_status": "NOT_RUN",
        "locked_2025_opened": False,
        "scientific_gate": scientific_payload,
    }
    _atomic_json(gate_path, gate_payload)
    _synchronize_phase_state(
        phase_directory,
        gate_path=gate_path,
        passed=passed,
        selected_checkpoint=str(source) if passed else None,
    )
    return {
        **gate_payload,
        "gate_path": str(gate_path),
        "daily_evidence_path": str(evidence_path),
    }


def materialize_r0_representation_capability(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | str | None = None,
    seed: int | None = None,
    corpus_factory: Callable[[Mapping[str, Any], Path], Any] = PreloadedV8Corpus,
    model_factory: Callable[..., nn.Module] = TriPlaneCausalWorldModelV8,
    anchor_loader: Callable[[str | Path], V8ResidualAnchor] = load_v8_residual_anchor,
) -> dict[str, Any]:
    """Load registered V8 inputs and materialize the formal R0 capability gate."""

    config, repository_root, paths = load_v8_config(config_path)
    active_seed = int(config["protocol"]["active_seed"] if seed is None else seed)
    registered_seeds = tuple(int(value) for value in config["protocol"]["seeds"])
    if active_seed not in registered_seeds:
        raise R0EvaluationError("R0 evaluation seed is not registered in the V8 protocol")
    phase_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else paths["output_root"] / f"seed={active_seed}" / "phase=r0"
    )
    source = (
        Path(checkpoint_path).resolve()
        if checkpoint_path is not None
        else phase_root / "checkpoints" / "best.pt"
    )
    anchor = anchor_loader(paths["residual_anchor"])
    if anchor.metadata.get("final_test_2025_opened") is not False:
        raise R0EvaluationError("V8 residual anchor does not prove 2025 stayed sealed")
    corpus = corpus_factory(config, repository_root)
    physical_core = getattr(corpus, "physical_core", None)
    if not isinstance(physical_core, nn.Module):
        raise R0EvaluationError("formal V8 corpus did not expose its frozen physical core")
    model = model_factory(
        anchor,
        config,
        physical_twin_core=physical_core,
    )
    return materialize_r0_checkpoint(
        model=model,
        corpus=corpus,
        checkpoint_path=source,
        phase_root=phase_root,
        seed=active_seed,
        batch_size=batch_size,
        device=device,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the 364-day V8 R0 Representation scientific gate"
    )
    parser.add_argument(
        "--config",
        default="configs/e2e_moe_dart_vpp_R4_trifusion_v8.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_r0_representation_capability(
        args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase": result["phase"],
                "gate_path": result["gate_path"],
                "daily_evidence_path": result["daily_evidence_path"],
                "next_phase_released": result["next_phase_released"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised through public CLI
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_REPLICATIONS",
    "DEFAULT_BATCH_SIZE",
    "EXPECTED_VALIDATION_DAYS",
    "R0EvaluationError",
    "materialize_r0_checkpoint",
    "materialize_r0_representation_capability",
]
