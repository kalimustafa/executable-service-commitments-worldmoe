from __future__ import annotations

"""Frozen formal-program orchestration for solver-free E2E-MoE R3.

The protocol layer deliberately keeps 2024 sealed while models are selected on
the declared 2022--2023 chronology.  Manifest/preflight operations inspect only
filenames and the canonical JSON index; they never open an NPZ payload.  The
only preparation operation which opens bundles is the explicitly requested
Price-aware EDF calibration. Price thresholds use the training window and the
pre-registered battery-rule grid uses validation only; 2024 remains sealed.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

from dcvpp.core import (
    atomic_json,
    atomic_text,
    load_yaml,
    utc_now,
)


FORMAL_SEEDS = (20240813, 20240829, 20240917)
SEED1 = FORMAL_SEEDS[0]
FORMAL_SPLITS = {
    "train": ("2022-01-01", "2023-06-30"),
    "validation": ("2023-07-01", "2023-12-31"),
    "formal_selection": ("2024-01-01", "2024-12-31"),
}
PHASE_BUDGETS = {
    "representation": (500, 10),
    "scenario": (500, 10),
    "policy_training": (500, 10),
    "a1": (500, 10),
    "a2": (500, 10),
}
TRAINING_SEMANTICS_REVISION = "b512_full_coverage_weighted_epoch500_v5"
TRAINING_PHASES = ("representation", "scenario", "policy", "a1", "a2")
BATCH_FALLBACK = (512, 256, 128, 64, 32, 4, 2, 1)
INTERVALS = 96
SCENARIOS = 20
ROOT_METADATA = (
    "formal_split_manifest.json",
    "provenance_manifest.json",
    "price_aware_edf_calibration.json",
)


class FormalProtocolError(ValueError):
    """The frozen formal protocol or its evidence chain is inconsistent."""


def _isolated_sha256(*, path: Path | None = None, value: Any = None) -> str:
    """Compute standard SHA-256 outside the NumPy/Torch process on Windows.

    The local Windows runtime has a reproducible native-library conflict when
    OpenSSL-backed ``hashlib`` is first exercised after NumPy/Torch in a Unicode
    workspace path.  A standard-library-only child process preserves the exact
    SHA-256 contract and prevents that native crash from terminating a formal
    training process.
    """

    if (path is None) == (value is None):
        raise ValueError("provide exactly one of path or value")
    if path is not None:
        script = (
            "import hashlib,sys; h=hashlib.sha256(); "
            "f=open(sys.argv[1],'rb'); "
            "exec(\"while True:\\n b=f.read(8388608)\\n if not b: break\\n h.update(b)\"); "
            "f.close(); print(h.hexdigest())"
        )
        command = [sys.executable, "-c", script, str(path)]
        input_text = None
    else:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        )
        script = (
            "import hashlib,sys; data=sys.stdin.buffer.read(); "
            "print(hashlib.sha256(data).hexdigest())"
        )
        command = [sys.executable, "-c", script]
        input_text = payload
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    digest = completed.stdout.strip().lower()
    if completed.returncode != 0 or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FormalProtocolError(
            "isolated SHA-256 failed: "
            + (completed.stderr.strip() or f"exit={completed.returncode}")
        )
    return digest


def sha256_file(path: Path) -> str:
    return _isolated_sha256(path=Path(path))


def sha256_json(value: Any) -> str:
    return _isolated_sha256(value=value)


def _repository_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    return parent.parent if parent.name == "configs" else parent


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalProtocolError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalProtocolError(f"{name} must be a mapping")
    return value


def _absolute(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else root / path).resolve()


def _reject_locked_name(path: Path | str) -> None:
    candidate = Path(path)
    for part in candidate.parts:
        token = str(part).lower()
        if token == "2025" or token.startswith("2025-") or token.startswith("2025_"):
            raise FormalProtocolError(f"locked 2025 path is closed: {candidate}")


def validate_formal_config(config: Mapping[str, Any]) -> None:
    """Reject any drift from the approved formal contract."""

    project = _mapping(config.get("project"), "project")
    protocol = _mapping(config.get("protocol"), "protocol")
    splits = _mapping(protocol.get("splits"), "protocol.splits")
    training = _mapping(config.get("training"), "training")
    formal = _mapping(training.get("formal"), "training.formal")
    runtime = _mapping(config.get("runtime"), "runtime")
    rolling = _mapping(config.get("rolling"), "rolling")
    scenarios = _mapping(config.get("scenarios"), "scenarios")

    _require(
        tuple(int(value) for value in protocol.get("seeds", ())) == FORMAL_SEEDS,
        f"protocol.seeds must be exactly {list(FORMAL_SEEDS)}",
    )
    for location, value in (
        ("protocol.active_seed", protocol.get("active_seed")),
        ("training.seed", training.get("seed")),
        ("run.seed", _mapping(config.get("run"), "run").get("seed")),
    ):
        _require(int(value) == SEED1, f"{location} must be {SEED1}")
    _require(3407 not in protocol.get("seeds", ()), "legacy seed 3407 is forbidden")
    _require(
        protocol.get("training_semantics_revision") == TRAINING_SEMANTICS_REVISION,
        f"protocol.training_semantics_revision must be {TRAINING_SEMANTICS_REVISION}",
    )

    for name, (expected_start, expected_end) in FORMAL_SPLITS.items():
        split = _mapping(splits.get(name), f"protocol.splits.{name}")
        _require(
            (str(split.get("start"))[:10], str(split.get("end"))[:10])
            == (expected_start, expected_end),
            f"{name} must remain {expected_start}..{expected_end}",
        )
    selection = _mapping(splits.get("formal_selection"), "formal_selection")
    _require(
        selection.get("sealed_until_all_methods_frozen") is True,
        "2024 formal selection must remain sealed until all methods are frozen",
    )
    _require(int(protocol.get("locked_test_year", 0)) == 2025, "2025 must remain closed")
    _require(int(project.get("locked_test_year", 0)) == 2025, "project 2025 lock drift")

    _require(str(training.get("mode", "")).lower() == "formal", "training.mode must be formal")
    _require(formal.get("enabled") is True, "training.formal.enabled must be true")
    for name, (epochs, patience) in PHASE_BUDGETS.items():
        phase = _mapping(training.get(name), f"training.{name}")
        _require(int(phase.get("epochs", -1)) == epochs, f"{name} epochs must be {epochs}")
        _require(
            int(phase.get("early_stopping_patience", -1)) == patience,
            f"{name} patience must be {patience}",
        )
    for name, section in (("training", training), ("training.formal", formal)):
        _require(int(section.get("microbatch_size", 0)) == 512, f"{name} microbatch must be 512")
        _require(int(section.get("default_batch_size", 0)) == 512, f"{name} default batch must be 512")
        _require(
            tuple(int(value) for value in section.get("batch_size_fallback", ()))
            == BATCH_FALLBACK,
            f"{name} fallback must be {list(BATCH_FALLBACK)}",
        )
        _require(int(section.get("gradient_accumulation", 0)) == 1, f"{name} accumulation must be 1")
    _require(training.get("forecast_amp") is False, "formal run does not claim unimplemented forecast AMP")
    _require(training.get("policy_amp") is False, "formal run does not claim unimplemented policy AMP")

    _require(int(scenarios.get("count", 0)) == SCENARIOS, "scenario count must be 20")
    _require(int(rolling.get("scenario_count", 0)) == SCENARIOS, "rolling scenario count must be 20")
    _require(int(rolling.get("intervals_per_day", 0)) == INTERVALS, "rolling day must have 96 origins")
    _require(runtime.get("backend") == "neural_constraint_shield", "runtime backend drift")
    _require(runtime.get("external_solver") is False, "external solver must remain false")
    _require("gurobi" not in str(project.get("output_root", "")).lower(), "formal root cannot be Gurobi-labelled")


def load_formal_config(config_path: Path | str) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    config = load_yaml(path)
    validate_formal_config(config)
    return config, _repository_root(path)


def _data_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    data = _mapping(config.get("data"), "data")
    project = _mapping(config.get("project"), "project")
    return {
        "selection": _absolute(root, data.get("selection_bundle_root")),
        "settlement": _absolute(root, data.get("realized_settlement_root")),
        "index": _absolute(root, data.get("realized_settlement_index")),
        "physical": _absolute(root, data.get("physical_artifact")),
        "output": _absolute(root, project.get("output_root")),
    }


def _dated_files(directory: Path) -> dict[str, Path]:
    _require(directory.is_dir(), f"missing data directory: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.npz")):
        _reject_locked_name(path)
        date = path.stem[:10]
        try:
            parsed = datetime.fromisoformat(date)
        except ValueError as exc:
            raise FormalProtocolError(f"non-canonical daily bundle filename: {path.name}") from exc
        if parsed.year == 2025:
            raise FormalProtocolError(f"locked 2025 file is closed: {path}")
        if parsed.year in {2022, 2023, 2024}:
            result[date] = path.resolve()
    return result


def build_split_manifest(config_path: Path | str) -> dict[str, Any]:
    """Build the canonical filename-only split manifest without opening NPZs."""

    config, root = load_formal_config(config_path)
    paths = _data_paths(config, root)
    selection = _dated_files(paths["selection"])
    settlement = _dated_files(paths["settlement"])
    _require(paths["index"].is_file(), f"missing canonical index: {paths['index']}")
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    days = _mapping(index.get("days"), "canonical index days")
    for date, entry in days.items():
        _reject_locked_name(str(date))
        if str(date).startswith("2025"):
            raise FormalProtocolError("canonical index contains locked 2025 metadata")
        if isinstance(entry, Mapping) and entry.get("path") is not None:
            _reject_locked_name(str(entry["path"]))

    split_payload: dict[str, Any] = {}
    for name, (start, end) in FORMAL_SPLITS.items():
        selection_dates = sorted(date for date in selection if start <= date <= end)
        settlement_dates = sorted(date for date in settlement if start <= date <= end)
        index_dates = sorted(str(date) for date in days if start <= str(date) <= end)
        _require(selection_dates, f"no eligible {name} selection days")
        _require(selection_dates == settlement_dates, f"{name} selection/settlement dates differ")
        _require(selection_dates == index_dates, f"{name} files differ from canonical index")
        split_payload[name] = {
            "start": start,
            "end": end,
            "count": len(selection_dates),
            "first_date": selection_dates[0],
            "last_date": selection_dates[-1],
            "dates": selection_dates,
            "selection_files": [
                selection[date].relative_to(root).as_posix() for date in selection_dates
            ],
            "settlement_files": [
                settlement[date].relative_to(root).as_posix() for date in selection_dates
            ],
        }
    manifest: dict[str, Any] = {
        "schema": "dcvpp.e2e_moe_r3.formal_split_manifest.v1",
        "protocol": str(_mapping(config.get("protocol"), "protocol").get("name")),
        "seeds": list(FORMAL_SEEDS),
        "canonical_index": paths["index"].relative_to(root).as_posix(),
        "canonical_index_sha256": sha256_file(paths["index"]),
        "splits": split_payload,
        "formal_2024_payload_opened": False,
        "locked_2025_opened": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    existing = _read_json(path)
    if existing is not None and existing.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise FormalProtocolError(
            "formal split manifest changed; create a new protocol root instead of overwriting evidence"
        )
    atomic_json(path, manifest)


def _update_provenance(
    path: Path,
    *,
    config_path: Path,
    root: Path,
    manifest: Mapping[str, Any],
    event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    existing = _read_json(path) or {}
    events = list(existing.get("events", [])) if isinstance(existing.get("events", []), list) else []
    if event is not None:
        event_key = sha256_json(event)
        if all(item.get("event_key") != event_key for item in events if isinstance(item, Mapping)):
            events.append({**dict(event), "event_key": event_key, "recorded_at_utc": utc_now()})
    paths = _data_paths(load_yaml(config_path), root)
    provenance = {
        "schema": "dcvpp.e2e_moe_r3.formal_provenance.v1",
        "protocol": str(_mapping(load_yaml(config_path).get("protocol"), "protocol").get("name")),
        "created_at_utc": existing.get("created_at_utc", utc_now()),
        "updated_at_utc": utc_now(),
        "config": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "split_manifest_sha256": manifest["manifest_sha256"],
        "canonical_index_sha256": manifest["canonical_index_sha256"],
        "physical_artifact": paths["physical"].relative_to(root).as_posix(),
        "physical_artifact_sha256": sha256_file(paths["physical"]) if paths["physical"].is_file() else None,
        "formal_2024_payload_opened": False,
        "locked_2025_opened": False,
        "events": events,
    }
    atomic_json(path, provenance)
    return provenance


def calibrate_price_aware_edf(
    config_path: Path | str,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit and freeze Price-aware EDF thresholds on training prices only."""

    config_path = Path(config_path).resolve()
    config, root = load_formal_config(config_path)
    paths = _data_paths(config, root)
    paths["output"].mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest or build_split_manifest(config_path))
    target = paths["output"] / "price_aware_edf_calibration.json"
    existing = _read_json(target)
    benchmark = _mapping(config.get("benchmark"), "benchmark")
    calibration_cfg = _mapping(benchmark.get("calibration"), "benchmark.calibration")
    if existing is not None and existing.get("battery_fractions_frozen") is True:
        _require(existing.get("split_manifest_sha256") == manifest.get("manifest_sha256"), "calibration split manifest mismatch")
        _require(existing.get("calibration_split") == "train", "calibration was not frozen from train")
        return existing

    if existing is None:
        train = _mapping(_mapping(manifest.get("splits"), "splits").get("train"), "train split")
        prices: list[np.ndarray] = []
        for relative in train.get("selection_files", []):
            path = _absolute(root, relative)
            _reject_locked_name(path)
            _require(path.stem[:10] <= FORMAL_SPLITS["train"][1], "calibration attempted to open post-train payload")
            with np.load(path, allow_pickle=False) as bundle:
                field = str(calibration_cfg.get("price_field", "rt_price_pool"))
                _require(field in bundle.files, f"calibration field {field!r} missing in {path.name}")
                value = np.asarray(bundle[field], dtype=np.float32).reshape(-1)
                _require(value.size > 0 and bool(np.isfinite(value).all()), f"non-finite calibration prices in {path.name}")
                prices.append(value)
        _require(bool(prices), "training price calibration has no observations")
        samples = np.concatenate(prices)
        # Identify the frozen input contract without hashing every daily NPZ.
        source_identity = {
            "split": "train",
            "first_date": train["first_date"],
            "last_date": train["last_date"],
            "day_count": train["count"],
            "date_list_sha256": sha256_json(train.get("dates", [])),
            "split_manifest_sha256": manifest["manifest_sha256"],
            "canonical_index_sha256": manifest["canonical_index_sha256"],
        }
        source_hash = sha256_json(source_identity)
        from .constrained_runtime import PriceAwareEDFShieldPolicy

        policy = PriceAwareEDFShieldPolicy(
            low_quantile=float(calibration_cfg.get("low_quantile", 0.35)),
            high_quantile=float(calibration_cfg.get("high_quantile", 0.65)),
            low_price_charge_fraction=0.0,
            high_price_discharge_fraction=0.0,
        ).calibrate(
            samples,
            split="train",
            source_hash=source_hash,
            source_label=(
                f"formal train {train['first_date']}..{train['last_date']} "
                f"({train['count']} eligible days)"
            ),
        )
        metadata = {
            "schema": "dcvpp.e2e_moe_r3.price_aware_edf_calibration.v1",
            "policy": PriceAwareEDFShieldPolicy.name,
            **policy.calibration_metadata(),
            "frozen": True,
            "frozen_at_utc": utc_now(),
            "objective": "causal price-aware earliest-deadline-first benchmark",
            "price_field": str(calibration_cfg.get("price_field", "rt_price_pool")),
            "train_start": train["first_date"],
            "train_end": train["last_date"],
            "train_day_count": train["count"],
            "split_manifest_sha256": manifest["manifest_sha256"],
            "calibration_input_identity": source_identity,
            "calibration_input_identity_sha256": source_hash,
            "daily_payloads_hashed": False,
            "refit_on_validation": False,
            "refit_on_formal_selection": False,
            "battery_candidate_values": [0.0, 0.5, 1.0],
            "battery_candidate_grid": [
                {
                    "low_price_charge_fraction": charge,
                    "high_price_discharge_fraction": discharge,
                }
                for charge in (0.0, 0.5, 1.0)
                for discharge in (0.0, 0.5, 1.0)
            ],
            "battery_candidate_selection_split": "validation",
            "battery_candidate_selection_order": [
                "feasibility",
                "sla_noninferiority",
                "lowest_cost",
            ],
            "battery_candidate_results": [],
            "battery_fractions_frozen": False,
            "formal_2024_payload_opened": False,
            "locked_2025_opened": False,
        }
        atomic_json(target, metadata)
        del samples, prices
    else:
        metadata = dict(existing)
        _require(metadata.get("split_manifest_sha256") == manifest.get("manifest_sha256"), "calibration split manifest mismatch")
        _require(metadata.get("calibration_split") == "train", "calibration was not fitted on train")

    # Select and freeze the pre-registered battery rule using all validation
    # days before any learned checkpoint is judged.  This helper is independent
    # of the learned model and never opens 2024/2025.
    try:
        from .formal_validation import calibrate_price_aware_battery_rules
        from .constrained_runtime import load_frozen_physical
    except (ImportError, AttributeError) as exc:
        raise FormalProtocolError(
            f"formal battery-rule calibrator is unavailable: {exc}"
        ) from exc
    validation = _mapping(_mapping(manifest.get("splits"), "splits").get("validation"), "validation split")
    selection_files = [_absolute(root, item) for item in validation.get("selection_files", [])]
    physical = load_frozen_physical(paths["physical"])
    from .trainer import R3DailyBundleDataset

    validation_dataset = R3DailyBundleDataset(
        selection_files,
        physical,
        realized_settlement_root=paths["settlement"],
    )
    calibration_kwargs: dict[str, Any] = {
        "runtime_config": config["runtime"],
        "calibration": metadata,
        "formal": True,
        "max_origins": INTERVALS,
        "logfile": None,
    }
    calibration_signature = inspect.signature(calibrate_price_aware_battery_rules)
    if "realized_settlement_root" in calibration_signature.parameters:
        calibration_kwargs["realized_settlement_root"] = paths["settlement"]
    result = calibrate_price_aware_battery_rules(
        validation_dataset,
        physical,
        **calibration_kwargs,
    )
    _require(result.get("status") == "COMPLETE", "Price-aware EDF validation-grid calibration did not complete")
    _require(int(result.get("completed_day_count", 0)) == int(validation["count"]), "battery-rule calibration did not cover all validation days")
    frozen = result.get("frozen_calibration")
    _require(isinstance(frozen, Mapping), "battery-rule calibration returned no frozen metadata")
    metadata = dict(frozen)
    metadata.update(
        schema="dcvpp.e2e_moe_r3.price_aware_edf_calibration.v1",
        split_manifest_sha256=manifest["manifest_sha256"],
        battery_candidate_results=result.get("candidate_results", []),
        battery_fractions_frozen=True,
        frozen=True,
        frozen_at_utc=utc_now(),
        formal_2024_payload_opened=False,
        locked_2025_opened=False,
    )
    atomic_json(target, metadata)
    return metadata


def prepare_protocol(
    config_path: Path | str,
    *,
    calibrate: bool = False,
) -> dict[str, Any]:
    """Materialize protocol metadata; calibration is explicit and train-only."""

    config_path = Path(config_path).resolve()
    config, root = load_formal_config(config_path)
    paths = _data_paths(config, root)
    paths["output"].mkdir(parents=True, exist_ok=True)
    manifest = build_split_manifest(config_path)
    _write_manifest(paths["output"] / "formal_split_manifest.json", manifest)
    calibration = calibrate_price_aware_edf(config_path, manifest=manifest) if calibrate else None
    provenance = _update_provenance(
        paths["output"] / "provenance_manifest.json",
        config_path=config_path,
        root=root,
        manifest=manifest,
        event={
            "type": "protocol_prepared",
            "calibration_consumed_training_payloads": bool(calibrate),
            "calibration_input_identity_sha256": (
                None if calibration is None else calibration.get("calibration_input_identity_sha256")
            ),
        },
    )
    return {
        "status": "COMPLETE",
        "root": str(paths["output"]),
        "manifest": manifest,
        "calibration": calibration,
        "provenance": provenance,
        "formal_2024_payload_opened": False,
        "locked_2025_opened": False,
    }


def evaluate_gate(
    learned: Mapping[str, Any],
    baseline: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen Seed1 validation gate without weakening thresholds."""

    numeric_values = [
        learned.get("feasibility_rate"),
        learned.get("rejected_non_exogenous_origins"),
        learned.get("total_cost_usd"),
        learned.get("normalized_sla_shortfall"),
        baseline.get("total_cost_usd"),
        baseline.get("normalized_sla_shortfall"),
    ]
    finite = all(
        isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
        for value in numeric_values
    )
    learned_cost = float(learned.get("total_cost_usd", math.nan))
    baseline_cost = float(baseline.get("total_cost_usd", math.nan))
    cost_reduction = baseline_cost - learned_cost
    sla_increase = float(learned.get("normalized_sla_shortfall", math.nan)) - float(
        baseline.get("normalized_sla_shortfall", math.nan)
    )
    checks = {
        "metrics_numeric_and_finite": finite,
        "feasibility_100_percent": finite and float(learned.get("feasibility_rate", 0.0)) == 1.0,
        "zero_rejected_non_exogenous": finite and int(learned.get("rejected_non_exogenous_origins", -1)) == 0,
        "strictly_positive_cost_reduction": finite and cost_reduction > 0.0,
        "normalized_sla_noninferiority": finite and sla_increase <= 0.001,
        "finite_metrics_evidence": evidence.get("finite_metrics") is True,
        "finite_checkpoints": evidence.get("checkpoints_finite") is True,
        "resumable_checkpoints": evidence.get("checkpoints_resumable") is True,
        "complete_validation": evidence.get("validation_complete") is True,
        "external_solver_disabled": evidence.get("external_solver") is False,
        "formal_2024_remained_sealed": evidence.get("formal_2024_opened") is False,
        "locked_2025_remained_closed": evidence.get("locked_2025_opened") is False,
    }
    return {
        "all_passed": all(checks.values()),
        "cost_reduction_usd": cost_reduction,
        "normalized_sla_shortfall_increase": sla_increase,
        "checks": checks,
    }


def _run_dir(
    output_root: Path,
    phase: str,
    split: str,
    *,
    seed: int | None = None,
    resume: bool = True,
) -> Path:
    seed_token = "" if seed is None else f"seed={seed}_"
    prefix = f"run_id={phase}_{split}_{seed_token}"
    candidates = sorted(path for path in output_root.glob(f"{prefix}*") if path.is_dir())
    if resume and candidates:
        return candidates[-1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = output_root / f"{prefix}{stamp}"
    suffix = 0
    while run.exists():
        suffix += 1
        run = output_root / f"{prefix}{stamp}_{suffix:02d}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoints").mkdir()
    return run


def _initialize_run(run: Path, config_path: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    (run / "runtime.log").touch(exist_ok=True)
    target = run / "config_used.yaml"
    if not target.exists():
        shutil.copyfile(config_path, target)


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        frame = pd.DataFrame([{"status": "NO_METRICS"}])
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _summary_lines(title: str, state: Mapping[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key in (
        "status",
        "phase",
        "split",
        "active_seed",
        "completed_day_count",
        "completed_origin_count",
        "reason",
        "resume_command",
    ):
        if key in state:
            lines.append(f"- {key}: `{state.get(key)}`")
    lines.extend(
        [
            "- external_solver: `False`",
            "- formal_2024_opened: `False`",
            "- locked_2025_opened: `False`",
        ]
    )
    return "\n".join(lines) + "\n"


def _complete_run(
    run: Path,
    *,
    title: str,
    state: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> None:
    atomic_json(run / "state.json", state)
    _write_parquet(run / "metrics.parquet", rows)
    atomic_text(run / "summary.md", _summary_lines(title, state))


def _calibration_path(config: Mapping[str, Any], root: Path) -> Path:
    return _data_paths(config, root)["output"] / "price_aware_edf_calibration.json"


def _load_calibration(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = _calibration_path(config, root)
    calibration = _read_json(path)
    if calibration is None:
        raise FormalProtocolError(
            "missing frozen Price-aware EDF calibration; run prepare --calibrate first"
        )
    _require(calibration.get("calibration_split") == "train", "benchmark calibration is not train-only")
    _require(calibration.get("frozen") is True, "benchmark price thresholds are not frozen")
    _require(
        calibration.get("battery_fractions_frozen") is True,
        "Price-aware EDF battery rules are not frozen from validation",
    )
    _require(calibration.get("formal_2024_payload_opened") is False, "calibration opened sealed 2024")
    _require(calibration.get("locked_2025_opened") is False, "calibration opened locked 2025")
    return calibration


def run_preflight(
    config_path: Path | str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the single formal preflight, including one audited 96-origin day."""

    config_path = Path(config_path).resolve()
    config, root = load_formal_config(config_path)
    paths = _data_paths(config, root)
    paths["output"].mkdir(parents=True, exist_ok=True)
    prepared = prepare_protocol(config_path, calibrate=False)
    manifest = prepared["manifest"]
    run = _run_dir(paths["output"], "preflight", "protocol", resume=resume)
    _initialize_run(run, config_path)
    existing = _read_json(run / "state.json")
    if (
        resume
        and existing is not None
        and existing.get("status") == "COMPLETE"
        and existing.get("split_manifest_sha256") == manifest.get("manifest_sha256")
    ):
        # Epoch budgets and trainer throughput semantics do not affect the
        # solver-free environment/data/96-origin preflight evidence.  Reuse the
        # one completed preflight and bind its consumption to the active config
        # without rerunning it.
        if existing.get("config_sha256") != sha256_file(config_path):
            _update_provenance(
                paths["output"] / "provenance_manifest.json",
                config_path=config_path,
                root=root,
                manifest=manifest,
                event={
                    "event": "preflight_reused_after_training_protocol_revision",
                    "preflight_run": str(run),
                    "preflight_config_sha256": existing.get("config_sha256"),
                    "active_config_sha256": sha256_file(config_path),
                    "training_semantics_revision": _mapping(
                        config.get("protocol"), "protocol"
                    ).get("training_semantics_revision"),
                },
            )
            return {
                **existing,
                "consumed_by_config_sha256": sha256_file(config_path),
                "training_semantics_revision": _mapping(
                    config.get("protocol"), "protocol"
                ).get("training_semantics_revision"),
                "preflight_reused_without_rerun": True,
            }
        return existing

    # A stale non-COMPLETE preflight is resumable after its explicit blocker
    # (for example, benchmark calibration or interpreter selection) changes.
    # Reinitialize its evidence rather than treating the earlier result as the
    # single completed formal preflight.

    blockers: list[str] = []
    failures: list[str] = []
    for key in ("selection", "settlement", "index", "physical"):
        if not paths[key].exists():
            blockers.append(f"missing {key}: {paths[key]}")
    calibration = _read_json(_calibration_path(config, root))
    if calibration is None:
        blockers.append("missing frozen train-only Price-aware EDF calibration")
    else:
        if calibration.get("split_manifest_sha256") != manifest.get("manifest_sha256"):
            blockers.append("calibration/split manifest mismatch")
        if calibration.get("calibration_split") != "train":
            blockers.append("Price-aware EDF was not calibrated on train")
        if calibration.get("battery_fractions_frozen") is not True:
            blockers.append("Price-aware EDF battery rules are not frozen from validation")
    try:
        from .formal_validation import FormalValidationEvaluator

        evaluator_available = callable(FormalValidationEvaluator)
    except (ImportError, AttributeError) as exc:
        evaluator_available = False
        blockers.append(f"FormalValidationEvaluator unavailable: {exc}")
    validation_count = int(manifest["splits"]["validation"]["count"])
    if validation_count < 1:
        blockers.append("validation split has no eligible dates")
    # Verify the actual active import graph in a fresh interpreter.  Looking at
    # configuration strings alone is insufficient evidence that a backend was
    # not imported transitively.
    banned_roots = {
        "pyomo",
        "gurobipy",
        "cvxpy",
        "cvxpylayers",
        "qpth",
        "optnet",
    }
    import_graph_modules: list[str] = []
    import_graph_error: str | None = None
    import_script = (
        "import json,sys; sys.path.insert(0,sys.argv[1]); "
        "import dcvpp.e2e_moe_r3; "
        "from dcvpp.e2e_moe_r3 import model,scenario_risk,policy,constrained_runtime,losses,trainer,cli; "
        "import dcvpp.e2e_moe_r3.formal_protocol; "
        "roots={'pyomo','gurobipy','cvxpy','cvxpylayers','qpth','optnet'}; "
        "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0].lower() in roots)))"
    )
    imported = subprocess.run(
        [sys.executable, "-c", import_script, str(root / "src")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if imported.returncode != 0:
        import_graph_error = imported.stderr.strip() or f"exit={imported.returncode}"
        failures.append(f"active import graph check failed: {import_graph_error}")
    else:
        try:
            import_graph_modules = list(json.loads(imported.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError, TypeError) as exc:
            import_graph_error = str(exc)
            failures.append(f"active import graph evidence is invalid: {exc}")
        if any(name.split(".")[0].lower() in banned_roots for name in import_graph_modules):
            failures.append(
                "active solver-free import graph loaded forbidden modules: "
                + ", ".join(import_graph_modules)
            )

    python_supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    if not python_supported:
        blockers.append(f"unsupported formal Python {sys.version.split()[0]}")
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name: str | None = None
    cuda_smoke_ok = False
    cuda_error: str | None = None
    if not cuda_available:
        blockers.append("CUDA is unavailable for the formal training program")
    else:
        try:
            cuda_device_name = torch.cuda.get_device_name(0)
            probe = torch.ones((32, 32), device="cuda", requires_grad=True)
            probe.square().mean().backward()
            cuda_smoke_ok = bool(
                probe.grad is not None and torch.isfinite(probe.grad).all().item()
            )
            if not cuda_smoke_ok:
                blockers.append("CUDA finite forward/backward smoke failed")
            del probe
        except (RuntimeError, OSError) as exc:
            cuda_error = str(exc)
            blockers.append(f"CUDA smoke failed: {exc}")

    locked_path_gate_verified = False
    preflight_date: str | None = None
    rolling_result: dict[str, Any] = {}
    residual_max = math.inf
    day_ahead_frozen = False
    if not blockers and not failures:
        from .constrained_runtime import (
            NeuralConstrainedRuntime,
            PriceAwareEDFShieldPolicy,
            R3DataError,
            load_frozen_physical,
            load_selection_day,
            validate_record,
        )

        try:
            load_selection_day(paths["selection"] / "2025-01-01.npz")
        except R3DataError:
            locked_path_gate_verified = True
        except Exception as exc:  # pragma: no cover - internal contract failure
            failures.append(f"2025 path gate raised the wrong error: {exc}")
        else:  # pragma: no cover - would be a critical lock regression
            failures.append("2025 path gate did not reject before payload access")

        if locked_path_gate_verified and not failures:
            preflight_date = str(manifest["splits"]["train"]["dates"][0])
            try:
                day = load_selection_day(
                    paths["selection"] / f"{preflight_date}.npz",
                    realized_settlement_path=(
                        paths["settlement"] / f"{preflight_date}.npz"
                    ),
                )
                physical = load_frozen_physical(paths["physical"])
                baseline = PriceAwareEDFShieldPolicy.from_calibration(calibration or {})
                runtime = NeuralConstrainedRuntime(
                    physical,
                    config["runtime"],
                    logfile=run / "runtime.log",
                )
                rolling_result = runtime.run_day(
                    day,
                    policy_callback=baseline,
                    max_origins=INTERVALS,
                    formal=True,
                )
                records = list(rolling_result.get("records", []))
                independent = [
                    validate_record(
                        record,
                        physical=physical,
                        day=day,
                        tolerance=float(config["runtime"].get("validation_tolerance", 1.0e-5)),
                    )
                    for record in records
                    if not record.get("rejected", False)
                ]
                numeric_residuals = [
                    float(value)
                    for audit in independent
                    for key, value in audit.items()
                    if key != "passed" and isinstance(value, (int, float, np.integer, np.floating))
                ]
                residual_max = max(numeric_residuals, default=math.inf)
                schedules = [
                    np.asarray(record["day_ahead_schedule_mw"], dtype=float)
                    for record in records
                    if not record.get("rejected", False)
                ]
                day_ahead_frozen = bool(
                    len(schedules) == INTERVALS
                    and all(np.array_equal(schedule, schedules[0]) for schedule in schedules[1:])
                )
                tolerance = float(config["runtime"].get("validation_tolerance", 1.0e-5))
                rolling_ok = (
                    rolling_result.get("status") == "COMPLETE"
                    and int(rolling_result.get("completed_origins", 0)) == INTERVALS
                    and int(rolling_result.get("scenario_count", 0)) == SCENARIOS
                    and len(independent) == INTERVALS
                    and all(audit.get("passed") is True for audit in independent)
                    and math.isfinite(residual_max)
                    and residual_max <= tolerance
                    and day_ahead_frozen
                )
                if not rolling_ok:
                    failures.append("the formal 96-origin audited day did not satisfy its contract")
            except Exception as exc:
                failures.append(f"formal 96-origin day failed: {type(exc).__name__}: {exc}")

    status = "FAILED" if failures else "BLOCKED" if blockers else "COMPLETE"
    reason_parts = [*failures, *blockers]
    state = {
        "schema_version": "dcvpp.e2e_moe_r3.formal_preflight.v1",
        "status": status,
        "experiment_id": str(_mapping(config.get("protocol"), "protocol").get("name")),
        "method": "e2e_moe_solver_free_r3",
        "phase": "preflight",
        "split": "protocol",
        "active_seed": SEED1,
        "predecessor": None,
        "protocol_seeds": list(FORMAL_SEEDS),
        "split_manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": sha256_file(config_path),
        "canonical_index_sha256": manifest["canonical_index_sha256"],
        "train_day_count": manifest["splits"]["train"]["count"],
        "validation_day_count": validation_count,
        "formal_selection_eligible_day_count": manifest["splits"]["formal_selection"]["count"],
        "formal_validation_evaluator_available": evaluator_available,
        "calibration_present": calibration is not None,
        "calibration_battery_fractions_frozen": bool(
            calibration and calibration.get("battery_fractions_frozen") is True
        ),
        "python_version": sys.version.split()[0],
        "python_supported": python_supported,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "cuda_smoke_ok": cuda_smoke_ok,
        "cuda_error": cuda_error,
        "active_import_graph_forbidden_modules": import_graph_modules,
        "active_import_graph_error": import_graph_error,
        "locked_2025_path_before_open_verified": locked_path_gate_verified,
        "preflight_operating_date": preflight_date,
        "preflight_day_status": rolling_result.get("status"),
        "preflight_completed_origins": int(rolling_result.get("completed_origins", 0)),
        "preflight_scenario_count": int(rolling_result.get("scenario_count", 0)),
        "preflight_independent_residual_max": (
            residual_max if math.isfinite(residual_max) else None
        ),
        "preflight_day_ahead_frozen": day_ahead_frozen,
        "blockers": blockers,
        "failures": failures,
        "reason": None if not reason_parts else "; ".join(reason_parts),
        "external_solver": False,
        "formal_2024_opened": False,
        "locked_2025_opened": False,
        "resumable": True,
        "updated_at_utc": utc_now(),
        "resume_command": (
            f"python scripts/run_e2e_moe_r3_formal.py preflight --config {config_path}"
        ),
    }
    _complete_run(
        run,
        title="E2E-MoE R3 formal preflight",
        state=state,
        rows=[
            {
                "status": status,
                "train_day_count": state["train_day_count"],
                "validation_day_count": validation_count,
                "formal_selection_eligible_day_count": state["formal_selection_eligible_day_count"],
                "blocker_count": len(blockers),
                "failure_count": len(failures),
                "preflight_operating_date": preflight_date,
                "preflight_completed_origins": state["preflight_completed_origins"],
                "preflight_scenario_count": state["preflight_scenario_count"],
                "preflight_independent_residual_max": state["preflight_independent_residual_max"],
                "preflight_day_ahead_frozen": day_ahead_frozen,
                "locked_2025_path_before_open_verified": locked_path_gate_verified,
                "external_solver": False,
                "formal_2024_opened": False,
                "locked_2025_opened": False,
            }
        ],
    )
    return state


def _validation_evaluator(
    config: Mapping[str, Any],
    root: Path,
    calibration: Mapping[str, Any],
    *,
    logfile: Path | None,
) -> Any:
    from .formal_validation import FormalValidationEvaluator

    kwargs: dict[str, Any] = {
        "runtime_config": config["runtime"],
        "calibration": calibration,
        "formal": True,
        "max_origins": INTERVALS,
        "logfile": logfile,
    }
    signature = inspect.signature(FormalValidationEvaluator)
    if "repository_root" in signature.parameters:
        kwargs["repository_root"] = root
    if "scenario_count" in signature.parameters:
        kwargs["scenario_count"] = SCENARIOS
    if "calibration_path" in signature.parameters:
        kwargs["calibration_path"] = _calibration_path(config, root)
    return FormalValidationEvaluator(**kwargs)


def _phase_run(
    config_path: Path,
    config: Mapping[str, Any],
    root: Path,
    *,
    phase: str,
    calibration: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    output = _data_paths(config, root)["output"]
    run = _run_dir(output, phase, "selection", seed=SEED1, resume=resume)
    _initialize_run(run, config_path)
    existing = _read_json(run / "state.json")
    if existing is not None and existing.get("status") == "COMPLETE":
        return existing
    from .trainer import run_training_phase

    evaluator = None
    if phase in {"policy", "a1", "a2"}:
        evaluator = _validation_evaluator(
            config,
            root,
            calibration,
            logfile=run / "runtime.log",
        )
    return run_training_phase(
        phase=phase,
        config=config,
        repository_root=root,
        run_dir=run,
        split="selection",
        resume=resume,
        validation_evaluator=evaluator,
    )


def run_seed1_lineage(
    config_path: Path | str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the exact Seed1 five-phase lineage after a COMPLETE preflight."""

    config_path = Path(config_path).resolve()
    config, root = load_formal_config(config_path)
    preflight = run_preflight(config_path, resume=True)
    if preflight.get("status") != "COMPLETE":
        return {
            "status": "BLOCKED",
            "stopped_before": "representation",
            "reason": preflight.get("reason"),
            "preflight": preflight,
        }
    calibration = _load_calibration(config, root)
    statuses: dict[str, str] = {}
    states: dict[str, Any] = {}
    for phase in TRAINING_PHASES:
        state = _phase_run(
            config_path,
            config,
            root,
            phase=phase,
            calibration=calibration,
            resume=resume,
        )
        states[phase] = state
        statuses[phase] = str(state.get("status"))
        if state.get("status") != "COMPLETE":
            return {
                "status": "PARTIAL" if state.get("status") != "BLOCKED" else "BLOCKED",
                "active_seed": SEED1,
                "stopped_after": phase,
                "phase_statuses": statuses,
                "reason": state.get("reason", state.get("error")),
                "resume_command": (
                    f"python scripts/run_e2e_moe_r3_formal.py run-seed1 "
                    f"--config {config_path} --resume"
                ),
            }
    return {
        "status": "COMPLETE",
        "active_seed": SEED1,
        "phase_statuses": statuses,
        "a2_checkpoint": states["a2"].get("checkpoint"),
        "external_solver": False,
        "formal_2024_opened": False,
        "locked_2025_opened": False,
    }


def _latest_complete_a2(
    config: Mapping[str, Any], root: Path
) -> tuple[Path, dict[str, Any]]:
    output = _data_paths(config, root)["output"]
    for run in sorted(
        (path for path in output.glob(f"run_id=a2_selection_seed={SEED1}_*") if path.is_dir()),
        reverse=True,
    ):
        state = _read_json(run / "state.json")
        best = run / "checkpoints" / "best.pt"
        if (
            state is not None
            and state.get("status") == "COMPLETE"
            and int(state.get("active_seed", state.get("seed", -1))) == SEED1
            and state.get("external_solver") is False
            and state.get("formal_2024_opened") is False
            and state.get("locked_2025_opened") is False
            and best.is_file()
        ):
            return best, state
    raise FormalProtocolError("no COMPLETE lineage-correct Seed1 A2 best checkpoint")


def _load_model_from_checkpoint(
    config: Mapping[str, Any], checkpoint: Path
) -> torch.nn.Module:
    from .cli import _model_kwargs
    from .model import E2EMoEDARTVPP, E2EMoER3Config

    model = E2EMoEDARTVPP(E2EMoER3Config(**_model_kwargs(config)))
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old torch compatibility
        payload = torch.load(checkpoint, map_location="cpu")
    _require(isinstance(payload, Mapping), "A2 checkpoint payload is invalid")
    _require("model_state_dict" in payload, "A2 checkpoint lacks model_state_dict")
    model.load_state_dict(payload["model_state_dict"])
    requested = str(_mapping(config.get("training"), "training").get("device", "auto"))
    device = (
        torch.device("cuda")
        if requested == "auto" and torch.cuda.is_available()
        else torch.device("cpu")
        if requested == "auto"
        else torch.device(requested)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FormalProtocolError("formal evaluation requests unavailable CUDA")
    model.to(device).eval()
    return model


def _record_rows(
    result: Mapping[str, Any],
    *,
    method: str,
    day_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in result.get("records", []):
        row = dict(source)
        row.update(
            method=method,
            day_status=result.get("status"),
            day_total_cost_usd=day_summary.get("total_cost_usd"),
            day_normalized_sla_shortfall=day_summary.get("normalized_sla_shortfall"),
            day_feasibility_rate=day_summary.get("feasibility_rate"),
            day_rejected_origins=day_summary.get("rejected_origins"),
            day_exogenous_infeasible_origins=day_summary.get("exogenous_infeasible_origins"),
        )
        rows.append(row)
    return rows


def _evaluate_one_date(
    *,
    config: Mapping[str, Any],
    root: Path,
    run: Path,
    model: torch.nn.Module,
    date: str,
    calibration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(FORMAL_SPLITS["validation"][0] <= date <= FORMAL_SPLITS["validation"][1], "post-A2 diagnostic is validation-only")
    paths = _data_paths(config, root)
    from .cli import _learned_policy_adapter
    from .constrained_runtime import (
        NeuralConstrainedRuntime,
        PriceAwareEDFShieldPolicy,
        load_frozen_physical,
        load_selection_day,
        run_policy_benchmarks,
    )

    day = load_selection_day(
        paths["selection"] / f"{date}.npz",
        realized_settlement_path=paths["settlement"] / f"{date}.npz",
    )
    physical = load_frozen_physical(paths["physical"])
    joint, weights, callback = _learned_policy_adapter(model, day, physical)
    learned_runtime = NeuralConstrainedRuntime(
        physical, config["runtime"], logfile=run / "runtime.log"
    )
    learned = learned_runtime.run_day(
        day,
        policy_callback=callback,
        joint_scenarios=joint,
        scenario_weights=weights,
        max_origins=INTERVALS,
        formal=True,
    )
    baseline_policy = PriceAwareEDFShieldPolicy.from_calibration(calibration)
    baseline = run_policy_benchmarks(
        physical,
        config["runtime"],
        day,
        policies={baseline_policy.name: baseline_policy},
        joint_scenarios=joint,
        scenario_weights=weights,
        max_origins=INTERVALS,
        formal=True,
        logfile=run / "runtime.log",
    )[baseline_policy.name]
    rows = _record_rows(
        learned,
        method="e2e_moe_solver_free_r3",
        day_summary=_mapping(learned.get("summary", {}), "learned summary"),
    ) + _record_rows(
        baseline,
        method=baseline_policy.name,
        day_summary=_mapping(baseline.get("summary", {}), "baseline summary"),
    )
    summary = {
        "date": date,
        "learned": dict(_mapping(learned.get("summary", {}), "learned summary")),
        "baseline": dict(_mapping(baseline.get("summary", {}), "baseline summary")),
        "learned_status": learned.get("status"),
        "baseline_status": baseline.get("status"),
        "learned_completed_origins": learned.get("completed_origins"),
        "baseline_completed_origins": baseline.get("completed_origins"),
        "common_scenario_count": SCENARIOS,
    }
    return rows, summary


def run_initial_validation_day(
    config_path: Path | str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the earliest eligible 96-origin validation diagnostic after A2."""

    config_path = Path(config_path).resolve()
    config, root = load_formal_config(config_path)
    prepared = prepare_protocol(config_path, calibrate=False)
    calibration = _load_calibration(config, root)
    date = str(prepared["manifest"]["splits"]["validation"]["dates"][0])
    checkpoint, checkpoint_state = _latest_complete_a2(config, root)
    output = _data_paths(config, root)["output"]
    run = _run_dir(output, "initial_validation_day", "validation", seed=SEED1, resume=resume)
    _initialize_run(run, config_path)
    existing = _read_json(run / "state.json")
    if (
        resume
        and existing is not None
        and existing.get("status") == "COMPLETE"
        and existing.get("operating_date") == date
    ):
        return existing
    model = _load_model_from_checkpoint(config, checkpoint)
    rows, result = _evaluate_one_date(
        config=config,
        root=root,
        run=run,
        model=model,
        date=date,
        calibration=calibration,
    )
    complete = (
        result["learned_status"] == "COMPLETE"
        and result["baseline_status"] == "COMPLETE"
        and int(result["learned_completed_origins"]) == INTERVALS
        and int(result["baseline_completed_origins"]) == INTERVALS
    )
    state = {
        "schema_version": "dcvpp.e2e_moe_r3.initial_validation_day.v1",
        "status": "COMPLETE" if complete else "PARTIAL",
        "phase": "initial_validation_day",
        "split": "validation",
        "operating_date": date,
        "requested_origins": INTERVALS,
        "completed_origin_count": len(rows),
        "active_seed": SEED1,
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_sha256": sha256_file(checkpoint),
        "a2_state_sha256": sha256_json(checkpoint_state),
        "result": result,
        "external_solver": False,
        "formal_2024_opened": False,
        "locked_2025_opened": False,
        "resumable": True,
        "updated_at_utc": utc_now(),
        "resume_command": (
            f"python scripts/run_e2e_moe_r3_formal.py validation-day --config {config_path} --resume"
        ),
    }
    _complete_run(run, title="Seed1 earliest formal validation day", state=state, rows=rows)
    return state


def _formal_evaluator_result(
    evaluator: Any,
    *,
    model: torch.nn.Module,
    dataset: Any,
    physical: Any,
) -> Mapping[str, Any]:
    if hasattr(evaluator, "evaluate"):
        return evaluator.evaluate(
            model,
            dataset,
            physical,
            phase="a2",
            device=next(model.parameters()).device,
        )
    return evaluator(
        model=model,
        dataset=dataset,
        physical=physical,
        phase="a2",
        device=next(model.parameters()).device,
    )


def run_full_validation(
    config_path: Path | str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate Seed1 on every eligible 2023-H2 day after the day diagnostic."""

    config_path = Path(config_path).resolve()
    config, root = load_formal_config(config_path)
    prepared = prepare_protocol(config_path, calibrate=False)
    manifest = prepared["manifest"]
    calibration = _load_calibration(config, root)
    pilot = run_initial_validation_day(config_path, resume=resume)
    if pilot.get("status") != "COMPLETE":
        return {
            "status": "BLOCKED",
            "phase": "validation_evaluation",
            "reason": "earliest 96-origin validation diagnostic is not COMPLETE",
            "initial_validation_day": pilot,
        }
    checkpoint, checkpoint_state = _latest_complete_a2(config, root)
    model = _load_model_from_checkpoint(config, checkpoint)
    output = _data_paths(config, root)["output"]
    run = _run_dir(output, "validation_evaluation", "validation", seed=SEED1, resume=resume)
    _initialize_run(run, config_path)
    existing = _read_json(run / "state.json")
    if (
        resume
        and existing is not None
        and existing.get("status") == "COMPLETE"
        and int(existing.get("completed_day_count", 0))
        == int(manifest["splits"]["validation"]["count"])
    ):
        return existing

    # The shared evaluator owns exact all-date paired runtime accounting and is
    # also the checkpoint-selection evaluator injected during policy/A1/A2.
    from .trainer import R3DailyBundleDataset
    from .constrained_runtime import load_frozen_physical

    paths = _data_paths(config, root)
    physical = load_frozen_physical(paths["physical"])
    validation_files = [
        _absolute(root, path)
        for path in manifest["splits"]["validation"]["selection_files"]
    ]
    dataset = R3DailyBundleDataset(
        validation_files,
        physical,
        realized_settlement_root=paths["settlement"],
    )
    evaluator = _validation_evaluator(
        config,
        root,
        calibration,
        logfile=run / "runtime.log",
    )
    result = dict(
        _formal_evaluator_result(
            evaluator,
            model=model,
            dataset=dataset,
            physical=physical,
        )
    )
    frozen_calibration = result.get("frozen_calibration")
    if isinstance(frozen_calibration, Mapping):
        selected = dict(frozen_calibration)
        selected.update(
            schema="dcvpp.e2e_moe_r3.price_aware_edf_calibration.v1",
            split_manifest_sha256=manifest["manifest_sha256"],
            battery_candidate_results=result.get("candidate_results", []),
            battery_fractions_frozen=True,
            frozen=True,
            frozen_at_utc=selected.get("frozen_at_utc", utc_now()),
            formal_2024_payload_opened=False,
            locked_2025_opened=False,
        )
        atomic_json(_calibration_path(config, root), selected)
        calibration = selected
    expected_days = int(manifest["splits"]["validation"]["count"])
    completed_days = int(
        result.get(
            "completed_day_count",
            result.get("validation_day_count", result.get("day_count", 0)),
        )
    )
    status = str(result.get("status", "COMPLETE" if completed_days == expected_days else "PARTIAL"))
    if completed_days != expected_days:
        status = "PARTIAL"
    rows_value = result.get("records", result.get("rows", []))
    rows = rows_value if isinstance(rows_value, Sequence) and not isinstance(rows_value, (str, bytes)) else []
    state = {
        "schema_version": "dcvpp.e2e_moe_r3.formal_validation_state.v1",
        "status": status,
        "phase": "validation_evaluation",
        "split": "validation",
        "active_seed": SEED1,
        "protocol_seeds": list(FORMAL_SEEDS),
        "expected_day_count": expected_days,
        "completed_day_count": completed_days,
        "completed_origin_count": int(result.get("completed_origin_count", completed_days * INTERVALS * 2)),
        "split_manifest_sha256": manifest["manifest_sha256"],
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_state_finite": checkpoint_state.get("best_checkpoint_finite_verified") is True,
        "checkpoint_resumable": checkpoint_state.get("resumable") is True,
        "evaluation_result": result,
        "external_solver": False,
        "formal_2024_opened": False,
        "locked_2025_opened": False,
        "resumable": True,
        "updated_at_utc": utc_now(),
        "resume_command": (
            f"python scripts/run_e2e_moe_r3_formal.py evaluate-validation --config {config_path} --resume"
        ),
    }
    _complete_run(run, title="Seed1 complete 2023-H2 validation", state=state, rows=rows)
    return state


def _aggregate_from_evaluation(
    evaluation: Mapping[str, Any], method: str
) -> dict[str, Any] | None:
    direct_keys = {
        "learned": ("learned", "learned_aggregate", "model_aggregate"),
        "baseline": (
            "baseline",
            "baseline_aggregate",
            "price_aware_edf_shield",
            "benchmark_aggregate",
        ),
    }
    for key in direct_keys[method]:
        value = evaluation.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    aggregates = evaluation.get("aggregates")
    if isinstance(aggregates, Mapping):
        candidates = (
            ("e2e_moe_solver_free_r3", "learned", "model")
            if method == "learned"
            else ("price_aware_edf_shield", "baseline", "benchmark")
        )
        for key in candidates:
            value = aggregates.get(key)
            if isinstance(value, Mapping):
                return dict(value)
    return None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _required_number(
    payload: Mapping[str, Any], *keys: str, label: str
) -> float:
    for key in keys:
        if key in payload and _finite_number(payload[key]):
            return float(payload[key])
    raise FormalProtocolError(f"measured report field is missing/non-finite: {label}")


def _required_mapping(
    payload: Mapping[str, Any], key: str, *, label: str
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise FormalProtocolError(f"measured report mapping is missing: {label}")
    return dict(value)


def _normalized_method_metrics(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    raw_categories = _required_mapping(
        payload,
        "raw_violation_category_counts",
        label=f"{label}.raw_violation_category_counts",
    )
    if not all(
        isinstance(key, str) and _finite_number(value) and float(value) >= 0.0
        for key, value in raw_categories.items()
    ):
        raise FormalProtocolError(
            f"{label}.raw_violation_category_counts must be finite nonnegative counts"
        )
    result = {
        "day_count": int(_required_number(payload, "day_count", label=f"{label}.day_count")),
        "completed_origin_count": int(
            _required_number(
                payload,
                "completed_origin_count",
                label=f"{label}.completed_origin_count",
            )
        ),
        "expected_origin_count": int(
            _required_number(
                payload,
                "expected_origin_count",
                label=f"{label}.expected_origin_count",
            )
        ),
        "feasibility_rate": _required_number(
            payload,
            "feasibility_rate",
            "post_shield_feasibility_fraction",
            label=f"{label}.feasibility_rate",
        ),
        "rejected_origins": int(
            _required_number(payload, "rejected_origins", label=f"{label}.rejected_origins")
        ),
        "rejected_non_exogenous_origins": int(
            _required_number(
                payload,
                "rejected_non_exogenous_origins",
                "non_exogenous_rejected_origins",
                label=f"{label}.rejected_non_exogenous_origins",
            )
        ),
        "exogenous_infeasible_origins": int(
            _required_number(
                payload,
                "exogenous_infeasible_origins",
                label=f"{label}.exogenous_infeasible_origins",
            )
        ),
        "total_cost_usd": _required_number(
            payload,
            "total_cost_usd",
            "operating_cost_usd",
            label=f"{label}.total_cost_usd",
        ),
        "cvar_95_usd": _required_number(
            payload,
            "cvar_95_usd",
            "weighted_particle_cvar_95_usd",
            label=f"{label}.cvar_95_usd",
        ),
        "sla_shortfall_work_units": _required_number(
            payload,
            "sla_shortfall_work_units",
            label=f"{label}.sla_shortfall_work_units",
        ),
        "normalized_sla_shortfall": _required_number(
            payload,
            "normalized_sla_shortfall",
            label=f"{label}.normalized_sla_shortfall",
        ),
        "corrective_intervention_rate": _required_number(
            payload,
            "corrective_intervention_rate",
            "correction_rate",
            label=f"{label}.corrective_intervention_rate",
        ),
        "corrective_magnitude_mean": _required_number(
            payload,
            "corrective_magnitude_mean",
            "corrective_intervention_mean",
            label=f"{label}.corrective_magnitude_mean",
        ),
        "corrective_magnitude_p95": _required_number(
            payload,
            "corrective_magnitude_p95",
            "corrective_intervention_p95",
            label=f"{label}.corrective_magnitude_p95",
        ),
        "policy_inference_time_seconds": _required_number(
            payload,
            "policy_inference_time_seconds",
            label=f"{label}.policy_inference_time_seconds",
        ),
        "constraint_shield_time_seconds": _required_number(
            payload,
            "constraint_shield_time_seconds",
            label=f"{label}.constraint_shield_time_seconds",
        ),
        "raw_violation_category_counts": {
            str(key): int(value) for key, value in sorted(raw_categories.items())
        },
    }
    if result["day_count"] <= 0 or result["expected_origin_count"] <= 0:
        raise FormalProtocolError(f"{label} has no measured validation coverage")
    if result["completed_origin_count"] != result["expected_origin_count"]:
        raise FormalProtocolError(f"{label} did not complete every validation origin")
    return result


def _paired_daily_evidence(
    evaluation: Mapping[str, Any], *, expected_days: int
) -> list[dict[str, Any]]:
    raw = evaluation.get("paired_day_results")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise FormalProtocolError("paired_day_results are absent from measured validation")
    rows: list[dict[str, Any]] = []
    numeric_aliases = {
        "learned_total_cost_usd": ("learned_total_cost_usd", "learned_cost_usd"),
        "baseline_total_cost_usd": ("baseline_total_cost_usd", "baseline_cost_usd"),
        "learned_cvar_95_usd": ("learned_cvar_95_usd", "learned_cvar_usd"),
        "baseline_cvar_95_usd": ("baseline_cvar_95_usd", "baseline_cvar_usd"),
        "learned_sla_shortfall_work_units": ("learned_sla_shortfall_work_units",),
        "baseline_sla_shortfall_work_units": ("baseline_sla_shortfall_work_units",),
        "learned_normalized_sla_shortfall": ("learned_normalized_sla_shortfall",),
        "baseline_normalized_sla_shortfall": ("baseline_normalized_sla_shortfall",),
        "learned_feasibility_rate": ("learned_feasibility_rate",),
        "baseline_feasibility_rate": ("baseline_feasibility_rate",),
        "learned_rejected_non_exogenous_origins": (
            "learned_rejected_non_exogenous_origins",
        ),
        "baseline_rejected_non_exogenous_origins": (
            "baseline_rejected_non_exogenous_origins",
        ),
        "learned_corrective_magnitude": ("learned_corrective_magnitude",),
        "baseline_corrective_magnitude": ("baseline_corrective_magnitude",),
    }
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise FormalProtocolError(f"paired day row {index} is not a mapping")
        date = str(item.get("operating_date", item.get("date", "")))[:10]
        if not (FORMAL_SPLITS["validation"][0] <= date <= FORMAL_SPLITS["validation"][1]):
            raise FormalProtocolError(f"paired day row has out-of-validation date: {date}")
        row: dict[str, Any] = {"operating_date": date}
        for output_key, aliases in numeric_aliases.items():
            row[output_key] = _required_number(
                item,
                *aliases,
                label=f"paired_day_results[{date}].{output_key}",
            )
        row["cost_delta_learned_minus_baseline_usd"] = (
            row["learned_total_cost_usd"] - row["baseline_total_cost_usd"]
        )
        row["cost_reduction_usd"] = -row["cost_delta_learned_minus_baseline_usd"]
        rows.append(row)
    rows.sort(key=lambda item: item["operating_date"])
    dates = [item["operating_date"] for item in rows]
    if len(rows) != expected_days or len(set(dates)) != expected_days:
        raise FormalProtocolError(
            f"paired daily evidence must contain {expected_days} unique dates, got {len(set(dates))}"
        )
    return rows


def _date_block_bootstrap_ci(
    values: Sequence[float],
    *,
    block_length_days: int = 7,
    replicates: int = 10_000,
    seed: int = SEED1,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < block_length_days or not np.isfinite(array).all():
        raise FormalProtocolError("paired daily deltas cannot support the preregistered block CI")
    rng = np.random.default_rng(seed)
    blocks_needed = int(math.ceil(array.size / block_length_days))
    estimates = np.empty(replicates, dtype=float)
    offsets = np.arange(block_length_days)
    for replicate in range(replicates):
        starts = rng.integers(0, array.size, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % array.size
        estimates[replicate] = float(array[indices.reshape(-1)[: array.size]].mean())
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimand": "mean daily cost reduction (baseline minus learned), USD",
        "method": "circular moving-block percentile bootstrap",
        "single_seed": SEED1,
        "block_length_days": block_length_days,
        "replicates": replicates,
        "confidence_level": 0.95,
        "point_estimate_usd": float(array.mean()),
        "lower_usd": float(lower),
        "upper_usd": float(upper),
    }


def _lineage_evidence(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    output = _data_paths(config, root)["output"]
    phases: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    errors: list[str] = []
    all_complete = True
    all_finite = True
    all_resumable = True
    for phase in TRAINING_PHASES:
        matches = sorted(
            path
            for path in output.glob(f"run_id={phase}_selection_seed={SEED1}_*")
            if path.is_dir()
        )
        state = _read_json(matches[-1] / "state.json") if matches else None
        phases[phase] = state
        all_complete &= bool(state and state.get("status") == "COMPLETE")
        all_finite &= bool(
            state
            and state.get("finite_metrics_verified") is True
            and state.get("checkpoint_finite_verified") is True
            and state.get("best_checkpoint_finite_verified") is True
        )
        all_resumable &= bool(state and state.get("resumable") is True)
        if state is None or not matches:
            errors.append(f"missing measured {phase} state")
            continue
        if int(state.get("active_seed", state.get("seed", -1))) != SEED1:
            errors.append(f"{phase} state seed mismatch")
        if state.get("training_regime") != "formal":
            errors.append(f"{phase} state is not formal")
        if state.get("external_solver") is not False:
            errors.append(f"{phase} state does not prove external_solver=false")
        if state.get("formal_2024_opened") is not False:
            errors.append(f"{phase} state opened sealed 2024")
        if state.get("locked_2025_opened") is not False:
            errors.append(f"{phase} state opened locked 2025")
        metrics_path = matches[-1] / "metrics.parquet"
        try:
            metrics = pd.read_parquet(metrics_path)
        except (OSError, ValueError, ImportError) as exc:
            errors.append(f"cannot read {phase} metrics: {exc}")
            continue
        required_columns = {"epoch", "total_loss", "validation_loss"}
        if not required_columns.issubset(metrics.columns):
            errors.append(f"{phase} metrics lack training/validation curve columns")
            continue
        curve_rows: list[dict[str, Any]] = []
        for _, metric in metrics.sort_values("epoch").iterrows():
            if not (
                _finite_number(metric["epoch"])
                and _finite_number(metric["total_loss"])
                and _finite_number(metric["validation_loss"])
            ):
                errors.append(f"{phase} contains non-finite curve metrics")
                curve_rows = []
                break
            curve_rows.append(
                {
                    "epoch": int(metric["epoch"]),
                    "training_loss": float(metric["total_loss"]),
                    "validation_loss": float(metric["validation_loss"]),
                    "rollout_horizon": (
                        int(metric["rollout_horizon"])
                        if "rollout_horizon" in metrics.columns
                        and _finite_number(metric.get("rollout_horizon"))
                        else None
                    ),
                }
            )
        if curve_rows:
            curves[phase] = curve_rows
    return {
        "phases": phases,
        "curves": curves,
        "errors": errors,
        "lineage_complete": all_complete,
        "finite": all_finite,
        "resumable": all_resumable,
    }


def write_seed1_report(config_path: Path | str) -> dict[str, Any]:
    """Write reports only when every required field has measured provenance."""

    def failed(reason: str) -> dict[str, Any]:
        return {
            "status": "MAINLINE_FAILED",
            "terminal_status": "MAINLINE_FAILED",
            "reason": reason,
            "reports_written": False,
        }

    try:
        config_path = Path(config_path).resolve()
        config, root = load_formal_config(config_path)
        output = _data_paths(config, root)["output"]
        validation_runs = sorted(
            path
            for path in output.glob(
                f"run_id=validation_evaluation_validation_seed={SEED1}_*"
            )
            if path.is_dir()
        )
        if not validation_runs:
            return failed("no measured Seed1 complete-validation run")
        validation_run = validation_runs[-1]
        state = _read_json(validation_run / "state.json")
        if state is None or state.get("status") != "COMPLETE":
            return failed("latest Seed1 validation state is not COMPLETE")
        for key, expected_value in (
            ("external_solver", False),
            ("formal_2024_opened", False),
            ("locked_2025_opened", False),
            ("active_seed", SEED1),
        ):
            if state.get(key) != expected_value:
                return failed(f"validation state has invalid or missing {key}")

        manifest = _read_json(output / "formal_split_manifest.json")
        provenance = _read_json(output / "provenance_manifest.json")
        calibration = _read_json(output / "price_aware_edf_calibration.json")
        if manifest is None or provenance is None or calibration is None:
            return failed("formal manifest, provenance, or frozen calibration is missing")
        validation_manifest = _mapping(
            _mapping(manifest.get("splits"), "manifest.splits").get("validation"),
            "manifest.splits.validation",
        )
        expected = int(validation_manifest.get("count", 0))
        completed = int(state.get("completed_day_count", 0))
        if expected <= 0 or completed != expected:
            return failed(
                f"Seed1 validation covers {completed}/{expected} eligible dates"
            )
        if state.get("split_manifest_sha256") != manifest.get("manifest_sha256"):
            return failed("validation/split manifest fingerprint mismatch")
        if calibration.get("battery_fractions_frozen") is not True:
            return failed("Price-aware EDF battery-rule selection is not frozen")
        candidate_results = calibration.get("battery_candidate_results")
        if not isinstance(candidate_results, Sequence) or len(candidate_results) != 9:
            return failed("complete nine-candidate benchmark calibration is missing")

        evaluation = _mapping(state.get("evaluation_result"), "evaluation_result")
        evaluation_completed = _required_number(
            evaluation,
            "completed_day_count",
            "validation_day_count",
            label="evaluation.completed_day_count",
        )
        if int(evaluation_completed) != expected:
            return failed("evaluation result does not cover the canonical validation split")
        if evaluation.get("formal_2024_opened") is not False:
            return failed("evaluation does not prove sealed 2024 remained closed")
        if evaluation.get("locked_2025_opened") is not False:
            return failed("evaluation does not prove locked 2025 remained closed")
        learned_raw = _aggregate_from_evaluation(evaluation, "learned")
        baseline_raw = _aggregate_from_evaluation(evaluation, "baseline")
        if learned_raw is None or baseline_raw is None:
            return failed("measured learned/baseline aggregates are absent")
        learned = _normalized_method_metrics(learned_raw, label="learned")
        baseline = _normalized_method_metrics(
            baseline_raw, label="price_aware_edf_shield"
        )
        if learned["day_count"] != expected or baseline["day_count"] != expected:
            return failed("method aggregates do not cover every validation date")
        expected_origins = expected * INTERVALS
        if (
            learned["expected_origin_count"] != expected_origins
            or baseline["expected_origin_count"] != expected_origins
        ):
            return failed("method aggregates have the wrong 96-origin denominator")
        paired_days = _paired_daily_evidence(evaluation, expected_days=expected)
        paired_learned_cost = float(
            sum(item["learned_total_cost_usd"] for item in paired_days)
        )
        paired_baseline_cost = float(
            sum(item["baseline_total_cost_usd"] for item in paired_days)
        )
        if not math.isclose(
            paired_learned_cost,
            learned["total_cost_usd"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        ) or not math.isclose(
            paired_baseline_cost,
            baseline["total_cost_usd"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            return failed("paired daily costs do not reconcile to method aggregates")

        lineage = _lineage_evidence(config, root)
        if (
            not lineage["lineage_complete"]
            or not lineage["finite"]
            or not lineage["resumable"]
            or lineage["errors"]
            or set(lineage["curves"]) != set(TRAINING_PHASES)
        ):
            return failed(
                "lineage/checkpoint/curve evidence is incomplete: "
                + "; ".join(lineage["errors"])
            )

        baseline_scale = abs(float(baseline["total_cost_usd"]))
        if baseline_scale <= 1e-12:
            return failed("cost-reduction percentage is undefined for zero baseline cost")
        cost_reduction_usd = float(
            baseline["total_cost_usd"] - learned["total_cost_usd"]
        )
        cost_reduction_percent = 100.0 * cost_reduction_usd / baseline_scale
        daily_reductions = [float(item["cost_reduction_usd"]) for item in paired_days]
        paired_summary = {
            "daily_cost_reduction_definition": "baseline minus learned; positive favors learned",
            "mean_daily_cost_reduction_usd": float(np.mean(daily_reductions)),
            "median_daily_cost_reduction_usd": float(np.median(daily_reductions)),
            "win_day_count": int(np.sum(np.asarray(daily_reductions) > 0.0)),
            "tie_day_count": int(np.sum(np.asarray(daily_reductions) == 0.0)),
            "loss_day_count": int(np.sum(np.asarray(daily_reductions) < 0.0)),
            "win_rate": float(np.mean(np.asarray(daily_reductions) > 0.0)),
            "date_block_95_percent_ci": _date_block_bootstrap_ci(daily_reductions),
            "days": paired_days,
        }
        evidence = {
            "finite_metrics": True,
            "checkpoints_finite": True,
            "checkpoints_resumable": True,
            "validation_complete": True,
            "external_solver": False,
            "formal_2024_opened": False,
            "locked_2025_opened": False,
        }
        gate = evaluate_gate(learned, baseline, evidence)
        terminal_status = (
            "MAINLINE_PASS" if gate["all_passed"] else "MAINLINE_NEEDS_IMPROVEMENT"
        )
        phase_lineage: dict[str, Any] = {}
        for phase, phase_state in lineage["phases"].items():
            assert isinstance(phase_state, Mapping)
            phase_lineage[phase] = {
                "status": phase_state.get("status"),
                "active_seed": phase_state.get("active_seed"),
                "completed_epochs": phase_state.get("completed_epochs"),
                "target_epochs": phase_state.get("target_epochs"),
                "early_stopped": phase_state.get("early_stopped"),
                "best_epoch": phase_state.get("best_epoch"),
                "best_validation_loss": phase_state.get("best_validation_loss"),
                "latest_checkpoint_sha256": phase_state.get(
                    "latest_checkpoint_sha256"
                ),
                "best_checkpoint_sha256": phase_state.get("best_checkpoint_sha256"),
                "predecessor_checkpoint_sha256": phase_state.get(
                    "predecessor_checkpoint_sha256"
                ),
                "resumable": phase_state.get("resumable"),
                "finite_metrics_verified": phase_state.get(
                    "finite_metrics_verified"
                ),
                "best_checkpoint_finite_verified": phase_state.get(
                    "best_checkpoint_finite_verified"
                ),
            }
        result = {
            "schema": "dcvpp.e2e_moe_r3.mainline_seed1_results.v2",
            "terminal_status": terminal_status,
            "generated_at_utc": utc_now(),
            "objective": {
                "method": "E2E-MoE-DART-VPP R3 solver-free constrained policy",
                "primary_comparator": "price_aware_edf_shield",
                "selection_split": "every eligible 2023-H2 validation date",
                "eligible_date_count": expected,
                "rolling_origins_per_day": INTERVALS,
                "common_learned_particles": SCENARIOS,
                "cvar_alpha": 0.95,
                "gate": {
                    "feasibility_rate": 1.0,
                    "maximum_non_exogenous_rejected_origins": 0,
                    "strictly_positive_cost_reduction": True,
                    "maximum_normalized_sla_shortfall_increase": 0.001,
                    "finite_resumable_lineage_required": True,
                },
            },
            "protocol": config["protocol"]["name"],
            "active_seed": SEED1,
            "protocol_seeds": list(FORMAL_SEEDS),
            "splits": FORMAL_SPLITS,
            "split_manifest_sha256": manifest["manifest_sha256"],
            "provenance_config_sha256": provenance.get("config_sha256"),
            "benchmark_calibration": calibration,
            "lineage": {
                "complete": True,
                "finite": True,
                "resumable": True,
                "phases": phase_lineage,
            },
            "training_validation_curves": lineage["curves"],
            "validation": {
                "run_dir": str(validation_run),
                "eligible_day_count": expected,
                "completed_day_count": completed,
                "origins_per_day": INTERVALS,
                "common_scenario_count": SCENARIOS,
            },
            "learned": learned,
            "price_aware_edf_shield": baseline,
            "paired_comparison": {
                "cost_reduction_usd": cost_reduction_usd,
                "cost_reduction_percent_of_absolute_baseline": cost_reduction_percent,
                **paired_summary,
            },
            "gate": gate,
            "claims_boundary": {
                "external_solver": False,
                "formal_2024_opened": False,
                "locked_2025_opened": False,
                "formal_2024_claims_allowed": False,
            },
        }
        reports = _mapping(
            _mapping(config.get("protocol"), "protocol").get("reports"),
            "protocol.reports",
        )
        json_path = _absolute(root, reports.get("results_json"))
        markdown_path = _absolute(root, reports.get("markdown"))
        markdown = [
            "# MAINLINE Seed1 Report",
            "",
            f"Terminal label: **{terminal_status}**",
            "",
            "## Objective and evidence boundary",
            "",
            f"- Seed: `{SEED1}` (the first of `{list(FORMAL_SEEDS)}`)",
            f"- Validation coverage: `{completed}/{expected}` eligible 2023-H2 days, `{INTERVALS}` rolling origins/day",
            f"- Risk accounting: exact 95% CVaR over the same `{SCENARIOS}` learned particles",
            "- External solver: `False`",
            "- 2024 formal selection payload opened: `False`",
            "- 2025 locked payload opened: `False`",
            "",
            "## Paired operating result",
            "",
            f"- Learned total cost: `{learned['total_cost_usd']}` USD",
            f"- Price-aware EDF total cost: `{baseline['total_cost_usd']}` USD",
            f"- Cost reduction: `{cost_reduction_usd}` USD (`{cost_reduction_percent}`% of absolute baseline cost)",
            f"- Daily win rate: `{paired_summary['win_rate']}` ({paired_summary['win_day_count']}/{expected})",
            f"- Date-block 95% CI for mean daily reduction: `[{paired_summary['date_block_95_percent_ci']['lower_usd']}, {paired_summary['date_block_95_percent_ci']['upper_usd']}]` USD",
            f"- Learned / baseline 20-particle CVaR: `{learned['cvar_95_usd']}` / `{baseline['cvar_95_usd']}` USD",
            f"- Learned / baseline SLA shortfall: `{learned['sla_shortfall_work_units']}` / `{baseline['sla_shortfall_work_units']}` work units",
            f"- Learned / baseline normalized SLA shortfall: `{learned['normalized_sla_shortfall']}` / `{baseline['normalized_sla_shortfall']}`",
            f"- Learned feasibility / non-exogenous rejects: `{learned['feasibility_rate']}` / `{learned['rejected_non_exogenous_origins']}`",
            "",
            "## Constraint-shield accounting",
            "",
            f"- Learned corrective rate / mean / P95: `{learned['corrective_intervention_rate']}` / `{learned['corrective_magnitude_mean']}` / `{learned['corrective_magnitude_p95']}`",
            f"- Baseline corrective rate / mean / P95: `{baseline['corrective_intervention_rate']}` / `{baseline['corrective_magnitude_mean']}` / `{baseline['corrective_magnitude_p95']}`",
            f"- Learned raw violation categories: `{json.dumps(learned['raw_violation_category_counts'], sort_keys=True)}`",
            f"- Baseline raw violation categories: `{json.dumps(baseline['raw_violation_category_counts'], sort_keys=True)}`",
            f"- Learned inference / shield latency: `{learned['policy_inference_time_seconds']}` / `{learned['constraint_shield_time_seconds']}` seconds",
            f"- Baseline inference / shield latency: `{baseline['policy_inference_time_seconds']}` / `{baseline['constraint_shield_time_seconds']}` seconds",
            "",
            "## Gate",
            "",
        ]
        markdown.extend(
            f"- {name}: `{'PASS' if passed else 'FAIL'}`"
            for name, passed in gate["checks"].items()
        )
        markdown.extend(
            [
                "",
                "## Lineage and curves",
                "",
                f"- All five phases complete, finite and resumable: `{lineage['lineage_complete'] and lineage['finite'] and lineage['resumable']}`",
                f"- Phase best epochs: `{json.dumps({phase: values['best_epoch'] for phase, values in phase_lineage.items()}, sort_keys=True)}`",
                f"- Training/validation curve points: `{json.dumps({phase: len(values) for phase, values in lineage['curves'].items()}, sort_keys=True)}`",
                "",
                "> Validation-stage evidence only. This report does not contain or imply 2024 formal-test results.",
                "",
            ]
        )
        # Stage both artifacts before publishing either exact path.  If staging
        # fails, neither final report exists.  Final renames are local and
        # atomic on the repository volume.
        json_stage = json_path.with_name(json_path.name + ".staged")
        markdown_stage = markdown_path.with_name(markdown_path.name + ".staged")
        atomic_json(json_stage, result)
        try:
            atomic_text(markdown_stage, "\n".join(markdown))
            json_stage.replace(json_path)
            markdown_stage.replace(markdown_path)
        finally:
            json_stage.unlink(missing_ok=True)
            markdown_stage.unlink(missing_ok=True)
        return {
            "status": terminal_status,
            "terminal_status": terminal_status,
            "reports_written": True,
            "markdown": str(markdown_path),
            "results_json": str(json_path),
            "gate": gate,
        }
    except (FormalProtocolError, KeyError, TypeError, ValueError) as exc:
        return failed(str(exc))


def run_mainline_seed1(
    config_path: Path | str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    prepared = prepare_protocol(config_path, calibrate=True)
    calibration = prepared.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("battery_fractions_frozen") is not True:
        return {
            "status": "BLOCKED",
            "stopped_before": "preflight",
            "reason": "primary Price-aware EDF benchmark was not fully frozen",
        }
    lineage = run_seed1_lineage(config_path, resume=resume)
    if lineage.get("status") != "COMPLETE":
        return lineage
    day = run_initial_validation_day(config_path, resume=resume)
    if day.get("status") != "COMPLETE":
        return day
    validation = run_full_validation(config_path, resume=resume)
    if validation.get("status") != "COMPLETE":
        return validation
    return write_seed1_report(config_path)


__all__ = [
    "FormalProtocolError",
    "build_split_manifest",
    "calibrate_price_aware_edf",
    "evaluate_gate",
    "load_formal_config",
    "prepare_protocol",
    "run_full_validation",
    "run_initial_validation_day",
    "run_mainline_seed1",
    "run_preflight",
    "run_seed1_lineage",
    "validate_formal_config",
    "write_seed1_report",
]
