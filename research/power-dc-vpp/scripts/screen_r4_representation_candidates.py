"""Screen every frozen R4 Representation candidate with the paper gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from dcvpp.e2e_moe_r4.artifact_io import atomic_json
from dcvpp.e2e_moe_r4.capability_evaluation import evaluate_representation
from dcvpp.e2e_moe_r4.capability_materialization import (
    HEADS,
    LiveRepresentationCapabilityEvaluator,
    _daily_metrics,
    _target,
)
from dcvpp.e2e_moe_r4.config import load_r4_config
from dcvpp.e2e_moe_r4.hashing import sha256_file, sha256_json
from dcvpp.e2e_moe_r4.physical_twin import build_formal_r4_model
from dcvpp.e2e_moe_r4.training_data import PreloadedR4Corpus


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root / "configs" / "e2e_moe_dart_vpp_R4_hybrid_solver_free_formal.yaml"
    )
    phase = (
        root
        / "artifacts"
        / "e2e_moe_hybrid_r4_v5_formal"
        / "seed=20240813"
        / "phase=representation"
    )
    registry_path = (
        phase
        / "provisional_best"
        / "representation"
        / "seed=20240813"
        / "provisional_registry.json"
    )
    output_root = root / "artifacts" / "run_control" / "r4_v5" / "candidate_screen"
    result_path = output_root / "candidate_screen.json"
    output_root.mkdir(parents=True, exist_ok=True)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidates = list(registry["candidates"])
    config = load_r4_config(config_path)
    corpus = PreloadedR4Corpus(config, root)
    baseline = LiveRepresentationCapabilityEvaluator(
        corpus, output_root, seed=20240813
    )
    records_by_date = {
        str(row["operating_date"]): row for row in corpus.records("validation")
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_formal_r4_model(config, root).to(device).eval()

    completed: dict[str, Any] = {}
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("registry_sha256") == sha256_file(registry_path):
            completed = {
                str(row["checkpoint_sha256"]): row
                for row in prior.get("candidates", [])
            }

    for candidate in candidates:
        epoch = int(candidate["epoch"])
        source = (
            phase
            / "provisional_best"
            / "representation"
            / "seed=20240813"
            / f"epoch{epoch:04d}.pt"
        )
        digest = sha256_file(source)
        if digest != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"candidate hash mismatch at epoch {epoch}")
        if digest in completed:
            continue
        payload = torch.load(source, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch in corpus.batches(
                "validation", 179, device=device, origin=0
            ):
                output = model(batch)
                forecasts = output["forecasts"]
                dates = [str(value) for value in batch["operating_dates"]]
                for index, day_text in enumerate(dates):
                    record = records_by_date[day_text]
                    baselines = baseline.validation_baselines[day_text][
                        baseline.selected_baseline
                    ]
                    row: dict[str, Any] = {
                        "date": day_text,
                        "checkpoint_sha256": digest,
                        "checkpoint_epoch": epoch,
                        "baseline_name": baseline.selected_baseline,
                        "baseline_sha256": baseline.baseline_sha256,
                        "causal_persistence_sha256": baseline.definition_hashes[
                            "causal_persistence"
                        ],
                        "seasonal_naive_sha256": baseline.definition_hashes[
                            "seasonal_naive"
                        ],
                        "future_leakage_audit_status": "NOT_RUN",
                    }
                    for head in HEADS:
                        target = _target(record, head)
                        prediction = forecasts[head][index].detach().cpu()
                        for name, value in _daily_metrics(prediction, target).items():
                            row[f"model_{head}_{name}"] = value
                        for name, value in _daily_metrics(
                            baselines[head], target
                        ).items():
                            row[f"baseline_{head}_{name}"] = value
                    identity = (
                        forecasts["da"][index]
                        - forecasts["rt"][index]
                        - forecasts["dart"][index]
                    )
                    row["da_rt_dart_identity_max_abs"] = float(
                        identity.abs().max().cpu()
                    )
                    row["evidence_sha256"] = sha256_json(row)
                    rows.append(row)
        decision = evaluate_representation(
            rows, baseline=baseline.selected_baseline, seed=20240813
        )
        completed[digest] = {
            "epoch": epoch,
            "cheap_validation": float(candidate["cheap_validation"]),
            "checkpoint": str(source),
            "checkpoint_sha256": digest,
            "status": decision.status,
            "passed": decision.passed,
            "reasons": list(decision.reasons),
            "metrics": dict(decision.metrics),
            "daily_evidence_sha256": sha256_json(rows),
        }
        atomic_json(
            result_path,
            {
                "schema": "dcvpp.r4.representation_candidate_screen.v1",
                "decision_rule": (
                    "paired_daily_nmae_model_minus_baseline_7day_block_"
                    "10000_one_sided_95_upper_strictly_below_zero"
                ),
                "registry": str(registry_path),
                "registry_sha256": sha256_file(registry_path),
                "candidate_count": len(candidates),
                "completed_count": len(completed),
                "candidates": sorted(completed.values(), key=lambda row: row["epoch"]),
            },
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "status": decision.status,
                    "reasons": list(decision.reasons),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
