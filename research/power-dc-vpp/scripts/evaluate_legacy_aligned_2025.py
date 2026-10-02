"""Supplement the historical R130 run with current seven-condition 2025 metrics.

Only inference is executed. The original checkpoints, historical results, and
the completed three-seed study's protocol/state are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# Preserve the verified Windows DLL loading order before importing Torch.
import pyarrow.dataset  # noqa: E402,F401
import evaluate_inference_ablation_2025 as core  # noqa: E402
from dcvpp.vpp_world_moe.r_ecmoe_runtime import RECMoEInferenceRuntime  # noqa: E402
from multiseed_common import now, read_json, write_json  # noqa: E402

OUTPUT = REPO / "artifacts/r_ecmoe_legacy_aligned_20260912"
OLD_RESULTS = REPO / "artifacts/r_ecmoe_mechanisms_2025_20260910/epoch130-seed20240813-20260910T204056"
SEED = 20240813
ROLE = "supplemental_historical_checkpoint_metric_alignment"


def inspect_legacy():
    launch = read_json(OLD_RESULTS / "launch_inputs.json", {})
    historical = read_json(OLD_RESULTS / "evaluation_2025_meta.json", {})
    checkpoint = core.base._path(historical["best_checkpoint"])
    w1_checkpoint = core.base._path(historical["w1_checkpoint"])
    payload = core.torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != 130 or int(payload["resolved_config"]["seed"]) != SEED:
        raise ValueError("Historical alignment requires the original R130/seed20240813")
    selection = core.base.validate_selected_checkpoint(payload)
    if core.base._path(payload["source_paths"][0]) != w1_checkpoint:
        raise ValueError("Historical R130 source W1 does not match the original evaluation")
    w1_state = read_json(w1_checkpoint.parent.parent / "state.json", {})
    parent = w1_state["parent_checkpoint"]
    if int(parent["source_epoch"]) != 96 or int(parent["source_update"]) != 732:
        raise ValueError("Historical W0 lineage differs from the selected epoch96 parent")
    w0_checkpoint = core.base._path(parent["path"])
    w0_state = read_json(w0_checkpoint.parent.parent / "state.json", {})
    if int(w0_state["epochs_completed"]) != 96:
        raise ValueError("Historical W0 completion record changed")
    model_mapping = launch["main_config_resolved"]["model"]
    if model_mapping.get("active_top_k") != 2 or model_mapping.get("mechanism_variant", "full") != "full":
        raise ValueError("Historical source must be the complete Top-2 model")
    continuation = read_json(checkpoint.parent.parent / "continuation_lineage.json", {})
    last_metrics = read_json(checkpoint.parent.parent / "latest_metrics.json", {})
    record = {"seed": SEED, "variant": "full", "checkpoint": str(checkpoint),
        "config": str(checkpoint.parent.parent / "config_resolved.yaml"),
        "w1_checkpoint": str(w1_checkpoint),
        "world_config": str(core.base._path(launch["main_config_resolved"]["source"]["world_config"])),
        "checkpoint_epoch": 130, "checkpoint_run_id": payload["run_id"],
        "train_frozen_cost_scale": float(payload["train_frozen_cost_scale"]),
        "configuration_snapshot": str(OLD_RESULTS / "launch_inputs.json"), **selection}
    lineage = {"w0_selected_epoch": 96, "w0_selected_update": 732,
        "w0_maximum_short_epochs": 96, "w0_checkpoint": str(w0_checkpoint),
        "w1_selected_epoch": 56, "w1_selected_update": 3360, "w1_maximum_epochs": 80,
        "w1_last_recorded_epoch": int(w1_state.get("stopped_training_resume_state", w1_state["resume_state"])["epoch"]),
        "w1_selection_role": w1_state.get("best_checkpoint_selection_role"),
        "r_selected_epoch": 130, "r_selected_update": int(payload["global_update"]),
        "r_initial_budget": int(continuation["source_epoch"]),
        "r_continuation_maximum_epoch": int(continuation["maximum_epoch"]),
        "r_last_recorded_training_epoch": int(last_metrics["epoch"]),
        "r_completion_note": "training_closed_historically; epoch168 is last recorded sentinel, not proven automatic early_stop"}
    return launch, historical, record, lineage


def _aligned_metadata(payload, protocol, lineage, *, started_at, elapsed_seconds):
    result = dict(payload)
    # These fields describe the new sequential training study and would be
    # misleading for a historical checkpoint selected under a different budget.
    for key in ("seed_base_training_completed", "completed_base_stages",
                "other_seeds_training_required_before_this_evaluation",
                "initial_configurations_frozen_before_first_test",
                "future_seed_configurations_frozen_before_first_test",
                "later_seed_training_after_prior_seed_test", "configuration_amendments"):
        result.pop(key, None)
    result.update(evaluation_role=ROLE, run_role="legacy_reference",
        historical_training_closed=True, prior_2025_already_evaluated=True,
        supplementary_evaluation_started_at=started_at,
        supplementary_evaluation_date=started_at[:10], elapsed_seconds=elapsed_seconds,
        original_results_directory=str(OLD_RESULTS), original_results_preserved=True,
        original_training_run_id=protocol["full_model_record"]["checkpoint_run_id"],
        historical_training_record=lineage, new_training_performed=False,
        training_performed=False, checkpoint_selection_performed=False,
        contributes_additional_unique_seed=False,
        configuration_revision_disclosure="historical_reference_supplemental_metric_alignment")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    launch, historical, record, lineage = inspect_legacy()
    if args.validate_only:
        print({"r_epoch": record["checkpoint_epoch"], "w1_epoch": lineage["w1_selected_epoch"],
               "w0_epoch": lineage["w0_selected_epoch"], "training_performed": False})
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = core.torch.device(args.device)
    if device.type == "cuda":
        if not core.torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        core.torch.backends.cuda.matmul.allow_tf32 = True
        core.torch.backends.cudnn.allow_tf32 = True
    protocol = {"schema": "dcvpp.legacy-aligned-evaluation.protocol.v1", "seed": SEED,
        "full_model_record": record, "comparison_design": core.DESIGN, "separately_retrained": False,
        "variants": list(core.VARIANTS), "expected_days": 362, "decisions_per_day": 6,
        "expected_replay_rows": 15204, "evaluation_role": ROLE,
        "prior_2025_already_evaluated": True, "training_performed": False,
        "checkpoint_selection_performed": False, "final_results_may_modify_selection": False,
        "historical_training_record": lineage,
        "original_results_directory": str(OLD_RESULTS), "contributes_additional_unique_seed": False}
    protocol_path = output / "alignment_protocol.json"
    previous = read_json(protocol_path)
    if previous is not None and previous != protocol:
        raise ValueError("Historical alignment protocol changed")
    if previous is None:
        write_json(protocol_path, protocol)
    normalizer = core.base._path(launch["world_config_resolved"]["data"]["normalization"]["artifact_path"])
    inputs = [Path(record["checkpoint"]), Path(record["w1_checkpoint"]), Path(lineage["w0_checkpoint"]),
              normalizer, OLD_RESULTS / "launch_inputs.json", OLD_RESULTS / "evaluation_2025_meta.json",
              protocol_path, Path(__file__), Path(core.__file__), Path(core.base.__file__)]
    hashes = {str(path): core.base._sha256(path) for path in inputs}
    manifest = {"schema": "dcvpp.legacy-aligned-evaluation.freeze.v1", "seed": SEED,
        "full_model_record": record, "protocol_path": str(protocol_path),
        "protocol_sha256": hashes[str(protocol_path)], "sha256": hashes,
        "evaluation_role": ROLE, "comparison_design": core.DESIGN, "separately_retrained": False,
        "historical_training_closed": True, "historical_training_record": lineage,
        "prior_2025_already_evaluated": True, "new_training_performed": False,
        "final_results_may_modify_selection": False}
    manifest_path = output / "freeze_manifest.json"
    previous = read_json(manifest_path)
    if previous is not None and previous != manifest:
        raise ValueError("Historical alignment frozen inputs changed")
    if previous is None:
        write_json(manifest_path, manifest)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    original_loader, original_writer, original_role = core.base._load_models, core.write_json, core.ROLE

    def load_legacy(records, runtime_device):
        if len(records) != 1 or records[0]["checkpoint"] != record["checkpoint"]:
            raise ValueError("Alignment cannot substitute another model")
        runtime = RECMoEInferenceRuntime.from_w1_checkpoint(
            launch["world_config_resolved"], launch["main_config_resolved"]["model"],
            record["w1_checkpoint"], runtime_device, r_checkpoint_path=record["checkpoint"])
        if runtime.checkpoint_load["epoch"] != 56 or runtime.checkpoint_load["update"] != 3360:
            raise ValueError("Loaded historical W1 is not the frozen epoch56 model")
        records[0].update(record)
        return launch["world_config_resolved"], {"full": runtime.model}

    def write_aligned(path, payload):
        if Path(path).name in {"metadata.json", "completed.json", "frozen_evaluation_marker.json"}:
            payload = _aligned_metadata(payload, protocol, lineage, started_at=started_at,
                                        elapsed_seconds=time.monotonic() - started)
        original_writer(path, payload)

    try:
        core.base._load_models, core.write_json, core.ROLE = load_legacy, write_aligned, ROLE
        result = core.evaluate_seed(manifest, manifest_path, output, device=device, batch_size=16)
        result = read_json(output / "metadata.json", result)
        print({"status": result["status"], "observed_replay_rows": result["observed_replay_rows"],
               "output": str(output), "elapsed_seconds": result.get("elapsed_seconds")}, flush=True)
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "updated_at": now(), "training_performed": False, "partial_results_preserved": True})
        raise
    finally:
        core.base._load_models, core.write_json, core.ROLE = original_loader, original_writer, original_role
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
