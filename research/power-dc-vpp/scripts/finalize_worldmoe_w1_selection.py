"""Finalize a paused W1 lineage on one explicitly selected formal checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch

from dcvpp.vpp_world_moe.gates import evaluate_phase_release_gate
from dcvpp.vpp_world_moe.runner import (
    PHASE_RELEASE_SCHEMA,
    RUN_STATE_SCHEMA,
    _sha256,
    _utc_now,
)
from dcvpp.vpp_world_moe.trainer import CHECKPOINT_SCHEMA


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    phase_root = args.phase_root.resolve()
    checkpoint = args.checkpoint.resolve()
    state_path = phase_root / "state.json"
    gate_path = phase_root / "gate.json"
    backup_path = phase_root / "state.before_user_epoch_selection.json"
    if not state_path.is_file() or not checkpoint.is_file():
        raise SystemExit("phase state or selected checkpoint is missing")
    if gate_path.exists():
        raise SystemExit("refusing to replace an existing W1 gate")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if (
        state.get("schema") != RUN_STATE_SCHEMA
        or state.get("phase") != "w1"
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("phase") != "w1"
        or payload.get("run_id") != state.get("run_id")
    ):
        raise SystemExit("W1 state/checkpoint lineage mismatch")
    resume_state = payload.get("resume_state")
    if (
        not isinstance(resume_state, Mapping)
        or int(resume_state.get("epoch", -1)) != args.epoch
        or int(resume_state.get("batch_in_short_epoch", -1)) != 0
    ):
        raise SystemExit("checkpoint is not the requested completed W1 epoch")
    extra = payload.get("extra")
    formal = extra.get("formal_selection") if isinstance(extra, Mapping) else None
    if not isinstance(formal, Mapping):
        raise SystemExit("selected checkpoint has no formal validation evidence")

    checkpoint_id = _sha256(checkpoint)
    native = evaluate_phase_release_gate(
        phase="w1", checkpoint_id=checkpoint_id, formal_selection=formal
    )
    if not native.passed:
        raise SystemExit("selected W1 checkpoint failed its native capability gate")

    selected_record = {
        "phase": "w1",
        "epoch": args.epoch,
        "primary_score": float(formal["primary_score"]),
        "eligible": bool(formal["eligible"]),
        "selection_metrics": dict(formal["selection_metrics"]),
        "selection_basis": "explicit_user_selected_formal_primary_checkpoint",
    }
    selected_at = _utc_now()
    gate = {
        "schema": PHASE_RELEASE_SCHEMA,
        "run_id": state["run_id"],
        "phase": "w1",
        "status": "PASS",
        "passed": True,
        "next_phase_released": True,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_selection_role": f"user_selected_formal_primary_epoch{args.epoch}",
        "release_candidate_existed": True,
        "native_evidence": native.as_dict(),
        "user_selection": selected_record,
        "locked_2025_opened": False,
        "evaluated_at_utc": selected_at,
    }

    if not backup_path.exists():
        shutil.copy2(state_path, backup_path)
    stopped_resume = dict(state.get("resume_state", {}))
    active_resume = dict(stopped_resume)
    active_resume["selected_record_json"] = json.dumps(
        selected_record, sort_keys=True, separators=(",", ":")
    )
    training_control = dict(state.get("training_control", {}))
    training_control["release_record"] = selected_record
    training_control["formal_primary_record"] = selected_record
    state.update(
        {
            "status": "CAPABILITY_GATE_PASSED",
            "best_validation": float(formal["primary_score"]),
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_exists": True,
            "best_checkpoint_selection_role": (
                f"user_selected_formal_primary_epoch{args.epoch}"
            ),
            "gate_path": str(gate_path.resolve()),
            "next_phase_released": True,
            "resume_state": active_resume,
            "training_control": training_control,
            "stopped_training_resume_state": stopped_resume,
            "user_selected_checkpoint": {
                "epoch": args.epoch,
                "path": str(checkpoint),
                "sha256": checkpoint_id,
                "formal_primary_score": float(formal["primary_score"]),
                "selected_at_utc": selected_at,
            },
            "updated_at_utc": selected_at,
        }
    )
    _atomic_json(gate_path, gate)
    _atomic_json(state_path, state)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_epoch": args.epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_id": checkpoint_id,
                "formal_primary_score": float(formal["primary_score"]),
                "gate": str(gate_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
