"""Serialization-only types for the R4 ablation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AblationExecutionState:
    schema_version: str
    status: str
    method_id: str
    architecture_id: str
    seed: int
    split: str
    phase_sequence: tuple[str, ...]
    phases: Mapping[str, Any]
    checkpoint_source: str
    ablation_lineage_id: str
    preregistration_sha256: str
    shared_contract_sha256: str
    reason: str | None
    source: Mapping[str, Any]
    resume_command: str

    def to_payload(self) -> dict[str, Any]:
        if self.status not in {"COMPLETE", "PARTIAL", "FAILED"}:
            raise ValueError(f"invalid ablation execution status {self.status!r}")
        if self.split != "train_validation_only":
            raise ValueError("ablation execution state cannot name 2024/2025")
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "method_id": self.method_id,
            "architecture_id": self.architecture_id,
            "seed": int(self.seed),
            "split": self.split,
            "phase_sequence": list(self.phase_sequence),
            "phases": dict(self.phases),
            "checkpoint_source": self.checkpoint_source,
            "ablation_lineage_id": self.ablation_lineage_id,
            "preregistration_sha256": self.preregistration_sha256,
            "shared_contract_sha256": self.shared_contract_sha256,
            "reason": self.reason,
            "source": dict(self.source),
            "formal_selection_2024_opened": False,
            "locked_test_2025_opened": False,
            "resume_command": self.resume_command,
        }


__all__ = ["AblationExecutionState"]
