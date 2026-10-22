from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EVIDENCE_ROLES = {
    "observed",
    "derived",
    "reconstructed",
    "replayed",
    "assumed",
    "oracle",
    "counterfactual",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, indent=2, sort_keys=True, default=str)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(value)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def code_tree_hash(root: Path) -> str:
    records: list[tuple[str, str]] = []
    for path in sorted((root / "src").rglob("*.py")):
        records.append((path.relative_to(root).as_posix(), sha256_file(path)))
    for path in sorted((root / "configs").glob("*.yaml")):
        records.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return sha256_json(records)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot(root: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(root),
        "code_tree_hash": code_tree_hash(root),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot["memory_bytes"] = {"total": int(memory.total), "available": int(memory.available)}
    except (ImportError, OSError):
        snapshot["memory_bytes"] = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        snapshot["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else [],
        }
    except (ImportError, RuntimeError):
        snapshot["torch"] = None
    return snapshot


@dataclass
class StageRecord:
    name: str
    status: str
    fingerprint: str
    started_at: str
    finished_at: str | None = None
    outputs: list[str] | None = None
    message: str | None = None


class PipelineState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "artifacts/state/pipeline_state.json"
        self.status_path = root / "artifacts/status/CURRENT_STAGE_STATUS.md"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "stages": {}}

    def completed(self, name: str, fingerprint: str) -> bool:
        record = self.data.get("stages", {}).get(name, {})
        if record.get("status") != "completed" or record.get("fingerprint") != fingerprint:
            return False
        return all((self.root / p).exists() for p in record.get("outputs", []))

    def set(self, record: StageRecord) -> None:
        self.data.setdefault("stages", {})[record.name] = asdict(record)
        self.data["updated_at"] = utc_now()
        atomic_json(self.path, self.data)
        atomic_text(
            self.status_path,
            f"# Current Stage Status\n\n- Run ID: `{self.data['run_id']}`\n- Stage: `{record.name}`\n- Status: `{record.status}`\n- Updated: `{self.data['updated_at']}`\n- Message: {record.message or 'none'}\n",
        )
