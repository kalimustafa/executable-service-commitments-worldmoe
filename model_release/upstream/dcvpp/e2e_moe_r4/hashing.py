"""Standard SHA-256 isolated from the long-lived NumPy/Torch process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import subprocess
import sys


class HashingError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    source = Path(path).resolve()
    script = (
        "import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],'rb'); "
        "exec(\"while True:\\n b=f.read(8388608)\\n if not b: break\\n h.update(b)\"); "
        "f.close(); print(h.hexdigest())"
    )
    return _run([sys.executable, "-c", script, str(source)])


def sha256_bytes(value: bytes) -> str:
    script = (
        "import hashlib,sys; data=sys.stdin.buffer.read(); "
        "print(hashlib.sha256(data).hexdigest())"
    )
    return _run([sys.executable, "-c", script], input_bytes=value)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _run(command: list[str], input_bytes: bytes | None = None) -> str:
    completed = subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    digest = completed.stdout.decode("ascii", errors="strict").strip().lower()
    if completed.returncode != 0 or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise HashingError(error or f"isolated SHA-256 failed with exit {completed.returncode}")
    return digest


__all__ = ["HashingError", "sha256_bytes", "sha256_file", "sha256_json"]
