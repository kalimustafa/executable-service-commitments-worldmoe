from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_bytes(path: Path, content: bytes) -> None:
    """Persist downloaded bytes without exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
