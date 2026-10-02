from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import requests


def resumable_download(
    url: str,
    destination: Path,
    *,
    stable_url: str | None = None,
    attempts: int = 4,
    timeout: tuple[int, int] = (20, 120),
) -> dict[str, Any]:
    """Download with a Range-resumable .part file and atomic completion."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "power-dc-vpp/0.1 research downloader"}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            request_headers = dict(headers)
            if offset:
                request_headers["Range"] = f"bytes={offset}-"
            with requests.get(url, headers=request_headers, stream=True, allow_redirects=True, timeout=timeout) as response:
                if offset and response.status_code == 200:
                    offset = 0
                    mode = "wb"
                elif response.status_code in (200, 206):
                    mode = "ab" if offset else "wb"
                else:
                    response.raise_for_status()
                    mode = "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(part, destination)
            digest = hashlib.sha256()
            with destination.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            return {
                "stable_url": stable_url or url,
                "retrieval_url": url,
                "final_url": response.url,
                "filename": destination.name,
                "byte_count": destination.stat().st_size,
                "sha256": digest.hexdigest(),
                "retrieved_at_epoch": time.time(),
            }
        except (OSError, requests.RequestException) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last_error}")

