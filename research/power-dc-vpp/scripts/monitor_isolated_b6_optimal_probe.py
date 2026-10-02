from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_metrics(pid: int) -> dict[str, object]:
    command = (
        f"$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        "if($p){[pscustomobject]@{running=$true;cpu_seconds=$p.CPU;"
        "working_set_bytes=$p.WorkingSet64}|ConvertTo-Json -Compress}"
        "else{'{\"running\":false}'}"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return dict(json.loads(completed.stdout.strip()))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"running": False, "monitor_error": completed.stderr.strip()}


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    heartbeat = args.probe_root / "heartbeat.json"
    stdout = args.probe_root / "solver.stdout.log"
    stderr = args.probe_root / "solver.stderr.log"
    while True:
        metrics = process_metrics(args.pid)
        payload = {
            "schema_version": "isolated_b6_probe_heartbeat_v1",
            "updated_at_utc": timestamp(),
            "pid": args.pid,
            **metrics,
            "solver_stdout_bytes": stdout.stat().st_size if stdout.exists() else 0,
            "solver_stderr_bytes": stderr.stat().st_size if stderr.exists() else 0,
        }
        write_json(heartbeat, payload)
        if not metrics.get("running"):
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
