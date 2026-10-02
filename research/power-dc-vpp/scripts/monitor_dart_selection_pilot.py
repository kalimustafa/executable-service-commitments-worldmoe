"""Record 30-second CPU/RAM/GPU/progress/ETA telemetry for a DART pilot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import psutil


FIELDS = (
    "timestamp_utc",
    "launcher_pid",
    "launcher_running",
    "tracked_root_pid",
    "active_process_count",
    "active_python_process_count",
    "process_tree_cpu_percent",
    "system_cpu_percent",
    "system_ram_used_gb",
    "system_ram_percent",
    "process_tree_rss_gb",
    "gpu_name",
    "gpu_utilization_percent",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "stage",
    "status",
    "completed_cell_count",
    "total_cell_count",
    "completed_unit_count",
    "total_unit_count",
    "shared_completed_unit_count",
    "shared_total_unit_count",
    "effective_workers",
    "elapsed_seconds",
    "eta_seconds",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _same_process(process: psutil.Process, identity: tuple[int, float]) -> bool:
    try:
        return process.pid == identity[0] and abs(process.create_time() - identity[1]) < 0.01
    except (psutil.Error, OSError):
        return False


def _descendants(pid: int) -> list[psutil.Process]:
    try:
        process = psutil.Process(pid)
        return [process, *process.children(recursive=True)]
    except (psutil.Error, OSError):
        return []


def _looks_like_pilot_python(process: psutil.Process) -> bool:
    try:
        name = process.name().lower()
        command = " ".join(process.cmdline()).lower()
    except (psutil.Error, OSError):
        return False
    return (
        name.startswith("python")
        and (
            "run_dart_lstm_jan2022_pilot.py" in command
            or "loky.backend.popen_loky" in command
            or "joblib" in command
        )
    )


def _process_identity(pid: int) -> tuple[int, float] | None:
    try:
        process = psutil.Process(pid)
        return process.pid, process.create_time()
    except (psutil.Error, OSError):
        return None


def _discover_descendants(parent_pids: set[int]) -> list[psutil.Process]:
    """Find children even after a short-lived Windows wrapper was reaped."""

    found: dict[int, psutil.Process] = {}
    frontier = set(parent_pids)
    while frontier:
        next_frontier: set[int] = set()
        for process in psutil.process_iter(["pid", "ppid"]):
            try:
                if process.ppid() in frontier and process.pid not in found:
                    found[process.pid] = process
                    next_frontier.add(process.pid)
            except (psutil.Error, OSError):
                continue
        frontier = next_frontier
    return list(found.values())


def _active_tree(
    launcher_identity: tuple[int, float] | None,
    launcher_pid: int,
    known: dict[int, float],
) -> tuple[bool, list[psutil.Process]]:
    """Track the stable PID identity plus Python descendants after wrapper exit."""

    launcher_alive = False
    launcher = None
    if launcher_identity is not None:
        try:
            launcher = psutil.Process(launcher_identity[0])
            launcher_alive = _same_process(launcher, launcher_identity) and launcher.is_running()
        except (psutil.Error, OSError):
            launcher_alive = False
            launcher = None
    if launcher_alive and launcher is not None:
        for process in _descendants(launcher.pid):
            try:
                known[process.pid] = process.create_time()
            except (psutil.Error, OSError):
                continue
    for process in _discover_descendants({launcher_pid, *known}):
        try:
            known.setdefault(process.pid, process.create_time())
        except (psutil.Error, OSError):
            continue
    active: dict[int, psutil.Process] = {}
    for pid, created in tuple(known.items()):
        try:
            process = psutil.Process(pid)
            if (
                abs(process.create_time() - created) < 0.01
                and process.is_running()
                and (
                    not hasattr(psutil, "STATUS_ZOMBIE")
                    or process.status() != psutil.STATUS_ZOMBIE
                )
            ):
                active[pid] = process
                for child in process.children(recursive=True):
                    try:
                        known[child.pid] = child.create_time()
                        active[child.pid] = child
                    except (psutil.Error, OSError):
                        continue
            else:
                known.pop(pid, None)
        except (psutil.Error, OSError):
            known.pop(pid, None)
    # A Windows .venv launcher can disappear while the base-interpreter child
    # keeps solving.  Once discovered, that child remains authoritative until
    # its exact PID+create_time identity exits.
    running = launcher_alive or bool(active)
    return running, list(active.values())


def _tree_resources(processes: list[psutil.Process]) -> tuple[float, dict[int, float]]:
    rss = 0
    cpu_seconds: dict[int, float] = {}
    for process in processes:
        try:
            rss += process.memory_info().rss
            times = process.cpu_times()
            cpu_seconds[process.pid] = float(times.user + times.system)
        except (psutil.Error, OSError):
            continue
    return rss / 1024**3, cpu_seconds


def _gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5.0, check=True
        )
        row = [item.strip() for item in completed.stdout.strip().splitlines()[0].split(",")]
        return {
            "gpu_name": row[0],
            "gpu_utilization_percent": float(row[1]),
            "gpu_memory_used_mb": float(row[2]),
            "gpu_memory_total_mb": float(row[3]),
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {
            "gpu_name": None,
            "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    launcher_identity = _process_identity(args.pid)
    known_processes = (
        {}
        if launcher_identity is None
        else {launcher_identity[0]: launcher_identity[1]}
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    launch_record = _read(args.output_root / "launch.json")
    recorded_identity = _process_identity(int(launch_record.get("pid", -1) or -1))
    if recorded_identity is not None:
        known_processes[recorded_identity[0]] = recorded_identity[1]
    for process in _discover_descendants({args.pid, *known_processes}):
        try:
            known_processes[process.pid] = process.create_time()
        except (psutil.Error, OSError):
            continue
    if launcher_identity is None and not any(
        _looks_like_pilot_python(psutil.Process(pid)) for pid in known_processes
    ):
        raise SystemExit(
            f"launcher PID {args.pid} is absent and no actual pilot Python child was found"
        )
    for process in _descendants(args.pid):
        try:
            known_processes[process.pid] = process.create_time()
            process.cpu_percent(interval=None)
        except (psutil.Error, OSError):
            continue
    csv_path = args.output_root / "device_monitor.csv"
    jsonl_path = args.output_root / "device_monitor.jsonl"
    snapshot_path = args.output_root / "device_monitor_latest.json"
    started = time.perf_counter()
    maximum_ram = 0.0
    maximum_tree_rss = 0.0
    previous_tree_cpu: dict[int, float] = {}
    previous_tree_sample = time.perf_counter()
    psutil.cpu_percent(interval=None)
    while True:
        if launcher_identity is None:
            recovered = _process_identity(args.pid)
            if recovered is not None:
                launcher_identity = recovered
                known_processes[recovered[0]] = recovered[1]
        running, active_processes = _active_tree(
            launcher_identity, args.pid, known_processes
        )
        tree_rss, current_tree_cpu = _tree_resources(active_processes)
        tree_sample = time.perf_counter()
        tree_sample_seconds = max(tree_sample - previous_tree_sample, 1e-9)
        tree_cpu = 100.0 * sum(
            max(0.0, seconds - previous_tree_cpu.get(pid, seconds))
            for pid, seconds in current_tree_cpu.items()
        ) / tree_sample_seconds
        previous_tree_cpu = current_tree_cpu
        previous_tree_sample = tree_sample
        python_processes = [
            process for process in active_processes if _looks_like_pilot_python(process)
        ]
        memory = psutil.virtual_memory()
        system_ram = memory.used / 1024**3
        cpu = psutil.cpu_percent(interval=None)
        maximum_ram = max(maximum_ram, system_ram)
        maximum_tree_rss = max(maximum_tree_rss, tree_rss)
        progress = _read(args.progress)
        active_root_raw = progress.get("active_worker_progress_root")
        if active_root_raw:
            active_root = Path(str(active_root_raw))
            worker_completed = sum(
                int(_read(path).get("completed_cell_count", 0) or 0)
                for path in active_root.glob("worker_*.json")
            )
        else:
            worker_completed = 0
        shared_root_raw = progress.get("active_shared_progress_root")
        if shared_root_raw:
            shared_root = Path(str(shared_root_raw))
            shared_completed = sum(
                int(_read(path).get("completed_shared_unit_count", 0) or 0)
                for path in shared_root.glob("date_*.json")
                if _read(path).get("panel_sha256") == progress.get("panel_sha256")
            )
        else:
            shared_completed = int(
                progress.get("shared_completed_unit_count", 0) or 0
            )
        elapsed = max(time.perf_counter() - started, 0.0)
        completed = min(
            int(progress.get("total_cell_count", 0) or 0),
            int(progress.get("completed_cell_count", 0) or 0) + worker_completed,
        )
        total = int(progress.get("total_cell_count", 0) or 0)
        eta = elapsed * (total - completed) / completed if completed > 0 else None
        record = {
            "timestamp_utc": _now(),
            "launcher_pid": args.pid,
            "launcher_running": running,
            "tracked_root_pid": (
                args.pid if launcher_identity is None else launcher_identity[0]
            ),
            "active_process_count": len(active_processes),
            "active_python_process_count": len(python_processes),
            "process_tree_cpu_percent": tree_cpu,
            "system_cpu_percent": cpu,
            "system_ram_used_gb": system_ram,
            "system_ram_percent": memory.percent,
            "process_tree_rss_gb": tree_rss,
            **_gpu(),
            "stage": progress.get("stage"),
            "status": (
                progress.get("status")
                if running
                else "STOPPED"
            )
            or ("WAITING" if running else "STOPPED"),
            "completed_cell_count": completed,
            "total_cell_count": total,
            "completed_unit_count": int(progress.get("completed_unit_count", 0) or 0),
            "total_unit_count": int(progress.get("total_unit_count", 0) or 0),
            "shared_completed_unit_count": min(
                int(progress.get("shared_total_unit_count", 0) or 0),
                shared_completed,
            ),
            "shared_total_unit_count": int(
                progress.get("shared_total_unit_count", 0) or 0
            ),
            "effective_workers": progress.get("effective_workers"),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "peak_system_ram_gb": maximum_ram,
            "peak_process_tree_rss_gb": maximum_tree_rss,
            "gpu_accelerates_highs_milp": False,
        }
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        csv_exists = csv_path.is_file()
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            if not csv_exists:
                writer.writeheader()
            writer.writerow(record)
        temporary = snapshot_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(snapshot_path)
        # A terminal progress record is not sufficient while a solver child is
        # still alive; monitor through the last exact process identity exit.
        if not running:
            break
        time.sleep(max(args.interval_seconds, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
