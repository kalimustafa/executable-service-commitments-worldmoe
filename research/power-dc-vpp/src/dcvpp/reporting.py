from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dcvpp.core import atomic_json, atomic_text, sha256_file, utc_now


def build_run_manifest(root: Path, artifact_root: Path, payload: dict[str, Any]) -> Path:
    artifact_paths = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            artifact_paths.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {"generated_at": utc_now(), **payload, "artifacts": artifact_paths}
    path = artifact_root / "manifests/run_manifest.json"
    atomic_json(path, manifest)
    return path


def build_demo_report(root: Path, artifact_root: Path, context: dict[str, Any]) -> Path:
    physical = pd.read_csv(artifact_root / "tables/nlr_twin_validation.csv")
    energy = pd.read_csv(artifact_root / "tables/nyiso_forecast_main.csv")
    workload = pd.read_csv(artifact_root / "tables/workload_forecast_main.csv")
    optimization = pd.read_csv(artifact_root / "tables/counterfactual_optimization_main.csv")
    text = f"""# Demo Run Summary

- Status: `DEMO_COMPLETE`
- Profile: `demo`
- Synthetic: `true`
- Generated: `{utc_now()}`
- Purpose: engineering smoke test only; not paper evidence

## Completed chain

Separate synthetic physical, market, and workload clocks were generated; critical domain gates passed; a physical twin and MILP surrogate were fitted; independent DA and RT demo models and a CfC-W demo model trained; scenarios were reduced; and B0-B8 were solved with HiGHS.

## Counts

- Physical models: {len(physical)}
- Energy forecast rows: {len(energy)}
- Workload forecast rows: {len(workload)}
- Optimization policies: {len(optimization)}

## Scientific boundary

These values are synthetic pipeline checks. They are not NLR observations, actual NYISO bids, realized revenue, observed demand response, or demonstrated grid benefit. Paper mode must use mandatory real sources and may terminate as blocked rather than substituting this demo evidence.
"""
    path = artifact_root / "report/run_summary.md"
    atomic_text(path, text)
    return path


def build_blocked_report(root: Path, status: str, blockers: list[dict[str, Any]], completed: list[str]) -> Path:
    path = root / "artifacts/report/run_summary.md"
    lines = [
        "# Research Run Summary",
        "",
        f"- Terminal status: `{status}`",
        f"- Generated: `{utc_now()}`",
        "- Counterfactual boundary: NLR was not operated or bid in NYISO.",
        "",
        "## Completed",
        "",
        *[f"- {item}" for item in completed],
        "",
        "## Blockers and failures",
        "",
        *[f"- `{item.get('code', 'BLOCKED')}`: {item.get('message', '')}" for item in blockers],
        "",
        "## Principal scientific limitations",
        "",
        "- NLR 295 contains measured node/GPU energy where monitoring exists, TDP-derived CPU energy where declared, and nulls where measurements are absent; these categories cannot be relabeled or zero-filled.",
        "- NLR 300 timestamps are timezone-naive and their storage timezone is not stated by its Parquet metadata or official README. Facility location is not timestamp-encoding evidence, so physical calibration and model-specific overlap were not attempted.",
        "- SRRL provides fixed-MST irradiance and meteorology, not measured PV output. Any PV series would be derived from irradiance and declared system assumptions.",
        "- Xcel publications are tariff/program-rule snapshots, not training labels or evidence of realized demand response; the local tariff/DR subcase was not reached after the upstream gate.",
        "- Historical NYISO ZIP hashes identify the acquired archive revision, but the archive does not supply an immutable original row-level publication vintage.",
        "- Capacity, hardware, battery, SLA, deadline, migration eligibility, checkpoint overhead, grid limits, and residency rules remain engineering/transfer assumptions unless explicitly observed.",
        "- No NLR-to-NYISO twin was instantiated. Any eventual NYISO deployment and market value remain trace-driven counterfactual simulation, never actual NLR operation, bids, dispatch, revenue, or realized savings.",
        "",
        "Paper-mode numerical claims remain unavailable where the required real evidence did not pass the minimum scientific gates.",
    ]
    atomic_text(path, "\n".join(lines) + "\n")
    atomic_json(root / "artifacts/report/blockers.json", {"status": status, "blockers": blockers})
    return path
