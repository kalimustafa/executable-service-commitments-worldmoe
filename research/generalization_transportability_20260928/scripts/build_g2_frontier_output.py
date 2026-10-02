from __future__ import annotations

import csv
import json
import math
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PACKAGE = TASK.parent / "ExecutableServiceCommitments_Revision_20260928"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, *keys: str, default=math.nan) -> float:
    for key in keys:
        try:
            value = float(row.get(key, ""))
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            continue
    return default


def main() -> None:
    source = read_csv(PACKAGE / "experiments/E04_frontier_nonpreemptive/frontier_results.csv")
    mapping_path = PACKAGE / "experiments/E04_frontier_nonpreemptive/mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    total_it = float(mapping["job_energy_mwh"])
    rows = []
    for item in source:
        summary_path = Path(item["source_json"])
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        jobs = int(num(item, "jobs"))
        late_jobs = int(num(item, "late_jobs"))
        closing_jobs = int(num(item, "closing_jobs"))
        saving = num(item, "saving_usd")
        reference = num(summary, "reference_cost_usd")
        if not math.isfinite(reference):
            reference = saving / (num(item, "saving_pct") / 100.0)
        policy = num(summary, "adjusted_cost_usd", "policy_cost_usd", default=reference-saving)
        shifted = num(summary, "shifted_IT_mwh", "shifted_it_mwh")
        late_it = num(item, "late_job_it_mwh")
        closing_it = num(item, "closing_job_it_mwh")
        zero_late = late_jobs == 0
        zero_closing = closing_jobs == 0
        rows.append({
            "pipeline": "WorldMoE_nonpreemptive",
            "seed": "20260920",
            "service_window_h": int(num(item, "window_h")),
            "total_jobs": jobs,
            "total_it_mwh": total_it,
            "completed_jobs": jobs-closing_jobs,
            "late_jobs": late_jobs,
            "late_it_mwh": late_it,
            "closing_jobs": closing_jobs,
            "closing_it_mwh": closing_it,
            "reference_cost_usd": reference,
            "policy_cost_usd": policy,
            "saving_usd": saving,
            "saving_pct": num(item, "saving_pct"),
            "shifted_it_mwh": shifted,
            "saving_per_total_it_mwh": saving/total_it if total_it else "NA",
            "saving_per_shifted_mwh": saving/shifted if math.isfinite(shifted) and shifted > 1e-8 else "NA",
            "zero_late": str(zero_late).lower(),
            "zero_closing": str(zero_closing).lower(),
            "feasible": str(zero_late and zero_closing).lower(),
            "notes": "Same-job earliest-service matched reference; year-end closing liability included. Zero-late and zero-closing are separate delivery conditions.",
        })
    dest = TASK / "outputs/g2_frontier/frontier_transfer_results.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with dest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Frontier window rows to {dest.relative_to(TASK).as_posix()}")


if __name__ == "__main__":
    main()
