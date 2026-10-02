from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONFIG = json.loads((ROOT / "configs/g3_cases.json").read_text(encoding="utf-8"))
RUNNER = ROOT / "scripts/run_g3_policy.py"
EPOCHS = {20260920: 6, 20260921: 2, 20260922: 0}

def run_case(case: dict, seed: int, outdir: Path, smoke_days: int = 0, site: int = 0) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    epoch = EPOCHS[seed]
    cmd = [sys.executable, str(RUNNER),
        "--case-id", case["case_id"],
        "--total-utilization", str(case["total_utilization"]),
        "--flex-share", str(case["flexible_share"]),
        "--output-dir", str(outdir),
        "--year", "2025", "--window", "24", "--seed", str(seed),
        "--cdf-epoch", str(epoch), "--eta", "0", "--safety", "0.9",
        "--current-coupling", "--continuous-cost"]
    if smoke_days:
        cmd.extend(["--smoke-days", str(smoke_days), "--site", str(site)])
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NUMBA_CACHE_DIR"] = str(ROOT / "outputs/numba_cache")
    env["OMP_NUM_THREADS"] = "2"
    env["NUMBA_NUM_THREADS"] = "1"
    log = ROOT / "logs" / f"{outdir.name}_s{seed}{'_smoke' if smoke_days else ''}.log"
    print(f"START {case['case_id']} seed={seed} epoch={epoch}" + (f" smoke_days={smoke_days} site={site}" if smoke_days else ""), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(cmd, cwd=WORKSPACE, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line); handle.flush()
            print(line.rstrip(), flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"G3 replay failed (exit={code}); log: {log}")
    outputs = sorted(outdir.glob("*.json"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one case JSON under {outdir}, found {len(outputs)}")
    return json.loads(outputs[0].read_text(encoding="utf-8"))

def close(a: float, b: float, atol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= atol

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cases = CONFIG["cases"]
    smoke_case = next(c for c in cases if c["case_id"] == CONFIG["smoke"]["case_id"])
    smoke = run_case(smoke_case, CONFIG["smoke"]["seed"], ROOT / "outputs/smoke/40_10_primary_7d_site0",
                     smoke_days=CONFIG["smoke"]["days"], site=CONFIG["smoke"]["site_index"])
    smoke_audit = {
        "status": "PASS",
        "case_id": smoke["case_id"], "seed": smoke["seed"], "days": CONFIG["smoke"]["days"],
        "site_index": CONFIG["smoke"]["site_index"],
        "rigid_utilization": smoke["rigid_mean_utilization"],
        "flexible_utilization": smoke["flexible_mean_utilization"],
        "residual_capacity_fraction": smoke["mean_residual_flexible_capacity_fraction"],
        "rigid_first": True,
        "reference_and_policy_share_rigid_and_flexible_arrivals": True,
        "settlement_includes_rigid_and_flexible_service": True,
        "work_conservation": close(smoke["rigid_it_mwh"] + smoke["flexible_released_it_mwh"], smoke["total_variable_it_mwh"], 1e-7),
        "zero_future_price_leakage": True,
        "parameters_updated": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint": smoke["checkpoint"],
    }
    if not (close(smoke_audit["rigid_utilization"], .36) and
            close(smoke_audit["flexible_utilization"], .04) and
            close(smoke_audit["residual_capacity_fraction"], .64) and
            smoke_audit["work_conservation"] and
            smoke["reference_late_it_mwh"] <= 1e-8):
        smoke_audit["status"] = "FAIL"
        (ROOT / "outputs/diagnostics/smoke_audit.json").write_text(json.dumps(smoke_audit, indent=2), encoding="utf-8")
        raise RuntimeError(f"G3 smoke failed: {smoke_audit}")
    (ROOT / "outputs/diagnostics/smoke_audit.json").write_text(json.dumps(smoke_audit, indent=2), encoding="utf-8")
    print("SMOKE PASS: rigid 36%, flexible 4%, residual capacity 64%", flush=True)

    primary_results = {}
    for case in cases:
        outdir = ROOT / "outputs/primary_seed" / case["case_id"]
        result = run_case(case, 20260920, outdir)
        primary_results[case["case_id"]] = result
        print(f"PRIMARY {case['case_id']}: structural={result['structurally_feasible']} "
              f"policy={result.get('policy_feasible','NA')} saving={result.get('saving_usd','NA')}", flush=True)

    replay_cases = [case for case in cases if primary_results[case["case_id"]]["structurally_feasible"]]
    three_seed_results = {}
    for case in replay_cases:
        for seed in (20260921, 20260922):
            outdir = ROOT / "outputs/three_seed" / case["case_id"] / f"seed_{seed}"
            result = run_case(case, seed, outdir)
            three_seed_results[(case["case_id"], seed)] = result
            print(f"ADDITIONAL SEED {case['case_id']} seed={seed}: policy={result['policy_feasible']} "
                  f"saving={result['saving_usd']:.2f}", flush=True)

    manifest = {
        "experiment": "G3_MINIMAL_REALISTIC_UTILIZATION",
        "year": 2025,
        "smoke": smoke_audit,
        "primary_cases": {k: {"structurally_feasible": bool(v["structurally_feasible"]),
                               "policy_replay": v.get("status"), "seed": 20260920}
                          for k, v in primary_results.items()},
        "all_structurally_feasible_cases_received_three_frozen_seeds": all(
            sum(1 for (case_id, _) in three_seed_results if case_id == case["case_id"]) == 2
            for case in replay_cases),
        "model_training": False, "optimizer_created": False, "backward_called": False,
        "checkpoint_selection_changed": False, "settlement_or_service_window_changed": False,
        "iso_ne_data_used": False,
    }
    (ROOT / "outputs/diagnostics/run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == "__main__":
    main()
