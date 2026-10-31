# Recovered historical WorldMoE service trajectories

This package restores the original nine WorldMoE records for 12 h, 16 h and 24 h contracts and the three frozen checkpoints used in Table 2. The records were found in the local September 2026 experiment archive. They were absent from the previously inspected public package. No model was trained, no policy was rerun, and no new trajectory was substituted for a historical output.

The original `forcing_2025.npz` includes the 35,040 by four admission array `eagle_arrivals`. The source function `ServiceCommitment_1to5_20260927/src/context.py::configure` reads this array. Applying its original earliest-service recurrence reproduces the published `reference_compute` array bit for bit. The prices, PV, PUE, idle power and work-power coefficients also match the published input archive bit for bit. The forcing SHA256 is `c600c9e01e8236bd2e1cb4e2dc08591e3ee264d05f1c26b4c785ec355c3147c8`.

## Contents

- `inputs/`: original forcing, physical constraints, published input archive, and nine unmodified historical JSON summaries, annual trajectory NPZs, complete service-book NPZs and daily CSVs. Complete books contain 35,040 observed intervals plus 8,192 stored tail intervals.
- `code/audit_historical_books.py`: independent service, electrical, terminal and settlement audit. Its release, deadline and total-work residuals use exact integer accumulation of stored binary64 numbers.
- `code/exact_repair_historical_books.py`: exact dyadic corridor repair and exact rational settlement. This creates separate feasible numerical witnesses without overwriting historical outputs.
- `results/historical_audit.json`: nine original-trajectory audit records and same-checkpoint contrasts.
- `results/historical_service_audit.csv`, `historical_settlement.csv`, `historical_paired_increments.csv` and `historical_monthly_ledger.csv`: concise evidence tables. Monthly aggregation uses UTC months and therefore has a short final 2026-01 bin; the market year is the original 2025 New York calendar.
- `results/audit_s*_w*.npz`: all 35,040 period-level service and monetary audit arrays.
- `results/exact_repaired_s*_w*.json.gz`: complete exact feasible numerical service witnesses, encoded as integer numerators with a common power-of-two denominator per site. All stored tail service is zero.
- `results/historical_exact_repair.json` and `.csv`: exact rational savings and repair-cost bounds.
- `supplement_historical_recovery.md`: manuscript-ready explanation, formulas and evidence boundaries.
- `source_manifest.json`: hashes, historical paths and selected-checkpoint identity.

## Reproduce

From any directory, run with Python and NumPy:

```text
python code/audit_historical_books.py
python code/exact_repair_historical_books.py
```

The first program reads the raw historical files and produces an audit. The second separately encodes and verifies exact feasible numerical witnesses. Every feasibility and monetary calculation in the second program uses integer or rational arithmetic. Its exact constraints are release precedence, per-cohort FIFO deadline completion, unit service capacity, electrical import capacity and complete service with zero year-end liability. No nominal ramp constraint is introduced; the original account has static import and service caps.

The original schedules contain tiny nonzero binary64 residuals and are therefore not described as exact symbolic witnesses. Their separate repairs change the objective by no more than US$0.000000016563 in any run. The exact repaired mean 24 h saving lies in [US$105,271.061841632, US$105,271.061841633]. The exact repaired mean 12-to-24 h contrast lies in [US$33,050.092227443, US$33,050.092227444]. All reported values remain unchanged to the nearest cent.

These are historical paired policy results with recovered trajectory verification. The small numerical repairs establish feasible schedules for the same physical account; they are not claimed to be new causal policy deployments. This audit does not reconstruct past first-release information or independently rerun neural inference. The policy contrast remains different from the perfect-information optimal contract-value increment and from ex-ante contract value.
