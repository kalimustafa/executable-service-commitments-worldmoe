# Reproduce the frozen WorldMoE system

Download `worldmoe-v1.4.0-full-inference.zip` from the GitHub release and extract it into a clean folder. The archive is self-contained for the first two levels below. It intentionally excludes provider datasets. Verify `SHA256SUMS.txt` with `python verify_package.py`.

## 1 Load all stages and run a decision

The verified environment is Python 3.13.5, PyTorch 2.9.0 and the exact package versions in `requirements-lock.txt`. The example runs on CPU. For the recorded CUDA build, install the pinned requirements with the official PyTorch CUDA wheel index:

```text
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.txt
python run_frozen_worldmoe.py --write-demo --seed 20260920 --output result.json
python run_frozen_worldmoe.py --input examples/synthetic_input.npz --metadata examples/synthetic_metadata.json --seed 20260921 --output result_s20260921.json
python run_frozen_worldmoe.py --input examples/synthetic_input.npz --metadata examples/synthetic_metadata.json --seed 20260922 --output result_s20260922.json
```

The program strictly loads W1, reconstructs early/middle/late world context and physical observations, runs R2 and APIC, loads a selected decision head, executes the original constrained action kernel, and checks work conservation and current overdue work. Each run reports loaded weight filenames, model parameter counts, APIC post-DA identity error, work served and remaining debt. The emitted settlement uses separately supplied ex-post prices only after decisions. An example result is an interface check, not a new annual result.

`--device cuda` is optional. The packaged model source is unchanged; the portable loader supplies paths relative to its own location. The original source still contains historical path identifiers in archival runners; these are not needed by the minimal loader. No neighboring private project directory is needed.

## 2 Reproduce executable service and accounting

The runtime kernel is `src/models/scenario_recourse.py`; its Gaussian partial-moment cost integration is in `src/models/continuous_cost.py`. The demonstration sends the neural outputs to this kernel with eta=0, safety=0.9, current physical coupling, and continuous costs. It subtracts action from the oldest due cohorts first and checks `opening debt = served + closing debt`. Work units convert to IT MWh by `1.639059066772461 × 0.25`. No neural output may override capacity or deadline feasibility.

The annual scripts in the repository under `research/ServiceCommitment_1to5_20260927/` and `research/R2_JointDecision_20260927/scripts/evaluate_scenario_recourse.py` implement full-year ledger, fixed DA positions, RT balancing, terminal debt restoration and detailed settlement. The mini example supports a nontrivial current queue and one step only. It does not replace terminal settlement. `source_reference/` includes the relevant original executor and training scripts for inspection.

## 3 Recalculate published annual statistics

Clone the same repository tag. The existing released source tables and figure builders remain unchanged:

```text
python -m pip install -r requirements-figures.txt
python tools/verify_release.py
python tools/reproduce_figures.py
```

These recalculate existing published statistics from derived annual/daily tables; they do not rerun training or regenerate provider inputs. The v1.4.0 model release preserves v1.3.0 and its checkpoint lineage.

## 4 Full retraining and annual replay

Obtain NYISO market records, the NLR Eagle physical data, NASA POWER/weather inputs, ISO-NE data for transfer, and the applicable Frontier workload data under their source terms. Restore the documented input fields, units, timestamps, masks and issue-time versions. Rebuild the original train-fitted normalization and caches only on their registered training years. W0/W1 sequential training code is under `research/power-dc-vpp/`; R2 training and 16-cohort refinement are under `research/R2_JointDecision_20260927/`. Freeze selection on 2024 before opening 2025. Then run the selected refinement with `--eta 0 --safety 0.9 --current-coupling --continuous-cost`, matching the selection record and archive epoch.

This level remains conditional on acquiring the provider inputs and reconstructing the full original calendar and mappings. The release does not claim to make those provider datasets public. Hyperparameters and final tensor hashes are available, but an end-to-end retraining run was not performed during packaging.

## External input schema

Use `examples/synthetic_input.npz` as a shape template; keep `fixture` marked accurately in the metadata. `fast_raw`, `medium_raw`, `slow_raw` are unnormalised typed observations shaped `[1,T,40,8]`, using original slot/feature definitions in `upstream/dcvpp/vpp_world_moe/data.py`. The supplied training-only world scaler is applied. Provide matching masks, source/quality identifiers and original graph fields for real packets. Optional `constraint_*` tensors override the demo's physical limits. This input interface consumes already constructed issue-legal packets; it is not a downloader for provider feeds.

`market_raw` is `[24,4,F]`, with feature order and train medians/IQR scales in `configs/r2_normalization.json`; zones are WEST, CAPITL, N.Y.C., LONGIL. `da_anchor` and `lead_hours` are `[24,4]`. The complete daily profile is masked before DA publication. `queue` is `[96,4]` work due by each future quarter hour, and must be feasible; `reference_queue`, `previous_pv` in MW and `previous_pue` are length four. `available_at` values describe when observations were available and are checked against `decision_at`.

Settlement fields are separate: `settlement_da_price` and `settlement_rt_price` in USD/MWh, `settlement_pv` and `da_position_mw` in MW, and dimensionless `settlement_pue`. They never enter the model/action code. The DA/RT energy account is 0.25 × [DA position × DA price + (import − DA position) × RT price]; full research accounting adds its documented PV, wear/transfer and terminal terms.
