# S32 Extension of the frozen policy to seven service windows

## S32.1 Design and common account

The service-window grid is extended to $W\in\{8,12,16,20,24,36,48\}$ h. The evaluation retains the three selected WorldMoE checkpoints: seed 20260920 at epoch 6, seed 20260921 at epoch 2, and seed 20260922 at epoch 0. No model parameter or observer coefficient is fitted or updated. All windows use the recovered 2025 Eagle admissions, the same four-site physical inputs, earliest-service reference, electricity-cost function, execution-change charge, static electrical limits and terminal account.

The 12 h and 16 h records are the recovered original historical evaluations. New evaluations dated 31 October 2026 cover 8 h, 20 h, 36 h and 48 h. The shared implementation is also rerun at 24 h for every checkpoint. These three 24 h anchors reproduce every element of the original complete service/reference books and the in-year compute, grid, daily-value, value, MTM and service-debt arrays bit for bit. All three monetary differences are exactly zero. This is a reduction check for the modified implementation; the original and repeated anchors are not treated as independent samples.

The curve is a same-checkpoint historical-calendar comparison. Additional windows supply intermediate and longer-right observations; they do not provide new market years, independent training seeds or a confidence interval for future calendar outcomes.

## S32.2 Fixed neural prediction and complete obligation queues

The neural forecast horizon remains exactly 96 quarter-hour slots. The model input remains a $96\times22$ tensor with its original lead coordinates and 4 h, 8 h, 12 h and 24 h backlog bins. The total-backlog feature includes all current admitted obligations, including those with deadlines outside the neural forecast horizon. No neural output is extrapolated by changing its lead coordinate or tensor dimensions.

For a contract with $D=4W$ quarter-hour slots, the service queue contains $H=\max(96,D)$ deadline bins. Thus the 36 h and 48 h runs retain 144 and 192 bins. Every known obligation enters the current-action feasibility guard and the continuation calculation. The implementation explicitly checks that no nonzero obligation lies beyond the continuation array passed to the action selector. It does not truncate the queue to the neural horizon.

The original executor's current-action guard is unchanged. If $q_d$ denotes known work in relative deadline bin $d$ and $C$ the original guaranteed site service capacity, its lower action bound is

$$
x_{\min}=\min\left\{C,\sum_dq_d,
\max\left(0,\max_d\left[\sum_{j=0}^{d}q_j-0.9Cd\right]\right)\right\},
\qquad x_{\max}=\min\left\{C,\sum_dq_d\right\}.
$$

The factor $0.9$ is the original future-capacity safety factor. The complete extended queue appears in these cumulative requirements. Current service uses the same candidate grid, physical/reference knots, expected-cost calculation and tie-breaking rule as the archived controller, and is assigned to outstanding work in deadline order.

## S32.3 Declared continuation beyond 24 h

Forecast scenarios for relative slots $h=0,\ldots,95$ use the original neural outputs, publication-aware market information and strictly lagged physical observations. For $96\le h<H$, the continuation is specified before evaluation as

$$
\widehat p_{zh}=p_z^{\mathrm{hist}},\qquad
\widehat{\mathrm{PV}}_{zh}=0,\qquad
\widehat{\mathrm{PUE}}_{zh}=\mathrm{PUE}^{\max}_z,\qquad
\widehat o_{zh}=0.
$$

Here $p_z^{\mathrm{hist}}$ is the frozen pre-2024 zonal median RT price stored in the original forcing record: $[28.37,41.01,40.56,45.96]$ US$/MWh. The declared maximum PUE is $1.0929484367370605$ at each site. The extra slots retain the same service capacity and accounting terms. The seven scenario columns receive the same deterministic tail values; the learned opportunity adjustment $\widehat o$ is zero beyond the trained prediction horizon. The earliest-service reference for already admitted work is continued using the unchanged causal reference-queue rule.

This is a declared deterministic continuation policy. A historical median is not a lower or upper bound on future RT prices. Zero PV and maximum PUE impose physical-stress assumptions, but do not certify conservative profit. Neither current nor future realised market residuals are supplied to the policy to construct this tail. All first-96-slot predictions remain unchanged in form, and the tail is inactive for the 24 h anchors.

## S32.4 Year-end operation and service accounting

The original year-end early-completion heuristic is retained for comparability. At observed slot $t$, the controller inserts newly admitted work into the relative target bin

$$
d_t^{\mathrm{target}}=\min\{D-1,N-1-t\}.
$$

It also applies the original earliest-service guard in the final hour. This is a stricter operating target when a cohort's contractual deadline extends beyond the observed year. It does not redefine the original exclusive contractual deadline $t+D$. Independent service verification uses that contractual deadline and the complete stored book. Residual work is retained in an explicit 8,192-slot tail and is subject to the original terminal valuation; it is never silently discarded.

Raw controller outputs and any subsequent exact rational feasibility repairs are separate evidence objects. New raw runs save complete service/reference books, quarter-hour compute, imports, monetary value, MTM value and service debt, plus daily settlement. Exact numerical repairs do not overwrite the raw controller trajectories or reclassify them as newly trained policies.

## S32.5 Reproduction and interpretation

`frozen_source_dependencies.json` identifies the original inputs, models, source functions and forecast caches by SHA256. The supplied `frozen_runtime/` contains the necessary archived Python source and frozen assets. `code/run_portable.py` selects that included runtime without relying on an external working directory. The observed environment is Python 3.13.5, NumPy 2.1.3, PyTorch 2.9.0+cu126, Numba 0.61.0 and Pandas 2.2.3; execution uses CPU inference with one Torch thread per process. The runtime source and checkpoint identities are recorded for every run.

Longer contractual rights enlarge the optimal feasible set under common assumptions. The realised saving of a fixed approximate controller need not increase monotonically at every adjacent window. The seven-point policy curve therefore describes this frozen controller and its stated continuation rule. Perfect-information interval results are reported separately, with their own complete feasible witnesses and upper-bound certificates. The repeated 24 h anchor demonstrates exact reduction to the original implementation, not optimality of the long-window control rule.

## S32.6 The observed 8 h service boundary

The frozen controller does not satisfy the 8 h service contract. Its three raw cash-account savings are US$50,947.45119, US$50,629.56573 and US$50,629.01365, with mean US$50,735.34352. The corresponding deadline-late work is 3.0119885083, 3.0104989686 and 3.0119885095 IT-MWh. These quantities are substantive service violations, not floating-point feasibility residuals; zero closing backlog does not remove an earlier deadline failure.

The 8 h policy result is therefore labelled cash-only and service-infeasible. It is excluded from feasible operating-value, opportunity-capture and certified policy-headroom calculations. It remains visible in the raw evidence and in a separately marked plot point. No new 8 h run, model tuning or undisclosed service repair is used to hide this outcome.

A feasible 8 h perfect-information certificate concerns a different object: it establishes that the recovered workload can be scheduled with hindsight under that service window. It does not establish that this frozen causal control rule meets all deadlines. The contrast identifies a contract and controller boundary that the original three-window grid could not show.

## S32.7 Seven-window raw operating results

The table reports means and checkpoint ranges on the common observed calendar. Deadline-late work and closing backlog are the raw archived accounting outputs. The separate exact numerical audit establishes feasibility for retained feasible-value claims; the 8 h row is excluded from those claims. CVaR is the original daily MTM loss CVaR at 95%, not an across-calendar confidence interval.

| Window, h | Provenance | Mean saving and checkpoint range, US$ | Maximum late work, IT-MWh | Closing backlog, IT-MWh | Mean daily MTM loss CVaR95, US$ |
|---:|---|---:|---:|---:|---:|
| 8 | New frozen evaluation; cash-only, deadline failure | 50,735.34 [50,629.01, 50,947.45] | 3.01199 | 0 | 331.95 |
| 12 | Historical recovery | 72,220.97 [72,099.98, 72,361.65] | 3.54483e-09 | 0 | 471.33 |
| 16 | Historical recovery | 86,038.55 [85,833.93, 86,257.34] | 2.1246e-09 | 0 | 429.12 |
| 20 | New frozen evaluation | 98,199.02 [97,853.91, 98,655.44] | 2.42483e-09 | 0 | 324.21 |
| 24 | New anchor; bitwise historical match | 105,271.06 [105,003.52, 105,565.67] | 1.76883e-09 | 0 | 313.38 |
| 36 | New frozen evaluation | 114,251.12 [113,954.92, 114,602.20] | 8.93917e-10 | 0 | 368.67 |
| 48 | New frozen evaluation | 123,101.20 [122,891.84, 123,247.54] | 1.37457e-09 | 0 | 325.53 |

Fifteen new annual raw evaluations are supplied: four new contract windows and one 24 h anchor for each of three checkpoints. Six original 12 h and 16 h records are reused with explicit historical provenance. Thus there are seven contract points, not twenty-one independent calendar replications. All new raw files, complete books, daily records, manifests and logs are retained. The 24 h anchors add a reproducibility check without adding independent observations.

The complete per-checkpoint table is `results/seven_window_raw_runs.csv`. Window aggregates, including the full CVaR ranges, mean and maximum deadline-late work, closing backlog, shifted IT energy, execution-change charges and grid imports, are in `results/seven_window_raw_summary.csv`. The historical summary copies and their original hashes are under `historical_reference/`.
