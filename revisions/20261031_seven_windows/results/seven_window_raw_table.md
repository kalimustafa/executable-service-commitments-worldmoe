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
