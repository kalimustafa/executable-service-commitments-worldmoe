# Contract economics revision materials

These materials separate historical policy outcomes, retrospective perfect-information opportunity and ex-ante nonanticipative value. They retain the original operating-cost account and do not alter the WorldMoE ablations.

## Files

- `main_section_draft.md`: focused main-paper insertion, with native LaTeX and a compact paired-value table.
- `supplement_S25_S26.md`: appendix-ready S25/S26; complete definitions, proofs, abstract counterexamples, paired observations, fee scenarios and conditional transfer endpoints. Final supplementary table numbers are 38–41.
- `economic_analysis_notes.md`: mathematical audit, evidence limits, source anchors and manuscript edit targets.
- `economic_value_analysis.py`: standalone arithmetic/re-settlement program; requires Python and NumPy.
- `reported_certificate_scalars.json`: original manuscript-reported values, explicitly flagged as not reconstructed original annual certificates.
- `economic_value_results.json`: source hashes, checks, exact-decimal descriptive calculations, independent 24 h ledger check and scope flags.
- `paired_upgrades.csv`, `upgrade_summary.csv`: all same-seed differences and their means/ranges.
- `historical_fee_scenarios.csv`: additional annual cost assumptions, with net outcomes explicitly marked analytical sensitivities.
- `conditional_transfer_frontiers.csv`: operator-side transfer ceilings; tenant-side costs remain symbolic/unobserved.
- `conditional_PI_increment_intervals.csv`: correct witness-to-ceiling arithmetic, conditional on original verification and all corrections.
- `energy_normalization.csv`: optional normalization by the common observed-year reference variable-IT energy, not a contract price.

## Reproduce

```sh
python economic_value_analysis.py --repo /path/to/executable-service-commitments-worldmoe --output-dir /path/to/results
```

The default manuscript-scalar JSON is next to the script. An alternative can be supplied with `--reported-scalars`. The script reads the public source files without modifying them. Its output directory must exist or be creatable.

## Evidence and interpretation

The nine annual summary rows establish matching checkpoint hashes and a common account across rights. The 24 h component is independently re-settled from complete annual physical/execution arrays; the inspected source does not contain the corresponding 12/16 h trajectories or original admissions. Thus paired contrasts are historical summary-supported evidence with an independently verified 24 h ledger component. They are not newly reconstructed deadline certificates.

The extra-cost grid and transfer frontiers are analytical assumptions, not observed fees or prices. Refinement ranges describe one common market calendar, not uncertainty across future years. No new probability law, future-year confidence interval, tenant type distribution or historical first-release replay is invented. The two counterexamples are abstract nested-action decision problems, not purported deadline-only experiments.

Any independently constructed replacement opportunity certificate has its own input domain and validation record. The missing-file status in these materials concerns the original annual witness and original certificate assets, not the validity of a separately constructed replacement.
