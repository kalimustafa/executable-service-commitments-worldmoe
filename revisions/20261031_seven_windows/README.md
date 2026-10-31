# Seven-window contract-value revision

Completed on 31 October 2026. The current main manuscript has 18 pages and the supplement 65 pages. Six main figures and twelve retained supplementary figures use short captions; the original detailed architecture and ablation images remain in the supplement.

This revision completes fifteen new annual frozen-checkpoint policy evaluations, including three 24 h anchors, and twenty-eight site–deadline perfect-information certificates. It also redraws the WorldMoE mechanism, adds contract-value and cash-flow diagrams in the original purple/pink palette, and restores a concise economic comparison with Van de Sande Araujo et al. (2025), Biswas et al. (2025), and Fei et al. (2026).

## Current results

All monetary values below use the recovered actual admissions and the same physical and financial account. The policy column is a three-checkpoint cash-account mean. A substantive service failure is explicitly excluded from feasible-value calculations.

| Deadline | Policy cash saving US$ | Service | Certified PI opportunity US$ |
|---:|---:|---|---:|
| 8 h | 50,735.34 | Fails; approximately 3.01 IT MWh late | 78,751.60–78,882.41 |
| 12 h | 72,220.97 | Exact feasible counterpart | 97,639.33–97,796.41 |
| 16 h | 86,038.55 | Exact feasible counterpart | 113,119.89–113,305.19 |
| 20 h | 98,199.02 | Exact feasible counterpart | 124,528.17–124,733.99 |
| 24 h | 105,271.06 | Exact feasible counterpart | 132,376.82–132,598.65 |
| 36 h | 114,251.12 | Exact feasible counterpart | 141,996.61–142,256.41 |
| 48 h | 123,101.20 | Exact feasible counterpart | 152,798.28–153,070.23 |

The 12/16/24 h curve retains the recovered historical policy records. New 24 h anchors reproduce the archived complete books and settlement trajectories bit for bit; they are not independent additional samples. The 8/20/36/48 h policy rows are new evaluations dated 31 October 2026. No training or refitting was performed.

The new 24 h remaining opportunity is **US$27,105.76–27,327.58**, its capture enclosure is **79.39%–79.53%**, and the actual-admission optimal 12→24 h rights premium is **US$34,580.41–34,959.31**. These displays use full-precision certificates and exact policy accounts before outward rounding. The wider earlier admission-compatible bounds remain valid as separate uniform guarantees; they are not mixed into the new actual-admission curve.

The seven-point study reveals two boundaries that three points did not establish. First, a feasible 8 h hindsight schedule exists, but the frozen controller misses deadlines, so its positive cash result is not a feasible operating-value endpoint. Second, certified opportunity per extra deadline hour rises from **US$783.16–823.30** over 24→36 h to **US$878.48–922.81** over 36→48 h, while the policy gains fall from **US$748.34** to **US$737.51**. The operating curve alone therefore does not identify the shape of the contractual opportunity curve.

## Evidence layout

- `manuscript/`: current Word manuscripts, editable Markdown source and reference-number mapping.
- `figures/`: current PNG/SVG/PDF figures, plot data and source code.
- `results/`: all fifteen new raw outputs, complete service books, quarter-hour trajectories, daily settlements, execution manifests and the three-anchor comparison.
- `audit/`: twelve separate exact feasible numerical witnesses, three unrepaired 8 h failure records, service-failure arrays and monetary reconciliation.
- `certificates/`: seven actual-admission opportunity intervals, twenty-eight complete dual and witness records, exact arithmetic and reproduction code.
- `analysis/`: joined exact accounts, service eligibility, seven-point tables and matched adjacent increments.
- `frozen_runtime/`: the frozen source, checkpoints, feature caches and inputs needed by the portable policy evaluator.
- `inputs/`: the common forcing and physical constraints used by the independent audit.
- `literature/`: verified publication metadata, primary-source links and the critical comparison used in the introduction.
- `logs/`: original output logs from the new annual policy evaluations and their orchestration.
- `prior_evidence/`: the preceding complete verification package, retained for provenance. Its older manuscripts, wider bounds and statements that additional windows had not yet been run describe the previous revision; this README, the current manuscripts, S31–S32 and the completed run records supersede that status.

All twelve service-valid new policy runs pass exact release, deadline, capacity, import and terminal checks after separate roundoff-scale repairs. The largest monetary repair allowance is below **US$1.10×10⁻⁸**. The three 8 h failures are not repaired into claimed policy successes. All twenty-eight PI witnesses are exactly feasible; the outer certificates permit legitimate closing work under the original terminal valuation.

## Reproduction

For an independent audit of the supplied new records and a fresh join of all seven points:

```text
python code/audit_new_policy.py
python code/assemble_results.py
python figures/build_window_extension_figure.py --policy analysis/policy.csv --certificates analysis/certificates.csv --anchor analysis/anchor.csv
```

The audit requires Python and NumPy. It resumes using source-book hashes and preserves raw outputs. Plotting additionally requires Matplotlib. Saved original-cost certificates can be checked without an LP solve:

```text
python certificates/code/run_actual_window_certificates.py --verify-only
```

That certificate program imports NumPy and SciPy. To regenerate its LP proposals and exact witnesses, omit `--verify-only`. For a new frozen-policy execution into a separate directory:

```text
python code/run_portable.py --fresh-output reproduced_policy --year 2025 --window 48 --seed 20260920 --cdf-epoch 6 --eta 0 --safety 0.9 --current-coupling --continuous-cost
```

The selected seed/epoch pairs are 20260920/6, 20260921/2 and 20260922/0. `README_RUNTIME.md` gives the observed dependency versions and portable-runtime scope. The actual fifteen annual runs used the original frozen workspace; the included portable entry point was checked for imports and existing-result reading, without adding a redundant sixteenth annual run. Frozen assets and source hashes are supplied.

The neural horizon stays at 24 h. The 36/48 h controller retains complete extended obligation queues and uses a declared deterministic continuation beyond the neural horizon; the historical median price is not a price bound. The source year-end early-completion target is a policy choice, separate from the original contractual deadline. These details appear in `PROTOCOL.md` and S32.

All outcomes concern the recorded simulation calendar. The package does not identify prospective contract prices, real-hardware performance or a journal decision. The contribution is a verifiable, service-matched and account-matched value interval and its separation from operating-policy capture.
