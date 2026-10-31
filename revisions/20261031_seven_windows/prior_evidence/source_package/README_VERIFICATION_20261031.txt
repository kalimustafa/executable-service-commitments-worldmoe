CONTRACT RIGHTS AND THE ECONOMIC VALUE OF DATA CENTRE FLEXIBILITY
Revised manuscript and reproducible verification — 31 October 2026

READ FIRST

manuscript/ contains the current 12-page main paper and 53-page supplement.
The main paper concentrates on contract-rights identification, value bounds,
information and net economic value. Original WorldMoE ablations and all
original supplementary tables are retained. The old six main-paper figure
images have moved into S28 without pixel changes.

The new annual certificates are conservative replacements constructed from
released annual evidence and documented source semantics. They are NOT a
recovery of the missing 29 October certificate files or the old
US$117,000.2387599464 hindsight assignment. Do not relabel the older scalar
audits or small test fixtures as full annual verification.

CURRENT VERIFIED RESULTS

                             12 h           16 h           24 h
Original-cost ceiling     100,625.48     115,735.23     134,437.85 USD
Independent epigraph      100,625.48     115,735.23     134,437.85 USD

Uniform constructive 24 h feasible-value floor: US$130,931.31.
Minimum 12-to-24 h optimal rights premium: US$30,305.84.
Minimum 16-to-24 h optimal rights premium: US$15,196.08.
Remaining 24 h bound-to-witness-value gap: at most US$3,506.54.

Differences use exact unrounded enclosures before conservative cent rounding.
The two independent annual ceilings differ by less than US$0.000000138.
Twelve site-contract certificates per formulation and four complete rational
surrogate schedules have been rechecked from saved files without calling the
LP solver. The epigraph verifier checks 2,095,170 facet/breakpoint comparisons.
Seven small regression fixtures are recorded separately from these annual
checks. The older 17 certificate tests, 13 repair tests and 16 original-source
boundary fixtures retain their explicitly limited scopes.

WHAT THE FEASIBLE-VALUE FLOOR MEANS

For every original admission array consistent with the published reference
recurrence and its rounding model, the saved surrogate schedule admits an
exact feasible 24 h repair. All true work finishes within the year. This is
a uniform constructive existence/value guarantee. The hidden actual arrivals
were not recovered; an arrival-specific final execution file was therefore
not materialised. It is not one universal executable schedule, and is not an
information-adapted policy or a future-revenue estimate.

The reference recurrence implies E=2.412e-12 work units per site. The total
electricity-cost repair allowance is below US$0.000836657841429. The source
MTM term is tau*(R_N-X_N), with the fixed reference prefix R_N, rather than
tau*(A_N-X_N). The terminal proof gives 0<tau<21 dollars/work; the additional
signed-reference allowance is delta_T=2.02608e-10 dollars. Each cost floor
subtracts delta_T, and the repaired cost upper bound adds delta_T. The complete
uniform original-account correction is below US$0.000836658044037. The common
DA charge US$2409.7842143056 enters each four-site account exactly once.

ECONOMIC EVIDENCE AND ITS BOUNDARY

Independent cash re-settlement reproduces the 24 h operating saving
US$105,271.0618416321. Matched-checkpoint published summaries give historical
mean increments US$13,817.58476055 (12-to-16 h), US$19,232.50746689 (16-to-24 h)
and US$33,050.09222744 (12-to-24 h). The same checkpoint and reference-account
checks are retained. Complete 12/16 h trajectories are not in the inspected
release, so their summary comparisons are not newly certified execution files.

Fee tables deduct explicit assumed payments/costs. They do not estimate an
observed contract price or tenant reservation value. Ex-ante value requires
a probability law and implementable information-adapted policies. The exact
information-gap identity and examples are in S25. Tenant payments reduce the
operator's margin but cancel from operator-plus-tenant surplus; actual delay
and implementation costs do not cancel. The full 2025 calendar is historical
evidence, not an out-of-sample confidence statement.

PACKAGE LAYOUT

manuscript/
  Current main DOCX, supplementary DOCX, editable main Markdown, and the new
  scientific figure in PNG/SVG. Native Word equations remain editable.

verification_20261031/certificate/
  Both new certificate programs, exact verification scripts, full multiplier
  archives, exact cost/facet and finite-bound records, four rational surrogate
  schedules, solver-free replay outputs, independent proof review and hashes.
  README_NEW_CERTIFICATES.txt explains each file and full regeneration.

verification_20261031/terminal_rate_audit.py/.json/.txt
  The exact four-equation terminal-coefficient enclosure, its source arithmetic
  and the intermediate CSV-parser error bound. The JSON retains exact rational
  matrices, inverse, error coefficients and positive lower endpoints.

verification_20261031/parser_provenance/
  Official Pandas 2.2.3 parser and selection sources, with the licence. These
  document the default parser algorithm assumed in the accounting proof; they
  do not assert recovery of the historical installed package version.

verification_20261031/economics/
  Paired source-summary calculations, exact decimal outputs, fee scenarios and
  S25/S26 derivations. Files prefixed conditional_ concern the old reported
  endpoints and retain that conditional status; Table 4/S27 use the new bounds.

source_snapshot/
  Minimal unchanged public source and input subset needed by the commands below.
  Original relative paths are preserved. This is not the full repository.
  Source commit: 323c2084e7ee668ba512fb1e33dd56b8394b32d4
  Repository: https://github.com/kalimustafa/executable-service-commitments-worldmoe
  The original MIT licence is included.

historical_reference_checks/
  Unchanged 30 October exact reference verifiers, repair routine, audits and
  saved results. Historical documentation describes the state before the new
  conservative construction. It is labelled by date and is not the current
  annual verification summary.

audit/
  Final document QA, original-table/image preservation evidence, source hashes
  and the package manifest. SHA256SUMS.txt covers every delivered file.

DEPENDENCIES

Recorded numerical runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. Fractions
and Decimal supply exact arithmetic. The SciPy solver proposes multipliers;
its success flag is not used as a mathematical proof. The verification-only
programs import SciPy but do not call its LP solver. Pandas is not required to
run the new verification programs; its archived code supports the parser proof.

REPRODUCIBLE COMMANDS

Run from the extracted package root. Keep regenerated results separate from
retained evidence. No model training, GPU, network or original missing arrivals
are needed for these retained-record checks.

  python verification_20261031/terminal_rate_audit.py --repo source_snapshot --output rerun/terminal_rate_audit.json

  python verification_20261031/certificate/verify_saved_envelope.py --repo source_snapshot --inputs source_snapshot/source_data/grid_metrics_inputs_2025.npz --results verification_20261031/certificate/results

  python verification_20261031/certificate/epigraph_certificate.py --repo source_snapshot --inputs source_snapshot/source_data/grid_metrics_inputs_2025.npz --output-dir verification_20261031/certificate/epigraph_results --verify-only

The two verification commands write their replay reports beside the retained
records after checking them; they do not alter saved multipliers or schedules.
To retain every package byte unchanged, first copy those two result directories
to a separate rerun location and pass the copied directory paths instead.

  python verification_20261031/economics/economic_value_analysis.py --repo source_snapshot --output-dir rerun/economics

  python -m unittest discover -s verification_20261031/certificate -p 'test_envelope_certificates.py' -v

  python -m unittest discover -s historical_reference_checks/code/certificates -p 'test_*.py' -v

  python historical_reference_checks/code/repair/exact_witness_repair.py --self-test --output rerun/repair_self_test.json

  python historical_reference_checks/code/source_contract_checks.py --repo source_snapshot --output-dir rerun

The source-boundary fixtures include tests that confirm known limitations of
archived code. A passing fixture is not a claim that those historical branches
were repaired or that the archived annual dispatch passed exact feasibility.

To regenerate the new LP proposals rather than just verify the saved records:

  python verification_20261031/certificate/envelope_certificate.py --repo source_snapshot --inputs source_snapshot/source_data/grid_metrics_inputs_2025.npz --output-dir rerun/new_outer --windows 12 16 24 --witness

  python verification_20261031/certificate/epigraph_certificate.py --repo source_snapshot --inputs source_snapshot/source_data/grid_metrics_inputs_2025.npz --output-dir rerun/new_epigraph --windows 12 16 24

Different solver/platform versions can propose different valid multipliers.
Use exact reevaluation to assess them; newly solved files need not match the
included NPZ hashes. The retained-record replay does require the archived
hashes and exact values to match.

REMAINING DATA REQUIREMENTS

The original admission array, old 29 October twelve-instance certificate set,
old annual hindsight assignment and immutable first-release input logs remain
unavailable in the inspected release. Actual tenant payment, delay-disutility
and implementation-cost observations are also absent. The revised manuscript
does not manufacture those inputs or claim prospective profitability without
them. Its new pathwise bound is complete under the stated reference/source,
nominal-physics and accounting premises, which are set out in S27 and the
terminal audit.
