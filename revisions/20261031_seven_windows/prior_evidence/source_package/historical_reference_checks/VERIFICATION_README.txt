VERIFICATION PACKAGE — 30 OCTOBER 2026
Restored service-contract mathematics and unchanged WorldMoE ablations

1. PURPOSE AND EVIDENCE STATUS

This package supplies the revised main paper and supplement, an audit of the
public calculation source, exact reference verification routines, and checks
that can be reproduced from the available public arrays and summaries.

Completed work has three distinct scopes:

  * Reference algorithms: 17 certificate tests (12 generic verifier tests and
    5 original-cost dual tests), plus 13 repair self-tests including an exhaustive
    check of 729 small rational paths. These are explicit algorithm fixtures.
  * Original-source behavior: 16 boundary fixtures against functions extracted
    from the pinned public source. Some passing fixtures CONFIRM limitations,
    including tail-lateness censoring, a late-job fallback, a year-end deadline
    clamp, and use of current realised physical forcing. Passing does not mean
    these limitations were repaired or that annual executions passed.
  * Supplied annual evidence: independent observed-year re-settlement of three
    complete policy arrays, and arithmetic reaggregation of 48 published site
    certificate rows into 12 four-site groups.

These counts must not be added together or described as complete annual
certificate verification. The original annual hindsight witness, complete
admissions/recovery inputs, annual dual vectors, and independently assembled
epigraph certificates remain unavailable in the inspected materials. The
conditional repair budget below is not an executed repair of that witness.

2. FILE LAYOUT

All commands below assume the current directory is the extracted package root.

manuscript/
  EnergyEconomics_Restored_Contracts_WorldMoE_20261030.docx
  EnergyEconomics_Supplementary_Verification_20261030.docx
    Restored paper and supplementary mathematical/evidence material.

audit/
  Contract_Calculation_Audit_20261030.txt
    Chinese explanation of corrections, reproduced results and missing inputs.
  CERTIFICATE_AUDIT.txt
    Actual certificate-source findings, dual formulas and evidence conditions.
  mathematical_and_contract_audit.txt
    Deadline conventions, repair proof and original contract-boundary findings.

code/certificates/
  reference_verifier.py, annual_dual_revaluation.py
  test_reference_verifier.py, test_annual_dual_revaluation.py, README.txt
    Exact finite-box dual, primal, facet, rounding and original-cost checks.

code/repair/
  exact_witness_repair.py, README.txt, repair_example.json
    Exact homogeneous cumulative repair; the four-slot example is synthetic.

code/
  source_contract_checks.py
  independent_settlement_check.py

results/
  reference_test_results.json, repair_self_test.json, repair_example_result.json
  source_boundary_test_results.json, independent_settlement_check.json
  published_bound_scalar_audit.json, conditional_money_bound.json
    Archived outputs with their individual evidence scopes.

inputs/
  grid_metrics_inputs_2025.npz
  cost_bound_zone_diagnostics.csv
  certified_value_bounds.csv
    Available public inputs, not substitutes for the missing annual admissions
    or the unpublished witness/certificate arrays.

evidence/
  complete_verification_manifest_template.json
  source_provenance.json
    Twelve-instance evidence checklist and provenance. Null/false template
    entries are unresolved requirements, not successful checks. The template
    does not bind the available public physical arrays to each missing annual
    certificate instance automatically.

3. DEPENDENCIES AND RUNNABLE CHECKS

The recorded environment used Python 3.12.14 and NumPy 2.3.5. The certificate
and exact-repair routines use only Python's standard library. NumPy is required
for source-boundary fixtures and annual re-settlement. No solver, neural-model
training framework or GPU is required for the included checks.

Install NumPy in the intended Python environment if needed:

    python -m pip install numpy==2.3.5

Keep regenerated output separate from the archived results:

    python -c "from pathlib import Path; Path('rerun').mkdir(exist_ok=True)"

Run the 17 reference certificate tests:

    python -m unittest discover -s code/certificates -p 'test_*.py' -v

Run the 13 repair self-tests and the explicit four-slot example:

    python code/repair/exact_witness_repair.py --self-test --output rerun/repair_self_test.json
    python code/repair/exact_witness_repair.py --input code/repair/repair_example.json --output rerun/repair_example_result.json

Re-settle the three supplied annual policy trajectories:

    python code/independent_settlement_check.py --inputs inputs/grid_metrics_inputs_2025.npz --output rerun/independent_settlement_check.json

That script uses NumPy longdouble for arithmetic. Last digits can differ across
platforms because longdouble precision is platform dependent; it is a numerical
re-settlement, not an exact rational proof of annual feasibility or a dual bound.

The 16 original-source fixtures additionally require the public repository at
the inspected commit. The full repository is not required for the preceding
checks. To obtain an independent checkout, run:

    git clone https://github.com/kalimustafa/executable-service-commitments-worldmoe.git external_source
    git -C external_source checkout --detach 323c2084e7ee668ba512fb1e33dd56b8394b32d4
    python code/source_contract_checks.py --repo external_source --output-dir rerun

The script extracts selected original functions with Python AST, supplies
explicit fixtures, and avoids training/data-loading side effects. It does not
modify the original source. An existing checkout at the same commit may be
supplied with --repo instead.

The generic verifier also has these CLI operations, whose JSON schemas and a
synthetic example are documented in code/certificates/README.txt:

    python code/certificates/reference_verifier.py bound certificate.json
    python code/certificates/reference_verifier.py facets facets.json
    python code/certificates/reference_verifier.py aggregate aggregation.json

Those three input filenames are examples to be supplied by the caller; they
are not missing annual records disguised as included data. Equality multipliers
are unrestricted; multipliers for inequalities G y <= h must be nonnegative.
All finite bounds, every original piecewise-linear breakpoint and any certified
allowance must be supplied. Decimal/rational strings are exact; a stored
binary64 value can be represented using the binary64_hex schema.

annual_dual_revaluation.py is an importable module, NOT a command-line annual
runner. Its revalue_zone(...) function requires the original arrivals, price
and physical arrays, reference imports, capacity/terminal coefficients and
complete equality multipliers. test_annual_dual_revaluation.py demonstrates
the function interface on small contracts. No annual call was made without
the missing inputs. The routine forms exact original-cost breakpoint minima
and rounds complete scalar terms downward at a declared quantum; three terms
per interval at 10^-12 dollars lose less than 0.00000010512 dollars per site
over 35,040 intervals, before any separate input/model allowances.

4. REPRODUCED ANNUAL SETTLEMENT AND PUBLISHED SCALARS

The input NPZ contains 35,040 intervals at four sites. Its SHA-256 is:
5be123c40f01b111cf6503b07882e703a4c2cc29a03dde3cfb7aa4321ce2cf72

Reconstructed observed-year savings before terminal-cost differences are:
  seed 20260920: 105565.6654617202 US dollars
  seed 20260921: 105003.5173783060 US dollars
  seed 20260922: 105244.0026848702 US dollars
  mean:         105271.0618416321 US dollars

The mean equals avoided energy value 172510.1075550115 minus recovery energy
cost 64971.8159527367 minus import-deviation fees 3348.7567679851 plus curtailment
benefit 1081.5270073425. The reconstructed reference cost is approximately
555168.205055697 dollars, including the common DA charge 2409.7842143056 once.
The original nominal-policy source reports zero terminal differences. This
check independently reproduces observed-year cash settlement; it does not
establish admissions, deadlines, post-year feasibility, first-release inputs,
checkpoint replay, or the separate 117000.2387599464-dollar hindsight witness.

The published CSV check has 48 site rows and 12 groups (two years, two traces,
three rights, four sites). Every recorded site floor is floor(raw dual)-1.
Reaggregation agrees with reported raw totals within 6.5e-11 dollars; the
reference-minus-cost-floor ceiling identity agrees within 1e-11 dollars.
These are arithmetic consistency results, not annual dual re-evaluations.
The archived one-dollar reserve is not itself a proof of numerical error.

The following POSIX-shell recipe independently repeats the CSV scalar check
using exact Decimal interpretations of the printed CSV fields:

    python - <<'PY'
    import csv, json
    from collections import defaultdict
    from decimal import Decimal as D, ROUND_FLOOR
    from pathlib import Path
    groups = defaultdict(list)
    with Path('inputs/cost_bound_zone_diagnostics.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            groups[(row['year'], row['trace'], row['window_h'])].append(row)
    with Path('inputs/certified_value_bounds.csv').open(newline='') as f:
        totals = list(csv.DictReader(f))
    assert sum(map(len, groups.values())) == 48 and len(totals) == 12
    max_raw = D(0)
    max_identity = D(0)
    for row in totals:
        sites = groups[(row['year'], row['trace'], row['window_h'])]
        assert len(sites) == 4 and len({s['zone'] for s in sites}) == 4
        assert all(D(s['conservative_usd']) == D(s['dual_usd']).to_integral_value(rounding=ROUND_FLOOR)-1 for s in sites)
        common = D(row['lower_bound_usd']) - sum(D(s['conservative_usd']) for s in sites)
        assert common == D('2409.7842143056' if row['year']=='2025' else '8256.6901964583')
        max_raw = max(max_raw, abs(sum(D(s['dual_usd']) for s in sites)+common-D(row['raw_dual_usd'])))
        max_identity = max(max_identity, abs(D(row['reference_cost_usd'])-D(row['lower_bound_usd'])-D(row['maximum_saving_upper_usd'])))
    assert max_raw <= D('6.5e-11') and max_identity <= D('1e-11')
    report = {'scope':'Published CSV arithmetic only; not annual certificate reconstruction',
              'site_rows':48, 'groups':12, 'all_floor_minus_one':True,
              'max_raw_residual_usd':str(max_raw), 'max_ceiling_identity_residual_usd':str(max_identity)}
    Path('rerun/published_scalar_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    PY

5. EXACT REPAIR AND THE CONDITIONAL MONETARY BUDGET

A D-slot right includes release slot r and ends at r+D-1; r+D is the first
ineligible slot. With quarter-hour intervals, D=4W and 24 hours means 96 slots.
The repair routine accepts initial cohorts with their retained deadlines and
uses exact rational arithmetic. It checks homogeneous release/deadline
prefixes, conservation and fixed service-capacity bounds. It rejects a raw
negative increment, capacity violation or infeasible corridor. Mixed deadlines,
electrical ramp coupling and nonpreemptive jobs require their original coupled
constraints; the homogeneous repair alone does not establish those properties.

If the printed prefix maxima 3.006750e-12 and 6.381505e-12 are confirmed as exact
bounds, including all endpoints, and the repair's feasibility assumptions hold,
the total four-site execution change is at most 1.3158578208e-6 work units.
The published physical/price extrema admit the outward Lipschitz coefficient
2026.14 dollars per work unit. Their product is less than 0.002667 dollars.
The absolute-price formula covers negative RT prices and includes the original
one-dollar import-deviation and curtailment terms.

This is the conditional electricity-cost correction in
results/conditional_money_bound.json. It excludes any changed terminal
liability and numerical error in BOTH reference and candidate cost evaluations;
these require separate allowances. The original annual assignment was not
available, so its hypotheses have not been certified on that assignment.
Assuming the reported ceilings are independently validated, the nominal
19197.8179185550-dollar 12-to-24-hour margin retains the downward-cent floor
19197.81 only if the total witness correction is at most 0.0079185550 dollars.
The analogous reported 16-to-24-hour cent floor has the same correction budget.
Savings ceilings round upward; cost floors and feasible-witness savings floors
round downward.

6. MATERIALS STILL REQUIRED TO CLOSE AN ANNUAL VERIFICATION

The public source inspected is:
  https://github.com/kalimustafa/executable-service-commitments-worldmoe
  commit 323c2084e7ee668ba512fb1e33dd56b8394b32d4

The authoritative original first-bound source is
research/ServiceCommitment_1to5_20260927/scripts/run_bounds.py. It already uses
a lower-convex-hull incremental-segment relaxation. A cumulative epigraph
implementation of the same complete envelope is an independent formulation
check; it is not automatically a different mathematical relaxation.

Closing the annual witness and two-formulation audit requires:

  1. The 29 October independent assembler code/compute_annual_cut.py and
     code/reproduce.py, their exact versions, and their complete input manifest.
  2. Original annual forcing/admissions, including the source-requested
     external_validation/data/forcing_2025.npz or a complete hash-matched
     equivalent; initial cohorts, full reference and recovery books, historical
     terminal prices, capacities and the original terminal accounting rule.
     Reproducing the additional 2024/Frontier cases also requires their year-
     specific forcing and frontier_YEAR/mapped_arrivals.npz equivalents.
  3. First-formulation annual multiplier records. run_bounds.py writes files
     named I1_Eagle_2025_w{12,16,24}_zone{0,1,2,3}_certificate.npz. Corresponding
     exact original-cost revaluation and allowance records are required.
  4. Twelve independently assembled epigraph NPZ certificates: WEST, CAPITL,
     N.Y.C. and LONGIL for 12/16/24-hour rights. Each must contain its separately
     assembled rows, objective, finite bounds, original breakpoints, all facets,
     raw/sign-corrected multipliers, solver logs and verified error budget.
     Their original individual filenames were not supplied; the manifest
     enumerates the twelve required site-contract instances explicitly.
  5. The original 24-hour annual hindsight assignment and its three local-window
     embedding records. Named records include certificate_annual_hindsight_bridge.csv,
     certificate_annual_hindsight_complete_book.npz and certificate_annual_bridge/.
     Supply the repaired exact cohort assignment, full constraint checks and
     re-settlement under the matched original terminal account.
  6. The annual-summary CSV/JSON, residual-diagnostic CSV, first-formulation
     revaluation JSON, solver/numerical-version record and hashes linking each
     record to the same original contract. Printed residuals are diagnostics,
     not exact feasibility repairs or certified rounding allowances.
  7. If decision-time implementability is claimed, immutable numerical first-
     release vintages with receipt/processing times, fixed checkpoints and
     action-level replay records. A perfect-information retrospective bound
     does not establish that separate information property. Retaining the
     earlier Frontier 30-hour zero-lateness revision also needs its matching
     replacement run and job-level output; the inspected source records 27.

No missing item has been replaced with invented annual data or a synthetic
annual result. Bind the recovered materials to the manifest and pass the exact
domain, monetary and provenance checks before asserting completed annual
witness feasibility or independently validated two-formulation exclusion.
