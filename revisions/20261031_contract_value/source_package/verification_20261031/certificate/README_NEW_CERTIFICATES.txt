NEW CONSERVATIVE ANNUAL CERTIFICATES — 31 OCTOBER 2026

WHAT IS VERIFIED

The newly supplied programs construct and verify a conservative replacement
from the public 2025 annual reference/physics/price arrays. They do not recover
the missing original admissions, the original 29 October dual archives, or the
old US$117,000.2387599464 hindsight assignment.

An exact admission envelope follows from the original work-conserving reference
recurrence. It contains every admission vector compatible with that source and
the published reference. Two independently assembled outer LPs are evaluated
with exact rational arithmetic: one through original-cost breakpoint minima,
the other through all cost facets and the full finite-box Lagrangian. Both
describe the same convex outer relaxation; their matrices and certificate
evaluations are separate.

The public reference and its absolute-deviation fee anchor remain fixed, as in
the original source account. The reference ledger is not relabelled as an
exact feasible schedule for unknown admissions. Exact feasibility concerns the
new candidate schedules and the admission-specific repair in the theorem.

RESULTS, WITH CONSERVATIVE MONETARY ROUNDING

  Right     Original-cost ceiling    Epigraph ceiling
  12 h      US$100,625.48             US$100,625.48
  16 h      US$115,735.23             US$115,735.23
  24 h      US$134,437.85             US$134,437.85

The uniform constructive 24-hour value floor is US$130,931.31. It exceeds both
12-hour ceilings by at least US$30,305.84 and both 16-hour ceilings by at least
US$15,196.08. Premiums are formed from the full exact enclosures before cent
rounding, not by subtracting independently rounded display cells. The 24-hour
remaining bound-to-value gap is at most US$3,506.54.

The longest saturated reference run is 35 intervals, with 158 busy blocks and
639 saturated intervals per site. A proved global rounding allowance is
E=2.412e-12 work units per site. Exact original-cost repair has a uniform
electricity-cost allowance 0.000836657841428138... dollars across all four
sites. The original MTM term is tau*(R_N-X_N), not tau*(A_N-X_N). Independently
proved 0<tau<21 dollars/work therefore requires the additional signed-terminal
allowance delta_T=4*21*E=0.000000000202608 dollars. Each outer cost floor is
reduced by delta_T and the repaired candidate's cost upper is increased by it.
The total uniform original-account correction is below US$0.000836658044037.
The common DA charge US$2409.7842143056 is counted once in each four-site cost
account, not once per site.

The four explicit rational surrogate schedules satisfy their stricter robust
prefix corridor exactly. For every source-compatible actual admission vector,
the constructive repair theorem supplies a schedule exactly feasible for those
admissions, finishing all true work within the year. The source MTM offset can
still be slightly signed, and is covered above. Actual admissions were not
recovered, so those admission-specific repaired schedules are not claimed to
have been materialised. The guarantee is uniform existence and value, with a
fully specified exact repair algorithm. It is retrospective perfect-information
opportunity, not an information-adapted execution result.

FILES IN THIS DIRECTORY

  envelope_certificate.py
      New cumulative-corridor segment LP, exact original-cost dual evaluation,
      reference non-identification demonstration, and robust-candidate generation.
  epigraph_certificate.py
      Independent x/cumulative/epigraph assembly, exact facet validation and
      finite-box dual evaluation, including signed terminal-offset allowance.
  verify_saved_envelope.py
      Re-evaluates saved original-cost multipliers and all four exact rational
      schedules without calling a solver; checks hashes and original cost.
  test_envelope_certificates.py
      Seven small exact regression tests; separate from the annual verification.
  results/envelope_results.json
      Twelve corrected annual site results, aggregate bounds, the uniform
      witness-value guarantee and all component allowances.
  results/outer_w{12,16,24}_zone{0,1,2,3}.npz
      Twelve saved multiplier records and their reference/input provenance.
  results/robust_w24_zone{0,1,2,3}.json
      Four complete rational surrogate execution arrays.
  results/solver_free_replay.json
      Saved-record replay of all twelve original-cost certificates and four
      robust schedules, including exact service boxes and every prefix.
  epigraph_results/epigraph_w{12,16,24}_zone{0,1,2,3}.npz
      Twelve independent epigraph equality/inequality multiplier records.
  epigraph_results/exact_epigraph_physics_zone{0,1,2,3}.json.gz
      Complete original breakpoints and exact facets for all annual slots.
  epigraph_results/epigraph_w{12,16,24}_zone{0,1,2,3}_bounds.json.gz
      All finite service, cumulative and epigraph bounds and complete row rules.
  epigraph_results/epigraph_results.json
  epigraph_results/epigraph_verify_only_results.json
      Original run and solver-free retained-file replay. All 2,095,170 facet/
      breakpoint comparisons pass; all finite-box terms are included.
  envelope_regression_results.json
      Seven small regression tests, including signed monetary formatting.
  supplement_S27.md
  MAIN_TEXT_CERTIFICATE_SECTION.txt
      Full proof and proposed concise main-text certificate section.
  independent_proof_review_math_audit.md
      Independent mathematical review of reference error, dual signs and repair.
  new_verification_manifest.json
      File hashes and an explicit distinction between new verification and
      unrecovered original records. Regenerate with make_manifest.py after any
      intentional file edit.

The accompanying terminal_rate_audit.py/.json/.txt are in the parent revision
directory. They prove the terminal coefficient enclosure from source accounting
and public MTM rows, including CSV parsing/round-trip error. Their documented
source-lineage and parser-arithmetic assumptions are premises of this audit;
the exact historical runtime version is not inferred from a tiny regression
residual. The prior 30 October verification code and original ablations are
retained unchanged.

DEPENDENCIES AND COMMANDS

Recorded environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, SciPy's bundled
HiGHS. Exact certificate arithmetic uses fractions.Fraction. The solver only
generates candidate multipliers or schedules; a success flag does not supply
the proof. The scripts presently import SciPy even for verification-only use.

Run these commands from this certificate directory. Replace the two absolute
input paths with the extracted public files. The pinned source repository is
https://github.com/kalimustafa/executable-service-commitments-worldmoe at commit
323c2084e7ee668ba512fb1e33dd56b8394b32d4. Generation reads its original reference
function and economic_constraints.json. Verification reads the latter; it does
not execute training or data acquisition.

    python -m unittest discover -p 'test_envelope_certificates.py' -v

Verify retained evidence without calling an LP solver:

    python verify_saved_envelope.py --repo /path/to/pinned_repo --inputs /path/to/grid_metrics_inputs_2025.npz --results results
    python epigraph_certificate.py --repo /path/to/pinned_repo --inputs /path/to/grid_metrics_inputs_2025.npz --output-dir epigraph_results --verify-only

Independently regenerate the new results into separate directories:

    python envelope_certificate.py --repo /path/to/pinned_repo --inputs /path/to/grid_metrics_inputs_2025.npz --output-dir regenerated_outer --windows 12 16 24 --witness
    python epigraph_certificate.py --repo /path/to/pinned_repo --inputs /path/to/grid_metrics_inputs_2025.npz --output-dir regenerated_epigraph --windows 12 16 24

Solver/platform differences may produce different valid multipliers or
candidate schedules. Re-evaluate them exactly; do not expect identical binary
NPZ hashes across newly solved runs. The retained verification checks do expect
the included archived file hashes and exact recorded bounds.

REMAINING ORIGINAL-EVIDENCE GAP

The old code/compute_annual_cut.py and code/reproduce.py, the original twelve
site-contract NPZs, complete original admissions/forcing and original annual
hindsight bridge/assignment remain absent. No new surrogate arrival array is
presented as those missing inputs. The new proof deliberately enlarges the
unknown-input domain and constructs a guaranteed value floor uniformly over
it. No claim is made of retraining, reproducing the old 29 October annual solve,
recovering the old 24-hour witness, or verifying historical first-release logs.
