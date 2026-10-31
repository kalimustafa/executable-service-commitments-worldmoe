INDEPENDENT CERTIFICATE VERIFICATION MATERIALS

This directory contains newly written reference checks. The package also
contains an audit of the public original code. It does not contain the missing original annual inputs,
the unpublished second-formulation NPZ certificates, or the exact repaired
24-hour hindsight witness. No test result here should be described as a full
annual rerun.

Files

../../audit/CERTIFICATE_AUDIT.txt
    Mathematical conditions, actual-source findings and evidence scope.
../../results/published_bound_scalar_audit.json
    Reaggregation of the 48 published site rows and 12 contract groups.
reference_verifier.py
    Standard-library exact finite-box dual, primal residual, PWL facet and
    monetary aggregation checks. All arrays and coefficients must be supplied.
test_reference_verifier.py
    Twelve small algebraic regression tests, distinct from annual results.
annual_dual_revaluation.py
    Exact original-nonconvex-cost revaluation of supplied annual multipliers.
test_annual_dual_revaluation.py
    Five tiny-contract tests for that exact revaluation formula.
../../evidence/complete_verification_manifest_template.json
    A structured evidence inventory; unknown records are explicitly null.

Run the small regression tests from this directory

    python -m unittest discover -v

Run a supplied finite-box certificate

    python reference_verifier.py bound certificate.json

The JSON schema for a finite-box certificate is illustrated below. This
two-variable example is synthetic and has exact optimum 4.

{
  "constant": "0",
  "objective": ["2", "1"],
  "lower": ["1", "0"],
  "upper": ["2", "3"],
  "equalities": [
    {"coefficients": [[0,"1"],[1,"1"]], "rhs":"3", "multiplier":"1"}
  ],
  "inequalities_le": [],
  "certified_allowance": "0",
  "primal": ["1", "2"]
}

Inequalities use G y <= h. Their multipliers must be nonnegative. Equality
multipliers are unrestricted. Sparse rows may list the same column more than
once; duplicate terms are added exactly. All variable bounds must be finite.
Use strings for exact decimal/rational coefficients; use
{"binary64_hex":"0x1.999999999999ap-4"} for the stored binary64 value of 0.1.

Run supporting-facet checks

    python reference_verifier.py facets facets.json

The input contains "points": [[x,f(x)], ...] at EVERY original breakpoint and
both endpoints, and "facets": [{"slope":a,"intercept":b}, ...]. This checks
continuous piecewise-linear costs only. A passing check is invalid if corners
are omitted or the original cost is nonlinear between the stated points.

Run a separable-site aggregation

    python reference_verifier.py aggregate aggregation.json

The input fields are baseline_cost, site_cost_lower_bounds, common_cost,
optionally witness_saving and witness_loss_upper. Site lower bounds must
exclude common_cost, which is added exactly once. A witness-loss bound must
be independently proved; a small constraint residual does not supply one.
