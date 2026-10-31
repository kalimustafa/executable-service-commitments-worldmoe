#!/usr/bin/env python3
"""Exact algebraic checks for supplied LP relaxation certificates.

This is a newly written independent reference verifier.  It is NOT the original
annual experiment code and does not reconstruct absent annual inputs.  Passing
these checks proves facts about the encoded model; outer-set containment, cost
dominance and provenance require the separate manifest and source-model audit.

All calculations use fractions.  JSON decimal strings preserve their written
values.  A binary64 coefficient may instead be encoded as {"binary64_hex": hex}.
An external solver is never called, and no solver success flag is trusted.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path


def Q(value):
    """Return the exact declared number, rejecting booleans/nonfinite values."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a coefficient")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction.from_float(value)
    if isinstance(value, dict):
        if set(value) == {"binary64_hex"}:
            return Fraction.from_float(float.fromhex(value["binary64_hex"]))
        if set(value) == {"numerator", "denominator"}:
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        raise ValueError("Unknown exact-number encoding")
    return Fraction(value)


def floor_to(value, decimals=2):
    """Downward rounding, including negative values."""
    q, scale = Q(value), 10 ** decimals
    return Fraction((q.numerator * scale) // q.denominator, scale)


def ceil_to(value, decimals=2):
    """Upward rounding, including negative values."""
    return -floor_to(-Q(value), decimals)


def fixed(value, decimals=2):
    """Format an already rounded multiple of 10**(-decimals), exactly."""
    scaled = Q(value) * 10 ** decimals
    if scaled.denominator != 1:
        raise ValueError("Apply floor_to or ceil_to before fixed")
    i = scaled.numerator
    sign = "-" if i < 0 else ""
    digits = str(abs(i)).zfill(decimals + 1)
    return sign + (digits[:-decimals] + "." + digits[-decimals:] if decimals else digits)


def exact_text(value):
    q = Q(value)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def sparse_row(row, n):
    result = {}
    for item in row.get("coefficients", []):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("A sparse coefficient is [column_index, exact_value]")
        i, value = item
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n:
            raise ValueError("Invalid column index")
        result[i] = result.get(i, Fraction(0)) + Q(value)
    return {i: v for i, v in result.items() if v}


def finite_box_bound(model, clip_inequality_multipliers=True):
    """Evaluate a cost lower bound for Ay=b, Gy<=h, l<=y<=u.

    L = c0 + lambda*b - mu*h + sum_i min(l_i*r_i, u_i*r_i)
    r = c - A.T*lambda + G.T*mu; lambda free; mu >= 0.

    A nonnegative allowance is subtracted after evaluation.  It must itself
    come from a documented proof, not merely a solver feasibility tolerance.
    """
    c = [Q(x) for x in model["objective"]]
    lower = [Q(x) for x in model["lower"]]
    upper = [Q(x) for x in model["upper"]]
    n = len(c)
    if len(lower) != n or len(upper) != n:
        raise ValueError("Objective and finite bounds must have equal lengths")
    if any(l > u for l, u in zip(lower, upper)):
        raise ValueError("An encoded box is empty")
    residual = list(c)
    constant = Q(model.get("constant", "0"))
    clipped = []
    for j, row in enumerate(model.get("equalities", [])):
        multiplier, rhs = Q(row["multiplier"]), Q(row["rhs"])
        constant += multiplier * rhs
        for i, a in sparse_row(row, n).items():
            residual[i] -= a * multiplier
    for j, row in enumerate(model.get("inequalities_le", [])):
        multiplier, rhs = Q(row["multiplier"]), Q(row["rhs"])
        if multiplier < 0:
            if not clip_inequality_multipliers:
                raise ValueError(f"Inequality multiplier {j} is negative")
            clipped.append(j)
            multiplier = Fraction(0)
        constant -= multiplier * rhs
        for i, g in sparse_row(row, n).items():
            residual[i] += g * multiplier
    box_terms = [min(l*r, u*r) for l, u, r in zip(lower, upper, residual)]
    raw = constant + sum(box_terms, Fraction(0))
    allowance = Q(model.get("certified_allowance", "0"))
    if allowance < 0:
        raise ValueError("Certified allowance must be nonnegative")
    bound = raw - allowance
    return {
        "raw_bound": raw,
        "corrected_bound": bound,
        "downward_cent_bound": floor_to(bound),
        "clipped_inequality_indices": clipped,
        "stationarity_residual": residual,
        "box_correction": sum(box_terms, Fraction(0)),
    }


def check_primal(model, vector):
    """Check encoded constraints exactly; a tolerance is not feasibility."""
    y = [Q(v) for v in vector]
    n = len(model["objective"])
    if len(y) != n:
        raise ValueError("Wrong primal-vector length")
    eq = []
    ineq = []
    for row in model.get("equalities", []):
        eq.append(abs(sum(a*y[i] for i, a in sparse_row(row, n).items()) - Q(row["rhs"])))
    for row in model.get("inequalities_le", []):
        ineq.append(max(Fraction(0), sum(g*y[i] for i, g in sparse_row(row, n).items()) - Q(row["rhs"])))
    box = [max(Fraction(0), Q(l)-v, v-Q(u)) for l, v, u in zip(model["lower"], y, model["upper"])]
    errors = eq + ineq + box
    return {"exactly_feasible": all(v == 0 for v in errors),
            "max_equality_residual": max(eq, default=Fraction(0)),
            "max_inequality_violation": max(ineq, default=Fraction(0)),
            "max_box_violation": max(box, default=Fraction(0))}


def verify_facets(points, facets):
    """Verify each affine minorant at every PWL breakpoint.

    The original cost must equal linear interpolation between these points on
    the FULL finite feasible domain.  For nonlinear pieces or missing corners,
    this test alone is insufficient.  Each facet is {'slope': a,'intercept': b}.
    """
    pts = [(Q(x), Q(y)) for x, y in points]
    if len(pts) < 2 or any(pts[i][0] >= pts[i+1][0] for i in range(len(pts)-1)):
        raise ValueError("Need at least two strictly ordered breakpoints")
    violations, slacks = [], []
    for j, facet in enumerate(facets):
        a, b = Q(facet["slope"]), Q(facet["intercept"])
        for i, (x, y) in enumerate(pts):
            slack = y - a*x - b
            slacks.append(slack)
            if slack < 0:
                violations.append({"facet": j, "breakpoint": i, "violation": -slack})
    return {"valid_minorants": not violations, "violations": violations,
            "minimum_slack": min(slacks, default=None),
            "breakpoint_count": len(pts), "facet_count": len(facets)}


def exact_piecewise_minimum(points, linear_shift="0"):
    """Exact global minimum of a complete finite PWL function minus shift*x."""
    pts = [(Q(x), Q(y)) for x, y in points]
    if len(pts) < 2 or any(pts[i][0] >= pts[i+1][0] for i in range(len(pts)-1)):
        raise ValueError("Need the complete ordered PWL domain")
    shift = Q(linear_shift)
    values = [(y - shift*x, x) for x, y in pts]
    return min(values)


def aggregate_certificates(baseline_cost, site_cost_lower_bounds, common_cost="0",
                           witness_saving=None, witness_loss_upper="0"):
    """Aggregate separable sites, adding their common fixed cost exactly once.

    site_cost_lower_bounds MUST exclude common_cost. Network-coupled sites need
    a joint bound and cannot be certified by this separable aggregation alone.
    """
    if not site_cost_lower_bounds:
        raise ValueError("At least one site bound is required")
    c0, common = Q(baseline_cost), Q(common_cost)
    lower = common + sum((Q(v) for v in site_cost_lower_bounds), Fraction(0))
    ceiling = c0 - lower
    result = {"cost_lower_bound": lower, "savings_ceiling": ceiling,
              "cost_lower_bound_cents": fixed(floor_to(lower)),
              "savings_ceiling_cents": fixed(ceil_to(ceiling))}
    if witness_saving is not None:
        loss = Q(witness_loss_upper)
        if loss < 0:
            raise ValueError("Witness loss upper bound must be nonnegative")
        witness_lower = Q(witness_saving) - loss
        premium = witness_lower - ceiling
        result.update(witness_saving_lower=witness_lower,
                      witness_saving_lower_cents=fixed(floor_to(witness_lower)),
                      premium_lower=premium,
                      premium_lower_cents=fixed(floor_to(premium)))
    return result


def jsonable(value):
    if isinstance(value, Fraction):
        return exact_text(value)
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("bound", "facets", "aggregate"))
    parser.add_argument("input_json", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input_json.read_text(), parse_float=Decimal)
    if args.operation == "bound":
        out = finite_box_bound(data)
        if "primal" in data:
            out["primal_diagnostics"] = check_primal(data, data["primal"])
    elif args.operation == "facets":
        out = verify_facets(data["points"], data["facets"])
    else:
        out = aggregate_certificates(**data)
    out["scope"] = "Exact checks of encoded data only; no original annual experiment is rerun."
    print(json.dumps(jsonable(out), indent=2))


if __name__ == "__main__":
    main()
