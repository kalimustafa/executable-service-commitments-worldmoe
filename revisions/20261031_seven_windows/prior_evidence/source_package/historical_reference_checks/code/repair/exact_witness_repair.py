"""Exact cumulative-path repair for homogeneous divisible service contracts.

This module does not contain or reconstruct the paper's annual hindsight witness.
It supplies a deterministic repair and verifier for supplied arrays. Numbers passed
as strings are exact decimals/rationals; Python floats retain their exact binary
values. The routine guarantees releases, homogeneous deadlines, conservation,
and fixed service-capacity bounds. It does NOT certify ramp constraints, a
heterogeneous deadline menu, a nonpreemptive job schedule, or first-release data.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def rational(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction.from_float(value)
    if hasattr(value, "item"):
        return rational(value.item())
    return Fraction(value)


def prefix(values: Iterable[Fraction]) -> list[Fraction]:
    out = [Fraction(0)]
    for value in values:
        out.append(out[-1] + value)
    return out


@dataclass
class ExactRepair:
    candidate: list[Fraction]
    repaired: list[Fraction]
    lower: list[Fraction]
    upper: list[Fraction]
    capacities: list[Fraction]
    candidate_prefix: list[Fraction]
    repaired_prefix: list[Fraction]
    lower_envelope: list[Fraction]
    upper_envelope: list[Fraction]
    release_error: Fraction
    deadline_error: Fraction
    correction_l1: Fraction
    correction_l1_bound: Fraction

    def report(self) -> dict:
        return {
            "arithmetic": "exact rational; float inputs interpreted as binary64 values",
            "scope": "release/deadline/conservation/fixed service capacity only; no ramp guarantee",
            "slots": len(self.repaired),
            "candidate_release_error_exact": str(self.release_error),
            "candidate_deadline_error_exact": str(self.deadline_error),
            "correction_l1_exact": str(self.correction_l1),
            "correction_l1_bound_exact": str(self.correction_l1_bound),
            "repaired_release_error_exact": "0",
            "repaired_deadline_error_exact": "0",
            "repaired_capacity_error_exact": "0",
            "repaired_nonnegative_error_exact": "0",
            "initial_executed_work_exact": str(self.repaired_prefix[0]),
            "final_executed_work_exact": str(self.repaired_prefix[-1]),
            "repaired_exact": [str(x) for x in self.repaired],
        }


def repair_cumulative_path(
    candidate: Sequence,
    capacities: Sequence,
    lower: Sequence,
    upper: Sequence,
) -> ExactRepair:
    """Repair a path into a feasible cumulative corridor using exact arithmetic.

    lower/upper have N+1 entries; entry u refers to service completed after u
    slots. The origin must be fixed at zero. Both corridor boundaries must be
    nondecreasing. Candidate increments must ALREADY satisfy [0,capacity]
    exactly, or this routine rejects them rather than silently clipping them.
    The returned path obeys any fixed endpoint encoded by lower[N]==upper[N].
    """
    x = list(map(rational, candidate))
    cap = list(map(rational, capacities))
    lo, hi = list(map(rational, lower)), list(map(rational, upper))
    n = len(x)
    if n == 0 or len(cap) != n or len(lo) != n+1 or len(hi) != n+1:
        raise ValueError("Inconsistent path dimensions or empty candidate")
    if lo[0] != 0 or hi[0] != 0:
        raise ValueError("Cumulative execution origin must be fixed at zero")
    if any(c < 0 for c in cap):
        raise ValueError("Negative service capacity")
    if any(v < 0 or v > c for v, c in zip(x, cap)):
        raise ValueError("Raw candidate violates exact nonnegativity or capacity")
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError("Infeasible pointwise service corridor")
    if any(v[i+1] < v[i] for v in (lo, hi) for i in range(n)):
        raise ValueError("Cumulative service boundaries must be nondecreasing")
    lower_env, upper_env = lo.copy(), hi.copy()
    for t in range(n-1, -1, -1):
        lower_env[t] = max(lo[t], lower_env[t+1]-cap[t])
    for t in range(1, n+1):
        upper_env[t] = min(hi[t], upper_env[t-1]+cap[t-1])
    if any(a > b for a, b in zip(lower_env, upper_env)):
        raise ValueError("Infeasible corridor after service-capacity propagation")
    raw_prefix = prefix(x)
    fixed_prefix = [min(b, max(a, v)) for v, a, b in zip(raw_prefix, lower_env, upper_env)]
    fixed = [fixed_prefix[t+1]-fixed_prefix[t] for t in range(n)]
    eps_r = max([Fraction(0)] + [v-u for v, u in zip(raw_prefix, hi)])
    eps_d = max([Fraction(0)] + [l-v for l, v in zip(lo, raw_prefix)])
    l1 = sum(abs(v-w) for v, w in zip(x, fixed))
    l1_bound = n*(eps_r+eps_d)
    # Independent exact certificate for the returned candidate.
    if any(v < 0 or v > c for v, c in zip(fixed, cap)):
        raise AssertionError("Internal repair violated a service capacity")
    if any(v < l or v > u for v, l, u in zip(fixed_prefix, lo, hi)):
        raise AssertionError("Internal repair violated its service corridor")
    if fixed_prefix[0] != 0 or l1 > l1_bound:
        raise AssertionError("Internal origin/correction certificate failed")
    if any(not -eps_r <= f-r <= eps_d for f, r in zip(fixed_prefix, raw_prefix)):
        raise AssertionError("Internal prefix perturbation certificate failed")
    return ExactRepair(x, fixed, lo, hi, cap, raw_prefix, fixed_prefix,
                       lower_env, upper_env, eps_r, eps_d, l1, l1_bound)


def repair_homogeneous(
    arrivals: Sequence,
    candidate: Sequence,
    capacities: Sequence,
    deadline_slots: int,
    *,
    initial_cohorts: Sequence[Mapping] = (),
    close_all: bool = True,
    terminal_remaining=None,
) -> ExactRepair:
    """Build the exact homogeneous release/deadline corridor and repair it.

    Slot r starts at boundary r, and new work at r is due at boundary r+D.
    Initial cohorts were released before boundary 0. Supply each as
    {"remaining": exact amount, "deadline": exclusive integer boundary}; their
    deadlines must lie in 1,...,D-1 to retain the homogeneous/FIFO assumptions.
    They are immediately available, and their remaining amounts comprise q0.

    close_all=True imposes exact completion by the supplied final boundary.
    Otherwise a fixed terminal_remaining may be supplied, or the original
    prefix obligations determine the permitted terminal state. Arrays must
    include any intended recovery tail; no extra slots are silently invented.
    """
    if not isinstance(deadline_slots, int) or deadline_slots < 1:
        raise ValueError("deadline_slots must be a positive integer")
    n = len(candidate)
    a = list(map(rational, arrivals))
    if len(a) > n or any(v < 0 for v in a):
        raise ValueError("Invalid arrivals or insufficient candidate horizon")
    a += [Fraction(0)]*(n-len(a))
    cohorts = []
    for item in initial_cohorts:
        amount, due = rational(item["remaining"]), item["deadline"]
        if amount < 0 or not isinstance(due, int) or not 1 <= due < deadline_slots:
            raise ValueError("Initial cohort violates homogeneous pre-origin deadline assumptions")
        cohorts.append((amount, due))
    q0 = sum((v for v, _ in cohorts), Fraction(0))
    accum = prefix(a)
    lo, hi = [Fraction(0)], [Fraction(0)]
    for completed_slots in range(1, n+1):
        due_initial = sum((v for v, d in cohorts if d <= completed_slots), Fraction(0))
        due_new = accum[max(0, completed_slots-deadline_slots+1)]
        lo.append(due_initial+due_new)
        hi.append(q0+accum[completed_slots])
    total = q0+accum[-1]
    if close_all:
        if terminal_remaining is not None and rational(terminal_remaining) != 0:
            raise ValueError("close_all conflicts with nonzero terminal remaining work")
        lo[-1] = hi[-1] = total
    elif terminal_remaining is not None:
        remaining = rational(terminal_remaining)
        target = total-remaining
        if remaining < 0 or not lo[-1] <= target <= hi[-1]:
            raise ValueError("Terminal remainder violates final service obligation")
        hi = [min(v, target) for v in hi]
        lo[-1] = hi[-1] = target
    return repair_cumulative_path(candidate, capacities, lo, hi)


def uniform_money_bound(slots: int, sites: int, release_eps, deadline_eps,
                        max_abs_rt, max_pue, max_work_power, dt="0.25") -> dict:
    """Conditional bound only; eps must first be certified on the actual arrays."""
    if slots <= 0 or sites <= 0:
        raise ValueError("Positive slot/site counts required")
    er, ed = rational(release_eps), rational(deadline_eps)
    price, pue, power, delta = map(rational, (max_abs_rt, max_pue, max_work_power, dt))
    if min(er, ed, price, pue, power, delta) < 0:
        raise ValueError("Negative input to an absolute-cost bound")
    coefficient = delta*power*pue*(price+2)
    l1 = slots*sites*(er+ed)
    return {
        "status": "conditional_theorem_evaluation_not_an_annual_witness_repair",
        "conditions": [
            "prefix error bounds verified in exact arithmetic including terminal endpoints",
            "candidate increments satisfy exact nonnegativity and fixed service-capacity bounds",
            "homogeneous service corridor is feasible",
            "repair respects all additional original physical constraints",
            "unchanged terminal liabilities or a separately added terminal allowance",
            "reference and witness cost evaluation allowance added separately"],
        "slots_per_site": slots, "sites": sites,
        "coefficient_exact": str(coefficient), "coefficient_usd_per_work": float(coefficient),
        "correction_l1_bound_exact": str(l1), "correction_l1_bound_work": float(l1),
        "cost_correction_bound_exact": str(coefficient*l1),
        "cost_correction_bound_usd": float(coefficient*l1),
        "terminal_allowance_included": False,
        "numerical_cost_evaluation_allowance_included": False,
    }


def self_test() -> dict:
    e = Fraction(1, 10**12)
    tests = []

    def case(label, **kwargs):
        repaired = repair_homogeneous(**kwargs)
        tests.append({"test": label, "passed": True,
                      "correction_l1_exact": str(repaired.correction_l1)})
        return repaired

    case("release_prefix_repair", arrivals=["0.5", "0.5"],
         candidate=[Fraction(1, 2)+e, Fraction(1, 2)-e, 0, 0],
         capacities=[1]*4, deadline_slots=2)
    case("deadline_prefix_repair", arrivals=["0.5", "0.5"],
         candidate=[0, Fraction(1, 2)-e, Fraction(1, 2)+e, 0],
         capacities=[1]*4, deadline_slots=2)
    case("exact_final_total_repair", arrivals=["0.5", "0.5"],
         candidate=["0.5", Fraction(1, 2)-e, 0, 0],
         capacities=[1]*4, deadline_slots=2)
    case("initial_cohort_retains_deadline", arrivals=["0.75", 0],
         candidate=[Fraction(1, 4)-e, Fraction(3, 4)+e],
         capacities=[1, 1], deadline_slots=2,
         initial_cohorts=[{"remaining": "0.25", "deadline": 1}])
    case("variable_capacity", arrivals=["0.5", "0.5", 0],
         candidate=[Fraction(1, 2)-e, "0.25", Fraction(1, 4)+e],
         capacities=["0.5", "0.25", "0.5"], deadline_slots=3)
    case("exact_nonzero_terminal_state", arrivals=[1, 1],
         candidate=["0.25", "0.75"], capacities=[1, 1], deadline_slots=3,
         close_all=False, terminal_remaining=1)
    case("terminal_endpoint_propagates_backward", arrivals=[2, 0],
         candidate=["0.5", "0.5"], capacities=[1, 1], deadline_slots=3,
         close_all=False, terminal_remaining=1)
    bad = [
        ("negative_candidate_rejected", dict(arrivals=[1],candidate=[-e,1+e],capacities=[1,2],deadline_slots=2)),
        ("capacity_violation_rejected", dict(arrivals=[1],candidate=[1+e,0],capacities=[1,1],deadline_slots=2)),
        ("infeasible_corridor_rejected", dict(arrivals=[2],candidate=[1],capacities=[1],deadline_slots=1)),
        ("unmatched_initial_deadline_rejected", dict(arrivals=[0],candidate=[0,1],capacities=[1,1],deadline_slots=2,initial_cohorts=[{"remaining":1,"deadline":3}])),
    ]
    for label, kwargs in bad:
        try:
            repair_homogeneous(**kwargs)
        except ValueError:
            tests.append({"test":label,"passed":True})
        else:
            raise AssertionError(label)
    exhaustive = 0
    for arrivals in product((Fraction(0),Fraction(1,2),Fraction(1)), repeat=3):
        for candidate in product((Fraction(0),Fraction(1,2),Fraction(1)), repeat=3):
            repair_homogeneous(arrivals,candidate,[1,1,1],2)
            exhaustive += 1
    tests.append({"test":"exhaustive_small_rational_paths","passed":True,"instances":exhaustive})
    assert rational(0.1) != rational("0.1")
    tests.append({"test":"binary_float_semantics_preserved","passed":True})
    return {"status":"passed","tests":tests,
            "empirical_annual_witness_repaired":False,
            "note":"These are algorithmic boundary fixtures, not new paper experiments."}


def main():
    parser=argparse.ArgumentParser()
    operation=parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--input", type=Path, help="JSON with repair_homogeneous arguments")
    parser.add_argument("--output", type=Path)
    operation.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    result=self_test() if args.self_test else repair_homogeneous(**json.loads(args.input.read_text())).report()
    text=json.dumps(result,indent=2)
    if args.output:
        args.output.write_text(text+"\n")
    else:
        print(text)


if __name__=="__main__":
    main()
