"""Exact original-cost dual revaluation for the public homogeneous-contract LP.

This module evaluates supplied annual multipliers; it does NOT regenerate absent
data, infer arrivals from immediate-service execution, or call a model solver.
It implements the actual release/deadline queue relaxation in public
ServiceCommitment_1to5_20260927/scripts/run_bounds.py without trusting its
floating-point convex hull.  IEEE binary64 arrays are interpreted exactly.

Every scalar original-cost minimum is evaluated at every exact PWL breakpoint.
Each exact summand is rounded downward to the declared dollar quantum before
integer accumulation, avoiding both catastrophic cancellation and denominator
growth.  For M summands, 0 <= exact dual - returned bound < M*quantum.
"""
from fractions import Fraction as F
from reference_verifier import Q, floor_to, ceil_to, fixed, exact_text


def original_slot_breakpoints(rt, pue, pv, idle, power, reference_grid, grid_limit,
                              compute_limit=1, dt=F(1,4), fee=1):
    rt, pue, pv, idle, power, reference_grid, grid_limit, compute_limit, dt, fee = map(
        Q, (rt,pue,pv,idle,power,reference_grid,grid_limit,compute_limit,dt,fee))
    if pue <= 0 or power <= 0 or compute_limit < 0 or reference_grid < 0 or fee < 0:
        raise ValueError("Invalid physical or ledger coefficient")
    cap = min(compute_limit, max(F(0), ((grid_limit+pv)/pue-idle)/power))
    if pue*idle-pv > grid_limit:
        raise ValueError("Idle load alone violates the stated import cap")
    points = {F(0), cap}
    for x in ((pv/pue-idle)/power, ((pv+reference_grid)/pue-idle)/power):
        if 0 < x < cap:
            points.add(x)
    def cost(x):
        load = pue*(idle+power*x)
        grid, spill = max(load-pv,F(0)), max(pv-load,F(0))
        return dt*(rt*grid + spill + fee*abs(grid-reference_grid))
    return [(x,cost(x)) for x in sorted(points)]


def revalue_zone(*, arrivals, rt, pue, pv, idle, power, reference_grid, grid_limit,
                 pue_max, terminal_rt, multipliers, window_slots, compute_limit=1,
                 dt=F(1,4), fee=1, decimals=12):
    """Return a proved lower bound for one site and one homogeneous contract.

    End-of-slot backlog Q_t may contain only arrivals from max(0,t-D+2)..t.
    The terminal coefficient matches the public archived liability rule; the
    relaxation does not certify a physically feasible post-year completion plan.
    """
    n=len(arrivals)
    if not isinstance(window_slots,int) or window_slots < 1:
        raise ValueError("window_slots must be a positive integer")
    if any(len(v)!=n for v in (rt,pue,pv,reference_grid,multipliers)):
        raise ValueError("Annual array lengths differ")
    a=[Q(float(v)) if hasattr(v,"dtype") else Q(v) for v in arrivals]
    if any(v<0 for v in a): raise ValueError("Negative arrival")
    lam=[Q(float(v)) if hasattr(v,"dtype") else Q(v) for v in multipliers]
    idle,power,grid_limit,pue_max,terminal_rt,dt,fee,compute_limit=map(
        Q,(idle,power,grid_limit,pue_max,terminal_rt,dt,fee,compute_limit))
    terminal=dt*power*pue_max*terminal_rt
    scale=10**decimals
    total_units=0; count=0; backlog_cap=F(0)
    def add_exact(term):
        nonlocal total_units,count
        total_units += (term.numerator*scale)//term.denominator
        count+=1
    for t in range(n):
        # In-year release/deadline bound for q after serving interval t.
        backlog_cap+=a[t]
        remove=t-window_slots+1
        if remove>=0: backlog_cap-=a[remove]
        if backlog_cap<0: raise AssertionError("Exact rolling arrivals became negative")
        points=original_slot_breakpoints(rt[t],pue[t],pv[t],idle,power,
            reference_grid[t],grid_limit,compute_limit,dt,fee)
        add_exact(lam[t]*a[t])
        add_exact(min(cost-lam[t]*x for x,cost in points))
        reduced=(lam[t+1] if t+1<n else terminal)-lam[t]
        add_exact(backlog_cap*min(F(0),reduced))
    lower=F(total_units,scale)
    return {"cost_lower_bound":lower,
            "cost_lower_bound_cents":fixed(floor_to(lower)),
            "arithmetic_loss_strict_upper":F(count,scale),
            "summand_count":count,"window_slots":window_slots,
            "scope":"Homogeneous, divisible service; in-year prefixes and priced terminal backlog; no post-year feasibility claim."}
