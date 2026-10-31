"""Conservative annual certificates from the published earliest-service book.

This is a NEW relaxation, not reproduction of absent admission/multiplier files.
The reference recurrence, zero initial queue, source accounting, and positivity
of historical terminal rates are separately documented proof obligations.
Floating LPs generate multipliers only. Exact original-cost minima and exact
finite-box evaluation certify the encoded outer corridor independently.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import platform
import sys
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def q(v):
    if isinstance(v, F): return v
    if isinstance(v, (float, np.floating)): return F.from_float(float(v))
    return F(v)


def rational_text(v):
    v=q(v)
    return f"{v.numerator}/{v.denominator}"


def down_units(v, scale):
    v=q(v)
    return v.numerator*scale//v.denominator


def up_units(v, scale):
    return -down_units(-q(v), scale)


def ceil_cents(v):
    c=up_units(v, 100)
    sign='-' if c<0 else '';c=abs(c)
    return f"{sign}{c//100}.{c%100:02d}"


def floor_cents(v):
    c=down_units(v, 100)
    sign='-' if c<0 else '';c=abs(c)
    return f"{sign}{c//100}.{c%100:02d}"


def reference_envelopes(reference):
    """Exact bounds on cumulative admissions consistent with the source book.

    beta=2u+u^2 bounds one queue update's combined addition/subtraction error.
    Each full run has m saturated slots and one unsaturated clearing slot.
    Unsaturated slots whose predecessor is also empty have no queue-rounding
    error: fl(0+a)=a and the stored queue becomes exactly zero.
    """
    b=[q(v) for v in reference]
    if not b or any(v<0 or v>1 for v in b): raise ValueError("Reference box")
    if b[-1] == 1: raise ValueError("Unobserved clearing boundary")
    n=len(b); R=[F(0)]
    for v in b:R.append(R[-1]+v)
    u=F(1,2**53); beta=2*u+u*u
    runs=[]; t=0
    while t<n:
        if b[t]<1:
            t+=1;continue
        start=t
        while t<n and b[t]==1:t+=1
        if t==n:raise ValueError("Final busy block is unclosed")
        m=t-start
        # m saturated intervals plus the clearing interval; deliberately loose.
        if F(m+2)*beta>=1:raise ValueError('Reference rounding denominator is not positive')
        err=F((m+1)*(m+2))*beta/(1-F(m+2)*beta)
        runs.append({"start_zero_based":start,"saturated_slots":m,
                     "clearing_slot_zero_based":t,"error":err})
        t+=1
    E=sum((v['error'] for v in runs),F(0))
    # Keep the exact outer corridor compact without rounding its error inward.
    E=F(up_units(E,10**15),10**15)
    next_clear=[0]*(n+1); nxt=n
    for t in range(n,0,-1):
        if b[t-1]<1:nxt=t
        next_clear[t]=nxt
    Amin=[F(0)]+[max(F(0),R[t]-E) for t in range(1,n+1)]
    Amax=[F(0)]+[R[next_clear[t]]+E for t in range(1,n+1)]
    return b,R,next_clear,E,Amin,Amax,runs


def original_points(rt,pue,pv,idle,power,ref,grid_limit):
    """Exact original cost and physical breakpoints, with exact matched baseline."""
    rt,pue,pv,idle,power,ref,grid_limit=map(q,(rt,pue,pv,idle,power,ref,grid_limit))
    if pue<=0 or power<=0:raise ValueError("Invalid physics")
    if pue*idle-pv>grid_limit:raise ValueError("Idle load violates import cap")
    cap=min(F(1),((grid_limit+pv)/pue-idle)/power)
    if cap<0:raise ValueError("Negative exact cap")
    refload=pue*(idle+power*ref)
    g0=max(refload-pv,F(0));v0=max(pv-refload,F(0))
    knots={F(0),cap}
    for x in ((pv/pue-idle)/power, ((pv+g0)/pue-idle)/power):
        if 0<x<cap:knots.add(x)
    def cost(x):
        load=pue*(idle+power*x)
        g=max(load-pv,F(0));v=max(pv-load,F(0))
        return (rt*g+v+abs(g-g0))/4
    return [(x,cost(x)) for x in sorted(knots)], (rt*g0+v0)/4, cap, cost


def lower_hull(points):
    out=[]
    for p in points:
        while len(out)>=2:
            x0,y0=out[-2];x1,y1=out[-1];x2,y2=p
            if (y1-y0)*(x2-x1)>=(y2-y1)*(x1-x0):out.pop()
            else:break
        out.append(p)
    return out


def solve_lp(points,L,U):
    n=len(points);slopes=[];width=[];slot=[];constant=0.
    for t,pts in enumerate(points):
        h=lower_hull([(float(x),float(y)) for x,y in pts]);constant+=h[0][1]
        for (x0,y0),(x1,y1) in zip(h[:-1],h[1:]):
            slopes.append((y1-y0)/(x1-x0));width.append(x1-x0);slot.append(t)
    m=len(slopes)
    # sum segment service + cumulative_previous - cumulative_current = 0.
    row=np.r_[slot,np.arange(n),np.arange(1,n)]
    col=np.r_[np.arange(m),m+np.arange(n),m+np.arange(n-1)]
    val=np.r_[np.ones(m),-np.ones(n),np.ones(n-1)]
    A=coo_matrix((val,(row,col)),shape=(n,m+n)).tocsr()
    lo=np.r_[np.zeros(m),np.array([float(v) for v in L[1:]])]
    up=np.r_[width,np.array([float(v) for v in U[1:]])]
    obj=np.r_[slopes,np.zeros(n)]
    res=linprog(obj,A_eq=A,b_eq=np.zeros(n),bounds=np.column_stack((lo,up)),
                method='highs',options={'primal_feasibility_tolerance':1e-8,
                                        'dual_feasibility_tolerance':1e-8})
    if not res.success:raise RuntimeError(res.message)
    service=np.bincount(np.asarray(slot),weights=res.x[:m],minlength=n)
    diagnostics={'status':int(res.status),'message':res.message,
                 'primal_objective_float':float(res.fun+constant),
                 'max_balance_residual_float':float(np.max(abs(A@res.x))),
                 'variables':m+n,'equalities':n}
    return res.eqlin.marginals,service,diagnostics


def exact_dual(points,L,U,multipliers,decimals=12):
    """L=sum min(f-lambda*x)+sum min((lambda_t-lambda_next)*[L,U])."""
    scale=10**decimals;total=0;count=0
    lam=[q(v) for v in multipliers]+[F(0)]
    for t,pts in enumerate(points):
        phi=min(c-lam[t]*x for x,c in pts)
        gamma=lam[t]-lam[t+1]
        psi=min(gamma*L[t+1],gamma*U[t+1])
        total+=down_units(phi,scale)+down_units(psi,scale);count+=2
    return F(total,scale),F(count,scale)


def exact_clamp(raw,L,U,caps):
    """Exact propagated-corridor repair after explicit clipping to capacity box."""
    x=[min(max(q(v),F(0)),cap) for v,cap in zip(raw,caps)]
    X=[F(0)]
    for v in x:X.append(X[-1]+v)
    low=list(L);high=list(U);n=len(x)
    for t in range(n-1,-1,-1):low[t]=max(low[t],low[t+1]-caps[t])
    for t in range(1,n+1):high[t]=min(high[t],high[t-1]+caps[t-1])
    if any(a>b for a,b in zip(low,high)):raise ValueError("Infeasible exact corridor")
    Y=[min(u,max(l,v)) for l,u,v in zip(low,high,X)]
    out=[Y[t+1]-Y[t] for t in range(n)]
    if any(v<0 or v>c for v,c in zip(out,caps)):raise AssertionError("Capacity repair")
    if any(v<l or v>u for v,l,u in zip(Y,L,U)):raise AssertionError("Prefix repair")
    return out,{"encoded_corridor_exactly_feasible":True,
                "raw_box_clipped":sum(q(v)!=w for v,w in zip(raw,x)),
                "explicit_repair_l1":rational_text(sum((abs(v-q(w)) for v,w in zip(out,raw)),F(0)))}


def nonuniqueness(reference,source):
    """Explicit inference counterexample, NOT a claimed original arrival array."""
    b=reference.copy();n=len(b)
    t=int(np.flatnonzero((b[:-1,0]==1)&(b[1:,0]>0))[0])
    a=b.copy();delta=b[t+1]/2;a[t]+=delta;a[t+1]-=delta
    tree=ast.parse(source.read_text())
    fn=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='reference')
    ns={'np':np,'c':SimpleNamespace(N=n)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
    out0=ns['reference'](b,1.);out1=ns['reference'](a,1.)
    assert np.array_equal(out0[:n],reference) and np.array_equal(out1,out0)
    return {"scope":"Non-identification counterexample for reference-book inversion; neither constructed arrival array is asserted original",
            "source_function_executed_unmodified":True,
            "distinct_admissions_identical_reference":True,
            "first_changed_slot_zero_based":t,
            "first_pair_arrivals_site0":[rational_text(b[t,0]),rational_text(b[t+1,0])],
            "second_pair_arrivals_site0":[rational_text(a[t,0]),rational_text(a[t+1,0])],
            "shifted_mass_site0":rational_text(delta[0])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--windows',type=int,nargs='+',default=[12,16,24])
    p.add_argument('--witness',action='store_true')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    with np.load(args.inputs,allow_pickle=False) as archive:
        z={key:archive[key] for key in archive.files}
    constraints_path=args.repo/'research/R2_PAPER_INTEGRATION_20260923/configs/economic_constraints.json'
    constraints=json.loads(constraints_path.read_text())['constraints']
    n,sites=z['reference_compute'].shape
    if n!=35040 or sites!=4:raise ValueError("Unexpected public annual shape")
    metadata={'scope':'New conservative admission-envelope certificate on published evidence; not reproduction of missing original annual certificates',
              'source_commit':'323c2084e7ee668ba512fb1e33dd56b8394b32d4',
              'input_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              'constraints_sha256':hashlib.sha256(constraints_path.read_bytes()).hexdigest(),
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'rounding_quantum_usd':'1e-12','terminal_cost_in_relaxation':'Source fixed-reference MTM term tau*(R_N-X_N) is bounded below by -21*E per site; separately proved 0<tau<=21 USD/work',
              'ramp_scope':'Nominal static import/compute caps; no ramp constraint is imposed by this original nominal account',
              'site_results':[],'witness_results':[]}
    metadata['reference_nonuniqueness']=nonuniqueness(z['reference_compute'],args.repo/'research/ServiceCommitment_1to5_20260927/vendor/reservation.py')
    common=F('2409.7842143056');scale=10**12;baseline_down=0;baseline_up=0
    for site in range(4):
        b,R,next_clear,E,Amin,Amax,runs=reference_envelopes(z['reference_compute'][:,site])
        points=[];cost_functions=[];caps=[];site_base_lo=0;site_base_hi=0
        for t in range(n):
            pts,base,cap,cost=original_points(z['rt_lmp_usd_mwh'][t,site],z['pue'][t,site],z['pv_mw'][t,site],
                z['idle_mw'][site],z['work_power_mw'][site],z['reference_compute'][t,site],constraints['grid_import_limit'][site])
            points.append(pts);caps.append(cap);cost_functions.append(cost)
            site_base_lo+=down_units(base,scale);site_base_hi+=up_units(base,scale)
        baseline_down+=site_base_lo;baseline_up+=site_base_hi
        if any(cap!=1 for cap in caps):raise ValueError('The reference-rounding witness proof currently requires all nominal caps equal one')
        site_info={'site':site,'saturated_intervals':sum(v['saturated_slots'] for v in runs),
                   'busy_blocks':len(runs),'max_saturated_run':max(v['saturated_slots'] for v in runs),
                   'reference_cumulative_error_upper_exact':rational_text(E),
                   'reference_cumulative_error_upper':float(E),
                   'all_exact_service_caps_one':True,
                   'reference_cost_down_exact':rational_text(F(site_base_lo,scale)),
                   'reference_cost_up_exact':rational_text(F(site_base_hi,scale))}
        print('site',site,'exact physical points ready; E=',float(E),flush=True)
        for hours in args.windows:
            D=4*hours
            if max(v['saturated_slots'] for v in runs)+1>D:
                raise ValueError('The observed reference does not prove feasibility at this duration')
            L=[F(0)]+[Amin[max(0,t-D+1)] for t in range(1,n+1)]
            U=Amax
            lam,service,diag=solve_lp(points,L,U)
            bound,loss=exact_dual(points,L,U,lam)
            slot_bound=bound;terminal_allowance=21*E;bound-=terminal_allowance
            name=f'outer_w{hours}_zone{site}'
            np.savez_compressed(out/(name+'.npz'),multipliers=lam,relaxed_service=service,
                reference=z['reference_compute'][:,site],input_sha256=metadata['input_sha256'])
            rec={**site_info,'window_hours':hours,'bound_cost_exact':rational_text(bound),
                 'slot_cost_lower_exact':rational_text(slot_bound),
                 'terminal_offset_allowance_exact':rational_text(terminal_allowance),
                 'bound_cost_usd':float(bound),'bound_cost_down_cents':floor_cents(bound),
                 'arithmetic_downward_loss_strict_upper':rational_text(loss),'solver_diagnostics':diag,
                 'saved_multiplier_sha256':hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest()}
            metadata['site_results'].append(rec)
            print(name,'proved original-cost dual',float(bound),flush=True)
        if args.witness:
            D=96
            L=[F(0)]+[R[next_clear[max(0,t-D+1)]] for t in range(1,n+1)]
            U=list(R);L[-1]=U[-1]
            _,raw,diag=solve_lp(points,L,U)
            repaired,repair_diag=exact_clamp(raw,L,U,caps)
            cost_units=sum(up_units(fn(x),scale) for fn,x in zip(cost_functions,repaired))
            # Exact unknown-admission corridor differs by at most E in each
            # prefix direction. Existence of its feasible earliest schedule is
            # proved separately using strictly positive clearing-slot slack.
            M=max(q(z['work_power_mw'][site])*q(z['pue'][t,site])*(abs(q(z['rt_lmp_usd_mwh'][t,site]))+2)/4 for t in range(n))
            uniform_l1=2*n*E;money=M*uniform_l1
            min_clear_slack=min(1-b[v['clearing_slot_zero_based']] for v in runs)
            assert E<min_clear_slack
            assert max(v['saturated_slots'] for v in runs)+2 < D
            saved={'scope':'Exact public-reference robust-corridor schedule; actual-admission feasibility/value is a uniform constructive repair theorem, not recovered original witness',
                   'site':site,'execution_exact':[rational_text(v) for v in repaired],
                   'all_encoded_prefixes_exactly_feasible':True}
            (out/f'robust_w24_zone{site}.json').write_text(json.dumps(saved,separators=(',',':'))+'\n')
            w={'site':site,'actual_admission_assignment_materialized':False,
               'exact_robust_corridor_checks':repair_diag,'solver_diagnostics':diag,
               'robust_execution_cost_upper_exact':rational_text(F(cost_units,scale)),
               'uniform_actual_admission_repair_l1_upper_exact':rational_text(uniform_l1),
               'uniform_actual_admission_repair_money_upper_exact':rational_text(money),
               'uniform_actual_admission_repair_money_upper_usd':float(money),
               'fixed_reference_terminal_offset_allowance_exact':rational_text(21*E),
               'min_reference_clearing_slack_exact':rational_text(min_clear_slack),
               'candidate_plus_uniform_repair_cost_upper_exact':rational_text(F(cost_units,scale)+money+21*E)}
            metadata['witness_results'].append(w)
            print('witness site',site,'cost',float(F(cost_units,scale)),'uniform repair $',float(money),flush=True)
        (out/'envelope_results.json').write_text(json.dumps(metadata,indent=2)+'\n')
    C0lo=F(baseline_down,scale)+common;C0hi=F(baseline_up,scale)+common
    metadata['baseline_cost_lower_exact']=rational_text(C0lo)
    metadata['baseline_cost_upper_exact']=rational_text(C0hi)
    metadata['common_cost_counted_once_exact']=rational_text(common)
    metadata['contract_aggregates']=[]
    for hours in args.windows:
        bound=common+sum(q(v['bound_cost_exact']) for v in metadata['site_results'] if v['window_hours']==hours)
        ceiling=C0hi-bound
        metadata['contract_aggregates'].append({'window_hours':hours,
            'cost_lower_exact':rational_text(bound),'savings_ceiling_exact':rational_text(ceiling),
            'savings_ceiling_usd':float(ceiling),'savings_ceiling_up_cents':ceil_cents(ceiling)})
    if args.witness:
        cost=common+sum(q(v['candidate_plus_uniform_repair_cost_upper_exact']) for v in metadata['witness_results'])
        H=C0lo-cost
        metadata['uniform_witness_value']={'scope':'Uniform existence/value guarantee after exact actual-admission repair; not an explicit recovered annual assignment',
            'savings_floor_exact':rational_text(H),'savings_floor_usd':float(H),'savings_floor_down_cents':floor_cents(H),
            'terminal_state_after_uniform_repair':'zero original admitted work remains; source signed MTM offset tau*(R_N-A_N) is covered by the separate 21*E allowance per site',
            'uniform_repair_money_total_upper':float(sum(q(v['uniform_actual_admission_repair_money_upper_exact']) for v in metadata['witness_results'])),
            'fixed_reference_terminal_offset_total_upper_exact':rational_text(sum(q(v['fixed_reference_terminal_offset_allowance_exact']) for v in metadata['witness_results']))}
        for row in metadata['contract_aggregates']:
            premium=H-q(row['savings_ceiling_exact'])
            row['uniform_24h_minus_ceiling_exact']=rational_text(premium)
            row['uniform_24h_minus_ceiling_down_cents']=floor_cents(premium)
    (out/'envelope_results.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'aggregates':metadata['contract_aggregates'],'uniform_witness_value':metadata.get('uniform_witness_value')},indent=2),flush=True)


if __name__=='__main__':main()
