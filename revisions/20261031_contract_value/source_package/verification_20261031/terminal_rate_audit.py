"""Identify a strictly positive enclosure for terminal rates from the released
cash/mark-to-market ledger and the documented binary64 accounting recurrence.

This recovers a coefficient enclosure, not the original training price file.
The proof conditions are documented in terminal_rate_audit.txt.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, csv, hashlib, json
import numpy as np
from scipy.linalg import qr


def ff(x):
    return F.from_float(float(x))


def inverse(a):
    n = len(a)
    rows = [list(r) + [F(int(i == j)) for j in range(n)] for i, r in enumerate(a)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        v = rows[j][j]
        rows[j] = [x / v for x in rows[j]]
        for i in range(n):
            if i != j:
                v = rows[i][j]
                rows[i] = [x - v*y for x, y in zip(rows[i], rows[j])]
    return [r[n:] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    npz = args.repo/'source_data/grid_metrics_inputs_2025.npz'
    daily_path = args.repo/'source_data/current/worldmoe_daily_by_seed.csv'
    z = np.load(npz)
    ref = z['reference_compute']
    k, idle, pue, pv, rt = [z[s] for s in ['work_power_mw','idle_mw','pue','pv_mw','rt_lmp_usd_mwh']]
    assert ref.shape == (35040,4) and np.all(k > 0)
    assert all(z[s].dtype == np.float64 for s in z.files if z[s].dtype.kind == 'f')
    labels = [str(datetime.fromisoformat(str(t)).astimezone(ZoneInfo('America/New_York')).date()) for t in z['timestamp_local']]
    day_ranges = []
    for t, day in enumerate(labels):
        if t == 0 or labels[t-1] != day:
            day_ranges.append([day,t,t])
        else:
            day_ranges[-1][2] = t
    assert len(day_ranges) == 365
    with daily_path.open(newline='') as f:
        tab = {r['date']:r for r in csv.DictReader(f)}
    assert set(tab) == set(labels)

    # These computations reproduce the deterministic coefficient and cash
    # stages of the documented source. Their output binary64 values are the
    # KNOWN constants of the terminal-rate inverse problem.
    def cash(x):
        load = pue * (idle+k*x)
        used = np.minimum(pv,load)
        grid = load-used
        return .25*(rt*grid+(pv-used)).sum(1),grid
    refcost, refgrid = cash(ref)
    records=[]
    qcache,vcache={},{}
    for seed in [20260920,20260921,20260922]:
        x=z[f'compute_seed_{seed}']
        assert np.all(x>=0) and np.all(x<=1)
        cost,grid=cash(x)
        value=refcost-cost-.25*np.abs(grid-refgrid).sum(1)
        q=.25*np.cumsum((ref-x)*k,axis=0)
        qcache[seed]=q; vcache[seed]=value
        for day,start,end in day_ranges:
            before=q[start-1] if start else np.zeros(4)
            mf=q[end]-before
            dv=float(np.sum(value[start:end+1],dtype=np.longdouble)-np.longdouble(float(tab[day][str(seed)])))
            records.append(dict(seed=seed,date=day,start=start,end=end,M=mf,d=dv))
    allM=np.stack([r['M'] for r in records])
    alld=np.array([r['d'] for r in records])
    _,_,piv=qr(allM.T,pivoting=True,mode='economic')
    selected=[records[int(i)] for i in piv[:4]]
    matrix=[]; rhs=[]; ar=[]; br=[]
    u=F(1,2**53)
    def gamma(n):return n*u/(1-n*u)
    g6=gamma(6)
    for r in selected:
        seed,start,end=r['seed'],r['start'],r['end']
        q=qcache[seed];v=vcache[seed]
        before=[ff(x) for x in q[start-1]] if start else [F(0)]*4
        after=[ff(x) for x in q[end]]
        matrix.append([a-b for a,b in zip(after,before)])
        # The LAST export round-trips the intermediate parser's binary64
        # value. The original daily file was parsed once by Pandas' default
        # precise_xstrtod before that export; its error is bounded below.
        csv_value=ff(float(tab[r['date']][str(seed)]))
        rhs.append(sum((ff(x) for x in v[start:end+1]),F(0))-csv_value)
        m=end-start+1
        av=sum((abs(ff(x)) for x in v[start:end+1]),F(0))
        L=lambda t: sum((abs(ff(x)) for x in q[t]),F(0)) if t>=0 else F(0)
        sL=sum((L(t)+L(t-1) for t in range(start,end+1)),F(0))
        endpoint=L(end)+L(start-1)
        # Binary64 subnormal roundoff is covered by an intentionally larger
        # rational 1e-100 additive reserve. No tolerance is used for signs.
        # Default shortest decimal export has relative error <= gamma_2.
        # precise_xstrtod uses <=34 Horner operations plus a cached decimal
        # power and <=2 divisions; gamma_40 encloses those operations. Its
        # 17-character digit cutoff can discard <1e-16 in fixed notation
        # (the only extra non-significant digits are in numbers below one).
        # Scientific notation has <=17 significant digits and no such loss.
        # A 1e-300 reserve covers tiny/subnormal conversion. If y0 is the
        # original daily float and y the observed re-export, then
        # |y-y0| <= eta*|y0|+delta <= (eta*|y|+delta)/(1-eta).
        eta=(1+gamma(2))*(1+gamma(40))-1
        delta=(1+gamma(40))*F(1,10**16)+F(1,10**300)
        csv_error=(eta*abs(csv_value)+delta)/(1-eta)
        a=(u+gamma(m)*(1+u))*av+csv_error+F(1,10**100)
        b=g6*endpoint+(u*(2+u)*(1+g6)+gamma(m)*(1+u)**2*(1+g6))*sL
        ar.append(a);br.append(b)
        r.update(a_bound=float(a),b_bound=float(b),csv_rounding_error_upper=float(csv_error))
    inv=inverse(matrix)
    estimate=[sum((a*b for a,b in zip(row,rhs)),F(0)) for row in inv]
    kappa=max(sum((abs(x) for x in row),F(0)) for row in inv)
    a,b=max(ar),max(br)
    alpha=kappa*b
    assert alpha<1
    norm=max(abs(x) for x in estimate)
    rho_max=(norm+kappa*a)/(1-alpha)
    radius=kappa*(a+b*rho_max)
    lows=[x-radius for x in estimate]
    highs=[x+radius for x in estimate]
    assert min(lows)>0
    # Exact check of the central inverse matrix itself.
    for i in range(4):
        for j in range(4):
            assert sum((inv[i][k0]*matrix[k0][j] for k0 in range(4)),F(0))==int(i==j)
    fit=np.array([float(x) for x in estimate])
    fit_resid=allM@fit-alld
    # Explanatory rates if the published maximum PUE is used to separate
    # rho=PUEmax*historical_rate. The proof only needs rho>0.
    pu=float(np.max(pue))
    result={
        'status':'strictly_positive_terminal_rate_enclosure_under_documented_source_account',
        'source_commit':'323c2084e7ee668ba512fb1e33dd56b8394b32d4',
        'inputs':{'annual_arrays_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'daily_mtm_sha256':hashlib.sha256(daily_path.read_bytes()).hexdigest()},
        'unknown_definition':'rho_z=PUEmax*historical_RT_median_z; terminal dollars per work unit=DT*k_z*rho_z',
        'conditions':['Released binary64 compute and physical arrays are the matched inputs of the published mark-to-market daily account.','The route uses the per-zone branch of vendor/economic_core.py record, no migration and no fee/grid overrides.','IEEE binary64 round-to-nearest arithmetic, shortest round-trip decimal CSV exports, and one intervening Pandas default precise_xstrtod parse as documented in summarize_worldmoe_final.py.','The same frozen terminal coefficients are used in the original annual contract account.'],
        'independent_equalities':4,'all_daily_rows_checked_descriptively':1095,
        'selected_rows':[{k0:r[k0] for k0 in ['seed','date','start','end','a_bound','b_bound','csv_rounding_error_upper']} for r in selected],
        'exact_inverse_verified':True,'inverse_inf_norm_upper':float(kappa),
        'csv_rounding_enclosure':{'eta_exact':str(eta),'delta_exact':str(delta),'formula':'(eta*abs(observed_reexport_float)+delta)/(1-eta)','parser_source':'https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/_libs/src/parser/tokenizer.c','default_selection_source':'https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/_libs/parsers.pyx'},
        'uniform_additive_error_a':float(a),'uniform_rate_error_coefficient_b':float(b),
        'self_bound_alpha':float(alpha),'rho_norm_upper':float(rho_max),
        'enclosure_radius':float(radius),
        'rho_center':[float(x) for x in estimate],
        'rho_lower':[float(x) for x in lows],
        'rho_upper':[float(x) for x in highs],
        'historical_rate_center_if_pue_max_is_published_max':[float(x)/pu for x in estimate],
        'terminal_dollars_per_work_lower':[float(F(1,4)*ff(k[i])*lows[i]) for i in range(4)],
        'terminal_dollars_per_work_upper':[float(F(1,4)*ff(k[i])*highs[i]) for i in range(4)],
        'strict_positive_sign_verified':True,
        'all_rows_descriptive_max_abs_residual_usd':float(np.max(np.abs(fit_resid))),
        'exact_certificate':{
            'matrix':[[str(x) for x in row] for row in matrix],
            'rhs':[str(x) for x in rhs],
            'inverse':[[str(x) for x in row] for row in inv],
            'center':[str(x) for x in estimate],
            'a':str(a),'b':str(b),'kappa':str(kappa),
            'rho_norm_upper':str(rho_max),'radius':str(radius),
            'lower':[str(x) for x in lows],'upper':[str(x) for x in highs],
        },
        'not_claimed':'Exact recovery of original training-price files or original annual dual/witness records.'
    }
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k0:result[k0] for k0 in ['status','selected_rows','inverse_inf_norm_upper','uniform_additive_error_a','uniform_rate_error_coefficient_b','enclosure_radius','rho_lower','historical_rate_center_if_pue_max_is_published_max','terminal_dollars_per_work_upper','all_rows_descriptive_max_abs_residual_usd']},indent=2))


if __name__=='__main__':main()
