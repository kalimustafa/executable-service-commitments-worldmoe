"""Independent epigraph assembly and exact finite-box certificate evaluation.

The service/cumulative/epigraph matrix is assembled here independently of the
incremental-segment matrix. Shared exact physics and reference-envelope inputs
define the same new conservative outer relaxation. This does not reconstruct
the unpublished original 29 October epigraph programme or certificates.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from envelope_certificate import (q,rational_text,down_units,up_units,ceil_cents,
    floor_cents,original_points,reference_envelopes)


def exact_facets(points):
    h=[]
    for x,y in points:
        while len(h)>=2:
            x0,y0=h[-2];x1,y1=h[-1]
            if (y1-y0)*(x-x1)>=(y-y1)*(x1-x0):h.pop()
            else:break
        h.append((x,y))
    facets=[]
    for (x0,y0),(x1,y1) in zip(h[:-1],h[1:]):
        a=(y1-y0)/(x1-x0);b=y0-a*x0
        if any(a*x+b>y for x,y in points):raise AssertionError('Invalid exact facet')
        facets.append((a,b))
    if not facets:raise ValueError('Epigraph implementation requires a nondegenerate service domain')
    return facets


def solve_epigraph(facets,points,L,U):
    n=len(facets);rows=[];cols=[];vals=[];rhs=[];facet_slot=[]
    for t,fs in enumerate(facets):
        for alpha,beta in fs:
            j=len(rhs);rows.extend([j,j]);cols.extend([t,2*n+t])
            vals.extend([float(alpha),-1.]);rhs.append(-float(beta));facet_slot.append(t)
    M=coo_matrix((vals,(rows,cols)),shape=(len(rhs),3*n)).tocsr()
    # Variables ordered x[0:N], cumulative[0:N], eta[0:N].
    er=np.r_[np.arange(n),np.arange(n),np.arange(1,n)]
    ec=np.r_[np.arange(n),n+np.arange(n),n+np.arange(n-1)]
    ev=np.r_[np.ones(n),-np.ones(n),np.ones(n-1)]
    A=coo_matrix((ev,(er,ec)),shape=(n,3*n)).tocsr()
    low=np.r_[np.zeros(n),[float(v) for v in L[1:]],
              [float(min(c for _,c in pts)) for pts in points]]
    high=np.r_[[float(pts[-1][0]) for pts in points],[float(v) for v in U[1:]],
               [float(max(c for _,c in pts)) for pts in points]]
    result=linprog(np.r_[np.zeros(2*n),np.ones(n)],A_eq=A,b_eq=np.zeros(n),
                  A_ub=M,b_ub=np.asarray(rhs),bounds=np.column_stack((low,high)),
                  method='highs',options={'primal_feasibility_tolerance':1e-8,
                                          'dual_feasibility_tolerance':1e-8})
    if not result.success:raise RuntimeError(result.message)
    # HiGHS' minimisation sensitivity for <= rows is nonpositive. Our
    # Lagrangian uses +mu*(M y-h), so mu=-that reported marginal.
    raw_mu=-result.ineqlin.marginals
    diag={'status':int(result.status),'message':result.message,
          'float_objective':float(result.fun),
          'max_equality_residual':float(np.max(abs(A@result.x))),
          'max_inequality_violation':float(max(0,np.max(M@result.x-np.asarray(rhs)))),
          'equalities':n,'inequalities':len(rhs),'variables':3*n,
          'raw_negative_mu_count':int(np.sum(raw_mu<0))}
    return result.eqlin.marginals,raw_mu,diag


def exact_box_revalue(facets,points,L,U,lambda_values,raw_mu_values,decimals=12):
    scale=10**decimals;n=len(facets);lam=[q(v) for v in lambda_values]+[F(0)]
    mu=[max(F(0),q(v)) for v in raw_mu_values]
    expected=sum(len(fs) for fs in facets)
    if len(mu)!=expected or len(lam)!=n+1:raise ValueError('Multiplier count')
    total=0;idx=0;clipped=sum(q(v)<0 for v in raw_mu_values)
    max_rx=F(0);max_reta=F(0);facet_checks=0
    for t,(fs,pts) in enumerate(zip(facets,points)):
        const=F(0);rx=-lam[t];reta=F(1)
        for alpha,beta in fs:
            multiplier=mu[idx];idx+=1
            const+=multiplier*beta;rx+=multiplier*alpha;reta-=multiplier
            # Exact dominance at every original breakpoint, not a mesh sample.
            for x,c in pts:
                if alpha*x+beta>c:raise AssertionError('Facet dominance')
                facet_checks+=1
        rs=lam[t]-lam[t+1];cap=pts[-1][0]
        eta_low=min(c for _,c in pts);eta_high=max(c for _,c in pts)
        terms=(const,min(F(0),rx*cap),min(rs*L[t+1],rs*U[t+1]),
               min(reta*eta_low,reta*eta_high))
        total+=sum(down_units(v,scale) for v in terms)
        max_rx=max(max_rx,abs(rx));max_reta=max(max_reta,abs(reta))
    return F(total,scale),{'arithmetic_loss_strict_upper_exact':rational_text(F(4*n,scale)),
         'clipped_negative_inequality_multipliers':clipped,
         'every_exact_breakpoint_facet_check_passed':True,'facet_breakpoint_checks':facet_checks,
         'max_x_stationarity_residual_exact':rational_text(max_rx),
         'max_eta_stationarity_residual_exact':rational_text(max_reta),
         'all_finite_boxes_evaluated_exactly':True,
         'finite_epigraph_embedding':'eta=f_original(x), with global original-cost min/max finite bounds'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--windows',type=int,nargs='+',default=[12,16,24])
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    with np.load(args.inputs,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    config=args.repo/'research/R2_PAPER_INTEGRATION_20260923/configs/economic_constraints.json'
    constraints=json.loads(config.read_text())['constraints'];n=len(z['reference_compute'])
    meta={'scope':'New independently assembled epigraph formulation of the public-reference admission-envelope relaxation; not the missing original certificate',
          'input_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
          'terminal_rule':'Source signed fixed-reference MTM term bounded below by -21*E per site, using independently proved 0<tau<=21',
          'verify_only':args.verify_only,'site_results':[]}
    expected={}
    if args.verify_only:
        retained=json.loads((out/'epigraph_results.json').read_text())
        if retained['input_sha256']!=meta['input_sha256']:raise ValueError('Retained annual input hash differs')
        expected={(v['site'],v['window_hours']):v for v in retained['site_results']}
    baseline_up=0;scale=10**12;common=F('2409.7842143056')
    for site in range(4):
        _,R,_,E,Amin,Amax,_=reference_envelopes(z['reference_compute'][:,site])
        points=[];facets=[];encoded=[]
        for t in range(n):
            pts,base,cap,_=original_points(z['rt_lmp_usd_mwh'][t,site],z['pue'][t,site],z['pv_mw'][t,site],
                z['idle_mw'][site],z['work_power_mw'][site],z['reference_compute'][t,site],constraints['grid_import_limit'][site])
            fs=exact_facets(pts);points.append(pts);facets.append(fs);baseline_up+=up_units(base,scale)
            encoded.append({'points':[[rational_text(x),rational_text(c)] for x,c in pts],
                            'facets':[[rational_text(a),rational_text(b)] for a,b in fs]})
        physics_file=out/f'exact_epigraph_physics_zone{site}.json.gz'
        if not args.verify_only:
            with gzip.open(physics_file,'wt',encoding='utf-8') as f:
                json.dump({'input_sha256':meta['input_sha256'],'site':site,'slots':encoded},f,separators=(',',':'))
        else:
            with gzip.open(physics_file,'rt',encoding='utf-8') as f:retained_physics=json.load(f)
            if retained_physics!={'input_sha256':meta['input_sha256'],'site':site,'slots':encoded}:
                raise ValueError('Retained exact physical breakpoints/facets differ from current reconstruction')
        print('epigraph exact facets ready for site',site,flush=True)
        for hours in args.windows:
            D=4*hours;L=[F(0)]+[Amin[max(0,t-D+1)] for t in range(1,n+1)];U=Amax
            file=out/f'epigraph_w{hours}_zone{site}.npz'
            if args.verify_only:
                with np.load(file,allow_pickle=False) as a:
                    if str(a['input_sha256'])!=meta['input_sha256']:raise ValueError('Saved multiplier input hash differs')
                    lam=a['lambda'];mu=a['mu_before_sign_clipping']
                if hashlib.sha256(file.read_bytes()).hexdigest()!=expected[(site,hours)]['multiplier_sha256']:
                    raise ValueError('Saved multiplier bytes differ from retained record')
                if hashlib.sha256(physics_file.read_bytes()).hexdigest()!=expected[(site,hours)]['exact_physics_sha256']:
                    raise ValueError('Saved exact physics bytes differ from retained record')
                with gzip.open(out/f'epigraph_w{hours}_zone{site}_bounds.json.gz','rt',encoding='utf-8') as f:bounds=json.load(f)
                if bounds['x_lower']!='0' or bounds['x_upper']!=[rational_text(p[-1][0]) for p in points]:raise ValueError('Saved service bounds differ')
                if bounds['cumulative_lower']!=[rational_text(v) for v in L[1:]] or bounds['cumulative_upper']!=[rational_text(v) for v in U[1:]]:raise ValueError('Saved cumulative bounds differ')
                if bounds['eta_lower']!=[rational_text(min(c for _,c in p)) for p in points] or bounds['eta_upper']!=[rational_text(max(c for _,c in p)) for p in points]:raise ValueError('Saved epigraph bounds differ')
                diag={'source':'Saved multiplier arrays; solver not called'}
            else:
                lam,mu,diag=solve_epigraph(facets,points,L,U)
                np.savez_compressed(file,**{'lambda':lam,'mu_before_sign_clipping':mu,
                    'input_sha256':meta['input_sha256'],'reference_error_upper':rational_text(E)})
                with gzip.open(out/f'epigraph_w{hours}_zone{site}_bounds.json.gz','wt',encoding='utf-8') as f:
                    json.dump({'x_lower':'0','x_upper':[rational_text(pts[-1][0]) for pts in points],
                      'cumulative_lower':[rational_text(v) for v in L[1:]],
                      'cumulative_upper':[rational_text(v) for v in U[1:]],
                      'eta_lower':[rational_text(min(c for _,c in pts)) for pts in points],
                      'eta_upper':[rational_text(max(c for _,c in pts)) for pts in points],
                      'equality_rows':'x_t + S_(t-1) - S_t = 0, S_0=0',
                      'inequality_rows':'alpha_tj*x_t-eta_t <= -beta_tj; exact facets in matching physics file',
                      'objective':'sum_t eta_t; no other variable coefficients or site constants'},f,separators=(',',':'))
            lower,checks=exact_box_revalue(facets,points,L,U,lam,mu)
            slot_lower=lower;terminal_allowance=21*E;lower-=terminal_allowance
            if args.verify_only and q(expected[(site,hours)]['lower_cost_exact'])!=lower:
                raise AssertionError('Saved exact epigraph lower bound does not reproduce')
            meta['site_results'].append({'site':site,'window_hours':hours,'lower_cost_exact':rational_text(lower),
                'slot_cost_lower_exact':rational_text(slot_lower),
                'terminal_offset_allowance_exact':rational_text(terminal_allowance),
                'lower_cost_usd':float(lower),'lower_cost_down_cents':floor_cents(lower),
                'checks':checks,'solver_diagnostics':diag,
                'multiplier_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                'exact_physics_sha256':hashlib.sha256(physics_file.read_bytes()).hexdigest()})
            print('epigraph',hours,site,'exact lower bound',float(lower),flush=True)
        (out/('epigraph_verify_only_results.json' if args.verify_only else 'epigraph_results.json')).write_text(json.dumps(meta,indent=2)+'\n')
    C0=F(baseline_up,scale)+common;meta['baseline_cost_upper_exact']=rational_text(C0)
    meta['contract_aggregates']=[]
    for hours in args.windows:
        lower=common+sum(q(r['lower_cost_exact']) for r in meta['site_results'] if r['window_hours']==hours)
        upper=C0-lower
        meta['contract_aggregates'].append({'window_hours':hours,'lower_cost_exact':rational_text(lower),
            'savings_ceiling_exact':rational_text(upper),'savings_ceiling_usd':float(upper),
            'savings_ceiling_up_cents':ceil_cents(upper)})
    (out/('epigraph_verify_only_results.json' if args.verify_only else 'epigraph_results.json')).write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta['contract_aggregates'],indent=2),flush=True)


if __name__=='__main__':main()
