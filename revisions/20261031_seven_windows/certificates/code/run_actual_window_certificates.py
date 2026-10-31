"""New 2026-10-31 actual-admission seven-duration perfect-information certificates.

Single solver thread, sequential sites/durations. Outer sets preserve year-end
closing work and the original terminal MTM. Feasible witnesses may voluntarily
complete by year end. Saved exact duals require no solver trust. Run with
--verify-only to verify all saved multipliers and exact witnesses without LPs.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, csv, gzip, hashlib, json, os, platform, sys, time, warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import scipy
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_matrix
import exact_certificate_core as e

WINDOWS=[8,12,16,20,24,36,48]
SCALE=10**12
q=e.q;txt=e.rational_text

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump_json(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def write_gz(p,obj):
    data=json.dumps(obj,separators=(',',':')).encode()
    p.write_bytes(gzip.compress(data,mtime=0))

def solve(points,L,U,tau):
    """Convex-envelope LP proposes dual multipliers and service only."""
    n=len(points);slopes=[];width=[];slot=[];constant=0.
    for t,pts in enumerate(points):
        h=e.lower_hull(pts);constant+=float(h[0][1])
        for (x0,y0),(x1,y1) in zip(h[:-1],h[1:]):
            slopes.append(float((y1-y0)/(x1-x0)-tau));width.append(float(x1-x0));slot.append(t)
    m=len(slopes)
    A=coo_matrix((np.r_[np.ones(m),-np.ones(n),np.ones(n-1)],
      (np.r_[slot,np.arange(n),np.arange(1,n)],np.r_[np.arange(m),m+np.arange(n),m+np.arange(n-1)])),shape=(n,m+n)).tocsr()
    lo=np.r_[np.zeros(m),[float(v) for v in L[1:]]]
    up=np.r_[width,[float(v) for v in U[1:]]]
    obj=np.r_[slopes,np.zeros(n)]
    started=time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected',category=OptimizeWarning)
        r=linprog(obj,A_eq=A,b_eq=np.zeros(n),bounds=np.column_stack((lo,up)),method='highs',
          options={'threads':1,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not r.success:raise RuntimeError(r.message)
    x=np.bincount(np.asarray(slot),weights=r.x[:m],minlength=n)
    return r.eqlin.marginals,x,{'status':int(r.status),'message':r.message,'solver_seconds':time.perf_counter()-started,
      'objective_float_excluding_terminal_constant':float(r.fun+constant),'variables':m+n,'equalities':n,
      'max_balance_residual_float':float(np.max(abs(A@r.x))),'solver_threads':1}

def dual(points,L,U,lam,tau,total):
    """Exact finite-box weak dual including tau*(A_N-X_N).

    Since X_N=sum x_t, terminal cost is tau*A_N - tau*sum x_t.
    Subtract lambda_t*(x_t+X_{t-1}-X_t), then minimise each
    original nonconvex slot term over its complete breakpoint set.
    """
    n=len(points);lf=[q(v) for v in lam]+[F(0)];units=0
    for t,pts in enumerate(points):
        phi=min(c-(tau+lf[t])*x for x,c in pts)
        gamma=lf[t]-lf[t+1]
        psi=min(gamma*L[t+1],gamma*U[t+1])
        units+=e.down_units(phi,SCALE)+e.down_units(psi,SCALE)
    return F(units,SCALE)+tau*total

def evaluate(x,functions,base,arrivals,reference_total,tau,physical,n):
    """Exact original slot costs and zero-backlog witness terminal account."""
    A=[F(0)];X=[F(0)]
    for a,v in zip(arrivals,x):A.append(A[-1]+a);X.append(X[-1]+v)
    assert X[-1]==A[-1] and reference_total==A[-1]
    lo=hi=blo=bhi=0;slack=None
    for t,(v,fn,bc) in enumerate(zip(x,functions,base)):
        c=fn(v);lo+=e.down_units(c,SCALE);hi+=e.up_units(c,SCALE)
        blo+=e.down_units(bc,SCALE);bhi+=e.up_units(bc,SCALE)
        rt,pue,pv,idle,k,cap=physical[t]
        grid=max(pue*(idle+k*v)-pv,F(0));ss=cap-grid
        assert ss>=0
        slack=ss if slack is None else min(slack,ss)
    terminal=tau*(A[-1]-X[-1]);assert terminal==0
    return F(lo,SCALE)+terminal,F(hi,SCALE)+terminal,F(blo,SCALE),F(bhi,SCALE),slack

def aggregate(out,inputs,windows):
    records=[json.loads((out/f'actual_w{w}_zone{z}.json').read_text()) for w in windows for z in range(4)]
    common=F('2409.7842143056')
    base=[v for v in records if v['window_hours']==windows[0]]
    C0lo=common+sum(F(v['reference_cost_lower_exact']) for v in base)
    C0hi=common+sum(F(v['reference_cost_upper_exact']) for v in base)
    rows=[]
    for w in windows:
        rr=[v for v in records if v['window_hours']==w]
        cost_floor=common+sum(F(v['outer_cost_lower_exact']) for v in rr)
        cost_upper=common+sum(F(v['witness_cost_upper_exact']) for v in rr)
        h=C0lo-cost_upper;u=C0hi-cost_floor
        assert h<=u
        rows.append({'window_hours':w,'raw_witness_savings_floor_exact':txt(h),'raw_ceiling_exact':txt(u),
           'raw_witness_savings_floor_usd':float(h),'raw_ceiling_usd':float(u),
           'raw_witness_floor_down_cents':e.floor_cents(h),'raw_ceiling_up_cents':e.ceil_cents(u),
           'witness_repair_l1_total_exact':txt(sum(F(v['witness_repair_l1_exact']) for v in rr)),
           'witness_terminal_backlog_exact':'0/1','outer_terminal_forced_complete':False})
    for i,row in enumerate(rows):
        hidx=max(range(i+1),key=lambda j:F(rows[j]['raw_witness_savings_floor_exact']))
        uidx=min(range(i,len(rows)),key=lambda j:F(rows[j]['raw_ceiling_exact']))
        h=F(rows[hidx]['raw_witness_savings_floor_exact']);u=F(rows[uidx]['raw_ceiling_exact'])
        row.update(combined_floor_exact=txt(h),combined_ceiling_exact=txt(u),combined_floor_usd=float(h),combined_ceiling_usd=float(u),
          combined_floor_down_cents=e.floor_cents(h),combined_ceiling_up_cents=e.ceil_cents(u),
          inherited_floor_window_hours=rows[hidx]['window_hours'],inherited_ceiling_window_hours=rows[uidx]['window_hours'],
          interval_width_up_cents=e.ceil_cents(u-h))
    adjacent=[]
    for a,b in zip(rows[:-1],rows[1:]):
        dw=b['window_hours']-a['window_hours'];lo=max(F(0),F(b['combined_floor_exact'])-F(a['combined_ceiling_exact']))
        hi=F(b['combined_ceiling_exact'])-F(a['combined_floor_exact'])
        adjacent.append({'from_hours':a['window_hours'],'to_hours':b['window_hours'],'additional_hours':dw,
          'premium_lower_exact':txt(lo),'premium_upper_exact':txt(hi),'premium_lower_down_cents':e.floor_cents(lo),'premium_upper_up_cents':e.ceil_cents(hi),
          'premium_per_hour_lower_exact':txt(lo/dw),'premium_per_hour_upper_exact':txt(hi/dw),
          'premium_per_hour_lower_down_cents':e.floor_cents(lo/dw),'premium_per_hour_upper_up_cents':e.ceil_cents(hi/dw)})
    result={'scope':'New actual-admission perfect-information opportunity calculations executed on 2026-10-31; not historical WorldMoE operating results',
      'source_evidence_scope':'Identical recovered actual admissions, original reference, prices, physical constraints and financial/terminal account at all seven windows',
      'baseline_cost_lower_exact':txt(C0lo),'baseline_cost_upper_exact':txt(C0hi),'input_sha256':inputs,
      'windows':rows,'adjacent_rights_premiums':adjacent,'site_record_count':len(records),
      'method':'One original-nonconvex-cost exact finite-box dual per site/window, separate exactly repaired feasible witness; no second formulation claimed',
      'nesting_combination':'H_W=max raw H_v over v<=W; U_W=min raw U_v over v>=W; separate raw endpoints retained',
      'terminal_scope':'Outer admits legitimate closing work and tau_z*(A_N-X_N); additional tail feasibility restrictions are relaxed. Witnesses elect zero true year-end backlog.',
      'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    dump_json(out/'seven_window_summary.json',result)
    for name,rr in [('seven_window_bounds.csv',rows),('adjacent_rights_premiums.csv',adjacent)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--verify-only',action='store_true')
    p.add_argument('--windows',nargs='+',type=int,default=WINDOWS)
    p.add_argument('--verify-one',nargs=2,type=int,metavar=('HOURS','SITE'),help='Verify one saved annual site-duration certificate without optimisation or changing the full summary')
    args=p.parse_args();root=args.root.resolve();out=root/'results';out.mkdir(exist_ok=True)
    if args.verify_one:
        args.verify_only=True;args.windows=[args.verify_one[0]];selected_sites=[args.verify_one[1]]
        assert args.verify_one[0] in WINDOWS and 0<=args.verify_one[1]<4
    else:selected_sites=range(4)
    forcing=root/'inputs/forcing_2025.npz';public=root/'inputs/grid_metrics_inputs_2025.npz';cp=root/'inputs/economic_constraints.json'
    with np.load(forcing,allow_pickle=False) as f:ff={k:f[k].copy() for k in f.files}
    with np.load(public,allow_pickle=False) as f:gg={k:f[k].copy() for k in f.files}
    constraints=json.loads(cp.read_text())['constraints'];n,zs=ff['eagle_arrivals'].shape;assert (n,zs)==(35040,4)
    for a,b in [('rt','rt_lmp_usd_mwh'),('pv','pv_mw'),('pue','pue'),('k','work_power_mw'),('idle','idle_mw')]:assert np.array_equal(ff[a],gg[b])
    ar=ff['eagle_arrivals'];queue=np.zeros(4);br=np.zeros_like(ar)
    for t in range(n):queue+=ar[t];br[t]=np.minimum(queue,1.);queue-=br[t]
    assert np.array_equal(br,gg['reference_compute']) and not queue.any()
    ih={'forcing':sha(forcing),'published_arrays':sha(public),'constraints':sha(cp),'core':sha(root/'code/exact_certificate_core.py')}
    for z in selected_sites:
        print('building exact original physics for site',z,flush=True);site_started=time.perf_counter()
        aa=[q(v) for v in ar[:,z]];A=[F(0)]
        for v in aa:assert v>=0;A.append(A[-1]+v)
        R=sum((q(v) for v in br[:,z]),F(0));assert R==A[-1]
        tau=q(.25)*q(ff['k'][z])*q(max(constraints['pue_max']))*q(ff['historical_rt'][z]);assert 0<tau<21
        points=[];base=[];caps=[];functions=[];physical=[]
        for t in range(n):
            pts,bc,cap,fn=e.original_points(ff['rt'][t,z],ff['pue'][t,z],ff['pv'][t,z],ff['idle'][z],ff['k'][z],br[t,z],constraints['grid_import_limit'][z])
            points.append(pts);base.append(bc);caps.append(cap);functions.append(fn)
            physical.append(tuple(map(q,(ff['rt'][t,z],ff['pue'][t,z],ff['pv'][t,z],ff['idle'][z],ff['k'][z],constraints['grid_import_limit'][z]))))
        assert all(c==1 for c in caps)
        for w in args.windows:
            prefix=out/f'actual_w{w}_zone{z}';done=prefix.with_suffix('.json');dualfile=out/f'actual_w{w}_zone{z}_dual.npz';witfile=out/f'actual_w{w}_zone{z}_witness.json.gz'
            if done.exists() and not args.verify_only:
                assert json.loads(done.read_text())['input_sha256']==ih,'Cached certificate inputs changed'
                print('cached',w,z,flush=True);continue
            D=4*w;L=[F(0)]+[A[max(0,t-D+1)] for t in range(1,n+1)];U=list(A);WL=list(L);WL[-1]=A[-1]
            started=time.perf_counter()
            if args.verify_only:
                with np.load(dualfile,allow_pickle=False) as zz:lam=zz['multipliers'];raw=zz['witness_raw_service']
                wr=json.loads(gzip.decompress(witfile.read_bytes()));x=[F(v) for v in wr['execution_exact']]
                diag={};wdiag={};repair={'explicit_repair_l1':wr['repair_l1_exact']}
            else:
                lam,outer,diag=solve(points,L,U,tau)
                _,raw,wdiag=solve(points,WL,U,F(0))
                x,repair=e.exact_clamp(raw,WL,U,caps)
                np.savez_compressed(dualfile,multipliers=lam,outer_relaxed_service=outer,witness_raw_service=raw,input_sha256=ih['forcing'])
            bound=dual(points,L,U,lam,tau,A[-1])
            X=[F(0)]
            for v,c in zip(x,caps):assert 0<=v<=c;X.append(X[-1]+v)
            assert all(l<=xx<=u for l,xx,u in zip(WL,X,U)) and X[-1]==A[-1]
            jlo,jhi,blo,bhi,slack=evaluate(x,functions,base,aa,R,tau,physical,n)
            if not args.verify_only:
                write_gz(witfile,{'scope':'New actual-admission feasible perfect-information witness, voluntarily complete by year end',
                  'window_hours':w,'site':z,'input_sha256':ih['forcing'],'execution_exact':[txt(v) for v in x],
                  'cumulative_execution_exact':[txt(v) for v in X],'repair_l1_exact':repair['explicit_repair_l1']})
            rec={'window_hours':w,'site':z,'input_sha256':ih,'outer_year_end_completion_enforced':False,
              'terminal_rate_exact':txt(tau),'source_total_arrival_work_exact':txt(A[-1]),'source_reference_minus_arrival_total_exact':txt(R-A[-1]),
              'outer_cost_lower_exact':txt(bound),'witness_cost_lower_exact':txt(jlo),'witness_cost_upper_exact':txt(jhi),
              'reference_cost_lower_exact':txt(blo),'reference_cost_upper_exact':txt(bhi),
              'witness_repair_l1_exact':repair['explicit_repair_l1'],'witness_all_prefixes_and_capacity_exact':True,
              'witness_all_import_limits_exact':True,'minimum_import_slack_mw':float(slack),'witness_terminal_backlog_exact':'0/1',
              'dual_arithmetic_loss_strict_upper_exact':txt(F(2*n,SCALE)),
              'witness_cost_rounding_width_exact':txt(jhi-jlo),'dual_sha256':sha(dualfile),'witness_sha256':sha(witfile),
              'outer_solver_diagnostics':diag,'witness_solver_diagnostics':wdiag,'elapsed_seconds':time.perf_counter()-started}
            if args.verify_only:
                old=json.loads(done.read_text())
                for k in ['input_sha256','dual_sha256','witness_sha256']:assert old[k]==rec[k],(w,z,k)
                for k in ['outer_cost_lower_exact','witness_cost_lower_exact','witness_cost_upper_exact','reference_cost_lower_exact','reference_cost_upper_exact']:
                    assert old[k]==rec[k],(w,z,k)
                if args.verify_one:dump_json(out/f'saved_path_verification_w{w}_zone{z}.json',{'passed':True,'checked_site_contracts':1,'window_hours':w,'site':z,'scope':'One saved annual certificate and witness reloaded and evaluated exactly; no LP solver invoked','input_sha256':ih,'dual_sha256':sha(dualfile),'witness_sha256':sha(witfile),'verifier_sha256':sha(Path(__file__)),'outer_cost_lower_exact':txt(bound),'witness_cost_lower_exact':txt(jlo),'witness_cost_upper_exact':txt(jhi),'all_prefix_capacity_import_terminal_checks_exact':True})
                print('verified saved',w,z,flush=True)
            else:
                dump_json(done,rec);print('W',w,'site',z,'H',float(blo-jhi),'U',float(bhi-bound),'seconds',round(rec['elapsed_seconds'],2),flush=True)
        print('site complete',z,'seconds',round(time.perf_counter()-site_started,2),flush=True)
    if args.verify_one:
        print('Single saved certificate verification complete',args.verify_one,flush=True);return
    result=aggregate(out,ih,args.windows)
    if args.verify_only:dump_json(out/'solver_free_verification.json',{'passed':True,'checked_site_contracts':4*len(args.windows),'windows':args.windows,'source_sha256':sha(Path(__file__))})
    print(json.dumps(result['windows'],indent=2),flush=True)

if __name__=='__main__':main()
