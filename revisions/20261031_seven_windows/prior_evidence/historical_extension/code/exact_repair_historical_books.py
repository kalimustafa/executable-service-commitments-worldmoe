"""Exact dyadic service repair and settlement of the recovered historical books.

All input binary64 numbers are interpreted exactly. No floating arithmetic is
used for feasibility, repair costs, or reported enclosures. NumPy is used only
to load source archives. Original historical files remain unchanged.
"""
from pathlib import Path
from fractions import Fraction as F
import csv, gzip, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1];INP=ROOT/'inputs';OUT=ROOT/'results'
SEEDS=[(20260920,6),(20260921,2),(20260922,0)]

def ratios(a): return [float(v).as_integer_ratio() for v in np.asarray(a).reshape(-1)]
def encode(r,Q): return [n*(Q//d) for n,d in r]
def text(v): return f'{v.numerator}/{v.denominator}'
def floor_scaled(v,s): return v.numerator*s//v.denominator
def ceil_scaled(v,s): return -floor_scaled(-v,s)
def decimal_bounds(v,scale=10**9):
    # JSON numeric endpoints are accompanied by exact integer endpoint encodings.
    lo=floor_scaled(v,scale);hi=ceil_scaled(v,scale)
    return {'lower_integer':lo,'upper_integer':hi,'scale':scale,'lower':lo/scale,'upper':hi/scale}

def repair(raw,a,Q,D):
    n=len(a);A=[0];X=[0]
    for v in a:A.append(A[-1]+v)
    clipped=[min(Q,max(0,v)) for v in raw]
    for v in clipped:X.append(X[-1]+v)
    L=[0]+[A[max(0,t-D+1)] for t in range(1,n+1)]
    U=A.copy();L[-1]=A[-1]
    for t in range(n-1,-1,-1):L[t]=max(L[t],L[t+1]-Q)
    for t in range(1,n+1):U[t]=min(U[t],U[t-1]+Q)
    assert all(l<=u for l,u in zip(L,U))
    Y=[min(u,max(l,v)) for l,u,v in zip(L,U,X)]
    x=[Y[t+1]-Y[t] for t in range(n)]
    assert all(0<=v<=Q for v in x)
    assert all(Y[t]<=A[t] for t in range(n+1))
    assert all(Y[min(n,t+D)]>=A[t+1] for t in range(n))
    assert Y[-1]==A[-1]
    return x,sum(abs(u-v) for u,v in zip(x,raw))

def main():
    OUT.mkdir(exist_ok=True)
    f=np.load(INP/'forcing_2025.npz',allow_pickle=False);n=len(f['times'])
    rows=[];all_sites=[]
    for seed,ep in SEEDS:
        for w in [12,16,24]:
            name=f'CdfR2_Eagle_2025_s{seed}_ep{ep:02d}_coupled_cdf_w{w}'
            z=np.load(INP/f'{name}_complete_book.npz',allow_pickle=False)
            assert np.all(z['compute'][n:]==0) and np.all(z['reference'][n:]==0)
            exact_raw=F(0);exact_repaired=F(0);bound=F(0);l1=F(0);saved=[];sites=[]
            for site in range(4):
                rr={key:ratios(f[key][:,site]) for key in ['eagle_arrivals','rt','pv','pue']}
                rr['raw']=ratios(z['compute'][:n,site]);rr['ref']=ratios(z['reference'][:n,site])
                rr['constants']=ratios([f['k'][site],f['idle'][site],f['grid_limit'][site]])
                Q=max(d for r in rr.values() for _,d in r)
                v={key:encode(r,Q) for key,r in rr.items()};k,idle,limit=v['constants']
                x,delta=repair(v['raw'],v['eagle_arrivals'],Q,4*w)
                raw_sum=repaired_sum=0;max_lipschitz=0
                Q2=Q*Q;Q3=Q2*Q;Q4=Q3*Q
                for t in range(n):
                    p,rt,pv=v['pue'][t],v['rt'][t],v['pv'][t]
                    # Every nominal electrical cap is at least the unit compute cap.
                    assert p*(idle*Q+k*Q) <= (limit+pv)*Q2
                    reference_load=p*(idle*Q+k*v['ref'][t]);pv3=pv*Q2
                    g0=max(reference_load-pv3,0);s0=max(pv3-reference_load,0)
                    for xx,which in [(v['raw'][t],0),(x[t],1)]:
                        load=p*(idle*Q+k*xx);grid=max(load-pv3,0);spill=max(pv3-load,0)
                        # [RT*(g0-g) + spill0-spill - |g-g0|] * 0.25
                        value=rt*(g0-grid)+(s0-spill-abs(grid-g0))*Q
                        if which==0:raw_sum+=value
                        else:repaired_sum+=value
                    max_lipschitz=max(max_lipschitz,k*p*(abs(rt)+2*Q))
                raw_value=F(raw_sum,4*Q4);fixed_value=F(repaired_sum,4*Q4)
                cost_bound=F(max_lipschitz*delta,4*Q4)
                assert abs(raw_value-fixed_value)<=cost_bound
                exact_raw+=raw_value;exact_repaired+=fixed_value;bound+=cost_bound;l1+=F(delta,Q)
                sr={'site':site,'denominator':str(Q),'changed_slots':sum(u!=v for u,v in zip(x,v['raw'])),
                    'repair_l1_work_exact':text(F(delta,Q)),'repair_l1_work':float(F(delta,Q)),
                    'cost_change_bound_usd_exact':text(cost_bound),'cost_change_bound_usd':float(cost_bound),
                    'exact_raw_saving_usd':text(raw_value),'exact_repaired_saving_usd':text(fixed_value),
                    'release_deadline_capacity_terminal_checks_exact':True,
                    'electrical_capacity_check_exact':True}
                sites.append(sr);saved.append({'site':site,'denominator':str(Q),'service_numerators':[str(i) for i in x]})
            payload={'name':name,'source_book_sha256':hashlib.sha256((INP/f'{name}_complete_book.npz').read_bytes()).hexdigest(),
                'encoding':'exact integer numerator / common power-of-two denominator per site',
                'intervals':n,'window_h':w,'terminal_service_after_observed_year_exact':'0/1','sites':saved}
            outpath=OUT/f'exact_repaired_s{seed}_w{w}.json.gz'
            with gzip.open(outpath,'wt',encoding='utf-8') as g:json.dump(payload,g,separators=(',',':'))
            row={'seed':seed,'window_h':w,'exact_repaired_saving_usd':text(exact_repaired),
                'repaired_saving_lower_1e9':floor_scaled(exact_repaired,10**9),
                'repaired_saving_upper_1e9':ceil_scaled(exact_repaired,10**9),
                'repair_l1_work_exact':text(l1),'repair_l1_work':float(l1),
                'repair_cost_bound_usd_exact':text(bound),'repair_cost_bound_usd':float(bound),
                'observed_cost_change_usd_exact':text(exact_repaired-exact_raw),
                'all_service_and_electrical_checks_exact':True,
                'repaired_witness_sha256':hashlib.sha256(outpath.read_bytes()).hexdigest()}
            rows.append(row);all_sites.append({'seed':seed,'window_h':w,'sites':sites})
            print(seed,w,'saving',float(exact_repaired),'repair bound',float(bound),'L1',float(l1),flush=True)
    means={w:sum((F(r['exact_repaired_saving_usd']) for r in rows if r['window_h']==w),F(0))/3 for w in [12,16,24]}
    result={'scope':'Exact feasible numerical repairs of recovered historical trajectories, not new trained-policy runs and not substitutions for the raw historical outputs.',
        'original_inputs_preserved':True,'all_nine_exact_feasible':True,
        'terminal_policy_and_reference_cost_difference_exact':'0/1',
        'mean_repaired_savings_usd_exact':{str(w):text(v) for w,v in means.items()},
        'mean_repaired_savings_enclosure':{str(w):decimal_bounds(v) for w,v in means.items()},
        'paired_12_to_24_mean_increment_exact':text(means[24]-means[12]),
        'paired_12_to_24_mean_increment_enclosure':decimal_bounds(means[24]-means[12]),
        'max_repair_cost_bound_usd':max(r['repair_cost_bound_usd'] for r in rows),'rows':rows,'site_details':all_sites}
    (OUT/'historical_exact_repair.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'historical_exact_repair.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','site_details']},indent=2))

if __name__=='__main__':main()
