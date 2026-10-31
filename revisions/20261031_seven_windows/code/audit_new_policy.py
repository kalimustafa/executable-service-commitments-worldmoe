"""Verify newly evaluated books once, preserving raw outputs and saving exact repairs.

Requires Python and NumPy. Runs on complete run manifests only and resumes by
source-book SHA256. The original reported cash account and the independent
exact service witness are kept distinct.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, csv, gzip, hashlib, importlib.util, json, shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit';OUT.mkdir(exist_ok=True)
INPUT=ROOT/'inputs';INPUT.mkdir(exist_ok=True)
WORKSPACE=ROOT.parent
PRIOR=WORKSPACE/'EnergyEconomics_Revision_20261031/historical_extension'
for name in ['forcing_2025.npz','economic_constraints.json']:
    if not (INPUT/name).exists():shutil.copy2(PRIOR/'inputs'/name,INPUT/name)
KERNEL=Path(__file__).with_name('exact_service_kernel.py')
if not KERNEL.exists():shutil.copy2(PRIOR/'code/exact_repair_historical_books.py',KERNEL)
spec=importlib.util.spec_from_file_location('exact_service_kernel',KERNEL)
kern=importlib.util.module_from_spec(spec);spec.loader.exec_module(kern)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(v):return F.from_float(float(v))
def ftext(v):return f'{v.numerator}/{v.denominator}'

def audit(meta_path,forcing,reference,config):
    j=json.loads(meta_path.read_text(encoding='utf-8'));seed=j['seed'];w=j['window_h'];name=meta_path.stem
    bookpath=ROOT/'results/trajectories'/f'{name}_complete_book.npz'
    bh=sha(bookpath);dest=OUT/f'policy_s{seed}_w{w}.json'
    if dest.exists() and json.loads(dest.read_text())['source_book_sha256']==bh:return
    z=np.load(bookpath,allow_pickle=False);n=len(forcing['times']);raw=z['compute'];ref=z['reference']
    assert np.array_equal(ref,reference)
    assert not raw[n:].any() and not ref[n:].any(), 'Nonzero tail requires explicit extended-account verification'
    assert raw.min()>=-1e-8 and raw.max()<=1+1e-8
    Afloat=np.cumsum(forcing['eagle_arrivals'],axis=0);Xfloat=np.cumsum(raw,axis=0)
    due=np.arange(n)+4*w-1
    deficit=np.maximum(Afloat-Xfloat[due],0.)
    late=np.maximum(Afloat-np.maximum(Afloat-forcing['eagle_arrivals'],Xfloat[due]),0.)
    late_mwh=float(.25*np.sum(late*forcing['k']))
    if deficit.max()>1e-8:
        # A substantive service failure is retained as an outcome, never repaired
        # into a claimed operating-policy success after looking at the result.
        load0=forcing['pue']*(forcing['idle']+forcing['k']*ref[:n]);g0=np.maximum(load0-forcing['pv'],0.);v0=np.maximum(forcing['pv']-load0,0.)
        load=forcing['pue']*(forcing['idle']+forcing['k']*raw[:n]);grid=np.maximum(load-forcing['pv'],0.);spill=np.maximum(forcing['pv']-load,0.)
        value=.25*(forcing['rt']*(g0-grid)+v0-spill-np.abs(grid-g0))
        saving=float(value.sum());assert abs(saving-j['saving_usd'])<1e-6
        assert abs(late_mwh-j['late_IT_mwh'])<1e-8
        np.savez_compressed(OUT/f'service_failure_s{seed}_w{w}.npz',timestamp_utc=forcing['times'],
            original_deadline_index=due,cohort_late_work=late,deadline_deficit_work=deficit,saving_usd=value)
        result={'run_name':name,'seed':seed,'window_h':w,'source_book_sha256':bh,'source_result_sha256':sha(meta_path),
            'forcing_sha256':sha(INPUT/'forcing_2025.npz'),'checkpoint_sha256':j['forecast_checkpoint_sha256'],
            'evidence_type':'new frozen-checkpoint evaluation with substantive original-deadline failure',
            'recorded_raw_saving_usd':j['saving_usd'],'independently_resettled_cash_saving_usd':saving,
            'late_IT_mwh':late_mwh,'max_deadline_deficit_work':float(deficit.max()),
            'service_feasible':False,'all_exact_checks':False,'repair_applied':False,
            'terminal_policy_and_reference_cost_difference_exact':'0/1',
            'eligible_as_feasible_value_or_capture':False,'repair_cost_bound_usd':None}
        dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('RETAINED SERVICE FAILURE',seed,w,'cash saving',saving,'late IT MWh',late_mwh,flush=True)
        return
    exact_raw=F(0);exact_fixed=F(0);bound=F(0);l1=F(0);sites=[];saved=[]
    for site in range(4):
        rr={key:kern.ratios(forcing[key][:,site]) for key in ['eagle_arrivals','rt','pv','pue']}
        rr['raw']=kern.ratios(raw[:n,site]);rr['ref']=kern.ratios(ref[:n,site])
        rr['constants']=kern.ratios([forcing['k'][site],forcing['idle'][site],forcing['grid_limit'][site]])
        Q=max(d for v in rr.values() for _,d in v);v={key:kern.encode(r,Q) for key,r in rr.items()}
        k,idle,limit=v['constants'];A=[0];X=[0]
        for x in v['eagle_arrivals']:A.append(A[-1]+x)
        for x in v['raw']:X.append(X[-1]+x)
        excess=max(0,max(x-a for x,a in zip(X,A)))
        deficit=max(0,max(A[t+1]-X[min(n,t+4*w)] for t in range(n)))
        mass=abs(A[-1]-X[-1])
        assert max(excess,deficit,mass)/Q<1e-8, 'Substantive raw service violation; do not conceal by numerical repair'
        x,delta=kern.repair(v['raw'],v['eagle_arrivals'],Q,4*w)
        rs=fs=mx=0;Q2=Q*Q;Q3=Q2*Q;Q4=Q3*Q
        for t in range(n):
            p,rt,pv=v['pue'][t],v['rt'][t],v['pv'][t]
            assert p*(idle*Q+k*Q)<=(limit+pv)*Q2
            load0=p*(idle*Q+k*v['ref'][t]);pv3=pv*Q2
            g0=max(load0-pv3,0);s0=max(pv3-load0,0)
            for xx,which in [(v['raw'][t],0),(x[t],1)]:
                load=p*(idle*Q+k*xx);g=max(load-pv3,0);spill=max(pv3-load,0)
                val=rt*(g0-g)+(s0-spill-abs(g-g0))*Q
                if which==0:rs+=val
                else:fs+=val
            mx=max(mx,k*p*(abs(rt)+2*Q))
        rvalue=F(rs,4*Q4);fvalue=F(fs,4*Q4);b=F(mx*delta,4*Q4)
        assert abs(fvalue-rvalue)<=b and b<F(1,10000)
        assert sum(v['ref'])==A[-1], 'Reference anchor offset must be included if nonzero'
        exact_raw+=rvalue;exact_fixed+=fvalue;bound+=b;l1+=F(delta,Q)
        sites.append({'site':site,'raw_release_excess_exact':ftext(F(excess,Q)),
            'raw_deadline_deficit_exact':ftext(F(deficit,Q)),'raw_mass_error_exact':ftext(F(mass,Q)),
            'repair_l1_exact':ftext(F(delta,Q)),'repair_cost_bound_exact':ftext(b),
            'all_exact_release_deadline_capacity_import_terminal_checks':True})
        saved.append({'site':site,'denominator':str(Q),'service_numerators':[str(i) for i in x]})
    assert abs(float(exact_raw)-j['saving_usd'])<1e-6
    repaired=OUT/f'exact_policy_s{seed}_w{w}.json.gz'
    with gzip.open(repaired,'wt',encoding='utf-8') as g:json.dump({'run_name':name,'source_book_sha256':bh,
        'window_h':w,'intervals':n,'terminal_service_exact':'0/1','sites':saved},g,separators=(',',':'))
    result={'run_name':name,'seed':seed,'window_h':w,'source_book_sha256':bh,'source_result_sha256':sha(meta_path),
        'forcing_sha256':sha(INPUT/'forcing_2025.npz'),'checkpoint_sha256':j['forecast_checkpoint_sha256'],
        'evidence_type':'new 2026-10-31 frozen-checkpoint evaluation with separate exact numerical witness',
        'recorded_raw_saving_usd':j['saving_usd'],'exact_raw_saving':ftext(exact_raw),
        'exact_repaired_saving':ftext(exact_fixed),'repaired_saving_enclosure':kern.decimal_bounds(exact_fixed),
        'repair_l1_work_exact':ftext(l1),'repair_cost_bound_exact':ftext(bound),'repair_cost_bound_usd':float(bound),
        'terminal_policy_and_reference_cost_difference_exact':'0/1','all_exact_checks':True,
        'repaired_witness_sha256':sha(repaired),'sites':sites,'service_feasible':True,'late_IT_mwh':late_mwh}
    dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('AUDITED',seed,w,'saving',float(exact_fixed),'repair_cost_bound',float(bound),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--window',type=int);args=p.parse_args()
    f=np.load(INPUT/'forcing_2025.npz',allow_pickle=False);a=f['eagle_arrivals'];n=len(a);ref=np.zeros((n+8192,4));qq=np.zeros(4)
    for t in range(n):qq+=a[t];ref[t]=np.minimum(qq,1.);qq-=ref[t]
    assert not qq.any()
    config=json.loads((INPUT/'economic_constraints.json').read_text())['constraints']
    for mp in sorted((ROOT/'results').glob('*_run_manifest.json')):
        manifest=json.loads(mp.read_text(encoding='utf-8'))
        if manifest['status']!='complete' or (args.window and manifest['window_h']!=args.window):continue
        jp=mp.with_name(mp.name.replace('_run_manifest',''))
        audit(jp,f,ref,config)
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('policy_s*_w*.json'))]
    if rows:
        fields=['seed','window_h','recorded_raw_saving_usd','late_IT_mwh','repair_cost_bound_usd','all_exact_checks','checkpoint_sha256']
        with (OUT/'audit_summary.csv').open('w',newline='',encoding='utf-8') as g:
            wr=csv.DictWriter(g,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    print('Completed audit records',len(rows),flush=True)
if __name__=='__main__':main()
