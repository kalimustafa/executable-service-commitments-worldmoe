"""WorldMoE-guided execution with current metered physics and FIFO deadlines.

No reservation reopening or adaptive locking is used. Future realized prices,
PV, PUE and arrivals are excluded from the decision. Current PV/PUE represent
the same execution-time physical feedback used in the electrical projector.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2024);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--safety',type=float,default=.5);p.add_argument('--fractions',type=float,nargs='+',default=[0,.1,.25]);p.add_argument('--physics-lag',type=int,default=1);p.add_argument('--site-cost',action='store_true');a=p.parse_args()
for fraction in a.fractions:
    name=f'O80C_Eagle_{a.year}_s{a.seed}_k{a.safety:g}_f{fraction:g}_lag{a.physics_lag}'+('_site' if a.site_cost else '');saved=RESULTS/f'{name}.json'
    if saved.exists():continue
    arr,ref,shared=configure(a.year);m=observer(a.year,a.seed,arr,24);cum=np.cumsum(arr,axis=0);served=np.zeros(4);book=np.zeros((c.N+8192,4));now_physical_reads=0
    start=time.perf_counter()
    for t in range(c.N):
        releases=np.arange(max(0,t-95),t+1);due=np.minimum(releases+95,c.N-1)
        required=np.maximum(cum[releases]-served-(due-t)[:,None]*a.safety,0).max(axis=0)
        ready=np.maximum(cum[t]-served,0.)
        observed=t-a.physics_lag
        solar=np.clip((c.pv[observed]/c.pue[observed]-c.idle)/c.k,0,1) if observed>=0 else np.zeros(4)
        now_physical_reads+=1
        service=np.minimum(ready,np.maximum(np.minimum(solar,ready),required));service=np.minimum(service,1.)
        for z in range(4):
            if ready[z]<=service[z]+1e-9 or service[z]>=1-1e-9:continue
            oldest=int(np.searchsorted(cum[:t+1,z],served[z]+1e-9))
            end=min(c.N-1,oldest+95)
            ids=np.arange(t,end+1);now=c.times.asi8[t]
            post=m.post_issue[ids]<=now;pre=m.pre_issue[ids]<=now
            price=np.where(post,m.post[ids,z,3],np.where(pre,m.pre[ids,z,3],m.base.historical[z]))
            if a.site_cost:
                physics=np.where(post,m.physical_post[ids],np.where(pre,m.physical_pre[ids],False))
                covered=np.clip((m.pv[ids,z]-m.pue[ids,z]*c.idle[z])/(m.pue[ids,z]*c.k[z]*.05),0,1)
                covered=np.where(physics,covered,0.)
                future=price*(1-covered)-covered
                future[0]=price[0]
                price=future
            if len(price)<=1 or price[0]+2<=np.quantile(price[1:],fraction):service[z]=min(ready[z],1.)
        book[t]=service;served+=service
    t=c.N
    while np.max(cum[-1]-served)>1e-9:
        service=np.minimum(np.maximum(cum[-1]-served,0.),1.);book[t]=service;served+=service;t+=1
        if t>=len(book):raise RuntimeError('Tail overflow')
    row=measure(name,book,ref,arr,shared,dict(experiment='O80C',year=a.year,trace='Eagle',seed=a.seed,window_h=24,role='selection' if a.year==2024 else 'exploratory_frozen_evaluation'),extra=dict(seconds=time.perf_counter()-start,safety_capacity=a.safety,price_rank_fraction=fraction,physical_observation_lag_intervals=a.physics_lag,current_physical_observation=a.physics_lag==0,future_realizations_used=False,price_forecasts='frozen WorldMoE native distribution',adaptive_locking=False,physical_cost_guidance=a.site_cost))
    bound=json.loads((RESULTS/f'I1_Eagle_{a.year}_w24.json').read_text());row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];saved.write_text(json.dumps(row,indent=2));print('CAPTURE',row['capture_lower_fraction'],'CLOSING',row['closing_it_mwh'],flush=True)
