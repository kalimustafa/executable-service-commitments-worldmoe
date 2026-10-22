"""Causal reservation-state bank; future observations are training labels only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import *
import numpy as np,pandas as pd,json
from collections import defaultdict
frame=pd.read_parquet(PAPER/'experiments/gate_hourly_panel.parquet').sort_values(['timestamp_utc','zone']).reset_index(drop=True)
frame.timestamp_utc=pd.to_datetime(frame.timestamp_utc,utc=True);zones=['WEST','CAPITL','N.Y.C.','LONGIL']
lookup={(t.value,z):i for i,(t,z) in enumerate(zip(frame.timestamp_utc,frame.zone))}
with np.load(LEGACY/'cache/development_physical_reference.npz') as z:
    dates=z['dates'];source_arr=z['arrivals'].copy();actual_pv=z['pv'].copy();actual_pue=z['pue'].copy()
with np.load(PREVIOUS/'cache/world_physical_forecasts.npz') as z:
    assert z['dates'].tolist()==dates.tolist();source_pv=z['issue_pv'];source_pue=z['issue_pue'];source_issue=pd.to_datetime(z['issue_utc'],utc=True).asi8
years=np.array([int(str(d)[:4]) for d in dates]);pue_ref=np.median(actual_pue[years<=2023],axis=(0,1))
historical=frame[frame.split.eq('train')].groupby('zone').rt_integrated.median().reindex(zones).to_numpy()
rng=np.random.default_rng(20260920);manif={}
for role,allowed,maximum in [('train',[2022,2023],12000),('selection',[2024],2000)]:
    times=pd.date_range(f'{min(allowed)}-01-01',f'{max(allowed)+1}-01-01',freq='15min',inclusive='left',tz='UTC');n=len(times)
    arr=np.zeros((n+96,4));prior=np.zeros((n+96,4));ppue=np.tile(pue_ref,(n+96,1));pv=np.full((n+96,4),np.nan);pue=pv.copy();issues=np.full(n+96,np.iinfo(np.int64).max,dtype=np.int64)
    valid_day=np.zeros(n+96,dtype=bool)
    for d,day in enumerate(dates):
        day=pd.Timestamp(str(day),tz='UTC')
        if day.year not in allowed:continue
        start=int((day-times[0]).total_seconds()/900)
        arr[start:start+96]=source_arr[d];prior[start:start+96]=source_pv[d];ppue[start:start+96]=source_pue[d];pv[start:start+96]=actual_pv[d];pue[start:start+96]=actual_pue[d];issues[start:start+96]=source_issue[d];valid_day[start:start+96]=True
    idx=np.full((n+96,4),-1,dtype=np.int64)
    for t,time in enumerate(times):
        for z,zone in enumerate(zones):idx[t,z]=lookup.get((time.floor('h').value,zone),-1)
    valid_rows=(idx>=0)&valid_day[:,None]
    for z in range(4):
        good=np.flatnonzero(idx[:,z]>=0);valid_rows[good,z]&=frame.iloc[idx[good,z]].date.str[:4].astype(int).isin(allowed).to_numpy()
    rt=np.full((n+96,4),np.nan);da=rt.copy()
    for z in range(4):
        good=idx[:,z]>=0;rt[good,z]=frame.iloc[idx[good,z]].rt_integrated.to_numpy();da[good,z]=frame.iloc[idx[good,z]].da_lbmp.to_numpy()
    local=times.tz_convert('America/New_York').tz_localize(None).normalize()
    pre=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(96,np.iinfo(np.int64).max)]
    post=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(96,np.iinfo(np.int64).max)]
    bad=np.vstack([np.zeros((1,4),dtype=np.int64),np.cumsum(~valid_rows,axis=0)])
    complete_window=(bad[np.arange(n)+96]-bad[np.arange(n)])==0
    candidates=np.argwhere((arr[:n]>1e-4)&complete_window);weights=np.sqrt(arr[candidates[:,0],candidates[:,1]])
    weights=.5/len(weights)+.5*weights/weights.sum();count=min(len(candidates),maximum)
    selected=set(int(t)*4+int(z) for t,z in candidates[rng.choice(len(candidates),count,replace=False,p=weights)])
    book=np.zeros((n+8192,4));events=[defaultdict(list) for _ in range(4)];pending=[defaultdict(float) for _ in range(4)]
    rows=[];states=[];costs=[];publications=[];amounts=[];outcomes=[];origins=[]
    for t in range(n):
        for z in range(4):
            for due,units in events[z].pop(t-1,[]):
                pending[z][due]-=units
                if pending[z][due]<1e-9:pending[z].pop(due,None)
            amount=float(arr[t,z])
            if amount<1e-10:continue
            ids=t+np.arange(96);due=t+95;pending[z][due]+=amount
            available=pre[ids]<=times.asi8[t];pub=post[ids]<=times.asi8[t];physical=(issues[ids]<=times.asi8[t])&available
            booked=book[ids,z].copy();free=np.maximum(1-booked,0.)
            if t*4+z in selected and len(rows)<maximum and valid_rows[ids,z].all() and free.sum()>=amount-1e-8:
                delta=np.minimum(np.maximum(free,1e-5),min(.05,amount))
                bins=np.array([sum(v for deadline,v in pending[z].items() if deadline<=t+h*4-1) for h in [4,8,12,24]])
                st=np.column_stack([booked,delta,prior[ids,z]*physical,np.where(physical,ppue[ids,z]/pue_ref[z],1.),np.arange(96)/96,
                    np.full(96,np.log1p(amount)/4),np.full(96,np.log1p(sum(pending[z].values()))/4),np.full(96,booked.mean()),available,physical,pub,1-np.arange(96)/96,
                    np.broadcast_to(np.log1p(bins)/4,(96,4))]).astype(np.float32)
                inc=pue[ids,z]*1.639059066772461*delta;load=pue[ids,z]*(.23069976270198822+1.639059066772461*booked)
                covered=np.clip((pv[ids,z]-load)/inc,0,1)
                label=pue[ids,z]/pue_ref[z]*((1-covered)*rt[ids,z]-covered)
                rows.append(idx[ids,z]);states.append(st);costs.append(label);publications.append(pub.astype(np.float32));amounts.append(amount);outcomes.append(np.column_stack([rt[ids,z],pv[ids,z],pue[ids,z]]));origins.append(str(times[t]))
            price=np.where(pub,np.nan_to_num(da[ids,z],nan=historical[z]),historical[z])
            delta=np.maximum(np.minimum(amount,free),1e-6);load=ppue[ids,z]*(.23069976270198822+1.639059066772461*booked);inc=ppue[ids,z]*1.639059066772461*delta
            coverage=np.where(physical,np.clip((prior[ids,z]-load)/inc,0,1),0.)
            score=price*(1-coverage)-coverage;score[1:]+=2;order=np.argsort(score,kind='stable');targets=ids[order]
            capacity=np.maximum(1-book[targets,z],0);take=np.minimum(capacity,np.maximum(amount-np.r_[0.,np.cumsum(capacity[:-1])],0));book[targets,z]+=take
            for slot,units in zip(targets[take>1e-12],take[take>1e-12]):events[z][int(slot)].append((due,float(units)))
            rest=amount-take.sum()
            if rest>1e-9:
                extra=np.arange(due+1,len(book));capacity=np.maximum(1-book[extra,z],0);more=np.minimum(capacity,np.maximum(rest-np.r_[0.,np.cumsum(capacity[:-1])],0));book[extra,z]+=more
                for slot,units in zip(extra[more>1e-12],more[more>1e-12]):events[z][int(slot)].append((due,float(units)))
        if t%9600==0:print('STATE BANK',role,t,'groups',len(rows),flush=True)
    np.savez_compressed(ROOT/'cache'/f'decision_{role}.npz',row_ids=np.array(rows,np.int64),states=np.array(states,np.float32),costs=np.array(costs,np.float32),publication=np.array(publications,np.float32),amounts=np.array(amounts,np.float32),outcomes=np.array(outcomes,np.float32),origins=np.array(origins),pue_reference=pue_ref.astype(np.float32))
    manif[role]=dict(groups=len(rows),slots_per_group=96,target_years=allowed,origin_min=min(origins),origin_max=max(origins),reservation_work_conservation_error=float(abs(book.sum()-arr.sum())))
    print(role,manif[role],flush=True)
(ROOT/'cache/decision_data_manifest.json').write_text(json.dumps(dict(groups=manif,price_training_years=[2021,2022,2023],physical_training_years=[2022,2023],state_features=16,state_source='causal published-price behaviour reservations; gaps have no observed arrivals and no supervised target windows',future_realizations='labels only',selection_fitting_forbidden=True,pue_reference=pue_ref.tolist()),indent=2),encoding='utf-8')
