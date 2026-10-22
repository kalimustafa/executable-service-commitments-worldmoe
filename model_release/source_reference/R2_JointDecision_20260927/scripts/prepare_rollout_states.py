"""Build causal 16-arrival episodes with exact initial reservation/deadline ledgers."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT,LEGACY,PREVIOUS,PAPER
from src.data.solar_nowcast import prepare,forecast
import numpy as np,pandas as pd
from collections import defaultdict

K=16;L=96+K-1;rng=np.random.default_rng(20260920)
with np.load(ROOT/'cache/rollout_forecast_features.npz') as z:
    frame_times=pd.DatetimeIndex(pd.to_datetime(z['timestamps'],utc=True));frame_zones=z['zone'];pue_ref=z['pue_reference'];historical=z['historical_rt']
    frame_dates=z['dates'];labels=z['labels']
lookup={(t.value,int(z)):i for i,(t,z) in enumerate(zip(frame_times,frame_zones))}
with np.load(LEGACY/'cache/development_physical_reference.npz') as z:
    dates=z['dates'];source_arr=z['arrivals'];actual_pv=z['pv'];actual_pue=z['pue']
with np.load(PREVIOUS/'cache/world_physical_forecasts.npz') as z:
    assert z['dates'].tolist()==dates.tolist();source_pv=z['issue_pv'];source_pue=z['issue_pue'];source_issue=pd.to_datetime(z['issue_utc'],utc=True).asi8

for role,allowed,maximum in [('train',[2022,2023],2400),('selection',[2024],320)]:
    times=pd.date_range(f'{min(allowed)}-01-01',f'{max(allowed)+1}-01-01',freq='15min',inclusive='left',tz='UTC');n=len(times)
    arr=np.zeros((n+L,4));prior=np.zeros_like(arr);ppue=np.broadcast_to(pue_ref,arr.shape).copy();pv=np.full_like(arr,np.nan);pue=pv.copy();issues=np.full(n+L,np.iinfo(np.int64).max,dtype=np.int64);valid_day=np.zeros(n+L,bool)
    for d,day in enumerate(dates):
        day=pd.Timestamp(str(day),tz='UTC')
        if day.year not in allowed:continue
        start=int((day-times[0]).total_seconds()/900)
        arr[start:start+96]=source_arr[d];prior[start:start+96]=source_pv[d];ppue[start:start+96]=source_pue[d];pv[start:start+96]=actual_pv[d];pue[start:start+96]=actual_pue[d];issues[start:start+96]=source_issue[d];valid_day[start:start+96]=True
    idx=np.full((n+L,4),-1,np.int64)
    for t,time in enumerate(times):
        for z in range(4):idx[t,z]=lookup.get((time.floor('h').value,z),-1)
    valid_rows=(idx>=0)&valid_day[:,None]
    for z in range(4):
        good=np.flatnonzero(idx[:,z]>=0);valid_rows[good,z]&=np.isin(np.array([int(d[:4]) for d in frame_dates[idx[good,z]]]),allowed)
    bad=np.vstack([np.zeros((1,4),np.int64),np.cumsum(~valid_rows,axis=0)]);complete=(bad[np.arange(n)+L]-bad[np.arange(n)])==0
    candidates=np.argwhere((arr[:n]>1e-4)&complete);weights=np.sqrt(arr[candidates[:,0],candidates[:,1]]);weights=.5/len(weights)+.5*weights/weights.sum()
    selected=set(int(t)*4+int(z) for t,z in candidates[rng.choice(len(candidates),min(maximum,len(candidates)),replace=False,p=weights)])
    local=times.tz_convert('America/New_York').tz_localize(None).normalize()
    pre=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(L,np.iinfo(np.int64).max)]
    post=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(L,np.iinfo(np.int64).max)]
    solar=prepare(times,pv[:n]);book=np.zeros((n+8192,4));events=[defaultdict(list) for _ in range(4)]
    records={k:[] for k in ['row_ids','states','claims','arrivals','outcomes','origins','zone']}
    for t in range(n):
        for z in range(4):
            events[z].pop(t-1,None);amount=float(arr[t,z])
            if amount<1e-10:continue
            if t*4+z in selected:
                ids=t+np.arange(L);claim=np.zeros((L,L),np.float32);legal=True
                for j,slot in enumerate(ids):
                    for due,units in events[z].get(int(slot),[]):
                        if due<t:legal=False
                        elif due<t+L:claim[j,due-t]+=units
                if legal:
                    features=np.zeros((K,96,22),np.float32)
                    for step in range(K):
                        origin=t+step;target=origin+np.arange(96);now=times.asi8[origin];available=pre[target]<=now;pub=post[target]<=now;phys=(issues[target]<=now)&available
                        solar_pv=forecast(origin,target,prior[target],phys,solar,6.)[:,z]
                        clock=times[target].hour.to_numpy()+times[target].minute.to_numpy()/60
                        st=features[step];st[:,2]=solar_pv;st[:,3]=np.where(phys,ppue[target,z]/pue_ref[z],1.);st[:,4]=np.arange(96)/96;st[:,8]=available;st[:,9]=1.;st[:,10]=pub;st[:,11]=1-np.arange(96)/96
                        st[:,17]=float(np.isfinite(pv[max(0,origin-16):origin,z]).sum())/16
                        st[:,18]=np.log1p(np.nan_to_num(pv[origin-1,z])) if origin else 0.;st[:,19]=np.log1p(prior[origin-1,z]) if origin else 0.
                        st[:,20]=np.sin(2*np.pi*clock/24);st[:,21]=np.cos(2*np.pi*clock/24)
                    records['row_ids'].append(idx[ids,z]);records['states'].append(features);records['claims'].append(claim);records['arrivals'].append(arr[t:t+K,z]);records['outcomes'].append(np.column_stack([labels[idx[ids,z],2],pv[ids,z],pue[ids,z]]));records['origins'].append(str(times[t]));records['zone'].append(z)
            ids=t+np.arange(96);pub=post[ids]<=times.asi8[t];phys=(issues[ids]<=times.asi8[t])&(pre[ids]<=times.asi8[t]);valid=idx[ids,z]>=0
            da=np.full(96,historical[z]);da[valid]=labels[idx[ids[valid],z],0];price=np.where(pub,da,historical[z])
            free=np.maximum(1-book[ids,z],0.);delta=np.maximum(np.minimum(amount,free),1e-6)
            coverage=np.where(phys,np.clip((prior[ids,z]-ppue[ids,z]*(.23069976270198822+1.639059066772461*book[ids,z]))/(ppue[ids,z]*1.639059066772461*delta),0,1),0.)
            score=price*(1-coverage)-coverage;score[1:]+=2;targets=ids[np.argsort(score,kind='stable')];capacity=np.maximum(1-book[targets,z],0.);take=np.minimum(capacity,np.maximum(amount-np.r_[0.,np.cumsum(capacity[:-1])],0.));book[targets,z]+=take
            for slot,units in zip(targets[take>1e-10],take[take>1e-10]):events[z][int(slot)].append((t+95,float(units)))
            rest=amount-take.sum()
            if rest>1e-9:
                tail=np.arange(t+96,len(book));capacity=np.maximum(1-book[tail,z],0.);more=np.minimum(capacity,np.maximum(rest-np.r_[0.,np.cumsum(capacity[:-1])],0.));book[tail,z]+=more
                for slot,units in zip(tail[more>1e-10],more[more>1e-10]):events[z][int(slot)].append((t+95,float(units)))
        if t%12000==0:print('ROLLOUT BANK',role,t,'episodes',len(records['zone']),flush=True)
    arrays={k:np.array(v,dtype=np.int64 if k in ['row_ids','zone'] else ('U32' if k=='origins' else np.float32)) for k,v in records.items()}
    np.savez_compressed(ROOT/'cache'/f'rollout_{role}.npz',**arrays)
    print('ROLLOUT COMPLETE',role,len(arrays['zone']),'shape',arrays['states'].shape,flush=True)
(ROOT/'configs/deep_architecture_protocol.json').write_text(json.dumps(dict(primary_seed_only=True,seed_gate_fraction=.8,world_and_original_market_encoder='frozen',trainable_architecture=['market residual adapter and APIC-preserving price shift','market and physical encoders','bidirectional cross-modal attention','two hourly temporal attention layers','workload-query attention and quarter-hour refinement','price/PV cross-moment decoder'],rollout_arrivals=K,candidate_quarters=96,solar_decay_hours=6,training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_year=2024,price_branch_queue_independent=True,gradient='hard feasible forward, soft straight-through gradient, unrolled through sequential reservation states',paper_edits_allowed=False),indent=2))
