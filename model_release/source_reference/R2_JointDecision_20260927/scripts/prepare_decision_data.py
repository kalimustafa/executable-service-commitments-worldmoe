"""Training-state augmentation; all future outcomes are labels, not features."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import *
import numpy as np
import pandas as pd
import json

frame=pd.read_parquet(PAPER/'experiments/gate_hourly_panel.parquet').sort_values(['timestamp_utc','zone']).reset_index(drop=True)
frame.timestamp_utc=pd.to_datetime(frame.timestamp_utc,utc=True)
zones=['WEST','CAPITL','N.Y.C.','LONGIL'];lookup={(t.value,z):i for i,(t,z) in enumerate(zip(frame.timestamp_utc,frame.zone))}
with np.load(LEGACY/'cache/development_physical_reference.npz') as z:
    dates=z['dates'];arr=z['arrivals'].copy();pv=z['pv'].copy();pue=z['pue'].copy()
with np.load(PREVIOUS/'cache/world_physical_forecasts.npz') as z:
    assert z['dates'].tolist()==dates.tolist();prior_pv=z['issue_pv'];prior_pue=z['issue_pue'];issued=pd.to_datetime(z['issue_utc'],utc=True).asi8
day_lookup={str(d):i for i,d in enumerate(dates)}
years=np.array([int(str(d)[:4]) for d in dates]);pue_ref=np.median(pue[years<=2023],axis=(0,1))
queue=np.zeros_like(arr);state=np.zeros(4);previous=None
for d,day in enumerate(dates):
    date=pd.Timestamp(day)
    if previous is None or date-previous!=pd.Timedelta(days=1):state[:]=0.
    for t in range(96):queue[d,t]=state;state=np.maximum(state+arr[d,t]-1,0)
    previous=date
offsets=np.array([0,1,4,8,12,16,24,32,40,48,56,64,72,80,88,95]);rng=np.random.default_rng(20260920)
written={}
for role,allowed,maximum in [('train',[2022,2023],40000),('selection',[2024],4000)]:
    choices=np.argwhere((arr>1e-6)&np.isin(years,allowed)[:,None,None]);order=rng.permutation(len(choices))
    rows=[];states=[];costs=[];publications=[];amounts=[];origin_dates=[]
    for index in order:
        d,t,z=map(int,choices[index]);origin=pd.Timestamp(str(dates[d]),tz='UTC')+pd.Timedelta(minutes=t*15)
        targets=origin+pd.to_timedelta(offsets*15,unit='min');target_dates=targets.strftime('%Y-%m-%d')
        di=np.array([day_lookup.get(x,-1) for x in target_dates]);slot=targets.hour.to_numpy()*4+targets.minute.to_numpy()//15
        if (di<0).any() or not np.isin(years[di],allowed).all():continue
        ids=np.array([lookup.get((target.floor('h').value,zones[z]),-1) for target in targets])
        if (ids<0).any() or not frame.iloc[ids].date.str[:4].astype(int).isin(allowed).all():continue
        local=targets.tz_convert('America/New_York').tz_localize(None).normalize()
        pre=(local-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize('America/New_York').tz_convert('UTC').asi8
        post=(local-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize('America/New_York').tz_convert('UTC').asi8
        available=pre<=origin.value;pub=post<=origin.value;physical=(issued[di]<=origin.value)&available
        # Booked-capacity augmentation is explicit, not claimed to be measured
        # policy reservations. Evaluation uses the actual conserved book.
        booked=rng.choice(np.array([0,.05,.1,.2,.4,.6,.8,.95,1.]),size=len(ids),p=[.2,.1,.1,.15,.15,.1,.08,.08,.04])
        quantity=min(.05,float(arr[d,t,z]));delta=np.full(len(ids),max(quantity,1e-5))
        st=np.column_stack([booked,delta,prior_pv[di,slot,z]*physical,
            np.where(physical,prior_pue[di,slot,z]/pue_ref[z],1.),offsets/96,
            np.full(len(ids),np.log1p(arr[d,t,z])/4),np.full(len(ids),np.log1p(queue[d,t,z])/4),np.full(len(ids),booked.mean()),
            available,physical,pub,1-offsets/96]).astype(np.float32)
        inc=pue[di,slot,z]*1.639059066772461*delta
        load=pue[di,slot,z]*(.23069976270198822+1.639059066772461*booked)
        coverage=np.clip((pv[di,slot,z]-load)/inc,0,1)
        actual_rt=frame.iloc[ids].rt_integrated.to_numpy()
        label=pue[di,slot,z]/pue_ref[z]*((1-coverage)*actual_rt-coverage)
        if not np.isfinite(label).all():continue
        rows.append(ids);states.append(st);costs.append(label);publications.append(pub.astype(np.float32));amounts.append(quantity);origin_dates.append(str(origin))
        if len(rows)>=maximum:break
    np.savez_compressed(ROOT/'cache'/f'decision_{role}.npz',row_ids=np.array(rows,np.int64),states=np.array(states,np.float32),costs=np.array(costs,np.float32),publication=np.array(publications,np.float32),amounts=np.array(amounts,np.float32),origins=np.array(origin_dates),pue_reference=pue_ref.astype(np.float32))
    written[role]=dict(groups=len(rows),slots_per_group=len(offsets),allowed_target_years=allowed,origin_min=min(origin_dates),origin_max=max(origin_dates))
    print(role,written[role],flush=True)
manifest=dict(groups=written,price_training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_fitting_forbidden=True,state_augmentation='sampled admissible booked capacities plus causal earliest-service queue',oracle_values='realized incremental cost labels only',forecast_and_physical_availability_masked=True,pue_reference=pue_ref.tolist(),offsets=offsets.tolist())
(ROOT/'cache/decision_data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
