"""Same measured physical forcing and frozen accounting for manuscript tests."""
from src.runtime import ROOT,WORKSPACE,LEGACY,PREVIOUS
from src.data.access_guard import enable_final_evaluation,install_guard
enable_final_evaluation();install_guard()
import json,time
import numpy as np
import pandas as pd
from vendor import economic_core as core
from vendor import reservation as engine
from vendor.forecast_provider import ForecastProvider,TZ
from src.candidate_runtime import build_candidate

def configure(year,trace='Eagle',proxy='registered'):
    folder=WORKSPACE/'external_validation/data'
    with np.load(folder/f'forcing_{year}.npz',allow_pickle=False) as z:
        core.times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True));core.N=len(core.times)
        for k in ['da','rt','pv','pue','k','idle']:setattr(core,k,z[k].copy())
        core.train_rt=z['historical_rt'].copy();shared=z['common_da_component'].copy()
        arr=z['eagle_arrivals'].copy()
    if trace=='Frontier':
        with np.load(folder/f'frontier_{year}/mapped_arrivals.npz',allow_pickle=False) as z:
            arr=z['resource_field7_arrivals' if proxy=='field7_sensitivity' else 'arrivals'].copy()
    core.c=json.loads((ROOT/'configs/economic_constraints.json').read_text())['constraints']
    core.day_index,core.days=pd.factorize(core.times.tz_convert(TZ).date,sort=True)
    core.ROWS=[];core.DAILY=[];core.TRAJ={}
    assert core.N==35040 if year==2025 else core.N==35136
    assert (np.diff(core.times.asi8)==900*10**9).all()
    return arr,engine.reference(arr,1.),shared

def physical_kwargs(year,base,capacity=1.):
    with np.load(LEGACY/'cache/development_physical_reference.npz',allow_pickle=False) as source:
        dates=pd.to_datetime(source['dates']);pue_ref=np.median(source['pue'][dates.year<=2023],axis=(0,1))
    path=ROOT/'cache/final_physical_priors.npz' if year==2025 else PREVIOUS/'cache/world_physical_forecasts.npz'
    with np.load(path,allow_pickle=False) as z:
        lookup={str(d):i for i,d in enumerate(z['dates'])}
        index=np.array([lookup.get(d,-1) for d in core.times.tz_convert(TZ).strftime('%Y-%m-%d')]);valid=index>=0
        slot=core.times.hour.to_numpy()*4+core.times.minute.to_numpy()//15
        pv=z['issue_pv'][index.clip(0),slot];pue=z['issue_pue'][index.clip(0),slot]
        issued=pd.DatetimeIndex(pd.to_datetime(z['issue_utc'],utc=True)).asi8[index.clip(0)]
    return dict(pv=pv,pue=pue,physical_pre=valid&(issued<=base.pre_issue),physical_post=valid&(issued<=base.post_issue),
                idle=core.idle,k=core.k,capacity=capacity,pue_reference=pue_ref)

def forecast_file(year,seed=20260920,ablation='normal'):
    return ROOT/f'results/predictions_{year}'/f'R2_seed{seed}_{ablation}_native.parquet'

def run_homogeneous(arr,reference,shared,*,year,trace,proxy,seed,window,policy,physical='particle_trailing4',ablation='normal'):
    name=f'{trace}_{year}_{proxy}_seed{seed}_{window}h_{policy}_{physical}_{ablation}'
    saved=ROOT/'results/paper_runs'/f'{name}.json';saved.parent.mkdir(exist_ok=True)
    if saved.exists():return json.loads(saved.read_text())
    frame=pd.read_parquet(forecast_file(year,seed,ablation))
    base=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame)
    meta=dict(year=year,trace=trace,proxy=proxy,seed=seed,window_h=window,policy=policy,physical_branch=physical,
              ablation=ablation,model='R2',market_branch='C1_ProfileResidual')
    observer=None
    if physical=='price_only':
        engine.rank=lambda w,p:base.ranked_slots(w,p)
    else:
        row=dict(name=name,**meta,price_forecast_source=str(forecast_file(year,seed,ablation)))
        observer,information,_=build_candidate(row,arr,physical_kwargs(year,base))
        engine.rank=lambda w,p:observer
    began=time.perf_counter();book,stats,events=engine.schedule(arr,[(1.,window)],'robust' if policy=='interval' else 'point',keep_events=True)
    if observer is not None:assert np.max(np.abs(book-observer.book))<1e-10
    return measure(name,book,reference,arr,shared,meta,stats,events,began,saved)

def measure(name,book,reference,arr,shared,meta,stats,events,began,saved):
    event=pd.DataFrame(events,columns=['release','deadline','execution','zone','work_units','window_h'])
    weight=core.DT*event.work_units.to_numpy()*core.k[event.zone.to_numpy(dtype=int)]
    strict_late=float(weight[event.execution.to_numpy()>event.deadline.to_numpy()].sum())
    assert (event.execution>=event.release).all() and abs(book.sum()-arr.sum())<1e-5
    row=core.record(name,'paper_final',book,reference,meta,late=strict_late)
    row.update(stats);row['late_IT_mwh']=strict_late
    refcost=float(core.cash(reference)[0].sum()+shared.sum()+core.DT*np.sum(reference[core.N:]*core.k*max(core.c['pue_max'])*core.train_rt))
    row.update(net_operating_cost_usd=refcost-row['value_usd'],reference_cost_usd=refcost,
        rt_normalized_cost_usd=refcost-shared.sum()-row['value_usd'],shared_da_component_usd=float(shared.sum()),
        saving_pct=100*row['value_usd']/refcost,strict_late_all_cohorts_IT_mwh=strict_late,
        requested_IT_mwh=float(core.DT*np.sum(arr*core.k)),seconds=time.perf_counter()-began)
    event.to_parquet(ROOT/'results/economic_trajectories'/f'{name}_assignments.parquet',index=False)
    np.savez_compressed(ROOT/'results/economic_trajectories'/f'{name}_complete_book.npz',compute=book,reference=reference)
    pd.DataFrame([r for r in core.DAILY if r['name']==name]).to_csv(ROOT/'results/economic_trajectories'/f'{name}_daily.csv',index=False)
    core.DAILY=[];core.TRAJ.pop(name,None)
    saved.write_text(json.dumps(row,indent=2))
    print(name,'COST',round(row['net_operating_cost_usd'],2),'VALUE',round(row['value_usd'],2),'LATE',round(strict_late,6),flush=True)
    return row
