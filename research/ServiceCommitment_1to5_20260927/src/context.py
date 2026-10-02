"""Independent extension of frozen fluid experiments; no discrete-job inputs."""
from pathlib import Path
from functools import lru_cache
import json, os, sys, time
import numpy as np
import pandas as pd
from paths import ROOT, WORKSPACE, ARCHIVE
from vendor import economic_core as c
from vendor import reservation as engine
from vendor.forecast_provider import ForecastProvider, TZ
from src.reservation_state_cost import ParticleReservationRanks

sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','2')
RESULTS=ROOT/'results'
TRAJECTORIES=RESULTS/'trajectories'
TRAJECTORIES.mkdir(exist_ok=True)
LEGACY=WORKSPACE/'新论文_市场可交付灵活性_20260917/revision_20260920_r2_executable'
PREVIOUS=WORKSPACE/'R2_WGR_COST_REDESIGN_20260923'

def configure(year, trace='Eagle'):
    with np.load(WORKSPACE/f'external_validation/data/forcing_{year}.npz',allow_pickle=False) as z:
        c.times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True));c.N=len(c.times)
        for key in ['da','rt','pv','pue','k','idle']:setattr(c,key,z[key].copy())
        c.train_rt=z['historical_rt'].copy();shared=z['common_da_component'].copy()
        arr=z['eagle_arrivals'].copy()
    if trace=='Frontier':
        with np.load(WORKSPACE/f'external_validation/data/frontier_{year}/mapped_arrivals.npz',allow_pickle=False) as z:arr=z['arrivals'].copy()
    c.c=json.loads((ARCHIVE/'configs/economic_constraints.json').read_text())['constraints']
    c.day_index,c.days=pd.factorize(c.times.tz_convert(TZ).date,sort=True)
    c.DATA=TRAJECTORIES;c.ROWS=[];c.DAILY=[];c.TRAJ={}
    return arr,engine.reference(arr,1.),shared

@lru_cache(maxsize=6)
def forecast_frame(year,seed):
    return pd.read_parquet(ARCHIVE/f'results/predictions_{year}/R2_seed{seed}_normal_native.parquet')

def observer(year,seed,arr,window,policy='interval',post_only=False):
    base=ForecastProvider(c.times,c.da,c.train_rt,kind='r2',frame=forecast_frame(year,seed))
    if post_only:
        base.pre=np.broadcast_to(c.train_rt[None,:,None],(c.N,4,7)).copy()
        base.pre_issue=np.full(c.N,np.iinfo(np.int64).max,dtype=np.int64)
    with np.load(LEGACY/'cache/development_physical_reference.npz',allow_pickle=False) as z:
        dates=pd.to_datetime(z['dates']);pue_ref=np.median(z['pue'][dates.year<=2023],axis=(0,1))
    path=ARCHIVE/'cache/final_physical_priors.npz' if year==2025 else PREVIOUS/'cache/world_physical_forecasts.npz'
    with np.load(path,allow_pickle=False) as z:
        lookup={str(d):i for i,d in enumerate(z['dates'])}
        local_days=c.times.tz_convert(TZ).strftime('%Y-%m-%d')
        ix=np.array([lookup.get(d,-1) for d in local_days]);valid=ix>=0
        slot=c.times.hour.to_numpy()*4+c.times.minute.to_numpy()//15
        pv=z['issue_pv'][ix.clip(0),slot];pue=z['issue_pue'][ix.clip(0),slot]
        issued_arr=pd.DatetimeIndex(pd.to_datetime(z['issue_utc'],utc=True)).asi8
        issued=issued_arr[ix.clip(0)];issued_map=dict(zip(z['dates'].tolist(),issued_arr))
    bundles=WORKSPACE/'power-dc-vpp/artifacts/dart_value_select/counterfactual_inputs/v1'/('locked_test' if year==2025 else 'selection')/'daily_bundles'
    holder={}
    @lru_cache(maxsize=6)
    def load_prior(day):
        record=json.loads((bundles/f'{day}.record.json').read_text())['record']
        if record['causal_gates']['pv'] is not True:raise RuntimeError('PV information gate')
        with np.load(bundles/f'{day}.npz',allow_pickle=False) as z:return z['pv_pool'].transpose(2,1,0).astype(float)
    def prior(day):
        if issued_map[day]>c.times.asi8[holder['model'].current_decision_index]:raise RuntimeError('Future PV prior')
        return load_prior(day)
    def prefix(day,now,zone,mode):
        start=int((pd.Timestamp(day,tz='UTC')-c.times[0]).total_seconds()/900)
        end=min(start+96,now,c.N);begin=max(start,0,now-16)
        if end<=begin:return np.empty(0,dtype=int),np.empty(0,dtype=float)
        return np.arange(begin,end)-start,c.pv[begin:end,zone].copy()
    physical_pre=valid&(issued<=base.pre_issue)
    if post_only:physical_pre=np.zeros(c.N,dtype=bool)
    fitted=json.loads((ARCHIVE/'configs/pv_particle_state_fitted.json').read_text())
    model=ParticleReservationRanks(base,arr,window,'robust' if policy=='interval' else 'point',
        pv=pv,pue=pue,physical_pre=physical_pre,physical_post=valid&(issued<=base.post_issue),
        idle=c.idle,k=c.k,capacity=1.,pue_reference=pue_ref,
        source_days=local_days.to_numpy(dtype='U10'),quarter_slots=slot,
        prior_paths=prior,observed_prefix=prefix,sigma=fitted['sigma'],context='trailing_four_hours')
    holder['model']=model
    return model

def cached_book(year,trace,seed,window):
    proxy='registered' if trace=='Eagle' else 'requested_nodes'
    name=f'{trace}_{year}_{proxy}_seed{seed}_{window}h_interval_particle_trailing4_normal'
    p=ARCHIVE/'results/economic_trajectories'/f'{name}_complete_book.npz'
    if p.exists():
        with np.load(p,allow_pickle=False) as z:return z['compute'].copy()
    return None

def persistent(arr,year,seed,window,post_only=False):
    model=observer(year,seed,arr,window,post_only=post_only)
    engine.rank=lambda w,p:model
    book,stats,_=engine.schedule(arr,[(1.,window)],'robust')
    if np.max(np.abs(model.book-book))>1e-9:raise RuntimeError('Reservation state mismatch')
    return book

def fifo_lateness(book,arr,window):
    cumulative=np.cumsum(book,axis=0);released=np.cumsum(arr,axis=0)
    previous=released-arr;due=np.arange(c.N)+int(window*4)-1
    late=np.maximum(released-np.maximum(previous,cumulative[due]),0.)
    return float(c.DT*np.sum(late*c.k))

def measure(name,book,reference,arr,shared,meta,late=None,extra=None):
    if abs(book.sum()-arr.sum())>1e-5:raise RuntimeError('Unserved work disappeared')
    if np.max(np.cumsum(book[:c.N],axis=0)-np.cumsum(arr,axis=0))>1e-7:raise RuntimeError('Service before release')
    if late is None:late=fifo_lateness(book,arr,meta['window_h'])
    c.ROWS=[];c.DAILY=[];c.TRAJ={}
    row=c.record(name,'extension',book,reference,meta,late=late)
    refcost=float(c.cash(reference)[0].sum()+shared.sum()+c.DT*np.sum(reference[c.N:]*c.k*max(c.c['pue_max'])*c.train_rt))
    _,grid=c.cash(book);_,refgrid=c.cash(reference)
    row.update(adjusted_cost_usd=refcost-row['value_usd'],reference_cost_usd=refcost,
        saving_usd=row['value_usd'],saving_pct=100*row['value_usd']/refcost,
        cvar95_usd=row['mtm_daily_loss_cvar95_usd'],closing_it_mwh=row['final_queue_IT_mwh'],
        max_fleet_recovery_mw=float(np.maximum(grid-refgrid,0).sum(1).max()),
        max_upward_site_ramp_mw=float(np.maximum(np.diff(grid,axis=0),0).max()),
        max_fleet_grid_mw=float(grid.sum(1).max()),
        pv_curtailment_mwh=float(c.DT*np.maximum(c.pv-c.pue*(c.idle+c.k*book[:c.N]),0).sum()),
        common_da_component_usd=float(shared.sum()))
    if extra:row.update(extra)
    np.savez_compressed(TRAJECTORIES/f'{name}_complete_book.npz',compute=book,reference=reference)
    pd.DataFrame(c.DAILY).to_csv(TRAJECTORIES/f'{name}_daily.csv',index=False)
    (RESULTS/f'{name}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    c.DAILY=[];c.TRAJ={}
    print(name,'saving',round(row['saving_usd'],2),'late',round(late,6),flush=True)
    return row

def run_persistent(year,trace,seed,window,post_only=False):
    name=f'I2_{trace}_{year}_s{seed}_w{window}_'+('post_only' if post_only else 'full')
    saved=RESULTS/f'{name}.json'
    if saved.exists():return json.loads(saved.read_text())
    arr,reference,shared=configure(year,trace);start=time.perf_counter()
    book=None if post_only else cached_book(year,trace,seed,window)
    if book is None:book=persistent(arr,year,seed,window,post_only)
    return measure(name,book,reference,arr,shared,dict(experiment='I2',year=year,trace=trace,seed=seed,window_h=window,policy='interval',information='post_only' if post_only else 'full'),extra={'seconds':time.perf_counter()-start})
