"""Reconstruct a declared R2 candidate with explicit causal observation APIs."""
from src.runtime import ROOT,WORKSPACE,REPO
from src.data.access_guard import read_development_parquet
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from vendor.forecast_provider import ForecastProvider,TZ,TAU,ZONES
from vendor import economic_core as core
from src.reservation_state_cost import ReservationStateRanks,NowcastReservationRanks,ParticleReservationRanks,DistributionalReservationRanks

PREVIOUS=WORKSPACE/'R2_WGR_COST_REDESIGN_20260923'
FAILED=WORKSPACE/'R2_WGR_DEEPFUSION_20260923'

def point_mean(q):
    ext=np.c_[2*q[:,0]-q[:,1],q,2*q[:,-1]-q[:,-2]];tau=np.r_[0,np.asarray(TAU),1]
    return np.sum(.5*(ext[:,1:]+ext[:,:-1])*np.diff(tau)[None],axis=1)

def build_candidate(row,arrivals,kwargs,observations=None):
    market=row.get('market_branch')
    if not isinstance(market,str):market='C0_Repaired'
    seed=int(row.get('seed',20260920))
    branch=FAILED if market.startswith('A') else PREVIOUS if seed==20260920 else ROOT
    source=branch/'results/predictions_2024'/f'{market}_seed{seed}_native.parquet'
    if row.get('price_forecast_source'):source=Path(row['price_forecast_source'])
    year=int(row.get('year',2024))
    columns=['date','timestamp_utc','zone','gate']+[f'{head}_q{tau:.2f}' for head in ['da','rt','dart'] for tau in TAU]
    frame=read_development_parquet(source,columns=columns)
    base=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame)
    model=str(row['model']);physical=row.get('physical_branch');name=str(row['name'])
    if not isinstance(physical,str):
        physical='particle_trailing4' if 'trailing_four_hours' in model else 'particle' if 'particle' in model else 'nowcast' if 'nowcast' in model else 'distributional' if 'distributional' in model else 'reservation'
    context='trailing_four_hours' if physical=='particle_trailing4' or 'trailing4' in name else 'completed_day_prefix'
    point_type=row.get('point_statistic');mean=point_type in ['mean','analytic_conditional_mean'] or name.startswith('mean_head_')
    window=int(row['window_h']);policy='robust' if row['policy']=='interval' else 'point'
    data=core.pv if observations is None else observations
    audit=dict(observation_calls=0,prior_calls=0,all_observations_precede_decision=True,all_priors_issued_before_use=True)
    holder={};folder=REPO/'artifacts/dart_value_select/counterfactual_inputs/v1'/('locked_test' if year==2025 else 'selection')/'daily_bundles'
    physical_source=ROOT/'cache/final_physical_priors.npz' if year==2025 else PREVIOUS/'cache/world_physical_forecasts.npz'
    with np.load(physical_source,allow_pickle=False) as cache:
        issued=dict(zip(cache['dates'].tolist(),pd.DatetimeIndex(pd.to_datetime(cache['issue_utc'],utc=True)).asi8))
    @lru_cache(maxsize=6)
    def load_prior(day):
        record=json.loads((folder/(day+'.record.json')).read_text())['record']
        assert record['causal_gates']['pv'] is True
        with np.load(folder/(day+'.npz'),allow_pickle=False) as source:return source['pv_pool'].transpose(2,1,0).astype(float)
    def prior(day):
        now=holder['observer'].current_decision_index
        assert issued[day]<=core.times.asi8[now]
        audit['prior_calls']+=1
        return load_prior(day)
    def prefix(day,now,z,mode):
        if physical=='particle_prior':return np.empty(0,int),np.empty(0,float)
        start=int((pd.Timestamp(day,tz='UTC')-core.times[0]).total_seconds()/900)
        end=min(start+96,now,core.N);begin=max(start,0)
        if mode=='trailing_four_hours':begin=max(begin,now-16)
        if end<=begin:return np.empty(0,int),np.empty(0,float)
        assert end<=now;audit['observation_calls']+=1
        return np.arange(begin,end)-start,data[begin:end,z].copy()
    def last(now,z):
        assert now>0 and core.times[now-1]+pd.Timedelta(minutes=15)<=core.times[now]
        audit['observation_calls']+=1;return float(data[now-1,z])
    parameters=dict(kwargs)
    if 'train_calibrated_point' in model:
        with np.load(ROOT/'cache/issued_PV_distribution.npz',allow_pickle=False) as cache:
            parameters['pv']=np.clip(parameters['pv']*cache['scale_train'],0,cache['capacity_train'])
    if physical.startswith('particle'):
        fitted=json.loads((ROOT/'configs/pv_particle_state_fitted.json').read_text())
        observer=ParticleReservationRanks(base,arrivals,window,policy,source_days=core.times.tz_convert(TZ).strftime('%Y-%m-%d').to_numpy(dtype='U10'),
            quarter_slots=core.times.hour.to_numpy()*4+core.times.minute.to_numpy()//15,
            prior_paths=prior,observed_prefix=prefix,sigma=fitted['sigma'],context=context,**parameters)
    elif physical=='nowcast':
        fitted=json.loads((ROOT/'configs/pv_error_dynamics_fitted.json').read_text())
        gains=[[1,0]]*4 if 'fixed_fast' in model else fitted['coefficients']
        observer=NowcastReservationRanks(base,arrivals,window,policy,previous_PV_observation=last,
            coefficients=gains,pv_capacity=fitted['capacity'],**parameters)
    elif physical=='distributional':
        with np.load(ROOT/'cache/issued_PV_distribution.npz',allow_pickle=False) as cache:
            lookup={str(d):i for i,d in enumerate(cache['dates'])};ix=np.array([lookup.get(d,0) for d in core.times.tz_convert(TZ).strftime('%Y-%m-%d')])
            slot=core.times.hour.to_numpy()*4+core.times.minute.to_numpy()//15
            quant=cache['quantiles'][ix,slot];levels=cache['levels']
        observer=DistributionalReservationRanks(base,arrivals,window,policy,pv_quantiles=quant,pv_levels=levels,**parameters)
    else:observer=ReservationStateRanks(base,arrivals,window,policy,**parameters)
    holder['observer']=observer
    if mean:
        index=pd.MultiIndex.from_product([core.times.floor('h'),ZONES],names=['timestamp_utc','zone']);means={}
        for gate in ['pre_DA','post_DA']:
            f=frame[frame.gate.eq(gate)].set_index(['timestamp_utc','zone']).reindex(index)
            means[gate]=(point_mean(f[[f'da_q{q:.2f}' for q in TAU]].to_numpy())-point_mean(f[[f'dart_q{q:.2f}' for q in TAU]].to_numpy())).reshape(-1,4)
        observer.set_conditional_mean_point_head(means['pre_DA'],means['post_DA'])
    description=dict(market_branch=market,physical_branch=physical,observation_context=context,
        output_statistic='conditional_mean' if mean else 'interval' if policy=='robust' else 'median',
        price_forecast_source=str(source),window_h=window,seed=seed,cost_output_precision='FP32')
    return observer,audit,description
