"""Causal multi-horizon physical features; future realized PV is labels only."""
import numpy as np
import pandas as pd
from src.runtime import ROOT,LEGACY,PREVIOUS,ARCHIVE,WORKSPACE
from src.data.solar_nowcast import prepare
from src.data.physical_observer_data import observer_inputs


def build(role):
    training=role=='train'
    if training:
        times=pd.date_range('2022-01-01','2024-01-01',freq='15min',inclusive='left',tz='UTC')
        obs=np.full((len(times),4),np.nan,np.float32)
        with np.load(LEGACY/'cache/development_physical_reference.npz') as z:
            dates=pd.DatetimeIndex(pd.to_datetime(z['dates'],utc=True));keep=dates.year<=2023
            for date,pv in zip(dates[keep],z['pv'][keep]):
                t=int((date-times[0]).total_seconds()/900);obs[t:t+96]=pv
        physical_file=PREVIOUS/'cache/world_physical_forecasts.npz'
    else:
        year=int(role)
        with np.load(WORKSPACE/f'external_validation/data/forcing_{year}.npz') as z:
            times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True));obs=z['pv'].astype(np.float32)
        physical_file=ARCHIVE/'cache/final_physical_priors.npz' if year==2025 else PREVIOUS/'cache/world_physical_forecasts.npz'
    n=len(times);history=observer_inputs(times,obs)
    with np.load(physical_file) as z:
        lookup={str(day):i for i,day in enumerate(z['dates'])}
        idx=np.array([lookup.get(day,-1) for day in times.strftime('%Y-%m-%d')]);valid=idx>=0;slot=times.hour.to_numpy()*4+times.minute.to_numpy()//15
        prior=np.r_[z['issue_pv'][idx.clip(0),slot]*valid[:,None],np.zeros((96,4))].astype(np.float32)
        issue_values=pd.DatetimeIndex(pd.to_datetime(z['issue_utc'],utc=True)).asi8
        issued=np.r_[np.where(valid,issue_values[idx.clip(0)],np.iinfo(np.int64).max),np.full(96,np.iinfo(np.int64).max)]
    local=times.tz_convert('America/New_York').tz_localize(None).normalize()
    pre=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(96,np.iinfo(np.int64).max)]
    high,middle,ratio,ready=prepare(times,obs)
    candidates=np.arange(0,n-96,4) if training else np.arange(n)
    if training:
        valid_target=np.isfinite(obs).all(-1);bad=np.r_[0,np.cumsum(~valid_target)]
        candidates=candidates[(bad[candidates+96]-bad[candidates]==0)&history['valid'][candidates]]
    future=np.empty((len(candidates),4,96,9),np.float32)
    target=np.zeros((len(candidates),4,96),np.float32)
    obs_extended=np.r_[np.nan_to_num(obs,nan=0.),np.zeros((96,4))]
    extended=times.append(pd.date_range(times[-1]+pd.Timedelta(minutes=15),periods=96,freq='15min'))
    clock=extended.hour.to_numpy()+extended.minute.to_numpy()/60
    season=extended.dayofyear.to_numpy()/366
    h=np.arange(96)
    for a in range(0,len(candidates),1024):
        origins=candidates[a:a+1024];ids=origins[:,None]+h
        available=(issued[ids]<=times.asi8[origins,None])&(pre[ids]<=times.asi8[origins,None])
        base=np.where(available[:,:,None],prior[ids],middle[ids])
        weight=np.exp(-h/24)[None,:,None]*ready[origins,None,:]
        nowcast=np.clip((1-weight)*base+weight*high[ids]*ratio[origins,None,:],0,1.5)
        factors=[nowcast,high[ids],base,np.broadcast_to(available[:,:,None],nowcast.shape),
                 np.broadcast_to(h[None,:,None]/96,nowcast.shape),
                 *[np.broadcast_to(v[:,:,None],nowcast.shape) for v in [np.sin(2*np.pi*clock[ids]/24),np.cos(2*np.pi*clock[ids]/24),np.sin(2*np.pi*season[ids]),np.cos(2*np.pi*season[ids])]]]
        future[a:a+len(origins)]=np.stack(factors,-1).transpose(0,2,1,3)
        target[a:a+len(origins)]=obs_extended[ids].transpose(0,2,1)
    return dict(history=history['history'][candidates],future=future,target=target,
                origins=times.asi8[candidates])
