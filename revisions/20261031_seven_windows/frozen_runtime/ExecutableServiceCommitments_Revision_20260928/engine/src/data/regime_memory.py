"""Market forecast-error memories following the project's registered D-2 gate."""
import numpy as np,pandas as pd,torch
from src.runtime import ROOT,PAPER
from src.models.coupled_rollout import ArchitectedR2QPHead


def market_history(include_2025=False):
    model=ArchitectedR2QPHead();model.load_state_dict(torch.load(ROOT/'checkpoints/QPR2_seed20260920_epoch10.pt',map_location='cpu',weights_only=False)['state_dict']);model.eval()
    with np.load(ROOT/'cache/rollout_forecast_features.npz') as z:
        data={k:z[k] for k in ['dates','timestamps','zone','labels','train_indices','selection_indices','rt_0','rt_1','decision_latent_0','decision_latent_1']}
    if include_2025:
        frame=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/gate_hourly_2025_with_context.parquet')
        frame=frame[frame.date.between('2025-01-01','2025-12-31')].sort_values(['timestamp_utc','zone']).reset_index(drop=True)
        from src.data.r2_data import ZONES
        zone=pd.Categorical(frame.zone,categories=ZONES).codes
        with np.load(ROOT/'cache/structured_market2025_s20260920_ep10.npz') as z:
            times=pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc,utc=True)).asi8;hi=np.searchsorted(z['hours'],times);valid=(hi<len(z['hours']));valid[valid]&=z['hours'][hi[valid]]==times[valid]
            frame=frame[valid].reset_index(drop=True);zone=zone[valid];hi=hi[valid]
            for gate in [0,1]:
                data[f'rt_{gate}']=np.concatenate([data[f'rt_{gate}'],z[f'rt_{gate}'][hi,zone]])
                data[f'decision_latent_{gate}']=np.concatenate([data[f'decision_latent_{gate}'],z[f'latent_{gate}'][hi,zone]])
        data['dates']=np.r_[data['dates'],frame.date.to_numpy(dtype=str)]
        data['timestamps']=np.r_[data['timestamps'],pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc,utc=True)).tz_localize(None).to_numpy(dtype='datetime64[ns]').astype(str)]
        data['zone']=np.r_[data['zone'],zone];data['labels']=np.r_[data['labels'],frame[['da_lbmp','spread','rt_integrated']].to_numpy()]
    weights=np.array([.075,.1,.2,.25,.2,.1,.075]);means=[];quantiles=[]
    with torch.inference_mode():
        for gate in [0,1]:
            _,shift=model.market_update(torch.tensor(data[f'decision_latent_{gate}']))
            q=data[f'rt_{gate}']+shift.numpy()[:,None];quantiles.append(q);means.append((q*weights).sum(-1))
    return data,np.asarray(means),np.asarray(quantiles)


def memory_features(data,means):
    times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'],utc=True));local=times.tz_convert('America/New_York')
    dates=pd.DatetimeIndex(pd.to_datetime(data['dates']));all_days=pd.date_range(dates.min(),dates.max(),freq='D');days=all_days.get_indexer(dates)
    columns=pd.MultiIndex.from_product([range(4),range(24)],names=['zone','hour']);position=data['zone']*24+local.hour.to_numpy()
    features=[]
    for gate in [0,1]:
        source=pd.DataFrame(dict(date=dates,zone=data['zone'],hour=local.hour,error=data['labels'][:,2]-means[gate]))
        hourly=source.groupby(['date','zone','hour']).error.mean().unstack(['zone','hour']).reindex(index=all_days,columns=columns)
        daily=source.groupby(['date','zone']).error.mean().unstack('zone').reindex(index=all_days,columns=range(4))
        hshift=hourly.shift(2);dshift=daily.shift(2)
        arrays=[]
        for length in [7,28]:
            hmean=hshift.rolling(length,min_periods=2).mean().to_numpy();arrays.append(hmean[days,position])
        for length in [7,28]:
            dmean=dshift.rolling(length,min_periods=2).mean().to_numpy();arrays.append(dmean[days,data['zone']])
        fleet=dshift.mean(1)
        for length in [7,28]:arrays.append(fleet.rolling(length,min_periods=2).mean().to_numpy()[days])
        features.append(np.nan_to_num(np.column_stack(arrays),nan=0.))
    # This reproduces the existing project's source_end < pre-DA issue rule.
    source_end=(dates-pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
    issue=(dates-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize('America/New_York').tz_convert('UTC')
    assert (source_end<issue).all()
    return np.asarray(features)


def export_correction(data,features,weights,year,checkpoint_hash):
    from src.runtime import WORKSPACE
    with np.load(WORKSPACE/f'external_validation/data/forcing_{year}.npz') as z:times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
    source_times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'],utc=True)).asi8
    keys=pd.MultiIndex.from_arrays([source_times,data['zone']]);assert not keys.duplicated().any()
    targets=pd.MultiIndex.from_arrays([np.repeat(times.floor('h').asi8,4),np.tile(np.arange(4),len(times))]);ix=keys.get_indexer(targets);good=ix>=0
    output={}
    for gate in [0,1]:
        delta=np.einsum('ij,ij->i',features[gate],weights[gate,data['zone']]);delta=np.clip(delta,-15,15)
        values=np.zeros(len(ix));values[good]=delta[ix[good]];output[f'correction_{gate}']=np.r_[values.reshape(len(times),4),np.zeros((96,4))].astype(np.float32)
    path=ROOT/'cache'/f'regime_mean_{year}_s20260920.npz'
    np.savez_compressed(path,**output,checkpoint_sha256=checkpoint_hash,latest_source_day='target local date minus two days')
    return path
