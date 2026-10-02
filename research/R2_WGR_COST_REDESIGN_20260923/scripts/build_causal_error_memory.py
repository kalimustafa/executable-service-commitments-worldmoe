from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json,hashlib
import numpy as np
import pandas as pd
import torch
from src.data.r2_data import R2Data,ZONES
from src.calibration.calibrate_and_predict import load_model

def historical_features(frame,errors,inside50,inside90):
    source=frame[['date','zone','hour']].copy();source['date']=pd.to_datetime(source.date)
    source['error']=errors;source['cover50']=inside50-.5;source['cover90']=inside90-.9
    days=pd.date_range(source.date.min(),source.date.max(),freq='D')
    keys=pd.MultiIndex.from_product([days,ZONES,range(24)],names=['date','zone','hour'])
    hourly=source.groupby(['date','zone','hour'])[['error','cover50','cover90']].mean().reindex(keys)
    group=hourly.groupby(['zone','hour'],sort=False)
    h={f'error_lag{lag}':group.error.shift(lag) for lag in [2,3,7]}
    shifted=group[['error','cover50','cover90']].shift(2)
    for length in [7,28]:
        for key in ['error','cover50','cover90']:
            h[f'{key}_mean{length}']=shifted[key].groupby(['zone','hour'],sort=False).transform(lambda s:s.rolling(length,min_periods=2).mean())
        h[f'error_std{length}']=shifted.error.groupby(['zone','hour'],sort=False).transform(lambda s:s.rolling(length,min_periods=2).std(ddof=0))
    hourly_features=pd.DataFrame(h)
    target_keys=pd.MultiIndex.from_frame(source[['date','zone','hour']])
    values=hourly_features.reindex(target_keys).to_numpy(np.float32)
    daily_keys=pd.MultiIndex.from_product([days,ZONES],names=['date','zone'])
    daily=source.groupby(['date','zone'])[['error']].mean().reindex(daily_keys)
    shifted_daily=daily.groupby('zone').error.shift(2)
    extra=[];names=list(hourly_features.columns)
    for length in [1,7,28]:
        means=shifted_daily.groupby('zone').transform(lambda s:s.rolling(length,min_periods=1).mean())
        extra.append(means.reindex(pd.MultiIndex.from_frame(source[['date','zone']])).to_numpy(np.float32))
        names.append(f'daily_error_mean{length}')
    values=np.column_stack([values,*extra]);available=np.isfinite(values).mean(-1)
    return values,available,names

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=R2Data(multiscale=True);model,cp=load_model('C0_Repaired',20260920,data)
    features=[];availability=[];names=None
    for publication in [0.,1.]:
        predictions=[]
        with torch.inference_mode():
            for ix in torch.arange(len(data.frame),device=data.device).split(4096):
                pub=torch.full((len(ix),),publication,device=data.device)
                out=model(data.batch(ix),pub,data.labels[ix,0]);predictions.append(out['dart'].cpu().numpy())
        q=np.concatenate(predictions);y=data.frame.spread.to_numpy()
        error=y-q[:,3];inside50=((y>=q[:,2])&(y<=q[:,4])).astype(float);inside90=((y>=q[:,0])&(y<=q[:,6])).astype(float)
        values,available,names=historical_features(data.frame,error,inside50,inside90)
        train=data.frame.split.eq('train').to_numpy()
        center=np.nanmedian(values[train],axis=0);scale=np.nanstd(values[train],axis=0).clip(.01)
        values=np.clip((np.where(np.isfinite(values),values,center)-center)/scale,-10,10)
        features.append(values.astype(np.float32));availability.append(available.astype(np.float32))
        print('Causal error state built',publication,values.shape,flush=True)
    dates=pd.to_datetime(data.frame.date)
    latest_day=dates-pd.Timedelta(days=2)
    source_end=(latest_day+pd.Timedelta(days=1)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    issue=(dates-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    assert (source_end<issue).all()
    np.savez_compressed(ROOT/'cache/causal_error_memory.npz',features=np.stack(features),available=np.stack(availability),
        dates=data.frame.date.to_numpy(dtype=str),source_end_utc=source_end.astype(str).to_numpy(dtype=str),names=np.asarray(names))
    (ROOT/'cache/causal_error_memory_manifest.json').write_text(json.dumps(dict(status='complete',rows=len(data.frame),feature_count=len(names),names=names,
        source_model='C0_Repaired',source_checkpoint_epoch=cp['epoch'],source_raw_not_calibrated=True,
        latest_source_day='target local date minus two days',min_hours_before_issue=float((issue-source_end).dt.total_seconds().min()/3600),
        normalization_fit='2021-2023 train only',trainable_updates_on_2024=False,locked_targets_accessed=False,
        row_keys_sha256=hashlib.sha256(data.frame[['timestamp_utc','zone']].astype(str).to_csv(index=False).encode()).hexdigest()),indent=2))

if __name__=='__main__':main()
