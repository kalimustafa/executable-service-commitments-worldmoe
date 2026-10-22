"""Training-only seasonal PV envelopes and strictly lagged clearness updates."""
from src.runtime import ROOT,LEGACY
import numpy as np,pandas as pd,json

def calendar_index(times):
    # A leap-year template preserves month/day alignment in every target year.
    return np.array([pd.Timestamp(f'2020-{t.month:02d}-{t.day:02d}').dayofyear-1 for t in times],np.int64)

def fit_envelope():
    output=ROOT/'cache/solar_seasonal_envelope.npz'
    if output.exists():return
    with np.load(LEGACY/'cache/development_physical_reference.npz') as z:
        dates=pd.DatetimeIndex(pd.to_datetime(z['dates']));mask=dates.year.isin([2022,2023]);pv=z['pv'][mask];dates=dates[mask]
    day=calendar_index(dates);upper=[];median=[]
    for d in range(366):
        distance=np.minimum((day-d)%366,(d-day)%366);sample=pv[distance<=21]
        upper.append(np.quantile(sample,.95,axis=0));median.append(np.median(sample,axis=0))
    np.savez_compressed(output,upper=np.array(upper,np.float32),median=np.array(median,np.float32))
    output.with_suffix('.json').write_text(json.dumps(dict(training_years=[2022,2023],days=len(dates),calendar_window_days=43,upper_quantile=.95,year2024_fitting=False,purpose='seasonal PV upper envelope and median, available before every target interval'),indent=2))

def prepare(times,observations):
    with np.load(ROOT/'cache/solar_seasonal_envelope.npz') as z:upper=z['upper'];median=z['median']
    n=len(times);extended=times.append(pd.date_range(times[-1]+pd.Timedelta(minutes=15),periods=96,freq='15min'))
    days=calendar_index(extended);slots=extended.hour.to_numpy()*4+extended.minute.to_numpy()//15
    high=upper[days,slots].astype(float);middle=median[days,slots].astype(float)
    obs=np.nan_to_num(observations,nan=0.);valid=np.isfinite(observations)&(high[:n]>.05)
    total=np.vstack([np.zeros((1,4)),np.cumsum(np.where(valid,obs,0),axis=0)])
    scale=np.vstack([np.zeros((1,4)),np.cumsum(np.where(valid,high[:n],0),axis=0)])
    t=np.arange(n);begin=np.maximum(0,t-4)
    ratio=(total[t]-total[begin])/(scale[t]-scale[begin]+1e-8)
    ready=(scale[t]-scale[begin])>.1
    previous=np.vstack([np.zeros((1,4)),obs[:-1]])
    prev_upper=np.vstack([np.zeros((1,4)),high[:n-1]])
    recent=previous/np.maximum(prev_upper,.02)
    ratio=np.where(prev_upper>.05,.6*recent+.4*ratio,ratio)
    return high,middle,np.clip(ratio,0,1.5),ready

def forecast(t,ids,issued,issued_available,prepared,hours):
    high,middle,ratio,ready=prepared
    base=np.where(issued_available[:,None],issued,middle[ids])
    weight=np.exp(-np.arange(len(ids))/(hours*4))[:,None]*ready[t]
    persistence=high[ids]*ratio[t]
    return np.clip((1-weight)*base+weight*persistence,0,1.5)
