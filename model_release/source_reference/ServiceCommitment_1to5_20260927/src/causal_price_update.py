"""Completed-hour settlement information; all fitted labels precede 2024."""
import numpy as np
import pandas as pd
from vendor.forecast_provider import TZ

FEATURES=['zone','lead_hours','target_hour_sin','target_hour_cos','target_dow','target_month','target_da',
          'rt_lag1','rt_lag2','rt_lag3','rt_lag6','rt_lag24',
          'spread_lag1','spread_lag2','spread_lag3','spread_lag6','spread_lag24',
          'rt_mean3','rt_mean6','rt_std6','rt_max6','rt_change3','target_da_minus_observed_da']

def features(times,da,rt,origins,lead):
    """Origins are hourly decisions. Observation indices are strictly < origin."""
    origins=np.asarray(origins,dtype=int);target=origins+lead;valid=(origins>=24)&(target<len(times));origins=origins[valid];target=target[valid]
    local=times[target].tz_convert(TZ);dates=local.tz_localize(None).normalize()
    published=(dates-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize(TZ).tz_convert('UTC').asi8
    known=published<=times[origins].asi8
    z=np.tile(np.arange(4),len(origins));o=np.repeat(origins,4);t=np.repeat(target,4)
    lagrt=[rt[o-l,z] for l in [1,2,3,6,24]];lagspread=[rt[o-l,z]-da[o-l,z] for l in [1,2,3,6,24]]
    six=np.column_stack([rt[o-l,z] for l in range(1,7)]);hour=np.repeat(local.hour.to_numpy(),4)
    X=np.column_stack([z,np.full(len(o),lead),np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.repeat(local.dayofweek.to_numpy(),4),np.repeat(local.month.to_numpy(),4),da[t,z],*lagrt,*lagspread,six[:,:3].mean(1),six.mean(1),six.std(1),six.max(1),lagrt[0]-lagrt[2],da[t,z]-da[o-1,z]]).astype(np.float32)
    mask=np.repeat(known,4)&np.isfinite(X).all(1)
    return X[mask],o[mask],t[mask],z[mask]
