"""Causal physical cost calibration from issued priors and completed PV."""
import numpy as np
import pandas as pd
from vendor.forecast_provider import TZ
FEATURES=['zone','lead_hours','target_hour_sin','target_hour_cos','doy_sin','doy_cos','issue_pv_target','issue_pue_target','past_pv_1q','past_pv_1h','past_pv_4h','same_hour_previous_day_pv','past_issue_pv','pv_innovation','past_pv_mean1h','published_da','da_available']

def features(times,prior,pue,pv,da,origins,lead):
    o=np.asarray(origins,dtype=int);t=o+lead;valid=(o>=192)&(t<len(times));o=o[valid];t=t[valid]
    local=times[t].tz_convert(TZ);date=local.tz_localize(None).normalize();pub=(date-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize(TZ).tz_convert('UTC').asi8
    known=pub<=times[o].asi8
    z=np.tile(np.arange(4),len(o));origin=np.repeat(o,4);target=np.repeat(t,4)
    local_hour=np.repeat(local.hour.to_numpy()+local.minute.to_numpy()/60,4);doy=np.repeat(local.dayofyear.to_numpy(),4)
    historical=target-96;historical=np.where(historical<origin,historical,historical-96)
    previous=pv[origin-1,z];prior_previous=prior[origin-1,z]
    X=np.column_stack([z,np.full(len(z),lead*.25),np.sin(2*np.pi*local_hour/24),np.cos(2*np.pi*local_hour/24),np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25),prior[target,z],pue[target,z],previous,pv[origin-4,z],pv[origin-16,z],pv[historical,z],prior_previous,previous-prior_previous,np.column_stack([pv[origin-j,z] for j in range(1,5)]).mean(1),np.where(np.repeat(known,4),da[target,z],0.),np.repeat(known,4)]).astype(np.float32)
    mask=np.isfinite(X).all(1)
    return X[mask],origin[mask],target[mask],z[mask]
