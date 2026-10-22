"""Quarter-hour PV observer fitted only on 2022-2023 sensor histories."""
from src.runtime import ROOT,LEGACY
from src.data.solar_nowcast import calendar_index
import numpy as np,pandas as pd,json

def fit_observer():
    path=ROOT/'cache/causal_observer.npz'
    if path.exists():return
    with np.load(LEGACY/'cache/development_physical_reference.npz') as z:
        dates=pd.DatetimeIndex(pd.to_datetime(z['dates']));keep=dates.year<=2023;dates=dates[keep];pv=z['pv'][keep].reshape(-1,4)
    with np.load(ROOT/'cache/solar_seasonal_envelope.npz') as z:upper=z['upper'][calendar_index(dates)].reshape(-1,4)
    previous=np.roll(pv,1,axis=0);previous_upper=np.roll(upper,1,axis=0);valid=np.ones(len(pv),bool);valid[0]=False
    for i in range(1,len(dates)):
        if (dates[i]-dates[i-1]).days!=1:valid[i*96]=False
    scaled=previous*np.clip(upper/np.maximum(previous_upper,.02),0,2);d=scaled-previous
    alpha=np.clip((d[valid]*(pv-previous)[valid]).sum(0)/np.maximum((d[valid]**2).sum(0),1e-9),0,1)
    point=np.clip(previous+alpha*d,0,1.5);error=(pv-point)/np.maximum(upper,.1)
    residual=np.zeros((96,4,7),np.float32);clock=np.arange(len(pv))%96;tau=[.05,.1,.25,.5,.75,.9,.95]
    for k in range(96):residual[k]=np.quantile(error[valid&(clock==k)],tau,axis=0).T
    np.savez_compressed(path,alpha=alpha.astype(np.float32),residual_quantiles=residual)
    path.with_suffix('.json').write_text(json.dumps(dict(training_years=[2022,2023],observations_strictly_before_decision=True,lag_quarters=1,alpha=alpha.tolist(),mae_persistence=float(np.abs(pv[valid]-previous[valid]).mean()),mae_observer=float(np.abs(pv[valid]-point[valid]).mean()),validation_fitted=False),indent=2))
