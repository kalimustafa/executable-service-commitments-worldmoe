import numpy as np
import pandas as pd
from src.context import WORKSPACE,PREVIOUS,LEGACY,c
from src.causal_pv_update import features,FEATURES
FIELDS=['rt_integrated_lag2','spread_lag2','day_std_mean7','day_rt_mean7']
NAMES=FEATURES+FIELDS+['booked_capacity','forecast_pv_surplus']
def market_arrays(times):
    panel=pd.read_parquet(WORKSPACE/'新论文_市场可交付灵活性_20260917/experiments/gate_hourly_panel.parquet')
    if times[-1].year>=2025:
        final=pd.read_parquet(WORKSPACE/'新论文_市场可交付灵活性_20260917/revision_20260918/forecast_2025/gate_hourly_2025_with_context.parquet')
        panel=pd.concat([panel,final],ignore_index=True)
    panel=panel.drop_duplicates(['timestamp_utc','zone'],keep='last')
    zones=['WEST','CAPITL','N.Y.C.','LONGIL'];out={}
    for field in ['da_lbmp','rt_integrated',*FIELDS]:out[field]=panel.pivot(index='timestamp_utc',columns='zone',values=field).reindex(index=times.floor('h'),columns=zones).to_numpy()
    return out
def feature_rows(times,prior,pue,pv,market,origins,lead):
    X,o,t,z=features(times,prior,pue,pv,market['da_lbmp'],origins,lead)
    extra=np.column_stack([market[field][t,z] for field in FIELDS]);valid=np.isfinite(extra).all(1)
    return np.column_stack([X,extra])[valid].astype(np.float32),o[valid],t[valid],z[valid]
