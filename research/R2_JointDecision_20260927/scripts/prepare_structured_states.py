"""Add strictly lagged PV evidence and candidate time features to the state bank."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT,LEGACY,PREVIOUS,PAPER
import numpy as np,pandas as pd
frame=pd.read_parquet(PAPER/'experiments/gate_hourly_panel.parquet',columns=['timestamp_utc','zone']).sort_values(['timestamp_utc','zone']).reset_index(drop=True)
zones=pd.Categorical(frame.zone,categories=['WEST','CAPITL','N.Y.C.','LONGIL']).codes
with np.load(LEGACY/'cache/development_physical_reference.npz') as z:dates=z['dates'];actual=z['pv'].copy()
with np.load(PREVIOUS/'cache/world_physical_forecasts.npz') as z:
    assert z['dates'].tolist()==dates.tolist();prior=z['issue_pv'].copy()
lookup={str(d):i for i,d in enumerate(dates)}
decay=np.exp(-np.arange(96)/16)
for role in ['train','selection']:
    with np.load(ROOT/'cache'/f'decision_{role}.npz') as z:pack={k:z[k] for k in z.files}
    additions=[]
    for i,origin in enumerate(pack['origins']):
        t=pd.Timestamp(str(origin));zone=zones[pack['row_ids'][i,0]]
        observed=[];predicted=[]
        for lag in range(1,17):
            time=t-pd.Timedelta(minutes=15*lag);d=lookup.get(time.strftime('%Y-%m-%d'))
            if d is None:continue
            slot=time.hour*4+time.minute//15;observed.append(actual[d,slot,zone]);predicted.append(prior[d,slot,zone])
        ratio=np.clip(np.log((sum(observed)+.2)/(sum(predicted)+.2)),-np.log(4),np.log(4)) if observed else 0.
        target=t+pd.to_timedelta(np.arange(96)*15,unit='min');hour=target.hour.to_numpy()+target.minute.to_numpy()/60
        additions.append(np.column_stack([ratio*decay/2,np.full(96,len(observed)/16),np.full(96,np.log1p(observed[0]) if observed else 0),np.full(96,np.log1p(predicted[0]) if predicted else 0),np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)]))
    pack['states']=np.concatenate([pack['states'],np.array(additions,np.float32)],-1)
    np.savez_compressed(ROOT/'cache'/f'structured_{role}.npz',**pack)
    print('STRUCTURED',role,pack['states'].shape,flush=True)
(ROOT/'configs/structured_protocol.json').write_text(json.dumps(dict(structure='ordered PV distribution and exact expected segment costs; pooled 96-slot context; bounded 5 USD/MWh opportunity and decision-price corrections',observation_lag_quarters=1,training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_year=2024,max_epochs=24,min_epochs=6,economic_evaluation_every_epochs=2,early_stop_patience_evaluations=3,economic_min_delta_usd=50,forecast_degradation_limit_fraction=.02,best_checkpoint_restored=True,price_base_frozen=True,world_backbone_frozen=True,gradient='hard feasible segment allocation forward, explicitly approximate straight-through soft allocation backward',other_seeds_condition='primary annual 2024 economic improvement and zero late/closing work',paper='rewrite after this round regardless of target outcome; claim only measured evidence'),indent=2))
