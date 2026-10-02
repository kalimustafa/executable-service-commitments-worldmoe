import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.causal_price_update import features,FEATURES
import lightgbm as lgb
O=ROOT/'optimization80';O.mkdir(exist_ok=True)
frame=pd.read_parquet(WORKSPACE/'新论文_市场可交付灵活性_20260917/experiments/gate_hourly_panel.parquet')
frame['timestamp_utc']=pd.to_datetime(frame.timestamp_utc,utc=True)
times=pd.date_range(frame.timestamp_utc.min().floor('h'),frame.timestamp_utc.max().floor('h'),freq='h')
da=frame.pivot(index='timestamp_utc',columns='zone',values='da_lbmp').reindex(index=times,columns=['WEST','CAPITL','N.Y.C.','LONGIL']).to_numpy()
rt=frame.pivot(index='timestamp_utc',columns='zone',values='rt_integrated').reindex(index=times,columns=['WEST','CAPITL','N.Y.C.','LONGIL']).to_numpy()
pieces=[];labels=[];years=[]
for lead in range(25):
    X,o,t,z=features(times,da,rt,np.arange(24,len(times)-lead),lead)
    local_year=times[t].tz_convert(TZ).year.to_numpy();valid=np.isin(local_year,[2021,2022,2023])&np.isfinite(rt[t,z])
    pieces.append(X[valid]);labels.append((rt[t,z]-da[t,z])[valid]);years.append(local_year[valid])
X=np.concatenate(pieces);y=np.concatenate(labels).astype(np.float32);years=np.concatenate(years)
assert set(np.unique(years))=={2021,2022,2023} and len(X)==len(y)
manifest=dict(training_years=sorted(np.unique(years).tolist()),training_rows=len(y),feature_names=FEATURES,max_observation_offset_hours=-1,selection_data_used_for_fit=False,evaluation_data_used_for_fit=False,model_role='Causal DA-anchored RT innovation blended with frozen WorldMoE RT distribution')
(O/'price_update_training_manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest,flush=True)
for objective in ['regression','regression_l1','huber']:
    out=O/f'price_update_{objective}.txt'
    if out.exists():continue
    model=lgb.LGBMRegressor(objective=objective,n_estimators=220,learning_rate=.04,num_leaves=31,min_child_samples=150,reg_lambda=20,max_bin=127,n_jobs=4,random_state=20260927,verbosity=-1)
    model.fit(X,y,feature_name=FEATURES);out.write_text(model.booster_.model_to_string(),encoding='utf-8')
    print('TRAINED',objective,flush=True)
