import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.causal_pv_update import features,FEATURES
import lightgbm as lgb
O=ROOT/'optimization80'
with np.load(PREVIOUS/'cache/world_physical_forecasts.npz') as z:
    dates=pd.to_datetime(z['dates']);prior=z['issue_pv'];pue=z['issue_pue']
with np.load(LEGACY/'cache/development_physical_reference.npz') as z:actual=z['pv']
times=pd.date_range('2022-01-01','2024-01-01',freq='15min',inclusive='left',tz='UTC');n=len(times)
forecast=np.full((n,4),np.nan);pue_array=forecast.copy();pv=forecast.copy()
for i,day in enumerate(dates):
    if day.year not in [2022,2023]:continue
    start=int((day-pd.Timestamp('2022-01-01')).total_seconds()/900);forecast[start:start+96]=prior[i];pue_array[start:start+96]=pue[i];pv[start:start+96]=actual[i]
panel=pd.read_parquet(WORKSPACE/'新论文_市场可交付灵活性_20260917/experiments/gate_hourly_panel.parquet')
da=panel.pivot(index='timestamp_utc',columns='zone',values='da_lbmp').reindex(index=times.floor('h'),columns=['WEST','CAPITL','N.Y.C.','LONGIL']).to_numpy()
xs=[];ys=[];leads=[0,1,2,3,4,6,8,12,16,24,32,48,64,80,95]
for lead in leads:
    X,o,t,z=features(times,forecast,pue_array,pv,da,np.arange(192,n-lead,4),lead)
    valid=np.isfinite(pv[t,z])&np.isin(times[t].year,[2022,2023]);xs.append(X[valid]);ys.append(pv[t[valid],z[valid]])
X=np.concatenate(xs);y=np.concatenate(ys).astype(np.float32)
manifest=dict(training_years=[2022,2023],allowed_training_years=[2021,2022,2023],training_rows=len(y),physical_cache_start_year=2022,feature_names=FEATURES,observations_strictly_before_decision=True,selection_data_used_for_fit=False)
(O/'pv_update_training_manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest,flush=True)
for objective in ['regression','regression_l1']:
    out=O/f'pv_update_{objective}.txt'
    if out.exists():continue
    model=lgb.LGBMRegressor(objective=objective,n_estimators=250,learning_rate=.04,num_leaves=31,min_child_samples=150,reg_lambda=10,max_bin=127,n_jobs=4,random_state=20260927,verbosity=-1)
    model.fit(X,y,feature_name=FEATURES);out.write_text(model.booster_.model_to_string(),encoding='utf-8');print('PV TRAINED',objective,flush=True)
