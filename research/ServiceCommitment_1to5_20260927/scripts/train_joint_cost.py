import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.joint_cost_data import feature_rows,market_arrays,NAMES
import lightgbm as lgb
O=ROOT/'optimization80'
with np.load(PREVIOUS/'cache/world_physical_forecasts.npz') as z:dates=pd.to_datetime(z['dates']);prior=z['issue_pv'];priorpue=z['issue_pue']
with np.load(LEGACY/'cache/development_physical_reference.npz') as z:actual=z['pv'];actualpue=z['pue']
mask=dates.year<=2023;pue_ref=np.median(actualpue[mask],axis=(0,1))
times=pd.date_range('2022-01-01','2024-01-01',freq='15min',inclusive='left',tz='UTC');n=len(times)
arrays=[np.full((n,4),np.nan,dtype=np.float32) for _ in range(4)]
for i,day in enumerate(dates):
    if day.year not in [2022,2023]:continue
    start=int((day-pd.Timestamp('2022-01-01')).total_seconds()/900)
    for target,source in zip(arrays,[prior,priorpue,actual,actualpue]):target[start:start+96]=source[i]
forecast,fpue,pv,pue=arrays;market=market_arrays(times);rng=np.random.default_rng(20260927);xs=[];ys=[]
k=1.639059066772461;idle=.23069976270198822
for lead in [0,1,2,3,4,6,8,12,16,24,32,48,64,80,95]:
    X,o,t,z=feature_rows(times,forecast,fpue,pv,market,np.arange(192,n-lead,4),lead)
    valid=np.isfinite(pv[t,z])&np.isfinite(pue[t,z])&np.isfinite(market['rt_integrated'][t,z])&X[:,16].astype(bool)
    X=X[valid];t=t[valid];z=z[valid]
    for lower,upper in [(0,0),(.0,.25),(.25,.55)]:
        booked=rng.uniform(lower,upper,len(t)) if upper>lower else np.zeros(len(t))
        pv_surplus=X[:,6]-X[:,7]*(idle+k*booked)
        xx=np.column_stack([X,booked,pv_surplus]).astype(np.float32)
        coverage=np.clip((pv[t,z]-pue[t,z]*(idle+k*booked))/(pue[t,z]*k*.05),0,1)
        y=pue[t,z]/pue_ref[z]*((1-coverage)*market['rt_integrated'][t,z]-coverage)
        xs.append(xx);ys.append(y.astype(np.float32))
X=np.concatenate(xs);y=np.concatenate(ys)
manifest=dict(training_years=[2022,2023],allowed_training_years=[2021,2022,2023],training_rows=len(y),feature_names=NAMES,target='realized incremental electricity and curtailment cost for 0.05 work units, in reference-PUE price units',publication_gate='published DA only',future_physical_values_used_as_labels_only=True,selection_year=2024,selection_data_used_for_fit=False)
(O/'joint_cost_training_manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest,flush=True)
for objective in ['regression','huber']:
    path=O/f'joint_cost_{objective}.txt'
    if path.exists():continue
    model=lgb.LGBMRegressor(objective=objective,n_estimators=240,learning_rate=.04,num_leaves=31,min_child_samples=200,reg_lambda=20,max_bin=127,n_jobs=4,random_state=20260927,verbosity=-1)
    model.fit(X,y,feature_name=NAMES);path.write_text(model.booster_.model_to_string(),encoding='utf-8');print('JOINT COST TRAINED',objective,flush=True)
