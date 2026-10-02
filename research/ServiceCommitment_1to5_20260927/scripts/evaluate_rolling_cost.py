import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.rolling_cost_dispatch import execute
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2024);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--safety',type=float,default=.5);p.add_argument('--weights',type=float,nargs='+',default=[0,.5,1]);a=p.parse_args()
for weight in a.weights:
    name=f'O80M_Eagle_{a.year}_s{a.seed}_k{a.safety:g}_b{weight:g}';saved=RESULTS/f'{name}.json'
    if saved.exists():print('CACHED',name,flush=True);continue
    arr,ref,shared=configure(a.year);m=observer(a.year,a.seed,arr,24)
    if weight:
        with np.load(ROOT/f'optimization80/pv_update_regression_l1_{a.year}_quarterly_causal.npz') as z:pred=z['predictions'].copy()
    else:pred=np.full((c.N,96,4),np.nan,dtype=np.float32)
    start=time.perf_counter()
    book=execute(arr,c.times.asi8,m.pre[:,:,3],m.post[:,:,3],m.pre_issue,m.post_issue,m.pv,m.pue,m.physical_pre,m.physical_post,pred,weight,c.pv,c.pue,c.idle,c.k,m.pue_reference,c.train_rt,a.safety)
    row=measure(name,book,ref,arr,shared,dict(experiment='O80M',year=a.year,trace='Eagle',seed=a.seed,window_h=24,role='selection' if a.year==2024 else 'exploratory_frozen_evaluation'),extra=dict(seconds=time.perf_counter()-start,safety_capacity=a.safety,pv_update_weight=weight,physical_observation_lag_intervals=1,future_realizations_used=False,adaptive_locking=False,executor='Receding known-work marginal-cost plan using frozen WorldMoE forecasts'))
    bound=json.loads((RESULTS/f'I1_Eagle_{a.year}_w24.json').read_text());row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];saved.write_text(json.dumps(row,indent=2));print('CAPTURE',row['capture_lower_fraction'],'CLOSING',row['closing_it_mwh'],flush=True)
