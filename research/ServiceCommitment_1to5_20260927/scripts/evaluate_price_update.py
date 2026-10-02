import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.causal_price_update import features
from src.optimized_allocator import allocate
import lightgbm as lgb
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2024);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--objective',default='regression');p.add_argument('--weights',type=float,nargs='+',default=[.25,.5,.75]);args=p.parse_args()
O=ROOT/'optimization80';arr,reference,shared=configure(args.year)
model_path=O/f'price_update_{args.objective}.txt';boost=lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))
cache=O/f'price_update_{args.objective}_{args.year}_causal.npz'
if cache.exists():
    with np.load(cache) as z:pred=z['predictions']
else:
    times=c.times[::4];da=c.da[::4];rt=c.rt[::4];pred=np.full((len(times),25,4),np.nan,dtype=np.float32)
    for lead in range(25):
        X,o,t,z=features(times,da,rt,np.arange(24,len(times)-lead),lead)
        pred[o,lead,z]=(da[t,z]+boost.predict(X,num_threads=2)).astype(np.float32)
    np.savez_compressed(cache,predictions=pred)
for weight in args.weights:
    name=f'O80P_Eagle_{args.year}_s{args.seed}_{args.objective}_a{weight:g}';saved=RESULTS/f'{name}.json'
    if saved.exists():print('CACHED',name,flush=True);continue
    observer_model=observer(args.year,args.seed,arr,24);observer_model.price_update=pred;observer_model.price_update_weight=weight
    start=time.perf_counter();book,late,extra=allocate(arr,observer_model,24,.05,0.,close_year=True)
    extra.update(seconds=time.perf_counter()-start,price_update_objective=args.objective,price_update_weight=weight,training_years='2021-2023',selection_year=2024)
    row=measure(name,book,reference,arr,shared,dict(experiment='O80P',year=args.year,trace='Eagle',seed=args.seed,window_h=24,role='selection' if args.year==2024 else 'exploratory_frozen_evaluation'),late=late,extra=extra)
    bound=json.loads((RESULTS/f'I1_Eagle_{args.year}_w24.json').read_text());row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];saved.write_text(json.dumps(row,indent=2));print('CAPTURE',row['capture_lower_fraction'],'CLOSING',row['closing_it_mwh'],flush=True)
