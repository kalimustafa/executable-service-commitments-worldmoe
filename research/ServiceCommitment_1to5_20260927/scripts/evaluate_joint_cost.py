import sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.joint_cost_data import feature_rows,market_arrays
from src.optimized_allocator import allocate
import lightgbm as lgb
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2024);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--objective',default='regression');p.add_argument('--weights',type=float,nargs='+',default=[.25,.5,.75]);args=p.parse_args()
O=ROOT/'optimization80';arr,reference,shared=configure(args.year);native=observer(args.year,args.seed,arr,24)
boost=lgb.Booster(model_str=(O/f'joint_cost_{args.objective}.txt').read_text(encoding='utf-8'));levels=np.array([0,.05,.1,.15,.2,.3,.4,.55],dtype=np.float32)
cache=O/f'joint_cost_{args.objective}_{args.year}_causal.npz'
if cache.exists():
    with np.load(cache) as z:pred=z['predictions']
else:
    market=market_arrays(c.times);pred=np.full((int(np.ceil(c.N/4)),25,4,len(levels)),np.nan,dtype=np.float32)
    for lead_h in range(25):
        X,o,t,z=feature_rows(c.times,native.pv[:c.N],native.pue[:c.N],c.pv,market,np.arange(192,c.N-lead_h*4,4),lead_h*4)
        for j,booked in enumerate(levels):
            surplus=X[:,6]-X[:,7]*(c.idle[z]+c.k[z]*booked)
            xx=np.column_stack([X,np.full(len(X),booked),surplus]).astype(np.float32)
            pred[o//4,lead_h,z,j]=boost.predict(xx,num_threads=2)
        if lead_h%6==0:print('JOINT CACHE',args.year,args.objective,lead_h,flush=True)
    np.savez_compressed(cache,predictions=pred)
for weight in args.weights:
    name=f'O80J_Eagle_{args.year}_s{args.seed}_{args.objective}_b{weight:g}';saved=RESULTS/f'{name}.json'
    if saved.exists():print('CACHED',name,flush=True);continue
    model=observer(args.year,args.seed,arr,24);original=model.cost_quantiles
    def corrected(self,q,ids,z,booked,delta,physical):
        base=original(q,ids,z,booked,delta,physical);now=self.current_decision_index
        values=pred[now//4,ids//4-now//4,z]
        j=np.searchsorted(levels,booked,side='right').clip(1,len(levels)-1)-1
        weight_b=np.clip((booked-levels[j])/(levels[j+1]-levels[j]),0,1)
        point=values[np.arange(len(ids)),j]*(1-weight_b)+values[np.arange(len(ids)),j+1]*weight_b
        valid=np.isfinite(point)&physical&(self.post_issue[ids]<=self.base.times.asi8[now])
        shift=np.where(valid,weight*(point-base[:,3]),0.)
        return base+shift[:,None]
    model.cost_quantiles=types.MethodType(corrected,model);start=time.perf_counter()
    book,late,extra=allocate(arr,model,24,.05,0.,close_year=True)
    extra.update(seconds=time.perf_counter()-start,joint_cost_objective=args.objective,joint_cost_weight=weight,training_years='2022-2023 within allowed 2021-2023',selection_year=2024)
    row=measure(name,book,reference,arr,shared,dict(experiment='O80J',year=args.year,trace='Eagle',seed=args.seed,window_h=24,role='selection' if args.year==2024 else 'exploratory_frozen_evaluation'),late=late,extra=extra)
    bound=json.loads((RESULTS/f'I1_Eagle_{args.year}_w24.json').read_text());row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];saved.write_text(json.dumps(row,indent=2));print('CAPTURE',row['capture_lower_fraction'],'CLOSING',row['closing_it_mwh'],flush=True)
