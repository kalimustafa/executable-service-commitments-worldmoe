import sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.causal_pv_update import features
from src.optimized_allocator import allocate
import lightgbm as lgb
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2024);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--objective',default='regression');p.add_argument('--weights',type=float,nargs='+',default=[.5,1.]);args=p.parse_args()
O=ROOT/'optimization80';arr,reference,shared=configure(args.year)
native=observer(args.year,args.seed,arr,24)
boost=lgb.Booster(model_str=(O/f'pv_update_{args.objective}.txt').read_text(encoding='utf-8'))
cache=O/f'pv_update_{args.objective}_{args.year}_quarterly_causal.npz'
if cache.exists():
    with np.load(cache) as z:pred=z['predictions']
else:
    pred=np.full((c.N,96,4),np.nan,dtype=np.float32)
    for lead in range(96):
        X,o,t,z=features(c.times,native.pv[:c.N],native.pue[:c.N],c.pv,c.da,np.arange(192,c.N-lead),lead)
        values=np.clip(boost.predict(X,num_threads=2),0,1.0);pred[o,lead,z]=values
        if lead%24==0:print('PV CACHE',args.year,args.objective,lead,flush=True)
    np.savez_compressed(cache,predictions=pred)
for weight in args.weights:
    name=f'O80V_Eagle_{args.year}_s{args.seed}_{args.objective}_b{weight:g}';saved=RESULTS/f'{name}.json'
    if saved.exists():print('CACHED',name,flush=True);continue
    model=observer(args.year,args.seed,arr,24);original=model.cost_quantiles
    def corrected(self,q,ids,z,booked,delta,physical):
        base=original(q,ids,z,booked,delta,physical)
        now=self.current_decision_index;pvhat=pred[now,ids-now,z]
        valid=np.isfinite(pvhat)&physical
        pue=self.pue[ids,z];inc=pue*self.k[z]*delta;load=pue*(self.idle[z]+self.k[z]*booked)
        coverage=np.clip((np.where(valid,pvhat,0)-load)/inc,0,1);ratio=pue/self.pue_reference[z]
        alternative=q*(ratio*(1-coverage))[:,None]-(ratio*coverage)[:,None]
        return np.where(valid[:,None],(1-weight)*base+weight*alternative,base)
    model.cost_quantiles=types.MethodType(corrected,model)
    start=time.perf_counter();book,late,extra=allocate(arr,model,24,.05,0.,close_year=True)
    extra.update(seconds=time.perf_counter()-start,pv_update_objective=args.objective,pv_update_weight=weight,training_years='2022-2023 within allowed 2021-2023',selection_year=2024)
    row=measure(name,book,reference,arr,shared,dict(experiment='O80V',year=args.year,trace='Eagle',seed=args.seed,window_h=24,role='selection' if args.year==2024 else 'exploratory_frozen_evaluation'),late=late,extra=extra)
    bound=json.loads((RESULTS/f'I1_Eagle_{args.year}_w24.json').read_text());row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];saved.write_text(json.dumps(row,indent=2));print('CAPTURE',row['capture_lower_fraction'],'CLOSING',row['closing_it_mwh'],flush=True)
