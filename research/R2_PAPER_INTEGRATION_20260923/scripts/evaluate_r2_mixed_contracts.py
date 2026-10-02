from src.evaluation.paper_economics import *
from src.evaluation.mixed_cost_adapter import MixedCostProvider,bound_executors
from vendor import paper_controls as control
import argparse
p=argparse.ArgumentParser();p.add_argument('--case',choices=['Eagle_2025','Frontier_2025','Frontier_2024','Eagle_2024']);args=p.parse_args()
assert json.loads((ROOT/'results/mixed_cost_binding_equivalence.json').read_text())['status']=='passed'
protocol=json.loads((ROOT/'configs/paper_support_protocol.json').read_text())
rows=[]
for year,trace in [(2025,'Eagle'),(2025,'Frontier'),(2024,'Frontier'),(2024,'Eagle')]:
    if args.case and args.case!=f'{trace}_{year}':continue
    proxy='registered' if trace=='Eagle' else 'requested_nodes'
    arr,_,shared=configure(year,trace,proxy)
    shares_list=protocol['mixed_shares'] if trace=='Eagle' and year==2025 else [[.25,.5,.25],[.5,.25,.25],[.25,.25,.5]]
    for shares in shares_list:
        classes=[(fraction,window) for fraction,window in zip(shares,[4,12,24])];control.CLASSES=classes
        reference,ref_late=control.edf_reference(arr)
        for policy in ['point','interval']:
            for recourse in [False,True]:
                code='_'.join(str(round(x*100)) for x in shares)
                name=f'{trace}_{year}_R2_seed20260920_mix{code}_{policy}_recourse{int(recourse)}'
                saved=ROOT/'results/paper_runs'/f'{name}.json'
                if saved.exists():rows.append(json.loads(saved.read_text()));continue
                provider=MixedCostProvider(arr,year,20260920,policy);fixed,repaired=bound_executors(provider,classes)
                began=time.perf_counter();mode='robust' if policy=='interval' else 'point'
                if recourse:
                    book,stats,event=repaired(arr,mode);events=list(event.itertuples(index=False,name=None))
                else:book,stats,events=fixed(arr,classes,mode,keep_events=True)
                meta=dict(year=year,trace=trace,proxy=proxy,seed=20260920,window_h='mixed',policy=policy,model='R2',
                    shares=shares,recourse=recourse,reference_late=ref_late)
                rows.append(measure(name,book,reference,arr,shared,meta,stats,events,began,saved))
                pd.DataFrame(rows).to_csv(ROOT/'results'/f'r2_mixed_contracts_{args.case or "all"}.csv',index=False)
pd.DataFrame(rows).to_csv(ROOT/'results'/f'r2_mixed_contracts_{args.case or "all"}.csv',index=False)
print('MIXED R2 CONTRACTS COMPLETE',flush=True)
