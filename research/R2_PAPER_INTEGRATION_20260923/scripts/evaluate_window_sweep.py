from src.evaluation.paper_economics import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--year',type=int,default=2025);p.add_argument('--trace',default='Eagle');p.add_argument('--exclude-primary',action='store_true');args=p.parse_args()
protocol=json.loads((ROOT/'configs/paper_support_protocol.json').read_text())
arr,reference,shared=configure(args.year,args.trace,'registered' if args.trace=='Eagle' else 'requested_nodes')
rows=[]
for window in protocol['window_sweep_h']:
    if args.exclude_primary and window in [12,24]:continue
    for policy in ['point','interval']:
        rows.append(run_homogeneous(arr,reference,shared,year=args.year,trace=args.trace,
            proxy='registered' if args.trace=='Eagle' else 'requested_nodes',seed=args.seed,window=window,policy=policy))
pd.DataFrame(rows).to_csv(ROOT/'results'/f'window_sweep_{args.trace}_{args.year}_seed{args.seed}.csv',index=False)
print('HOURLY CONTRACT SWEEP COMPLETE',args.seed,args.trace,args.year,flush=True)
