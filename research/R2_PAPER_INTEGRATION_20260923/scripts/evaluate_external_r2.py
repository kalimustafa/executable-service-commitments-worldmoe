from src.evaluation.paper_economics import *
rows=[]
for year,trace in [(2024,'Frontier'),(2025,'Frontier'),(2024,'Eagle')]:
    proxy='requested_nodes' if trace=='Frontier' else 'registered'
    arr,reference,shared=configure(year,trace,proxy)
    for seed in [20260920,20260921,20260922]:
        for window in [12,24]:
            for policy in ['point','interval']:
                rows.append(run_homogeneous(arr,reference,shared,year=year,trace=trace,proxy=proxy,seed=seed,window=window,policy=policy))
                pd.DataFrame(rows).to_csv(ROOT/'results/external_r2_replications.csv',index=False)
print('ALL R2 EXTERNAL REPLICATIONS COMPLETE',flush=True)
