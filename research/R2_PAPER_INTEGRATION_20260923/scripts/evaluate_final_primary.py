from src.evaluation.paper_economics import *
arr,reference,shared=configure(2025)
rows=[]
for seed in [20260920,20260921,20260922]:
    for window in [12,24]:
        for policy in ['point','interval']:
            rows.append(run_homogeneous(arr,reference,shared,year=2025,trace='Eagle',proxy='registered',seed=seed,window=window,policy=policy))
pd.DataFrame(rows).to_csv(ROOT/'results/multiseed_economics_2025.csv',index=False)
print('FINAL PRIMARY THREE SEEDS COMPLETE',flush=True)
