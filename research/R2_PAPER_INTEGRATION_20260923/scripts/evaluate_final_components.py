from src.evaluation.paper_economics import *
arr,reference,shared=configure(2025)
rows=[]
for physical,ablation in [('price_only','normal'),('reservation','normal'),('particle_prior','normal'),('particle_trailing4','force_gate_0')]:
    for window in [12,24]:
        for policy in ['point','interval']:
            rows.append(run_homogeneous(arr,reference,shared,year=2025,trace='Eagle',proxy='registered',seed=20260920,
                window=window,policy=policy,physical=physical,ablation=ablation))
            pd.DataFrame(rows).to_csv(ROOT/'results/final_component_ablations.csv',index=False)
print('FINAL COMPONENT ABLATIONS COMPLETE',flush=True)
