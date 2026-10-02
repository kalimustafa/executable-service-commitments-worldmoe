from src.runtime import ROOT,FAILED
from src.data.access_guard import install_guard,read_development_parquet
install_guard()
import json
import argparse
import numpy as np
import pandas as pd
from src.evaluation.economic_dev_eval import configure_development,run_one,choose_policy

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['C0_Repaired','C1_ProfileResidual']);args=parser.parse_args()
    arr,reference=configure_development();new=[]
    cost=json.loads((ROOT/'results/cost_scope.json').read_text())['reference_operating_cost_usd']
    for name in args.models:
        path=ROOT/'results/predictions_2024'/f'{name}_seed20260920_native.parquet'
        frame=read_development_parquet(path)
        new+=run_one(name,20260920,frame,arr,reference,windows=(12,24))
    fresh=pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'results/economic_trajectories').glob('C*_normal_*_fp32v1_metrics.json')])
    old=pd.read_csv(FAILED/'results/economic_dev_metrics_2024.csv')
    old=old[old.model.isin(['A0_Base','A1_Shallow','RASC'])&old.seed.isin([0,20260920])&old.window_h.isin([12,24])]
    table=pd.concat([old,fresh],ignore_index=True)
    table['reference_operating_cost_usd']=cost;table['net_operating_cost_usd']=cost-table.value_usd
    table.to_csv(ROOT/'results/pilot_economic_metrics_2024.csv',index=False)
    selected=[]
    for (model,window),f in table.groupby(['model','window_h']):selected.append(choose_policy(f.to_dict('records')))
    pd.DataFrame(selected).to_csv(ROOT/'results/pilot_policy_selection_2024.csv',index=False)
    # Strongest existing zero-late control, chosen identically for both new
    # candidates. Selection does not choose a favorable seed.
    checks=[]
    for window in [12,24]:
        controls=table[table.model.isin(['A0_Base','A1_Shallow','RASC'])&table.window_h.eq(window)&table.late_it_mwh.lt(1e-8)]
        comparator=controls.sort_values('net_operating_cost_usd').iloc[0]
        for name in sorted(fresh.model.unique()):
            row=pd.DataFrame(selected).query('model == @name and window_h == @window').iloc[0]
            difference=comparator.net_operating_cost_usd-row.net_operating_cost_usd
            checks.append(dict(model=name,seed=20260920,window_h=window,selected_policy=row.policy,
                comparator_model=comparator.model,comparator_policy=comparator.policy,
                comparator_net_cost_usd=comparator.net_operating_cost_usd,net_operating_cost_usd=row.net_operating_cost_usd,
                cost_reduction_usd=difference,cost_reduction_pct=100*difference/comparator.net_operating_cost_usd,
                selection_score_delta=row.selection_score-comparator.selection_score,
                late_it_mwh=row.late_it_mwh,closing_work_mwh=row.closing_work_mwh))
    pd.DataFrame(checks).to_csv(ROOT/'results/pilot_cost_success_check.csv',index=False)
    print(pd.DataFrame(checks).to_string(index=False),flush=True)

if __name__=='__main__':main()
