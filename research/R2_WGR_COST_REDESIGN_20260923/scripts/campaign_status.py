from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json
import pandas as pd
from src.evaluation.economic_dev_eval import choose_policy

REFERENCE=3343.4716892558006
TARGET=4346.513196032541
files=list((ROOT/'results/economic_trajectories').glob('C*_normal_24h_*_fp32v1_metrics.json'))
records=[json.loads(p.read_text()) for p in files]
rows=[]
for model,group in pd.DataFrame(records).groupby('model'):
    chosen=choose_policy(group.to_dict('records'))
    rows.append(dict(model=model,seed=20260920,window_h=24,policy=chosen['policy'],value_usd=chosen['value_usd'],
        savings_gain_pct=100*(chosen['value_usd']/REFERENCE-1),gap_to_target_usd=TARGET-chosen['value_usd'],
        zero_late_feasible=chosen['late_it_mwh']<1e-8,closing_work_mwh=chosen['closing_work_mwh'],
        selection_score=chosen['selection_score'],target_met=chosen['value_usd']>=TARGET and chosen['late_it_mwh']<1e-8))
frame=pd.DataFrame(rows).sort_values('value_usd',ascending=False)
frame.to_csv(ROOT/'results/campaign_savings_2024.csv',index=False)
(ROOT/'results/campaign_status.json').write_text(json.dumps(dict(target_savings_usd=TARGET,fixed_reference_savings_usd=REFERENCE,
    target_met=bool(frame.target_met.any()),best=frame.iloc[0].to_dict(),status='target_measured' if frame.target_met.any() else 'continue',
    interpretation='30% greater savings, not 30% lower total cost; 2024 development only',other_seeds_started=False),indent=2))
print(frame.to_string(index=False),flush=True)
