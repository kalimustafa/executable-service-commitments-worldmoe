from src.runtime import ROOT,FAILED
from src.data.access_guard import install_guard
install_guard()
import json
import numpy as np
import pandas as pd
from src.evaluation.economic_dev_eval import configure_development
from vendor import economic_core as core

arr,reference=configure_development()
reference_cash,reference_grid=core.cash(reference)
reference_terminal=float(core.DT*np.sum(reference[core.N:]*core.k*np.max(core.c['pue_max'])*core.train_rt))
reference_cost=float(reference_cash.sum()+reference_terminal)
frame=pd.read_csv(FAILED/'results/economic_dev_metrics_2024.csv')
rows=[]
for row in frame.itertuples():
    total=reference_cost-row.value_usd
    rows.append(dict(model=row.model,seed=row.seed,window_h=row.window_h,policy=row.policy,
        reference_operating_cost_usd=reference_cost,net_operating_cost_usd=total,
        savings_usd=row.value_usd,savings_share_of_reference_pct=100*row.value_usd/reference_cost,
        late_it_mwh=row.late_it_mwh,closing_work_mwh=row.closing_work_mwh))
pd.DataFrame(rows).to_csv(ROOT/'results/existing_cost_denominators_2024.csv',index=False)
data=dict(reference_operating_cost_usd=reference_cost,reference_cash_cost_usd=float(reference_cash.sum()),
    reference_terminal_work_liability_usd=reference_terminal,reference_grid_import_mwh=float(core.DT*reference_grid.sum()),
    definition='same executor cash cost plus original fees and terminal-work liability; net cost = reference cost - recorded value',
    target='substantial reduction in net operating cost, not model compute',locked_targets_accessed=False)
(ROOT/'results/cost_scope.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2),flush=True)
print(pd.DataFrame(rows).query('window_h in [12,24]').groupby(['model','window_h','policy'])[['net_operating_cost_usd','savings_usd','savings_share_of_reference_pct']].mean().to_string(),flush=True)
