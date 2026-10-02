"""2024 hindsight-price diagnostic; never a feasible forecasting candidate."""
from src.runtime import ROOT,PAPER
from src.data.access_guard import install_guard,read_development_parquet
install_guard()
import json
import numpy as np
import pandas as pd
from src.evaluation.economic_dev_eval import configure_development,run_one
from vendor.r2_market import QUANTILES

arr,reference=configure_development()
base=read_development_parquet(PAPER/'experiments/gate_hourly_panel.parquet',columns=['timestamp_utc','date','zone','da_lbmp','spread','rt_integrated'])
base=base[base.date>='2024-07-01'].copy()
pieces=[]
for publication in ['pre_DA','post_DA']:
    f=base.copy();f['gate']=publication
    for tau in QUANTILES:
        f[f'rt_q{tau:.2f}']=f.rt_integrated
        f[f'dart_q{tau:.2f}']=f.spread
    pieces.append(f)
frame=pd.concat(pieces,ignore_index=True)
rows=run_one('HINDSIGHT_PRICE_DIAGNOSTIC',0,frame,arr,reference)
reference_cost=json.loads((ROOT/'results/cost_scope.json').read_text())['reference_operating_cost_usd']
for row in rows:
    row['net_operating_cost_usd']=reference_cost-row['value_usd']
    row['savings_share_reference_pct']=100*row['value_usd']/reference_cost
    row['usable_forecaster']=False
    row['interpretation']='infeasible hindsight-price diagnostic with unchanged greedy executor, not a rigorous optimum or guaranteed attainable improvement'
pd.DataFrame(rows).to_csv(ROOT/'results/hindsight_price_diagnostic_2024.csv',index=False)
print(pd.DataFrame(rows)[['window_h','policy','value_usd','net_operating_cost_usd','savings_share_reference_pct']].to_string(index=False),flush=True)
