"""Collect measured all-flexible counterfactual utilization cases."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE/'engine/results'
source=BASE/'source_data'
base='CdfR2_Eagle_2025_s20260920_ep06_coupled_cdf_w24'
with np.load(BASE.parent/'external_validation/data/forcing_2025.npz') as f:
    nominal_util=float(np.sum(f['eagle_arrivals']*f['k'])/(len(f['times'])*np.sum(f['k'])))
rows=[]
for target,suffix in [(None,''),(.4,'_util40_flex100'),(.6,'_util60_flex100'),(.8,'_util80_flex100')]:
    path=ROOT/f'{base}{suffix}.json'
    if not path.exists():continue
    data=json.loads(path.read_text())
    util=target if target is not None else nominal_util
    rows.append(dict(target_utilization=util,flexible_fraction=1.0,
        counterfactual=target is not None,saving_usd=data['saving_usd'],
        saving_pct=data['saving_pct'],reference_cost_usd=data['reference_cost_usd'],
        policy_cost_usd=data['adjusted_cost_usd'],it_energy_mwh=data['IT_energy_mwh'],
        saving_per_flexible_mwh_usd=data['saving_usd']/data['IT_energy_mwh'],
        shifted_IT_mwh=data['shifted_IT_mwh'],cvar95_usd=data['cvar95_usd'],
        late_IT_mwh=data['late_IT_mwh'],closing_it_mwh=data['closing_it_mwh'],
        max_fleet_recovery_mw=data['max_fleet_recovery_mw'],seconds=data['seconds'],
        source_json=str(path)))
frame=pd.DataFrame(rows)
frame.to_csv(HERE/'results.csv',index=False)
frame.to_csv(source/'extfig2_workload_scaling.csv',index=False)
print(frame[['target_utilization','saving_usd','saving_per_flexible_mwh_usd','late_IT_mwh']].to_string(index=False))
