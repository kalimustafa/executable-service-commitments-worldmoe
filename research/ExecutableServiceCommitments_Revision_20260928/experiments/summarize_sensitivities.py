"""Summarize measured no-PV and current-action grid cases."""
from pathlib import Path
import json
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'engine/results'
OUT=BASE/'source_data'
base='CdfR2_Eagle_2025_s20260920_ep06_coupled_cdf_w24'

def load(name):
    path=ROOT/name
    if not path.exists():return None
    return json.loads(path.read_text())

grid=[]
for step,suffix in [(.05,'_grid0.05'),(.025,''),(.0125,'_grid0.0125')]:
    data=load(base+suffix+'.json')
    if data is None:continue
    grid.append(dict(action_grid_work_units=step,saving_usd=data['saving_usd'],
        saving_pct=data['saving_pct'],seconds=data['seconds'],late_IT_mwh=data['late_IT_mwh'],
        closing_it_mwh=data['closing_it_mwh'],source_json=str(ROOT/(base+suffix+'.json'))))
frame=pd.DataFrame(grid)
if len(frame)==3:
    finest=frame.loc[frame.action_grid_work_units.eq(.0125),'saving_usd'].iloc[0]
    frame['difference_from_fine_usd']=frame.saving_usd-finest
frame.to_csv(OUT/'extfig2_numerical_convergence.csv',index=False)
frame.to_csv(BASE/'experiments/E08_numerical_convergence/results.csv',index=False)

physical=[]
for fraction,suffix in [(0,'_pv0'),(.5,'_pv0.5'),(1,'')]:
    data=load(base+suffix+'.json')
    if data is None:continue
    physical.append(dict(pv_fraction_of_nominal=fraction,saving_usd=data['saving_usd'],
        saving_pct=data['saving_pct'],reference_cost_usd=data['reference_cost_usd'],
        policy_cost_usd=data['adjusted_cost_usd'],shifted_IT_mwh=data['shifted_IT_mwh'],
        cvar95_usd=data['cvar95_usd'],late_IT_mwh=data['late_IT_mwh'],
        closing_it_mwh=data['closing_it_mwh'],source_json=str(ROOT/(base+suffix+'.json'))))
pv=pd.DataFrame(physical)
pv.to_csv(OUT/'extfig2_physical_sensitivity.csv',index=False)
pv.to_csv(BASE/'experiments/E06_physical_sensitivity/results.csv',index=False)
print('Grid\n',frame[['action_grid_work_units','saving_usd','seconds']].to_string(index=False))
print('PV\n',pv[['pv_fraction_of_nominal','saving_usd','saving_pct']].to_string(index=False))
