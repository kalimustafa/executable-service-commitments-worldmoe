"""Summarise completed raw runs without recalculating their economic account."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
files=sorted((ROOT/'results').glob('WindowExtension20261031_CdfR2*_coupled_cdf_w*.json'))
rows=[]
for p in files:
    if p.name.endswith('_run_manifest.json'):continue
    r=json.loads(p.read_text());
    if r.get('evidence_type')!='new frozen-checkpoint evaluation':continue
    rows.append({k:r[k] for k in ['name','seed','window_h','saving_usd','reference_cost_usd','adjusted_cost_usd','late_IT_mwh','closing_it_mwh','terminal_liability_difference_usd','shifted_IT_mwh','fee_usd','pv_curtailment_mwh','grid_import_mwh','cvar95_usd','seconds','forecast_checkpoint_sha256','run_date','forecast_horizon_slots','planning_horizon_slots','evidence_type']})
if not rows:raise SystemExit('No completed new runs yet')
with (ROOT/'results/new_window_raw_runs.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=[]
for window in sorted({r['window_h'] for r in rows}):
    s=[r for r in rows if r['window_h']==window]
    groups.append({'window_h':window,'n':len(s),'saving_mean_usd':float(np.mean([r['saving_usd'] for r in s])),
        'saving_min_usd':min(r['saving_usd'] for r in s),'saving_max_usd':max(r['saving_usd'] for r in s),
        'maximum_reported_late_IT_mwh':max(r['late_IT_mwh'] for r in s),
        'mean_reported_late_IT_mwh':float(np.mean([r['late_IT_mwh'] for r in s])),
        'minimum_reported_late_IT_mwh':min(r['late_IT_mwh'] for r in s),
        'maximum_closing_IT_mwh':max(r['closing_it_mwh'] for r in s),
        'mean_closing_IT_mwh':float(np.mean([r['closing_it_mwh'] for r in s])),
        'mean_daily_mtm_loss_cvar95_usd':float(np.mean([r['cvar95_usd'] for r in s])),
        'min_daily_mtm_loss_cvar95_usd':min(r['cvar95_usd'] for r in s),
        'max_daily_mtm_loss_cvar95_usd':max(r['cvar95_usd'] for r in s),
        'mean_shifted_IT_mwh':float(np.mean([r['shifted_IT_mwh'] for r in s])),
        'mean_execution_change_charge_usd':float(np.mean([r['fee_usd'] for r in s])),
        'mean_grid_import_mwh':float(np.mean([r['grid_import_mwh'] for r in s])),
        'service_status':'raw deadline failure; cash-only, excluded from feasible-value and capture claims' if max(r['late_IT_mwh'] for r in s)>1e-6 else 'raw residuals below 1e-6 IT-MWh; exact audit supplied separately',
        'evidence_type':'new frozen-checkpoint evaluation; raw numerical result, exact audit separate'})
with (ROOT/'results/new_window_raw_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
combined=list(rows);combined_groups=list(groups)
for window in [12,16]:
    history=[]
    for p in sorted((ROOT/'historical_reference').glob(f'CdfR2*_w{window}.json')):
        r=json.loads(p.read_text());r.update(run_date='historical archive',forecast_horizon_slots=96,planning_horizon_slots=96,evidence_type='recovered original historical paired policy result')
        history.append({k:r[k] for k in rows[0]})
    combined.extend(history)
    if history:
        s=history
        combined_groups.append({'window_h':window,'n':len(s),
            'saving_mean_usd':float(np.mean([r['saving_usd'] for r in s])),
            'saving_min_usd':min(r['saving_usd'] for r in s),'saving_max_usd':max(r['saving_usd'] for r in s),
            'maximum_reported_late_IT_mwh':max(r['late_IT_mwh'] for r in s),
            'mean_reported_late_IT_mwh':float(np.mean([r['late_IT_mwh'] for r in s])),
            'minimum_reported_late_IT_mwh':min(r['late_IT_mwh'] for r in s),
            'maximum_closing_IT_mwh':max(r['closing_it_mwh'] for r in s),
            'mean_closing_IT_mwh':float(np.mean([r['closing_it_mwh'] for r in s])),
            'mean_daily_mtm_loss_cvar95_usd':float(np.mean([r['cvar95_usd'] for r in s])),
            'min_daily_mtm_loss_cvar95_usd':min(r['cvar95_usd'] for r in s),
            'max_daily_mtm_loss_cvar95_usd':max(r['cvar95_usd'] for r in s),
            'mean_shifted_IT_mwh':float(np.mean([r['shifted_IT_mwh'] for r in s])),
            'mean_execution_change_charge_usd':float(np.mean([r['fee_usd'] for r in s])),
            'mean_grid_import_mwh':float(np.mean([r['grid_import_mwh'] for r in s])),
            'service_status':'raw residuals below 1e-6 IT-MWh; exact audit supplied in historical extension',
            'evidence_type':'recovered original historical paired policy result; not a new evaluation'})
with (ROOT/'results/seven_window_raw_runs.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(combined,key=lambda r:(r['window_h'],r['seed'])))
with (ROOT/'results/seven_window_raw_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(sorted(combined_groups,key=lambda r:r['window_h']))
print(json.dumps(groups,indent=2))
