"""Record the actual-economics gate before authorizing additional seed runs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=sorted((root/'results').glob('JointR2_Eagle_2024_s20260920_*.json'))
rows=[json.loads(p.read_text()) for p in files]
baseline={r.get('allocation_mode','marginal'):r for r in rows if r['model_epoch']==0}
initial=json.loads((root/'cache/market2024_s20260920_ep00.json').read_text())['metrics']
comparisons=[]
for row in rows:
    if row['model_epoch']==0:continue
    mode=row.get('allocation_mode','marginal');base=baseline[mode]
    forecast=json.loads((root/'cache'/f"market2024_s20260920_ep{row['model_epoch']:02d}.json").read_text())['metrics']
    comparisons.append(dict(name=row['name'],epoch=row['model_epoch'],allocation_mode=mode,saving_usd=row['saving_usd'],baseline_saving_usd=base['saving_usd'],incremental_saving_usd=row['saving_usd']-base['saving_usd'],relative_change_pct=100*(row['saving_usd']/base['saving_usd']-1),capture_lower_fraction=row['capture_lower_fraction'],late_it_mwh=row['late_IT_mwh'],closing_it_mwh=row['closing_it_mwh'],prediction_guard_passed=forecast['price_score']<=initial['price_score']*1.02))
passed=[r for r in comparisons if r['incremental_saving_usd']>0 and r['late_it_mwh']<1e-8 and r['closing_it_mwh']<1e-8 and r['prediction_guard_passed']]
result=dict(status='primary_seed_complete',seed=20260920,epochs=20,training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_year=2024,decision_gradient_reaches_R2_adapter=True,comparisons=comparisons,secondary_seed_effect_gate_passed=bool(passed),other_seeds_started=False,year2025_new_model_evaluated=False,original_target_2025_savings_usd=106084.33667311305,manuscript_rewrite_triggered=False,interpretation='Proxy loss improvement alone does not authorize additional seeds or an 80% claim; require actual same-executor annual economic improvement.',efficiency=json.loads((root/'logs/efficiency_summary.json').read_text()))
(root/'results/primary_round_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
