"""Create figure source rows from measured final checkpoint replays only."""
from pathlib import Path
import json
import shutil
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
WORKSPACE=BASE.parent
ENGINE=BASE/'engine/results'
CURRENT=WORKSPACE/'R2_JointDecision_20260927/results'
SOURCE=BASE/'source_data'
EPOCH={20260920:6,20260921:2,20260922:0}

def row(path, source_type):
    data=json.loads(path.read_text(encoding='utf-8'))
    keys=['seed','window_h','adjusted_cost_usd','reference_cost_usd','saving_usd','saving_pct',
          'capture_lower_fraction','certified_savings_ceiling_usd','shifted_IT_mwh','cvar95_usd',
          'late_IT_mwh','closing_it_mwh','max_fleet_recovery_mw','seconds','forecast_checkpoint_sha256']
    result={k:data.get(k) for k in keys}
    result['source_type']=source_type
    result['source_json']=str(path)
    return result

service=[]
for window in (12,16,24):
    for seed,epoch in EPOCH.items():
        base=f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf'
        path=(ENGINE/f'{base}_w{window}.json') if window!=24 or seed==20260920 else (CURRENT/f'{base}.json')
        if not path.exists():raise RuntimeError(f'Missing E01 measured replay: {path}')
        r=row(path,'new frozen WorldMoE replay' if path.parent==ENGINE else 'current frozen WorldMoE result')
        r['window_h']=window
        if window==24 and seed!=20260920:
            r['certified_savings_ceiling_usd']=132605.42084139131
        service.append(r)
windows=pd.DataFrame(service)
windows.to_csv(SOURCE/'fig3_service_windows.csv',index=False)
windows.to_csv(BASE/'experiments/E01_service_windows/per_seed_results.csv',index=False)

ablation=[]
for seed,epoch in EPOCH.items():
    base=f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf'
    r=row(CURRENT/f'{base}.json','current frozen WorldMoE result')
    r['variant']='full'
    ablation.append(r)
    for variant in ('no_opportunity','no_cost_bias','price_only'):
        matches=list(ENGINE.glob(f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_*_w24_e02_{variant}.json'))
        if len(matches)!=1:raise RuntimeError(f'Need one measured E02 {variant} seed {seed}, found {len(matches)}')
        r=row(matches[0],'new same-checkpoint signal intervention')
        r['variant']=variant
        ablation.append(r)
ab=pd.DataFrame(ablation)
scores=pd.read_csv(SOURCE/'worldmoe_forecast_scores.csv').set_index('seed')
ab['post_rt_pinball_usd_mwh']=ab.seed.map(scores.post_rt_pinball_usd_mwh)
ab.to_csv(SOURCE/'fig3_ablation.csv',index=False)
ab.to_csv(BASE/'experiments/E02_component_attribution/per_seed_results.csv',index=False)

copies={
    'worldmoe_three_seed_results.csv':'fig2_seed_economics.csv',
    'external_benchmarks_24h.csv':'fig3_benchmarks.csv',
    'worldmoe_information_by_seed.csv':'fig4_information_gate.csv',
    'worldmoe_grid_summary.csv':'fig5_electrical_constraints.csv',
    'worldmoe_event_sensitivity.csv':'fig6_event_influence.csv',
}
for src,dst in copies.items():shutil.copy2(SOURCE/src,SOURCE/dst)
print('E01',windows.groupby('window_h')[['saving_usd','saving_pct','capture_lower_fraction','late_IT_mwh']].mean().round(5).to_string())
print('E02',ab.groupby('variant')[['saving_usd','saving_pct','late_IT_mwh']].mean().round(5).to_string())
