from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
records=pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'results/paper_runs').glob('*.json')])
assert not records.name.duplicated().any()
mixed=records[records.name.str.contains('_R2_seed20260920_mix')]
mixed.to_csv(ROOT/'results/r2_mixed_contracts.csv',index=False)
primary=records[(records.model=='R2')&(records.year==2025)&(records.trace=='Eagle')&
    (records.physical_branch=='particle_trailing4')&(records.ablation=='normal')&records.window_h.isin(range(6,25))]
primary.to_csv(ROOT/'results/full_three_seed_window_matrix.csv',index=False)
expected={
    'causal_RT_lag_control':(3,len(records[records.name.str.startswith('control_RT_lag_')])),
    'matched_base_price_cost_control':(4,len(records[records.name.str.startswith('control_Base_price_cost_')])),
    'main_window_sweep':(114,len(primary)),
    'new_R2_mixed_contracts':(68,len(mixed)),
    'original_external_windows':(152,len(records[records.name.str.startswith('external_')&records.proxy.isin(['registered','requested_nodes'])&records.policy.isin(['myopic_da','persistent_da'])])),
    'original_external_mixed':(64,len(records[records.name.str.startswith('external_')&records.proxy.isin(['registered','requested_nodes'])&records.name.str.contains('_mix')])),
    'original_resource_field_windows':(4,len(records[records.name.str.startswith('external_')&(records.proxy=='field7_sensitivity')&records.policy.isin(['myopic_da','persistent_da'])])),
    'original_resource_field_mixed':(4,len(records[records.name.str.startswith('external_')&(records.proxy=='field7_sensitivity')&records.name.str.contains('_mix')])),
    'baseline_windows':(95,len(records[records.name.str.startswith('baseline_')])),
    'new_R2_external_homogeneous':(36,len(pd.read_csv(ROOT/'results/external_r2_replications.csv'))),
    'parameter_sensitivities':(36,len(pd.read_csv(ROOT/'results/original_parameter_sensitivities.csv'))),
    'component_ablations':(16,len(pd.read_csv(ROOT/'results/final_component_ablations.csv'))),
    'missing_context_replacement':(4,len(pd.read_csv(ROOT/'results/missing_context_replacement.csv'))),
    'missing_context_exclusion':(12,len(pd.read_csv(ROOT/'results/missing_context_day_exclusion.csv'))),
    'observed_settlement_tail':(12,len(pd.read_csv(ROOT/'results/settlement_tail_2026.csv')))}
status={k:dict(expected=n,observed=m,complete=n==m) for k,(n,m) in expected.items()}
complete=all(s['complete'] for s in status.values())
(ROOT/'results/PAPER_SUPPORT_STATUS.json').write_text(json.dumps(dict(status='complete' if complete else 'running',requirements=status),indent=2))
main=records[(records.year==2025)&(records.trace=='Eagle')&(records.window_h==24)&(records.policy=='interval')&
    (records.model=='R2')&(records.physical_branch=='particle_trailing4')&(records.ablation=='normal')]
summary=dict(seeds=main.seed.tolist(),mean_cost_usd=float(main.net_operating_cost_usd.mean()),
    mean_savings_usd=float(main.value_usd.mean()),mean_savings_pct=float(main.saving_pct.mean()),
    cost_range_usd=[float(main.net_operating_cost_usd.min()),float(main.net_operating_cost_usd.max())],
    zero_late_all_seeds=bool((main.late_IT_mwh<1e-8).all()),zero_closing_all_seeds=bool((main.final_queue_IT_mwh<1e-8).all()),
    primary_seed=main[main.seed==20260920].iloc[0].dropna().to_dict())
(ROOT/'results/PAPER_PRIMARY_RESULT.json').write_text(json.dumps(summary,indent=2))
print('SUPPORT', 'COMPLETE' if complete else 'RUNNING',json.dumps({k:v for k,v in status.items() if not v['complete']}),flush=True)
print('PRIMARY',summary['mean_savings_pct'],summary['zero_late_all_seeds'],summary['zero_closing_all_seeds'],flush=True)
