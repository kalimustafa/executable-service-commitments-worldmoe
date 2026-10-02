from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json,hashlib
import numpy as np
import yaml
from src.evaluation.economic_dev_eval import configure_development
from vendor import economic_core as core

configure_development()
folder=ROOT/'results/economic_trajectories'
names=['C0_Repaired_seed20260920_normal_24h_robust_fp32v1','C7_IssueSiteCost_seed20260920_normal_24h_robust_fp32v1']
records=[json.loads((folder/(name+'_metrics.json')).read_text()) for name in names]
trajectories=[]
for name in names:
    with np.load(folder/(name+'_trajectory.npz'),allow_pickle=False) as value:
        trajectories.append({k:value[k] for k in ['compute','grid']})
old,new=trajectories
old_load=core.pue*(core.idle+core.k*old['compute']);new_load=core.pue*(core.idle+core.k*new['compute'])
old_used=old_load-old['grid'];new_used=new_load-new['grid']
settlement=float(core.DT*np.sum(core.rt*(old['grid']-new['grid'])))
curtailment=float(core.DT*np.sum(new_used-old_used))
fees=records[0]['fee_usd']-records[1]['fee_usd']
gain=records[1]['value_usd']-records[0]['value_usd']
assert abs(settlement+curtailment+fees-gain)<1e-7
decomposition=dict(extra_savings_usd=gain,avoided_grid_settlement_cost_usd=settlement,
    avoided_existing_PV_curtailment_cost_usd=curtailment,fee_reduction_usd=fees,
    additional_PV_used_mwh=curtailment,facility_energy_change_mwh=float(core.DT*np.sum(new_load-old_load)),
    grid_import_reduction_mwh=float(core.DT*np.sum(old['grid']-new['grid'])),
    mean_delay_old_hours=records[0]['mean_delay_hours'],mean_delay_new_hours=records[1]['mean_delay_hours'],
    p95_delay_old_hours=records[0]['p95_delay_hours'],p95_delay_new_hours=records[1]['p95_delay_hours'],
    source='decomposition of existing measured schedules; no new optimization or candidate')
(ROOT/'results/savings_decomposition_2024.json').write_text(json.dumps(decomposition,indent=2))
report=ROOT/'reports/SAVINGS_30PCT_RESULT.md'
with report.open('a',encoding='utf-8') as stream:
    stream.write(f'\n## 已测轨迹的节省分解与限制\n\n新增节省 ${gain:.2f} = 避免的 grid settlement 成本 ${settlement:.2f} + 原 PV curtailment 费用减少 ${curtailment:.2f} + 手续费减少 ${fees:.2f}。该分解来自同一已执行轨迹，不是新的优化目标。新增利用 PV {curtailment:.3f} MWh，网购电减少 {decomposition["grid_import_reduction_mwh"]:.3f} MWh。\n\n')
    stream.write(f'平均等待由 {records[0]["mean_delay_hours"]:.3f} h 增至 {records[1]["mean_delay_hours"]:.3f} h，P95 由 {records[0]["p95_delay_hours"]:.2f} h 增至 {records[1]["p95_delay_hours"]:.2f} h；仍满足相同 24h 合同，late work 为 0。不能把“硬服务约束不变”写成“等待时间不变”。\n\n')
    stream.write('成本头名义 90% 区间的实际 coverage 为 pre-DA 75.65%、post-DA 86.46%；PV/PUE 点估计的条件变换仍有不确定性校准不足，不能把这些区间当作已经验证的 90% 风险保证。+54.30% 是已测开发回放成本结果，不是外部泛化或概率保证。\n')
path=ROOT/'configs/continued_goal_30pct.yaml';goal=yaml.safe_load(path.read_text())
goal.update(status='measured_target_met_search_stopped',selected_model='C7_IssueSiteCost',measured_savings_usd=records[1]['value_usd'],
            measured_savings_gain_pct=100*gain/records[0]['value_usd'],additional_seeds_started=False)
path.write_text(yaml.safe_dump(goal,allow_unicode=True,sort_keys=False),encoding='utf-8')
with (ROOT/'REVISED_GOAL.md').open('a',encoding='utf-8') as stream:
    stream.write('\n## Completion\n\nThe latest user-defined savings goal was measured at +54.2962% for the primary seed and 24h interval policy. New search has stopped. No decision-loss amendment was used and no additional seed started. See results/SAVINGS_TARGET_MET.json and reports/SAVINGS_30PCT_RESULT.md for exact scope and limitations.\n')
(ROOT/'SOURCE_PATHS.yaml').write_text(yaml.safe_dump(dict(selected_model='C7_IssueSiteCost',
    exact_source_provenance='results/issued_physical_provenance.json',selected_market_checkpoint='checkpoints/C0_Repaired_seed20260920_best.ckpt',
    selected_calibrator='checkpoints/C0_Repaired_seed20260920_calibration.json',physical_forecast_cache='cache/world_physical_forecasts.npz',
    source_code_manifest='frozen_candidate/source_manifest.json',source_years=[2021,2022,2023,2024],
    original_project_access='READ ONLY',locked_data_accessed=False,large_W1_checkpoint_copied=False,
    counterfactual_issued_prior_contract=True),allow_unicode=True,sort_keys=False),encoding='utf-8')
print(json.dumps(decomposition,indent=2),flush=True)
