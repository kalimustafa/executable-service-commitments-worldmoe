"""Replace every 2026-09-27 World-ME observation with measured final WorldMoE.

The retained comparator pipelines are preserved as 2025 complete-pipeline
references. Missing final-WorldMoE windows/conditions stay missing; no value
is interpolated or copied from the earlier World-ME policy.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'NatureEnergy_Revised_Comparisons_20260927/source_data'
NEW=BASE/'source_data'
E04=BASE/'experiments/E04_frontier_nonpreemptive'

def write(frame,name):
    frame.to_csv(NEW/name,index=False)
    print(name,len(frame))

old=pd.read_csv(OLD/'fluid_model_window_summary.csv')
old=old[old.model.ne('World-ME')].copy()
old['provenance']='2026-09-27 frozen comparator pipeline'
current=pd.read_csv(NEW/'fig3_service_windows.csv')
rows=[]
for window,group in current.groupby('window_h'):
    ref=float(group.reference_cost_usd.iloc[0])
    rows.append(dict(model='World-ME',policy='final',window_h=int(window),
        saving=group.saving_usd.mean(),saving_min=group.saving_usd.min(),saving_max=group.saving_usd.max(),
        cost=group.adjusted_cost_usd.mean(),cost_min=group.adjusted_cost_usd.min(),cost_max=group.adjusted_cost_usd.max(),
        late=group.late_IT_mwh.mean(),late_max=group.late_IT_mwh.max(),
        closing=group.closing_it_mwh.mean(),closing_max=group.closing_it_mwh.max(),
        shifted=group.shifted_IT_mwh.mean(),cvar=group.cvar95_usd.mean(),
        cvar_min=group.cvar95_usd.min(),cvar_max=group.cvar95_usd.max(),
        n=len(group),reference=ref,provenance='current final WorldMoE frozen 2024-selected checkpoints'))
fluid=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
write(fluid,'0927_adapted_fluid_comparisons.csv')

mixed=pd.read_csv(OLD/'mixed_deadline_all_comparators.csv')
mixed=mixed[mixed.model.ne('World-ME')].copy()
mixed['provenance']='2026-09-27 frozen comparator pipeline'
e03=pd.read_csv(NEW/'extfig1_heterogeneous_deadlines.csv')
mixed_rows=[]
for pathway,recourse in [('H0 homogeneous 24 h request fifo',False),('H1 mixed deadline aware WorldMoE edf',True)]:
    group=e03[e03.pathway.eq(pathway)]
    mixed_rows.append(dict(model='World-ME',policy='final',shares='[0.25, 0.5, 0.25]',urgent_share=.25,
        recourse=recourse,saving=group.annual_saving_usd.mean(),late=group.late_work_it_mwh.mean(),
        source='current final WorldMoE mixed-deadline replay' if recourse else 'current final WorldMoE 24 h book audited under mixed deadlines',
        provenance='current final WorldMoE; H0 is service-infeasible' if not recourse else 'current final WorldMoE; H1 is service-feasible'))
mixed=pd.concat([mixed,pd.DataFrame(mixed_rows)],ignore_index=True)
write(mixed,'0927_adapted_mixed_comparisons.csv')

jobs=pd.read_csv(OLD/'discrete_model_window_comparison.csv')
jobs=jobs[jobs.model.ne('World-ME')].copy()
jobs['provenance']='2026-09-27 frozen comparator pipeline'
e04=pd.read_csv(NEW/'extfig1_frontier.csv')
job_rows=[]
for r in e04.itertuples():
    result=json.loads((E04/f'Frontier_2025_w{int(r.window_h)}_summary.json').read_text())
    job_rows.append(dict(model='World-ME',window_h=int(r.window_h),saving=r.saving_usd,
        late=int(r.late_jobs),closing=int(r.closing_jobs),cvar=r.cvar95_usd,
        shifted=result['shifted_IT_mwh'],jobs=int(r.jobs),source=r.source_json,
        job_file=str(E04/f'Frontier_2025_w{int(r.window_h)}_worldmoe_jobs.parquet'),
        provenance='current final WorldMoE causal 24 h slot-score job replay'))
jobs=pd.concat([jobs,pd.DataFrame(job_rows)],ignore_index=True)
write(jobs,'0927_adapted_frontier_comparisons.csv')
old_cdf=pd.read_csv(OLD/'Fig6_completion_cdf_all_models.csv')
old_cdf=old_cdf[old_cdf.model.ne('World-ME')].copy()
current_job_file=E04/'Frontier_2025_w48_worldmoe_jobs.parquet'
completions=pd.read_parquet(current_job_file,columns=['submit_time','completion_seconds'])
duration_h=np.sort((completions.completion_seconds.to_numpy()-completions.submit_time.to_numpy())/3600)
grid=np.arange(0,48.5,.5)
new_cdf=pd.DataFrame({'model':'World-ME','hours':grid,
    'completed_pct':100*np.searchsorted(duration_h,grid,side='right')/len(duration_h),
    'source':str(current_job_file)})
write(pd.concat([old_cdf,new_cdf],ignore_index=True),'0927_adapted_frontier_completion_cdf.csv')

old_forecast=pd.read_csv(OLD/'apic_and_forecast_controls.csv')
old_forecast=old_forecast[~old_forecast.model.str.startswith('A')].copy()
old_forecast['provenance']='2026-09-27 frozen comparator forecast'
ab=pd.read_csv(NEW/'fig3_ablation.csv')
forecast=pd.read_csv(NEW/'worldmoe_forecast_scores.csv').set_index('seed')
labels={'full':'Full WorldMoE','no_opportunity':'No opportunity score',
    'no_cost_bias':'No task cost bias','price_only':'Market-mean signal'}
ab_rows=[]
for variant,g in ab.groupby('variant'):
    values=g.seed.map(forecast.post_rt_coverage90_pct)
    ab_rows.append(dict(model='Final_'+variant,label=labels[variant],saving=g.saving_usd.mean(),
        saving_min=g.saving_usd.min(),saving_max=g.saving_usd.max(),
        cvar=g.cvar95_usd.mean(),cvar_min=g.cvar95_usd.min(),cvar_max=g.cvar95_usd.max(),
        pinball=g.post_rt_pinball_usd_mwh.mean(),coverage90=values.mean(),
        tail_coverage90=np.nan,reassign=np.nan,changed_jobs=np.nan,job_saving=np.nan,
        job_late=np.nan,economic_n=len(g),provenance='current final WorldMoE fixed-checkpoint signal intervention'))
controls=pd.concat([pd.DataFrame(ab_rows),old_forecast],ignore_index=True)
write(controls,'0927_adapted_forecast_controls.csv')

rasc=fluid[(fluid.model.eq('RASC-ME'))&(fluid.policy.eq('interval'))&(fluid.window_h.isin([12,16,24]))].set_index('window_h')
world=fluid[(fluid.model.eq('World-ME'))&(fluid.policy.eq('final'))].set_index('window_h')
contrast=[]
for window in (12,16,24):
    contrast.append(dict(window_h=window,worldmoe_saving_usd=world.loc[window,'saving'],
        rasc_saving_usd=rasc.loc[window,'saving'],
        difference_usd=world.loc[window,'saving']-rasc.loc[window,'saving'],
        paired_interval_available=window==24))
write(pd.DataFrame(contrast),'0927_adapted_window_contrasts.csv')

meta=dict(source_paper='NatureEnergy_Integrated_20260927.docx',
    retained_comparator_scope='Frozen old-paper complete pipelines, same 2025 calendar and earliest-service reference; distinct methods and calibration',
    current_worldmoe_windows=[12,16,24],current_frontier_windows=[24,30,48],
    current_mixed_contract='25/50/25% 4/12/24 h only',
    missing_worldmoe_cells='Intentionally blank; no interpolation or earlier World-ME values retained')
(NEW/'0927_adapted_comparison_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
for name,out in [('Fig2_hourly_workload.csv','0927_hourly_workload.csv'),('Fig2_capacity_floor.csv','0927_capacity_floor.csv')]:
    shutil.copy2(OLD/name,NEW/out)
