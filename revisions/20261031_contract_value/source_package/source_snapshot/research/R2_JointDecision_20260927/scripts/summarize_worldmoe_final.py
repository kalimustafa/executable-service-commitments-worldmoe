"""Aggregate completed WorldMoE decision-module replications and extensions."""
import sys, json, hashlib, shutil
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT, EXTENSION, WORKSPACE

OUT=WORKSPACE/'WorldMoE_Revision_20260928'
SEEDS=[20260920,20260921,20260922]
rows=[];selections=[];daily=[];post_daily=[];logs=[]
for seed in SEEDS:
    completed=json.loads((ROOT/'results'/('cdf_primary_complete.json' if seed==20260920 else f'cdf_complete_s{seed}.json')).read_text())
    assert completed['status']=='complete' and completed['best_restored']
    selected=json.loads((ROOT/'results'/('cdf_selection.json' if seed==20260920 else f'cdf_selection_s{seed}.json')).read_text())
    epoch=selected['refinement_epoch'];stem=f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf'
    row=json.loads((ROOT/'results'/f'{stem}.json').read_text())
    cp=ROOT/'checkpoints'/f'CdfR2_seed{seed}_epoch{epoch:02d}.pt'
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==row['forecast_checkpoint_sha256']==selected['forecast_checkpoint_sha256']
    for suffix,collection in [('',daily),('_postonly',post_daily)]:
        d=pd.read_csv(ROOT/'results/trajectories'/f'{stem}{suffix}_daily.csv').set_index('date')
        assert len(d)==365 and d.index.is_unique
        assert abs(d.mtm_value_usd.sum()-json.loads((ROOT/'results'/f'{stem}{suffix}.json').read_text())['saving_usd'])<1e-6
        collection.append(d.mtm_value_usd.rename(str(seed)))
    history=pd.read_csv(ROOT/'logs'/f'cdf_training_seed{seed}.csv');history['seed']=seed;logs.append(history)
    if epoch:
        validation=history.loc[history.epoch.eq(epoch)].iloc[0].to_dict()
        price_score=float(validation['price_score'])
    else:price_score=.42173296213150024
    row.update(selected_epoch=epoch,completed_epochs=completed['epochs'],early_stopped=completed['early_stopped'],
               selection_savings_usd=selected['selected_2024_saving_usd'],selection_price_score=price_score,
               module_parameters=227527,model='WorldMoE (joint decision)',training_seconds=completed['seconds'])
    rows.append(row);selections.append(selected)

grid=pd.read_csv(ROOT/'results/cdf_grid_constraints.csv')
assert len(grid)==30 and set(grid.seed)==set(SEEDS)
for folder in ['source_data','figures','manuscript','scripts','qa']:(OUT/folder).mkdir(parents=True,exist_ok=True)
D=OUT/'source_data';frame=pd.DataFrame(rows);frame.to_csv(D/'worldmoe_three_seed_results.csv',index=False)
pd.concat(logs,ignore_index=True).to_csv(D/'worldmoe_training_history.csv',index=False)
daily=pd.concat(daily,axis=1);post_daily=pd.concat(post_daily,axis=1)
assert daily.notna().all().all() and post_daily.notna().all().all()
daily.index.name='date';post_daily.index.name='date'
daily.to_csv(D/'worldmoe_daily_by_seed.csv');post_daily.to_csv(D/'worldmoe_post_publication_daily_by_seed.csv')
old=pd.read_csv(EXTENSION/'source_data/information_daily_24h.csv').set_index('date').reindex(daily.index)
assert old.notna().all().all()
comparison=pd.DataFrame({'worldmoe_joint':daily.mean(1),'worldmoe_joint_post':post_daily.mean(1),
                        'rasc_me_original_executor':old.rasc,'seasonal_me_original_executor':old.seasonal})
comparison.to_csv(D/'paired_daily_comparisons.csv')
monthly=comparison.copy();monthly.index=pd.to_datetime(monthly.index)
monthly=monthly.groupby(monthly.index.strftime('%Y-%m')).sum();monthly.index.name='month'
monthly.to_csv(D/'monthly_savings.csv')

def interval(x,block=7):
    x=np.asarray(x,float);n=len(x);rng=np.random.default_rng(20260928)
    starts=rng.integers(0,n,(2000,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
    return np.quantile(x[ix].sum(1),[.025,.975]).tolist()

contrasts={
    'Joint WorldMoE vs earliest service':comparison.worldmoe_joint,
    'Joint WorldMoE vs original RASC-ME':comparison.worldmoe_joint-comparison.rasc_me_original_executor,
    'Joint WorldMoE vs original Seasonal-ME':comparison.worldmoe_joint-comparison.seasonal_me_original_executor,
    'Joint WorldMoE early information pathway':comparison.worldmoe_joint-comparison.worldmoe_joint_post,
}
paired=[]
for label,values in contrasts.items():
    lo,hi=interval(values)
    paired.append(dict(contrast=label,difference_usd=float(values.sum()),lower95_usd=lo,upper95_usd=hi,
                       block_days=7,resamples=2000,scope='same-calendar paired financial variability; not across-market inference'))
pd.DataFrame(paired).to_csv(D/'paired_financial_intervals.csv',index=False)
info=[];info_raw=[]
for seed,sel,row in zip(SEEDS,selections,rows):
    s=str(seed);native=float(daily[s].sum());post=float(post_daily[s].sum())
    restricted=json.loads((ROOT/'results'/f'{row["name"]}_postonly.json').read_text());info_raw.append(restricted)
    info.append(dict(seed=seed,epoch=sel['refinement_epoch'],native_usd=native,post_only_usd=post,early_path_usd=native-post,
                     restricted_late_it_mwh=restricted['late_IT_mwh'],restricted_closing_it_mwh=restricted['closing_it_mwh'],restricted_cvar95_usd=restricted['cvar95_usd']))
pd.DataFrame(info).to_csv(D/'worldmoe_information_by_seed.csv',index=False)
(D/'worldmoe_information_raw.json').write_text(json.dumps(info_raw,indent=2))

states=pd.read_csv(EXTENSION/'source_data/extreme_state_definition_inputs.csv')
states.to_csv(D/'market_state_definition_inputs.csv',index=False)
event_rows=[];influence=[];blocks=[];event_summary={}
for label,values in contrasts.items():
    if 'earliest service' in label:continue
    v=values.to_numpy();total=float(v.sum());best=int(v.argmax());date=values.index[best]
    threshold=float(states.loc[states.year.eq(2024),'spread_range'].quantile(.95))
    flag=states.loc[states.year.eq(2025)].set_index('date').loc[values.index,'spread_range'].gt(threshold).to_numpy()
    worst_week=min((dict(start=i,value=float(v[i:i+7].sum())) for i in range(len(v)-6)),key=lambda r:-r['value'])
    remaining=v.copy();remaining[best]=0
    ev=dict(total_usd=total,threshold=threshold,extreme_days=int(flag.sum()),
            extreme_usd=float(v[flag].sum()),ordinary_usd=float(v[~flag].sum()),
            extreme_positive_days=int((v[flag]>0).sum()),extreme_median_usd=float(np.median(v[flag])),
            ordinary_median_usd=float(np.median(v[~flag])),top_date=str(date),top_date_usd=float(v[best]),
            top_date_share=float(v[best]/total) if abs(total)>1e-12 else None,
            without_top_total_usd=float(remaining.sum()),without_top_extreme_usd=float(remaining[flag].sum()),
            top_week_start=str(values.index[worst_week['start']]),top_week_end=str(values.index[worst_week['start']+6]),
            without_top_week_total_usd=total-worst_week['value'])
    event_summary[label]=ev
    for n in [1,7]:
        for start in range(len(v)-n+1):
            keep=np.ones(len(v),bool);keep[start:start+n]=False
            influence.append(dict(contrast=label,excluded_days=n,start_date=values.index[start],end_date=values.index[start+n-1],
                                  retained_total_usd=float(v[keep].sum()),retained_extreme_usd=float(v[keep&flag].sum())))
    for block in [3,7,14,28]:
        lo,hi=interval(np.where(flag,v,0),block)
        blocks.append(dict(contrast=label,block_days=block,extreme_usd=float(v[flag].sum()),lower95_usd=lo,upper95_usd=hi))
    for definition in ['spread_range','max_abs_spread','max_rt']:
        for q in [.9,.95,.975]:
            threshold=float(states.loc[states.year.eq(2024),definition].quantile(q))
            flag=states.loc[states.year.eq(2025)].set_index('date').loc[values.index,definition].gt(threshold).to_numpy()
            lo,hi=interval(np.where(flag,v,0))
            event_rows.append(dict(contrast=label,definition=definition,quantile=q,threshold=threshold,
                                   extreme_days=int(flag.sum()),extreme_usd=float(v[flag].sum()),ordinary_usd=float(v[~flag].sum()),
                                   total_usd=total,extreme_lower95=lo,extreme_upper95=hi))
pd.DataFrame(event_rows).to_csv(D/'worldmoe_event_sensitivity.csv',index=False)
pd.DataFrame(influence).to_csv(D/'worldmoe_financial_exclusions.csv',index=False)
pd.DataFrame(blocks).to_csv(D/'worldmoe_event_block_intervals.csv',index=False)
(D/'worldmoe_event_summary.json').write_text(json.dumps(event_summary,indent=2))

grid.to_csv(D/'worldmoe_grid_raw.csv',index=False)
gs=grid.groupby(['experiment','scenario'],sort=False).agg(saving_usd=('saving_usd','mean'),saving_min=('saving_usd','min'),
      saving_max=('saving_usd','max'),policy_cost_usd=('adjusted_cost_usd','mean'),reference_cost_usd=('reference_cost_usd','mean'),
      late_mean=('late_IT_mwh','mean'),late_max=('late_IT_mwh','max'),closing_max=('closing_it_mwh','max'),
      cvar95_usd=('cvar95_usd','mean'),upward_ramp_mw=('max_upward_site_ramp_mw','max'),
      cap_excess_mw=('max_import_cap_excess_mw','max'),ramp_excess_mw=('max_upward_ramp_excess_mw','max'),
      reference_late_max=('reference_late_it_mwh','max'),n=('seed','size')).reset_index()
gs['savings_retained_pct']=100*gs.saving_usd/frame.saving_usd.mean()
gs.to_csv(D/'worldmoe_grid_summary.csv',index=False)
bound=json.loads((EXTENSION/'results/I1_Eagle_2025_w24.json').read_text())['maximum_saving_upper_usd']
metrics={}
for key in ['saving_usd','adjusted_cost_usd','saving_pct','capture_lower_fraction','shifted_IT_mwh','cvar95_usd',
            'late_IT_mwh','closing_it_mwh','pv_curtailment_mwh','grid_import_mwh','fee_usd','max_upward_site_ramp_mw',
            'max_fleet_recovery_mw','selection_savings_usd','selection_price_score']:
    x=frame[key].to_numpy(float);metrics[key]=dict(mean=float(x.mean()),min=float(x.min()),max=float(x.max()),sd=float(x.std(ddof=1)))
summary=dict(status='complete',model='WorldMoE',module='prediction- and state-conditioned joint decision module',
             internal_family='CdfR2',seeds=SEEDS,selected_epochs=frame.selected_epoch.tolist(),
             max_completed_epochs=frame.completed_epochs.tolist(),metrics=metrics,
             fixed_reference_cost_usd=float(frame.reference_cost_usd.iloc[0]),certified_savings_ceiling_usd=bound,
             target_80_usd=.8*bound,target_met=bool(metrics['saving_usd']['mean']>=.8*bound and frame.late_IT_mwh.max()<1e-8 and frame.closing_it_mwh.max()<1e-8),
             threshold_waived_by_user=True,all_zero_late_to_tolerance=bool(frame.late_IT_mwh.max()<1e-8),
             all_zero_closing=bool(frame.closing_it_mwh.max()<1e-8),late_tolerance_it_mwh=1e-8,
             training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_year=2024,evaluation_year=2025,
             refinement_parameters=227527,world_backbone_parameters=59156961,world_backbone_updated=False,
             replication_scope='Three optimization seeds from one shared pretrained QPR2 initialization and frozen WorldMoE/R2 feature cache; not independent backbone or full-pipeline initializations.',
             evaluation_scope='Temporal holdout excluded from parameter fitting and per-seed checkpoint selection. Development had inspected 2025 outcomes; not a never-inspected preregistered blind test.',
             paired_intervals=paired,
             information_mean= {k:float(pd.DataFrame(info)[k].mean()) for k in ['native_usd','post_only_usd','early_path_usd']},
             information_feasibility=dict(max_late_it_mwh=float(pd.DataFrame(info).restricted_late_it_mwh.max()),max_closing_it_mwh=float(pd.DataFrame(info).restricted_closing_it_mwh.max())),
             added_frozen_policy_cases={'information':3,'electrical':30},
             innovation_scope=[1,2,3,5],excluded_innovations=[4,6])
(ROOT/'results/worldmoe_final_summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'results_summary.json').write_text(json.dumps(summary,indent=2))
original=WORKSPACE/'NatureEnergy_Revised_Comparisons_20260927/source_data'
target=D/'original';target.mkdir(exist_ok=True)
for name in ['fluid_model_window_summary.csv','apic_and_forecast_controls.csv','Fig2_hourly_workload.csv','Fig2_capacity_floor.csv']:
    shutil.copy2(original/name,target/name)
target=D/'extensions';target.mkdir(exist_ok=True)
bounds=pd.read_csv(EXTENSION/'source_data/certified_value_bounds.csv')
bounds=bounds.drop(columns=[c for c in ['policy_saving_usd','policy_cost_usd','value_capture_lower_fraction','cost_gap_upper_usd','seeds'] if c in bounds.columns])
bounds.to_csv(target/'certified_value_bounds.csv',index=False)
certificates=D/'certificates';certificates.mkdir(exist_ok=True)
for p in (EXTENSION/'results').glob('I1_*_certificate.npz'):shutil.copy2(p,certificates/p.name)
shutil.copy2(EXTENSION/'results/cost_bound_zone_diagnostics.csv',certificates/'cost_bound_zone_diagnostics.csv')
forecasts=pd.read_csv(ROOT/'results/worldmoe_final_forecast_scores.csv')
forecasts.to_csv(D/'worldmoe_forecast_scores.csv',index=False)
summary['forecast_scores']={c:dict(mean=float(forecasts[c].mean()),min=float(forecasts[c].min()),max=float(forecasts[c].max())) for c in forecasts.columns if c.startswith('post_')}
controls=pd.read_csv(original/'fluid_model_window_summary.csv')
controls=controls[controls.window_h.eq(24)&~controls.model.isin(['World-ME','World-price'])]
controls=controls[controls.policy.eq('interval')|controls.model.isin(['DA-persistent','DA-myopic','RT-lag','DAC'])]
controls.to_csv(D/'external_benchmarks_24h.csv',index=False)
(ROOT/'results/worldmoe_final_summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'results_summary.json').write_text(json.dumps(summary,indent=2))
manifest=dict(checkpoints=[dict(seed=r['seed'],epoch=r['selected_epoch'],sha256=r['forecast_checkpoint_sha256'],result=r['name']+'.json') for r in rows],
              shared_initialization=json.loads((ROOT/'results/replication_authorization.json').read_text()),
              figure_scope='All main figures redrawn around the current WorldMoE and innovations 1,2,3,5. No old-versus-new WorldMoE mechanism comparison. External frozen baselines compare complete pipelines under common physics and settlement.',
              data_files=[str(p.relative_to(OUT)) for p in D.rglob('*') if p.is_file()])
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(status='complete',mean_saving=metrics['saving_usd']['mean'],capture=metrics['capture_lower_fraction']['mean'],selected_epochs=summary['selected_epochs'],output=str(OUT)),indent=2))
