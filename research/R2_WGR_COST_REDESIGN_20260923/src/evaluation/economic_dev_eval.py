from src.runtime import ROOT, REPO, PAPER, LEGACY
from src.data.access_guard import install_guard,assert_target_year,assert_years,read_development_parquet,register_historical
install_guard()
import argparse
import json
import time
import numpy as np
import pandas as pd
from vendor import economic_core as core
from vendor import reservation as engine
from vendor.forecast_provider import ForecastProvider,ZONES,TZ

def configure_development():
    with np.load(LEGACY/'cache/development_physical_reference.npz',allow_pickle=False) as source:
        dates=assert_years(source['dates'],'physical reference target dates')
        split=json.loads((ROOT/'configs/splits_2024_frozen.json').read_text())['economic_selection']
        select=(dates>=pd.Timestamp(split['start_utc']))&(dates<pd.Timestamp(split['end_exclusive_utc']))
        assert int(select.sum())==split['days']
        assert np.all(np.diff(dates[select].asi8)==86400*10**9)
        times=pd.date_range(dates[select][0],periods=int(select.sum())*96,freq='15min')
        assert_years(times,'economic physical block')
        arr=source['arrivals'][select].reshape(-1,4)
        pv=source['pv'][select].reshape(-1,4);pue=source['pue'][select].reshape(-1,4)
        train=dates.year<=2023
        train_pue=source['pue'][train]
        assert int(train.sum())==722
    # Reconstruct the exact FP32 packet constants from their original frozen
    # physical source and training-only envelopes, avoiding the old mixed-year
    # completed.json that also contains locked economics.
    physical_path=REPO/'artifacts/research/physical/run_id=paper-physical-v1/frozen_physical_artifact.json'
    physical=json.loads(physical_path.read_text(encoding='utf-8'))
    const={k:np.asarray(physical[v],np.float32).astype(float).tolist() for k,v in [
        ('work_power_per_unit','work_mw_per_unit'),('idle_power','idle_mw'),('grid_import_limit','grid_import_limit_mw'),('compute_capacity','capacity_work_units')]}
    const['pue_min']=np.maximum(train_pue.min((0,1)),1.).astype(np.float32).astype(float).tolist()
    const['pue_max']=np.maximum(train_pue.max((0,1)),const['pue_min']).astype(np.float32).astype(float).tolist()
    const['grid_export_limit']=[0.]*4
    (ROOT/'configs/economic_constraints.json').write_text(json.dumps(dict(constraints=const,physical_source=str(physical_path),
        envelope_source='2022-2023 physical-reference PUE, same amin/amax as data._fit_empirical_envelopes',
        precision='FP32 tensor constants widened to NumPy float64, as in formal executor',locked_results_accessed=False),indent=2))
    prices=read_development_parquet(PAPER/'experiments/gate_hourly_panel.parquet')
    index=pd.MultiIndex.from_product([times.floor('h'),ZONES],names=['timestamp_utc','zone'])
    panel=prices.set_index(['timestamp_utc','zone']).reindex(index)
    da=panel.da_lbmp.to_numpy().reshape(-1,4);rt=panel.rt_integrated.to_numpy().reshape(-1,4)
    assert np.isfinite(da).all() and np.isfinite(rt).all()
    core.N=len(times);core.times=times;core.da=da;core.rt=rt;core.pv=pv;core.pue=pue;core.c=const
    core.k=np.asarray(const['work_power_per_unit']);core.idle=np.asarray(const['idle_power'])
    core.train_rt=prices[prices.split.eq('train')].groupby('zone').rt_integrated.median().reindex(ZONES).to_numpy()
    core.day_index,core.days=pd.factorize(times.tz_convert(TZ).date,sort=True)
    core.ROWS=[];core.DAILY=[];core.TRAJ={}
    protocol=dict(**split,rows=len(times),local_days=len(core.days),windows_h=[12,24],diagnostic_h=7,initial_queue=0,continuous=True,
                  selector='zero late first; maximize value_usd - .1 * local_days * mtm_daily_loss_cvar95_usd; if none feasible minimize late then maximize score',
                  evaluation_role='2024 development-year decision value',locked_targets_accessed=False)
    (ROOT/'configs/economic_protocol.json').write_text(json.dumps(protocol,indent=2))
    return arr,engine.reference(arr,1.)

def run_one(model,seed,frame,arr,reference,windows=(12,24),ablation='normal',kind='r2'):
    provider=ForecastProvider(core.times,core.da,core.train_rt,kind=kind,frame=frame)
    engine.rank=lambda window,policy:provider.ranked_slots(window,policy)
    rows=[]
    for window in windows:
        for policy in ['point','robust']:
            name=f'{model}_seed{seed}_{ablation}_{window}h_{policy}_fp32v1'
            cache=ROOT/'results/economic_trajectories'/f'{name}_metrics.json'
            if cache.exists():rows.append(json.loads(cache.read_text()));continue
            started=time.perf_counter();book,stats,_=engine.schedule(arr,[(1.,window)],policy)
            row=core.record(name,'validation_2024',book,reference,dict(model=model,seed=seed,ablation=ablation,
                policy='interval' if policy=='robust' else 'point',window_h=window,workload='original_2024'),late=stats['late_IT_mwh'])
            row.update(stats)
            row.update(selection_score=row['value_usd']-.1*len(core.days)*row['mtm_daily_loss_cvar95_usd'],
                       cvar95=row['mtm_daily_loss_cvar95_usd'],shifted_it_mwh=row['shifted_IT_mwh'],late_it_mwh=row['late_IT_mwh'],
                       closing_work_mwh=row['final_queue_IT_mwh'],zero_late_feasible=row['late_IT_mwh']<1e-8,
                       post_DA_identity_max_error=provider.max_identity_error,seconds=time.perf_counter()-started,
                       evaluation_role='2024 development-year decision value',evaluation_protocol='fp32_v1')
            cache.write_text(json.dumps(row,indent=2));rows.append(row)
            # The unmodified accounting function retains a trajectory; drop
            # only its in-memory reference after the file is safely written.
            core.TRAJ.pop(name,None)
            print(model,seed,ablation,window,policy,'value',round(row['value_usd'],3),'late',round(row['late_it_mwh'],8),flush=True)
    provider.rank_cache.clear()
    return rows

def choose_policy(rows):
    f=pd.DataFrame(rows);feasible=f[f.late_it_mwh<1e-8]
    chosen=(feasible.sort_values('selection_score',ascending=False) if len(feasible) else f.sort_values(['late_it_mwh','selection_score'],ascending=[True,False])).iloc[0]
    return chosen.to_dict()

def aggregate_results():
    files=sorted((ROOT/'results/economic_trajectories').glob('*_fp32v1_metrics.json'))
    rows=[json.loads(p.read_text()) for p in files]
    if not rows:return
    frame=pd.DataFrame(rows)
    normal=frame[frame.ablation.eq('normal')]
    normal.to_csv(ROOT/'results/economic_dev_metrics_2024.csv',index=False)
    normal[normal.window_h.eq(7)].to_csv(ROOT/'results/service_boundary_7h_2024.csv',index=False)
    selected=[]
    for _,f in normal[normal.window_h.isin([12,24])].groupby(['model','seed','window_h']):selected.append(choose_policy(f.to_dict('records')))
    pd.DataFrame(selected).to_csv(ROOT/'results/economic_policy_selection_2024.csv',index=False)
    frame[~frame.ablation.eq('normal')].to_csv(ROOT/'results/ablation_economics_2024.csv',index=False)
    if core.DAILY:pd.DataFrame(core.DAILY).to_csv(ROOT/'results'/f'economic_daily_part_{time.time_ns()}.csv',index=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['RASC','A0_Base','A1_Shallow','A2_Deep_NoGate','A3_WGR'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[20260920,20260921,20260922]);parser.add_argument('--target-year',type=int,default=2024)
    args=parser.parse_args();assert_target_year(args.target_year)
    arr,reference=configure_development()
    for name in args.models:
        if name=='RASC':
            path=PAPER/'revision_20260918/forecast_2025/development_forecast_only_2024h2.parquet'
            register_historical(path)  # Exact 2024 filename, then read dates first.
            frame=read_development_parquet(path)
            rows=run_one('RASC',0,frame,arr,reference,windows=(12,24,7),kind='old_rasc')
            # A single existing development-only reference checks that the
            # reused executor/constants reproduce the official 12h comparator.
            old=pd.read_csv(LEGACY/'validation_economics/policy_comparison_2024.csv')
            comparisons=[]
            for row in rows:
                if row['window_h']!=12:continue
                policy='robust' if row['policy']=='interval' else 'point'
                expected=old[old.model.eq('Old_RASC')&old.policy.eq(policy)].iloc[0]
                delta=abs(float(expected.value_usd)-row['value_usd'])
                comparisons.append(dict(policy=policy,expected=float(expected.value_usd),observed=row['value_usd'],absolute_error=delta))
                assert delta<max(.01,1e-7*abs(float(expected.value_usd))), 'Formal executor reproduction failed'
            (ROOT/'results/economic_executor_reproduction.json').write_text(json.dumps(dict(passed=True,comparisons=comparisons),indent=2))
        else:
            for seed in args.seeds:
                path=ROOT/'results/predictions_2024'/f'{name}_seed{seed}_native.parquet'
                frame=read_development_parquet(path)
                run_one(name,seed,frame,arr,reference,windows=(12,24,7) if seed==20260920 else (12,24))
        aggregate_results()

if __name__=='__main__':main()
