from src.runtime import ROOT,LEGACY
from src.data.access_guard import install_guard,assert_years,read_development_parquet
install_guard()
import json,hashlib,time,argparse
import numpy as np
import pandas as pd
from vendor.forecast_provider import ForecastProvider,ZONES,TZ
from vendor import economic_core as core
from vendor import reservation as engine
from src.evaluation.economic_dev_eval import configure_development,choose_policy
from src.models.world_site_cost import marginal_cost_coefficients,transform_quantiles

def physical_forecasts(times,kind):
    with np.load(ROOT/'cache/world_physical_forecasts.npz',allow_pickle=False) as source:
        dates=source['dates'];assert_years(dates)
        lookup={str(d):i for i,d in enumerate(dates)}
        local_dates=times.tz_convert(TZ).strftime('%Y-%m-%d')
        index=np.array([lookup.get(d,-1) for d in local_dates]);valid=index>=0
        position=times.hour.to_numpy()*4+times.minute.to_numpy()//15
        # Same delivery-date world source as formal R2. For the UTC tail
        # (local evening), explicitly extend its same-time-of-day forecast.
        pv=source[kind+'_pv'][index.clip(0),position]
        pue=source[kind+'_pue'][index.clip(0),position]
        issued=pd.DatetimeIndex(pd.to_datetime(source['issue_utc'],utc=True)).asi8[index.clip(0)]
        issued=np.where(valid,issued,np.iinfo(np.int64).max)
        extended=(times.strftime('%Y-%m-%d').to_numpy()!=np.asarray(local_dates))&valid
    return pv,pue,issued,valid,extended

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recompute',action='store_true');args=parser.parse_args()
    arr,reference=configure_development();frame=read_development_parquet(ROOT/'results/predictions_2024/C0_Repaired_seed20260920_native.parquet')
    with np.load(LEGACY/'cache/development_physical_reference.npz',allow_pickle=False) as physical:
        dates=assert_years(physical['dates']);pue_ref=np.median(physical['pue'][dates.year<=2023],axis=(0,1))
    baseline=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame)
    identity=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame)
    ones=np.ones_like(core.da);zeros=np.zeros_like(core.da)
    identity.pre=transform_quantiles(identity.pre,ones,zeros);identity.post=transform_quantiles(identity.post,ones,zeros)
    for window in [12,24]:
        for policy in ['point','robust']:
            assert np.array_equal(identity.ranked_slots(window,policy),baseline.ranked_slots(window,policy)), 'Identity cost head changed ranking'
    del identity
    allrows=[];audit=[]
    for name,kind in [('C7_WorldSiteCost','world'),('C7_IssueSiteCost','issue')]:
        pv,pue,issued,valid,extended=physical_forecasts(core.times,kind)
        a,b=marginal_cost_coefficients(pv,pue,core.idle,core.k,np.asarray(core.c['compute_capacity']),pue_ref)
        provider=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame)
        # The physical source must exist before BOTH the declared pre-DA
        # forecast issue and the later post-DA issue. Otherwise use identity.
        pre_available=valid&(issued<=provider.pre_issue)
        post_available=valid&(issued<=provider.post_issue)
        for field,available in [('pre',pre_available),('post',post_available)]:
            aa=np.where(available[:,None],a,1.);bb=np.where(available[:,None],b,0.)
            setattr(provider,field,transform_quantiles(getattr(provider,field),aa,bb))
        engine.rank=lambda window,policy:provider.ranked_slots(window,policy)
        audit.append(dict(model=name,physical_source=kind,pre_available_fraction=float(pre_available.mean()),
            post_available_fraction=float(post_available.mean()),UTC_tail_persistence_fraction=float(extended.mean()),
            pue_reference=pue_ref.tolist(),cost_coefficient_min=float(a.min()),cost_coefficient_max=float(a.max()),
            physical_forecast_issue_verified=True,market_price_forecasts_modified=False,rank_method_modified=False,
            executor_modified=False,new_fitted_parameters=0))
        np.savez_compressed(ROOT/'results'/f'{name}_cost_head_2024.npz',a=a,b=b,physical_issue_utc=issued,pre_available=pre_available,
                            post_available=post_available,UTC_tail_persistence=extended,timestamps_utc=core.times.tz_localize(None).to_numpy())
        for window in [12,24]:
            for policy in ['point','robust']:
                label=f'{name}_seed20260920_normal_{window}h_{policy}_fp32v1'
                path=ROOT/'results/economic_trajectories'/f'{label}_metrics.json'
                if path.exists() and not args.recompute:row=json.loads(path.read_text())
                else:
                    started=time.perf_counter();book,stats,_=engine.schedule(arr,[(1.,window)],policy)
                    row=core.record(label,'validation_2024',book,reference,dict(model=name,seed=20260920,ablation='normal',
                        window_h=window,policy='interval' if policy=='robust' else 'point',workload='original_2024'),late=stats['late_IT_mwh'])
                    row.update(stats);row.update(selection_score=row['value_usd']-.1*len(core.days)*row['mtm_daily_loss_cvar95_usd'],
                        cvar95=row['mtm_daily_loss_cvar95_usd'],shifted_it_mwh=row['shifted_IT_mwh'],late_it_mwh=row['late_IT_mwh'],
                        closing_work_mwh=row['final_queue_IT_mwh'],evaluation_protocol='fp32_v1',seconds=time.perf_counter()-started,
                        decision_quantity='capacity-average site marginal cost, not market RT price',market_price_source='C0_Repaired',
                        savings_gain_vs_fixed_reference_pct=100*(row['value_usd']/3343.4716892558006-1))
                    path.write_text(json.dumps(row,indent=2));core.TRAJ.pop(label,None)
                allrows.append(row);print(name,window,policy,'savings',row['value_usd'],'gain_pct',row['savings_gain_vs_fixed_reference_pct'],'late',row['late_it_mwh'],flush=True)
        provider.rank_cache.clear()
    pd.DataFrame(allrows).to_csv(ROOT/'results/world_site_cost_economics_2024.csv',index=False)
    if core.DAILY:pd.DataFrame(core.DAILY).to_csv(ROOT/'results/world_site_cost_daily_2024.csv',index=False)
    (ROOT/'results/world_site_cost_audit.json').write_text(json.dumps(dict(identity_transform_ranks_exact=True,models=audit,
        marginal_cost_formula_derived_from='vendor/economic_core.py cash()',ranking_source='vendor/forecast_provider.py unchanged ranked_slots()',
        price_forecast_files_unchanged=True,physical_source_manifest='cache/world_physical_forecasts_manifest.json',locked_targets_accessed=False),indent=2))
    for name in ['C7_WorldSiteCost','C7_IssueSiteCost']:
        rows=[r for r in allrows if r['model']==name and r['window_h']==24]
        print('SELECTED',json.dumps(choose_policy(rows)),flush=True)

if __name__=='__main__':main()
