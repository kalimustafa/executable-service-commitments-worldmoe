from src.runtime import ROOT,PREVIOUS,LEGACY
from src.data.access_guard import install_guard,assert_years
install_guard()
import json
import time
import numpy as np
import pandas as pd
from vendor import economic_core as core
from vendor import reservation as engine
from vendor.forecast_provider import ForecastProvider,TZ
from src.evaluation.economic_dev_eval import configure_development
from src.candidate_runtime import build_candidate

def setup():
    arr,reference=configure_development()
    base=ForecastProvider(core.times,core.da,core.train_rt,kind='published_da')
    with np.load(LEGACY/'cache/development_physical_reference.npz',allow_pickle=False) as source:
        dates=assert_years(source['dates']);pue_ref=np.median(source['pue'][dates.year<=2023],axis=(0,1))
    with np.load(PREVIOUS/'cache/world_physical_forecasts.npz',allow_pickle=False) as source:
        lookup={str(d):i for i,d in enumerate(source['dates'])}
        index=np.array([lookup.get(d,-1) for d in core.times.tz_convert(TZ).strftime('%Y-%m-%d')]);valid=index>=0
        slot=core.times.hour.to_numpy()*4+core.times.minute.to_numpy()//15
        pv=source['issue_pv'][index.clip(0),slot];pue=source['issue_pue'][index.clip(0),slot]
        issued=pd.DatetimeIndex(pd.to_datetime(source['issue_utc'],utc=True)).asi8[index.clip(0)]
    kwargs=dict(pv=pv,pue=pue,physical_pre=valid&(issued<=base.pre_issue),physical_post=valid&(issued<=base.post_issue),
                idle=core.idle,k=core.k,capacity=1.,pue_reference=pue_ref)
    return arr,reference,kwargs

def main():
    protocol=json.loads((ROOT/'configs/locked_protocol.json').read_text())
    arr,reference,kwargs=setup();refcost=float(core.cash(reference)[0].sum());rows=[]
    for seed in [protocol['primary_seed']]+protocol['completion_seeds']:
        for window in [12,24]:
            for policy in ['point','interval']:
                name=f'R2_seed{seed}_{window}h_{policy}_development'
                path=ROOT/'results'/f'{name}_metrics.json'
                if path.exists():rows.append(json.loads(path.read_text()));continue
                row=dict(name=name,seed=seed,market_branch=protocol['market_branch'],physical_branch=protocol['physical_branch'],
                         model='R2_StateCost',point_statistic='median' if policy=='point' else 'interval',policy=policy,window_h=window)
                observer,information,description=build_candidate(row,arr,kwargs)
                engine.rank=lambda w,p:observer
                started=time.perf_counter()
                book,stats,events=engine.schedule(arr,[(1.,window)],'robust' if policy=='interval' else policy,keep_events=True)
                difference=float(np.max(np.abs(book-observer.book)));assert difference<1e-10
                event=pd.DataFrame(events,columns=['release','deadline','execution','zone','work_units','window_h'])
                missed=event.execution>event.deadline
                strict_late=float(core.DT*np.sum(event.loc[missed,'work_units']*core.k[event.loc[missed,'zone'].astype(int)]))
                met=core.record(name,'validation_2024',book,reference,{k:v for k,v in row.items() if k!='name'},late=strict_late)
                met.update(stats);met.update(strict_late_all_cohorts_IT_mwh=strict_late,net_operating_cost_usd=refcost-met['value_usd'],
                    reference_cost_usd=refcost,observer_max_difference=difference,information_audit=information,
                    seconds=time.perf_counter()-started)
                path.write_text(json.dumps(met,indent=2));rows.append(met)
                event.to_parquet(ROOT/'results/economic_trajectories'/f'{name}_assignments.parquet',index=False)
                core.TRAJ.pop(name,None)
                print(json.dumps({k:met[k] for k in ['seed','window_h','policy','net_operating_cost_usd','value_usd','mtm_daily_loss_cvar95_usd','strict_late_all_cohorts_IT_mwh']}),flush=True)
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/multiseed_economics_2024.csv',index=False)
    if core.DAILY:pd.DataFrame(core.DAILY).to_csv(ROOT/'results/multiseed_daily_2024.csv',index=False)
    main=table[(table.window_h==24)&(table.policy=='interval')]
    summary=dict(status='complete',seeds=main.seed.tolist(),mean_net_cost_usd=float(main.net_operating_cost_usd.mean()),
        minimum_net_cost_usd=float(main.net_operating_cost_usd.min()),maximum_net_cost_usd=float(main.net_operating_cost_usd.max()),
        all_seeds_zero_strict_late=bool((main.strict_late_all_cohorts_IT_mwh<1e-8).all()),
        all_seeds_zero_closing=bool((main.final_queue_IT_mwh<1e-8).all()),seed_selection=False)
    (ROOT/'results/MULTISEED_DEVELOPMENT_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)

if __name__=='__main__':main()
