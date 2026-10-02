"""Select the R2 calibration on continuous pre-2025 economics, not test MAE."""
from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent
EF=PAPER/'revision_20260920_market_expansion/experiment';sys.path.insert(0,str(EF))
import run_screen as core
import deepen_scheduling as engine
from ef_forecast_provider import ForecastProvider,ZONES,TZ

def configure_validation():
    source=np.load(ROOT/'cache/development_physical_reference.npz',allow_pickle=False)
    dates=pd.DatetimeIndex(pd.to_datetime(source['dates'],utc=True));eligible=np.flatnonzero(dates>=pd.Timestamp('2024-07-01',tz='UTC'))
    blocks=[];current=[]
    for i in eligible:
        if current and dates[i]-dates[current[-1]]!=pd.Timedelta(days=1):blocks.append(current);current=[]
        current.append(int(i))
    if current:blocks.append(current)
    selected=max(blocks,key=lambda b:(len(b),-b[0]))
    start=dates[selected[0]];times=pd.date_range(start,periods=len(selected)*96,freq='15min')
    assert times[-1]<pd.Timestamp('2025-01-01',tz='UTC')
    arr=source['arrivals'][selected].reshape(-1,4);pv=source['pv'][selected].reshape(-1,4);pue=source['pue'][selected].reshape(-1,4)
    prices=pd.read_parquet(PAPER/'experiments/gate_hourly_panel.parquet')
    index=pd.MultiIndex.from_product([times.floor('h'),ZONES],names=['timestamp_utc','zone'])
    data=prices.set_index(['timestamp_utc','zone']).reindex(index)
    da=data.da_lbmp.to_numpy().reshape(-1,4);rt=data.rt_integrated.to_numpy().reshape(-1,4)
    assert np.isfinite(da).all() and np.isfinite(rt).all()
    constants=json.loads((PAPER/'revision_20260920_service_frontier/experiment/results/completed.json').read_text())['constraints']
    core.N=len(times);core.times=times;core.da=da;core.rt=rt;core.pv=pv;core.pue=pue;core.c=constants
    core.k=np.asarray(constants['work_power_per_unit']);core.idle=np.asarray(constants['idle_power'])
    core.train_rt=prices[prices.split.eq('train')].groupby('zone').rt_integrated.median().reindex(ZONES).to_numpy()
    core.day_index,core.days=pd.factorize(times.tz_convert(TZ).date,sort=True)
    core.DATA=ROOT/'validation_economics';core.DATA.mkdir(exist_ok=True);core.ROWS=[];core.DAILY=[];core.TRAJ={}
    protocol=dict(selection_period='longest available contiguous post-calibration UTC block',
        start_utc=times[0].isoformat(),end_exclusive_utc=(times[-1]+pd.Timedelta(minutes=15)).isoformat(),
        days=len(selected),steps=len(times),initial_queue=0,continuous=True,service_window_h=12,
        objective='highest total value minus 0.1 times number of local dates times service-adjusted daily CVaR95 among zero-late policies; if none, lowest late work then that score',
        risk_weight=.1,period_chosen_by='availability only, before model comparisons',test_year_used=False)
    (ROOT/'validation_economics/protocol.json').write_text(json.dumps(protocol,indent=2))
    np.savez_compressed(ROOT/'validation_economics/forcing.npz',times=times.tz_localize(None).to_numpy(),arrivals=arr,pv=pv,pue=pue,da=da,rt=rt)
    return arr,protocol

def main():
    began=time.perf_counter();arr,protocol=configure_validation();reference=engine.reference(arr,1.)
    old=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/development_forecast_only_2024h2.parquet')
    specs=[('Published_DA','published_da',None,['point']),('Old_RASC','old_rasc',old,['point','robust'])]
    for variant in ['full','no_world']:
        for cal in ['raw','native','external']:
            name=f'R2_{variant}_{cal}';frame=pd.read_parquet(ROOT/'forecast_development'/(name+'.parquet'))
            specs.append((name,'r2',frame,['point','robust']))
    rows=[]
    for name,kind,frame,policies in specs:
        provider=ForecastProvider(core.times,core.da,core.train_rt,kind=kind,frame=frame)
        engine.rank=lambda w,p:provider.ranked_slots(w,p)
        for policy in policies:
            book,stats,_=engine.schedule(arr,[(1.,12)],policy)
            row=core.record(name+'_'+policy,'validation_2024',book,reference,dict(model=name,policy=policy,workload='original_2024',window_h=12),late=stats['late_IT_mwh'])
            row.update(stats);row['selection_score']=row['value_usd']-.1*len(core.days)*row['mtm_daily_loss_cvar95_usd']
            row['post_DA_identity_max_error']=provider.max_identity_error;rows.append(row)
            print(name,policy,'value',round(row['value_usd'],2),'late',round(row['late_IT_mwh'],6),'score',round(row['selection_score'],2),flush=True)
        provider.rank_cache.clear()
    f=pd.DataFrame(rows);f.to_csv(ROOT/'validation_economics/policy_comparison_2024.csv',index=False)
    pd.DataFrame(core.DAILY).to_csv(ROOT/'validation_economics/daily_values.csv',index=False)
    candidates=f[f.model.str.startswith('R2_full_')&f.policy.eq('robust')].copy()
    feasible=candidates[candidates.late_IT_mwh<1e-8]
    if len(feasible):winner=feasible.sort_values('selection_score',ascending=False).iloc[0]
    else:winner=candidates.sort_values(['late_IT_mwh','selection_score'],ascending=[True,False]).iloc[0]
    chosen=dict(status='selected_on_2024_only',primary_model=winner.model,primary_calibration=winner.model.removeprefix('R2_full_'),
        primary_interval='central 50%',selection_score=float(winner.selection_score),validation_value_usd=float(winner.value_usd),
        validation_late_IT_mwh=float(winner.late_IT_mwh),representation_ablation='same calibration applied to no_world head',
        optional_slot_ranking='not selected; assess only in a separate pre-2025 stage',
        evaluation_2025_opened=False,seconds=time.perf_counter()-began)
    (ROOT/'validation_economics/selected_calibration.json').write_text(json.dumps(chosen,indent=2));print('SELECTED',chosen,flush=True)
if __name__=='__main__':main()
