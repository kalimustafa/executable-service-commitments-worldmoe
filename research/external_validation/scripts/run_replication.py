"""Frozen Published-DA execution mechanisms on four primary trace/year pairs.
No forecast model is run or trained. New runs live outside the frozen archive.
"""
import sys,time,json
import numpy as np,pandas as pd
from paths import *
RR=PAPER/'analysis/reviewer_response_20260920'
sys.path.insert(0,str(RR/'experiment'))
import common as H
from task_a_executor_ablation import myopic_execute
from ef_forecast_provider import ForecastProvider
c,d,repair=H.c,H.d,H.repair
CONS=json.loads((PAPER/'revision_20260920_service_frontier/experiment/results/completed.json').read_text())['constraints']
DT=.25

def configure(year):
    z=np.load(DATA/f'forcing_{year}.npz')
    c.N=len(z['times']);c.DT=DT;c.times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
    c.da=z['da'];c.rt=z['rt'];c.pv=z['pv'];c.pue=z['pue'];c.k=z['k'];c.idle=z['idle'];c.c=CONS;c.train_rt=z['historical_rt']
    c.day_index,c.days=pd.factorize(c.times.tz_convert('America/New_York').date,sort=True)
    c.OUT=RESULTS;c.DATA=BASE/'runs';c.DAILY=[];c.ROWS=[];c.TRAJ={};c.RANKS={}
    p=ForecastProvider(c.times,c.da,c.train_rt,kind='published_da');d.rank=lambda w,pol:p.ranked_slots(w,pol)
    return z,p

def floor(arr,capacity):
    queue=np.zeros(4);max_prefix=0.;max_steps=0;peak_t=0
    for t,a in enumerate(arr):
        ready=queue+a
        if ready.max()>max_prefix:max_prefix=float(ready.max());peak_t=t
        max_steps=max(max_steps,int(np.ceil(ready.max()/capacity-1e-10)))
        queue=np.maximum(ready-capacity,0.)
    return dict(floor_h=max_steps*DT,peak_outstanding_prefix=max_prefix,peak_step=peak_t)

def reference_late(arr,book,window):
    released=arr.cumsum(0);served=book.cumsum(0)
    deadline=np.arange(c.N)+int(window*4)-1
    rem=np.minimum(arr,np.maximum(released-served[deadline],0.))
    return float(DT*np.sum(rem*c.k))

def events_audit(events,book,arr):
    if not isinstance(events,pd.DataFrame):events=pd.DataFrame(events,columns=['release_step','deadline_step','execution_step','site','work_units','window_h'])
    assert (events.execution_step>=events.release_step).all()
    executed=np.zeros_like(book)
    np.add.at(executed,(events.execution_step.to_numpy(dtype=int),events.site.to_numpy(dtype=int)),events.work_units.to_numpy())
    # Recourse emits an execution record at its historical execution time.
    # Agreement with the final book proves later repairs did not alter it.
    history_error=float(np.abs(executed-book).max())
    assert history_error<1e-7
    released=np.zeros_like(arr)
    np.add.at(released,(events.release_step.to_numpy(dtype=int),events.site.to_numpy(dtype=int)),events.work_units.to_numpy())
    assert np.max(np.abs(released-arr))<1e-7
    e=DT*events.work_units.to_numpy()*c.k[events.site.to_numpy(dtype=int)]
    late_all=float(e[events.execution_step.to_numpy()>events.deadline_step.to_numpy()].sum())
    return events,dict(no_future_arrivals=True,executed_history_max_error=history_error,all_deadline_late_it_mwh=late_all)

def measure(trace,year,proxy,window,policy,book,ref,arr,K,late,stats=None,events=None):
    name=f'{trace}_{year}_{proxy}_{window}_{policy}'
    r=H.enrich(H.record(book,ref,name,dict(window_h=window),late=late),book,ref,arr)
    cost,g=c.cash(book);base,g0=c.cash(ref)
    refliab=float(DT*np.sum(ref[c.N:]*c.k*np.max(CONS['pue_max'])*c.train_rt))
    liability=float(DT*np.sum(book[c.N:]*c.k*np.max(CONS['pue_max'])*c.train_rt))
    basecash=float(base.sum()+K.sum());baseadjusted=basecash+refliab
    cash=float(cost.sum()+K.sum()+r['fee_usd']);adjusted=cash+liability
    saving=baseadjusted-adjusted
    assert abs(saving-r['value_usd'])<1e-7
    residual=float(DT*np.sum((book.sum(0)-arr.sum(0))*c.k))
    future_error=float(np.maximum(np.cumsum(book[:c.N],axis=0)-np.cumsum(arr,axis=0),0).max())
    assert abs(residual)<1e-7 and future_error<1e-7 and book.max()<=1+1e-8
    assert not r['grid_limit_exceeded']
    audit=dict(work_conservation_residual_it_mwh=residual,no_future_work_cumulative_error=future_error,executed_history_max_error=0.,all_deadline_late_it_mwh=late)
    if events is not None:
        events,a=events_audit(events,book,arr);audit.update(a)
        events.to_parquet(BASE/'runs'/(name+'_assignments.parquet'),index=False)
    row=dict(trace=trace,workload_trace=trace,market_year=year,proxy=proxy,service_window_h=window,policy=policy,
        annual_variable_it_mwh=float(DT*np.sum(arr*c.k)),annual_cash_cost_usd=cash,annual_adjusted_cost_usd=adjusted,
        earliest_cash_cost_usd=basecash,earliest_adjusted_cost_usd=baseadjusted,
        saving_vs_earliest_usd=saving,saving_usd=saving,saving_rate_pct=100*saving/baseadjusted if baseadjusted>0 else np.nan,
        cash_saving_usd=basecash-cash,terminal_liability_usd=liability,shared_da_component_usd=float(K.sum()),
        rt_normalised_adjusted_cost_usd=adjusted-float(K.sum()),rt_normalised_earliest_cost_usd=baseadjusted-float(K.sum()),
        shifted_it_mwh=r['shifted_IT_mwh'],shifted_share_pct=r['shifted_share_pct'],late_it_mwh=late,closing_work_mwh=r['final_queue_IT_mwh'],
        cvar95_usd=r['mtm_daily_loss_cvar95_usd'],max_drawdown_usd=r['mtm_max_drawdown_usd'],
        peak_recovery_mw=float(np.maximum(g-g0,0.).sum(1).max()),grid_displacement_mwh=r['grid_displacement_mwh'],
        **audit,**({k:v for k,v in (stats or {}).items() if k in ['repair_calls','max_repair_conservation_error','p95_delay_hours']}))
    np.savez_compressed(BASE/'runs'/(name+'_trajectory.npz'),compute=book,grid=g,reference_grid=g0)
    pd.DataFrame(c.DAILY).to_csv(BASE/'runs'/(name+'_daily.csv'),index=False);c.DAILY=[]
    print(f'{name}: saving={saving:.2f}; rate={row["saving_rate_pct"]:.3f}%; late={late:.5f}',flush=True)
    return row

def main():
    began=time.perf_counter();rows=[];mixrows=[];floors=[];diag=[];alt=[];altmix=[]
    for year in [2025,2024]:
        z,provider=configure(year);K=z['common_da_component']
        for trace,proxy,arr in [('Eagle','registered',z['eagle_arrivals']),('Frontier','requested_nodes',np.load(DATA/f'frontier_{year}/mapped_arrivals.npz')['arrivals']),('Frontier','field7_sensitivity',np.load(DATA/f'frontier_{year}/mapped_arrivals.npz')['resource_field7_arrivals'])]:
            is_alt=proxy=='field7_sensitivity'
            if is_alt:windows=[12]
            else:windows=[7,12,24]
            fl={cap:floor(arr,cap) for cap in [1.,.75,.5]}
            frow=dict(trace=trace,market_year=year,proxy=proxy,annual_variable_it_mwh=float(DT*np.sum(arr*c.k)),peak_outstanding_prefix=fl[1.]['peak_outstanding_prefix'],
                physical_floor_h_100pct=fl[1.]['floor_h'],physical_floor_h_75pct=fl[.75]['floor_h'],physical_floor_h_50pct=fl[.5]['floor_h'],peak_step=fl[1.]['peak_step'])
            floors.append(frow)
            ref=d.reference(arr,1.)
            localrows=[]
            for w in windows:
                late0=reference_late(arr,ref,w)
                localrows.append(measure(trace,year,proxy,w,'earliest',ref,ref,arr,K,late0))
                book,st=myopic_execute(arr,float(w),provider.ranked_slots(w,'point'))
                localrows.append(measure(trace,year,proxy,w,'myopic_da',book,ref,arr,K,st['late_IT_mwh'],st))
                book,st,ev=d.schedule(arr,[(1.,float(w))],'point',keep_events=True)
                localrows.append(measure(trace,year,proxy,w,'persistent_da',book,ref,arr,K,st['late_IT_mwh'],st,ev))
                inc=localrows[-1]['saving_usd']-localrows[-2]['saving_usd']
                for r in localrows[-3:]:
                    r['physical_floor_h']=fl[1.]['floor_h'];r['physical_window_feasible']=w+1e-10>=fl[1.]['floor_h']
                    r['persistence_increment_usd']=inc;r['persistence_share_pct']=100*inc/localrows[-1]['saving_usd'] if abs(localrows[-1]['saving_usd'])>1e-9 else np.nan
            (alt if is_alt else rows).extend(localrows)
            if not is_alt:
                book,st=myopic_execute(arr,12.,provider.ranked_slots(12,'point'),hedge=False)
                diag.append(measure(trace,year,proxy,12,'unhedged_myopic_da',book,ref,arr,K,st['late_IT_mwh'],st))
            mref,edflate=repair.edf_reference(arr)
            fixed,fs,fe=d.schedule(arr,repair.CLASSES,'point',keep_events=True)
            fr=measure(trace,year,proxy,'mixed','fixed_mixed_da',fixed,mref,arr,K,fs['late_IT_mwh'],fs,fe)
            repaired,rs,re=repair.run(arr,'point')
            rr=measure(trace,year,proxy,'mixed','recourse_mixed_da',repaired,mref,arr,K,rs['late_IT_mwh'],rs,re)
            mr=dict(trace=trace,workload_trace=trace,market_year=year,proxy=proxy,mix='4/12/24 h;25/50/25%',
                fixed_late_mwh=fr['all_deadline_late_it_mwh'],recourse_late_mwh=rr['all_deadline_late_it_mwh'],late_eliminated_mwh=fr['all_deadline_late_it_mwh']-rr['all_deadline_late_it_mwh'],
                fixed_value_usd=fr['saving_usd'],recourse_value_usd=rr['saving_usd'],value_change_usd=rr['saving_usd']-fr['saving_usd'],
                value_change_pct=100*(rr['saving_usd']-fr['saving_usd'])/abs(fr['saving_usd']) if abs(fr['saving_usd'])>1e-9 else np.nan,
                edf_reference_late_mwh=edflate,executed_history_max_error=rr['executed_history_max_error'],repair_calls=rs['repair_calls'],max_repair_conservation_error=rs['max_repair_conservation_error'])
            (altmix if is_alt else mixrows).append(mr)
            pd.DataFrame(rows+alt+diag).to_csv(RESULTS/'all_mechanism_runs.csv',index=False)
            pd.DataFrame(mixrows+altmix).to_csv(RESULTS/'all_recourse_runs.csv',index=False)
            pd.DataFrame(floors).to_csv(RESULTS/'trace_service_floor_all_proxies.csv',index=False)
            save(BASE/'progress.json',dict(status='running',completed_case=f'{trace}-{year}-{proxy}',elapsed_seconds=time.perf_counter()-began))
    allf=pd.DataFrame(rows);mixed=pd.DataFrame(mixrows)
    allf[allf.trace.eq('Frontier')].to_csv(RESULTS/'frontier_external_validation.csv',index=False)
    mixed[mixed.trace.eq('Frontier')].to_csv(RESULTS/'frontier_recourse_validation.csv',index=False)
    allf[allf.market_year.eq(2024)].to_csv(RESULTS/'nyiso_2024_mechanism_replication.csv',index=False)
    pd.DataFrame(alt).to_csv(RESULTS/'frontier_resource_field_sensitivity.csv',index=False)
    pd.DataFrame(altmix).to_csv(RESULTS/'frontier_resource_field_recourse_sensitivity.csv',index=False)
    pd.DataFrame(diag).to_csv(RESULTS/'unhedged_execution_diagnostic.csv',index=False)
    pd.DataFrame([r for r in floors if r['proxy']!='field7_sensitivity']).to_csv(RESULTS/'trace_service_floor.csv',index=False)
    summary=[]
    for (trace,year),g in allf[allf.service_window_h.eq(12)].groupby(['trace','market_year']):
        g=g.set_index('policy');m=mixed[mixed.trace.eq(trace)&mixed.market_year.eq(year)].iloc[0]
        summary.append(dict(trace=trace,market_year=int(year),annual_variable_it_mwh=g.loc['earliest','annual_variable_it_mwh'],physical_floor_h=g.loc['earliest','physical_floor_h'],
            earliest_cost=g.loc['earliest','annual_adjusted_cost_usd'],myopic_da_saving=g.loc['myopic_da','saving_usd'],myopic_da_saving_pct=g.loc['myopic_da','saving_rate_pct'],
            persistent_da_saving=g.loc['persistent_da','saving_usd'],persistent_da_saving_pct=g.loc['persistent_da','saving_rate_pct'],
            persistence_increment=g.loc['persistent_da','persistence_increment_usd'],persistence_share_pct=g.loc['persistent_da','persistence_share_pct'],
            myopic_late_mwh=g.loc['myopic_da','late_it_mwh'],persistent_late_mwh=g.loc['persistent_da','late_it_mwh'],
            fixed_mixed_late=m.fixed_late_mwh,recourse_mixed_late=m.recourse_late_mwh,recourse_value_change=m.value_change_usd,
            counterfactual_cross_trace=True,R2_independent_test=False))
    summary=pd.DataFrame(summary)
    summary.to_csv(RESULTS/'external_validity_summary.csv',index=False)
    summary[summary.trace.eq('Eagle')].to_csv(RESULTS/'market_year_replication_summary.csv',index=False)
    # Required frozen anchor check uses file-backed values, never hard-coded scores.
    frozen=pd.read_csv(PAPER/'revision_20260922_completion/results/traditional_baseline_economics_12h.csv').set_index('policy')
    old=allf[allf.trace.eq('Eagle')&allf.market_year.eq(2025)&allf.service_window_h.eq(12)].set_index('policy')
    checks={}
    for new,prior in [('earliest','Earliest service'),('myopic_da','Myopic + Published DA'),('persistent_da','Published DA persistent')]:
        checks[new]=dict(cost_residual=float(old.loc[new,'annual_adjusted_cost_usd']-frozen.loc[prior,'annual_service_adjusted_cost_usd']),saving_residual=float(old.loc[new,'saving_usd']-frozen.loc[prior,'incremental_value_vs_earliest_usd']))
    assert max(abs(v) for r in checks.values() for v in r.values())<.01
    for r in json.loads((BASE/'frozen_2025_manifest.json').read_text()):assert sha(r['path'])==r['sha256']
    save(RESULTS/'replication_completed.json',dict(status='complete',primary_cases=4,primary_policy_rows=len(rows),proxy_sensitivity_rows=len(alt),unhedged_rows=len(diag),recourse_pairs=len(mixrows),frozen_2025_checks=checks,
        frozen_2025_hashes_unchanged=True,all_work_conserved=True,no_future_arrivals=True,executed_history_preserved=True,seconds=time.perf_counter()-began))
    save(BASE/'progress.json',dict(status='experiments_complete',elapsed_seconds=time.perf_counter()-began))
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
