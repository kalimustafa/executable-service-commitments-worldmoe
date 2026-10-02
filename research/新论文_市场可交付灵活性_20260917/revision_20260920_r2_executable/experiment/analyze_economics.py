"""Common financial, physical and attribution tables from completed trajectories."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;O=ROOT/'results'
sys.path.insert(0,str(PAPER/'revision_20260920_market_expansion/experiment'))
import run_screen as c
z=np.load(PAPER/'revision_20260918/annual_continuous_2025/data/annual_forcing_2025.npz')
c.times=pd.DatetimeIndex(pd.to_datetime(z['timestamps_utc'],utc=True));c.day_index,c.days=pd.factorize(c.times.tz_convert(c.TZ).date,sort=True)
f=pd.read_csv(O/'service_window_R2_results.csv');daily=pd.read_csv(O/'ef_daily_2025.csv');complete=(ROOT/'neural_2025/completed.json').exists()
if complete:
    nc=json.loads((ROOT/'neural_2025/completed.json').read_text());k=np.asarray(nc['constraints']['work_power_per_unit']);dt=.25
    ledger=pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'neural_2025').glob('ledger_*.parquet'))],ignore_index=True)
    rows=[];dr=[];ood_rows=[]
    from ood_distribution import describe,metrics
    od=pd.read_csv(ROOT/'w1_ood_metrics.csv');bounds=od[od.source.eq('W1_train')].set_index('metric')
    def append_ood(name,values):
        described=describe(name,values)
        for r in described:
            v=np.asarray(values[r['metric']]).ravel();v=v[np.isfinite(v)];b=bounds.loc[r['metric']];lo,hi=b.p001,b.p999;span=max(hi-lo,1e-6)
            r['outside_training_998pct_fraction']=float(((v<lo)|(v>hi)).mean());r['far_outside_training_envelope_fraction']=float(((v<lo-.1*span)|(v>hi+.1*span)).mean())
        ood_rows.extend(described)
    for w,g in ledger.groupby('window_h'):
        g=g.sort_values('step');assert np.array_equal(g.step.to_numpy(),np.arange(35040))
        def a(name):return g[[f'{name}_{s}' for s in range(4)]].to_numpy()
        compute=a('compute');ref=a('reference_compute');grid=a('grid');refgrid=a('reference_grid')
        cash=g.value.to_numpy();liab=g.service_liability.to_numpy();value=cash.copy();value[-1]-=liab[-1];mtm=cash-np.diff(np.r_[0,liab])
        met,dv=c.risk(value);mm,mv=c.risk(mtm);shift=dt*np.maximum((ref-compute)*k,0).sum();requested=dt*(a('arrivals')*k).sum();late=g.missed_IT_energy_mwh.sum()
        name=f'Old_R_action_w{w}_native';row=dict(name=name,model='Old_R_action',policy='native',window_h=w,route='native_learned',**met,
            **{'mtm_'+key:v for key,v in mm.items()},cash_value_usd=cash.sum(),terminal_liability_difference_usd=liab[-1],
            final_queue_IT_mwh=g.queue_IT_energy_mwh.iloc[-1],reference_final_queue_IT_mwh=dt*np.dot(a('reference_queue')[-1],k),
            IT_energy_mwh=g.IT_service_energy_mwh.sum(),shifted_IT_mwh=shift,shifted_share_pct=100*shift/requested,requested_IT_mwh=requested,
            value_per_shifted_mwh=met['value_usd']/shift if shift>1e-9 else np.nan,late_IT_mwh=late,deadline_feasible=bool(late<1e-8),
            electrical_shortage_mwh=g.unserved_energy_mwh.sum(),fee_usd=g.fee.sum(),grid_import_mwh=dt*grid.sum(),
            recovery_energy_mwh=dt*np.maximum((compute-ref)*k,0).sum(),peak_recovery_rate_mw=np.maximum((compute-ref)*k,0).sum(1).max(),grid_displacement_mwh=dt*np.abs(grid-refgrid).sum())
        rows.append(row);dr.extend(dict(name=name,date=str(day),value_usd=v,mtm_value_usd=m) for day,v,m in zip(c.days,dv,mv))
        append_ood(f'selected_Old_R_{w}h',metrics(compute,grid,a('queue_after'),a('pv_used'),ramp_axis=0))
        np.savez_compressed(ROOT/'ef_2025'/(name+'_trajectory.npz'),compute=compute,grid=grid,daily=dv,value=value,mtm=mtm,service_debt=liab)
    forcing=z['actual_21_hourly_pricing'];arr=forcing[:,12,None]*np.ones((1,4))/4
    for w in [7,12,24]:
        tr=np.load(ROOT/'ef_2025'/f'R2_full_native_w{w}_robust_trajectory.npz');compute=tr['compute'];grid=tr['grid'];queue=np.maximum(np.cumsum(arr-compute,axis=0),0.)
        used=np.clip(forcing[:,17:21]*(.23069976270198822+k*compute)-grid,0,forcing[:,13:17])
        append_ood(f'R2_EF_{w}h',metrics(compute,grid,queue,used,ramp_axis=0))
    od=od[~od.source.str.startswith(('selected_Old_R_','R2_EF_'))]
    pd.concat([od,pd.DataFrame(ood_rows)],ignore_index=True).to_csv(ROOT/'w1_ood_metrics.csv',index=False)
    f=pd.concat([f,pd.DataFrame(rows)],ignore_index=True);daily=pd.concat([daily,pd.DataFrame(dr)],ignore_index=True)
f['primary_comparison']=f.model.isin(['Published_DA','Old_RASC','R2_full_native','Old_R_action'])
for w,g in f.groupby('window_h'):
    elig=g[g.primary_comparison&g.deadline_feasible];best=elig.loc[elig.value_usd.idxmax()]
    f.loc[g.index,'VRR']=g.value_usd/best.value_usd;f.loc[g.index,'RF']=g.shifted_IT_mwh/best.shifted_IT_mwh
    f.loc[g.index,'VCR']=(g.value_usd/g.shifted_IT_mwh)/(best.value_usd/best.shifted_IT_mwh)
    f.loc[g.index,'ratio_comparator']=best['name']
f.to_csv(O/'policy_comparison_2025.csv',index=False);daily.to_csv(O/'all_daily_2025.csv',index=False)
pivot=daily.pivot(index='date',columns='name',values='value_usd').sort_index();assert len(pivot)==365
rng=np.random.default_rng(20260920);ix=((rng.integers(0,365,size=(2000,53))[:,:,None]+np.arange(7))%365).reshape(2000,-1)[:,:365]
draw=pivot.to_numpy()[ix].sum(1)
pd.DataFrame(dict(name=pivot.columns,lower95=np.quantile(draw,.025,axis=0),upper95=np.quantile(draw,.975,axis=0))).to_csv(O/'annual_bootstrap.csv',index=False)
contrasts=[]
for w in [7,12,24]:
    pairs=[(f'R2_full_native_w{w}_{p}',f'Old_RASC_w{w}_{p}','forecast') for p in ['point','robust']]
    pairs+=[(f'R2_full_native_w{w}_{p}',f'R2_no_world_native_w{w}_{p}','representation') for p in ['point','robust']]
    pairs += [(f'R2_full_native_w{w}_point',f'Published_DA_w{w}_point','versus_published_DA'),(f'R2_full_native_w{w}_robust',f'R2_full_native_w{w}_point','risk_gate')]
    for first,second,role in pairs:
        delta=pivot[first]-pivot[second];samples=delta.to_numpy()[ix].sum(1)
        contrasts.append(dict(window_h=w,first=first,second=second,role=role,difference_usd=float(delta.sum()),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975))))
pd.DataFrame(contrasts).to_csv(O/'paired_policy_differences.csv',index=False)
water=[]
if complete:
    chain=['Old_R_action_w12_native','Published_DA_w12_point','Old_RASC_w12_point','R2_full_native_w12_point','R2_full_native_w12_robust'];prev=0.;lut=f.set_index('name')
    for i,name in enumerate(chain):
        value=float(lut.loc[name,'value_usd']);water.append(dict(chain='execution_forecast_risk',step=i,name=name,increment_usd=value-prev,cumulative_value_usd=value,
            interpretation=['learned baseline','joint change to reservation execution and published-DA score; not pure causal architecture effect','old forecast under identical execution','R2 forecast under identical execution','interval selection under identical R2/execution'][i]));prev=value
    water.append(dict(chain='execution_forecast_risk',step=5,name='R2 interval total',increment_usd=prev,cumulative_value_usd=prev,interpretation='total, not an additional increment'))
for p in ['point','robust']:
    a=f[f.model.eq('R2_full_native')&f.policy.eq(p)&f.window_h.eq(12)].iloc[0];b=f[f.model.eq('R2_no_world_native')&f.policy.eq(p)&f.window_h.eq(12)].iloc[0]
    water.append(dict(chain='representation_ablation',step=0,name=p,increment_usd=a.value_usd-b.value_usd,cumulative_value_usd=a.value_usd,interpretation='matched full-minus-no-world, trained separately before 2025'))
pd.DataFrame(water).to_csv(O/'attribution_waterfall.csv',index=False)
metadata=dict(status='complete' if complete else 'EF_complete_neural_running',neural_complete=complete,rows=len(f),
    ratios='denominator is the highest-value deadline-feasible primary comparator at the same service window; raw/external calibration and no-world ablations are not primary selectors',
    risk='exact fractional 5% daily loss tail; primary downside subtracts changes in work liability; drawdown at quarter-hour resolution',
    recovery='positive IT service deviation from earliest service, not an event-specific recovery ratio',
    grid_displacement='absolute paired grid deviation, summed over sites and all intervals',
    bootstrap='2000 paired circular seven-day block resamples; calendar-year variability, not independent replications',
    waterfall='ordered conditional differences, not Shapley effects or invariant causal percentages')
(O/'analysis_completed.json').write_text(json.dumps(metadata,indent=2))
print(f[f.window_h.eq(12)][['model','policy','value_usd','shifted_IT_mwh','mtm_daily_loss_cvar95_usd','VRR','VCR']].to_string(index=False))
print(pd.DataFrame(contrasts).query('window_h==12').to_string(index=False))
