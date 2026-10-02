"""Frozen 2025 comparisons through the unchanged executable-flexibility kernel."""
from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent
sys.path.insert(0,str(PAPER/'revision_20260920_market_expansion/experiment'))
import run_screen as c
import deepen_scheduling as d
import repair_reservations as repair
from ef_forecast_provider import ForecastProvider
O=ROOT/'results';DATA=ROOT/'ef_2025';DATA.mkdir(exist_ok=True)

def enrich(row,book,ref,arr):
    delta=(book[:c.N]-ref[:c.N])*c.k
    row.update(requested_IT_mwh=float(c.DT*np.sum(arr*c.k)),shifted_share_pct=100*row['shifted_IT_mwh']/(c.DT*np.sum(arr*c.k)),
        value_per_shifted_mwh=row['value_usd']/row['shifted_IT_mwh'] if row['shifted_IT_mwh']>1e-9 else np.nan,
        deadline_feasible=bool(row['late_IT_mwh']<1e-8),recovery_energy_mwh=float(c.DT*np.maximum(delta,0).sum()),
        peak_recovery_rate_mw=float(np.maximum(delta,0).sum(1).max()),
        grid_displacement_mwh=float(c.DT*np.abs(c.cash(ref)[1]-c.cash(book)[1]).sum()))
    return row
def main():
    start=time.perf_counter();freeze=json.loads((ROOT/'EVALUATION_FREEZE.json').read_text());assert not freeze['post_evaluation_parameter_changes_allowed']
    c.OUT=O;arr=c.load();c.DATA=DATA;c.ROWS=[];c.DAILY=[];c.TRAJ={};reference=d.reference(arr,1.)
    old=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/bridge_forecast_only_hourly.parquet')
    specs=[('Published_DA','published_da',None,['point']),('Old_RASC','old_rasc',old,['point','robust'])]
    for model in ['R2_full_native','R2_no_world_native','R2_full_raw','R2_full_external']:
        specs.append((model,'r2',pd.read_parquet(ROOT/'forecast_2025'/(model+'.parquet')),['point','robust']))
    protocol=dict(windows_h=[7,12,24],mixed_classes=repair.CLASSES,physical_steps=c.N,
        information='pre-DA 04:55 previous local day; post-DA 11:00 previous local day; historical medians beyond issued targets',
        forecast_primary=freeze['primary_model'],all_parameters_frozen=True,continuous_queue=True,service_work_lost=False,
        fixed_vs_recourse='only mixed deadlines; same selected slot priorities',mixed_models=['Published_DA','Old_RASC','R2_full_native','R2_no_world_native'],
        calibration_sensitivity_windows=[12],penalties=0,friction_usd_per_grid_MWh=1.,terminal_work_valuation='pre-2024 median RT times PUE upper bound',
        no_neural_execution_in_EF=True)
    (DATA/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];mixed=[];mref,mlate=repair.edf_reference(arr)
    for model,kind,frame,policies in specs:
        provider=ForecastProvider(c.times,c.da,c.train_rt,kind=kind,frame=frame);d.rank=lambda w,p:provider.ranked_slots(w,p)
        windows=[12] if model in ['R2_full_raw','R2_full_external'] else [7,12,24]
        for window in windows:
            for policy in policies:
                name=f'{model}_w{window}_{policy}'
                book,stats,events=d.schedule(arr,[(1.,window)],policy,keep_events=window==12 and model=='R2_full_native')
                row=c.record(name,'R2_EF',book,reference,dict(model=model,policy=policy,window_h=window),late=stats['late_IT_mwh'])
                row.update(stats);enrich(row,book,reference,arr);rows.append(row)
                if events:pd.DataFrame(events,columns=['release_step','deadline_step','execution_step','site','work_units','window_h']).to_parquet(DATA/(name+'_assignments.parquet'),index=False)
                print(name,'value',round(row['value_usd'],2),'late',round(row['late_IT_mwh'],5),'seconds',round(time.perf_counter()-start,1),flush=True)
                pd.DataFrame(rows).to_csv(O/'service_window_R2_results.csv',index=False)
            provider.rank_cache.clear()
        if model not in ['R2_full_raw','R2_full_external']:
            for policy in policies:
                for recourse in [False,True]:
                    if recourse:book,stats,events=repair.run(arr,policy)
                    else:book,stats,events=d.schedule(arr,repair.CLASSES,policy)
                    name=f'{model}_mixed_{policy}_'+('recourse' if recourse else 'fixed')
                    row=c.record(name,'R2_EF_mixed',book,mref,dict(model=model,policy=policy,window_h=None,recourse=recourse),late=stats['late_IT_mwh'])
                    row.update(stats);row['reference_late_IT_mwh']=mlate;enrich(row,book,mref,arr);mixed.append(row)
                    if recourse and model=='R2_full_native':events.to_parquet(DATA/(name+'_assignments.parquet'),index=False)
                    print(name,'value',round(row['value_usd'],2),'late',round(row['late_IT_mwh'],5),'seconds',round(time.perf_counter()-start,1),flush=True)
                    pd.DataFrame(mixed).to_csv(O/'heterogeneous_deadline_R2_results.csv',index=False)
                provider.rank_cache.clear()
    pd.DataFrame(c.DAILY).to_csv(O/'ef_daily_2025.csv',index=False)
    (DATA/'completed.json').write_text(json.dumps(dict(status='complete',uniform_scenarios=len(rows),mixed_scenarios=len(mixed),steps=c.N,seconds=time.perf_counter()-start),indent=2))
    print('EF COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
