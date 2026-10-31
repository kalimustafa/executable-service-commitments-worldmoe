"""Create a complete leap-year NYISO forcing without modifying frozen 2025.

Uses the archived official 2024 DA/integrated RT panel, dated NASA-derived PV,
the existing deterministic Eagle source-date map, and frozen physical twin.
No W0/W1/R2 inference, fitting, calibration or outcome-based selection occurs.
"""
from datetime import date
import sys,json
import numpy as np,pandas as pd,torch
from paths import *
sys.path.insert(0,str(REPO/'src'))
from dcvpp.vpp_world_moe.r_ecmoe_final_2025_inputs import _template_map,_circular_distance,_causal_source_dates
from dcvpp.e2e_moe_r4.physical_twin import load_frozen_physical_twin_core
from dcvpp.e2e_moe_r4.training_data import _allocate_total_workload
ZONES=['WEST','CAPITL','N.Y.C.','LONGIL']

def main():
    torch.set_num_threads(4)
    constraints_path=PAPER/'revision_20260920_service_frontier/experiment/results/completed.json'
    c=json.loads(constraints_path.read_text())['constraints']
    k=np.array(c['work_power_per_unit']);idle=np.array(c['idle_power']);gridlimit=np.array(c['grid_import_limit'])
    panel_path=PAPER/'experiments/gate_hourly_panel.parquet'
    panel=pd.read_parquet(panel_path,columns=['timestamp_utc','zone','da_lbmp','rt_integrated','date','split'])
    historical=panel[panel.split.eq('train')].groupby('zone').rt_integrated.median().reindex(ZONES).to_numpy()
    source2025=PAPER/'revision_20260918/annual_continuous_2025/data/annual_forcing_2025.npz'
    old=np.load(source2025);a=old['actual_21_hourly_pricing'];times=pd.DatetimeIndex(pd.to_datetime(old['timestamps_utc'],utc=True))
    K=pd.read_csv(PAPER/'revision_20260922_completion/results/shared_da_settlement_component.csv')
    assert pd.DatetimeIndex(pd.to_datetime(K.timestamp_utc,utc=True)).equals(times)
    np.savez_compressed(DATA/'forcing_2025.npz',times=times.tz_localize(None).to_numpy(),da=a[:,:4],rt=a[:,4:8],pv=a[:,13:17],pue=a[:,17:21],eagle_arrivals=a[:,12,None]*np.ones((1,4))/4,
        common_da_component=K.shared_da_basis_adjustment_usd.to_numpy(),historical_rt=historical,k=k,idle=idle,grid_limit=gridlimit)
    year=2024;t0=pd.Timestamp('2024-01-01',tz='America/New_York').tz_convert('UTC');t1=pd.Timestamp('2025-01-01',tz='America/New_York').tz_convert('UTC')
    index=pd.date_range(t0,t1,freq='15min',inclusive='left');N=len(index)
    g=panel[panel.timestamp_utc.ge(t0)&panel.timestamp_utc.lt(t1)]
    assert not g.duplicated(['timestamp_utc','zone']).any()
    prices={key:g.pivot(index='timestamp_utc',columns='zone',values=key).reindex(index=index.floor('h'),columns=ZONES).to_numpy() for key in ['da_lbmp','rt_integrated']}
    assert all(np.isfinite(v).all() for v in prices.values())
    days=tuple(pd.date_range('2024-01-01','2025-01-01',freq='D').strftime('%Y-%m-%d'))
    selection=REPO/'artifacts/dart_value_select/counterfactual_inputs/v1/selection/daily_bundles'
    _,forcing_map=_template_map(selection,days)
    sources=tuple(sorted(forcing_map))
    # Same nearest-season/same-weekday rule as 2025, with an issue-legal
    # pre-target-year candidate pool. The Eagle work map itself is unchanged.
    candidates=[(date.fromisoformat(p.stem),p) for p in sorted(selection.glob('*.npz')) if p.stem<'2024-01-01']
    art=REPO/'artifacts/research/physical/run_id=paper-physical-v1'
    core=load_frozen_physical_twin_core(artifact_path=art/'frozen_physical_artifact.json',surrogate_path=art/'frozen_piecewise_surrogate.json',calibration_manifest_path=art/'physical_calibration_manifest.json')
    capacity=core.capacity_work_units.numpy().astype(np.float32)
    pv_path=REPO/'data/processed/nyiso_zone_pv_15m.parquet'
    pv=pd.read_parquet(pv_path,columns=['timestamp_utc','zone','pv_dc_availability_mw_per_mwdc'])
    pv=pv[pv.timestamp_utc.ge(t0)&pv.timestamp_utc.lt(t1)]
    assert not pv.duplicated(['timestamp_utc','zone']).any()
    pv_array=pv.pivot(index='timestamp_utc',columns='zone',values='pv_dc_availability_mw_per_mwdc').reindex(index=index,columns=ZONES).to_numpy(dtype=np.float32)
    assert np.isfinite(pv_array).all()
    cache={};work_all=[];mult_all=[];util_all=[];qda_all=[];time_all=[];records=[]
    for day in days:
        target=date.fromisoformat(day);pool=[x for x in candidates if x[0].weekday()==target.weekday()]
        _,p=min(pool or candidates,key=lambda x:(_circular_distance(x[0],target),-x[0].toordinal()))
        if p not in cache:
            with np.load(p) as z:cache[p]={n:z[n].copy() for n in z.files}
        a=cache[p];issue=pd.Timestamp(day,tz='UTC')-pd.Timedelta(days=1)
        assert _causal_source_dates(a,issue)
        workday=sources[(target.toordinal()-date(2022,1,1).toordinal())%len(sources)]
        work=forcing_map[workday]
        mult=a['physical_multiplier'].astype(np.float64)
        util=_allocate_total_workload(work,capacity)/capacity[None]
        fw=a['workload_pool'].mean(axis=0).astype(np.float32)
        fp=a['pv_pool'].mean(axis=0).T.astype(np.float32)
        issue_work=_allocate_total_workload(fw,capacity)
        issue_grid=np.maximum(mult[None]*(idle[None]+k[None]*issue_work)-fp,0.)
        qda=np.minimum(issue_grid,gridlimit[None]).astype(np.float32)
        work_all.append(work);mult_all.append(mult);util_all.append(util);qda_all.append(qda)
        time_all.extend(pd.date_range(day,periods=96,freq='15min',tz='UTC'))
        records.append(dict(utc_day=day,workload_source_date=workday,issue_template=str(p),issue_template_sha256=sha(p),issue_time=str(issue),source_gate_passed=True))
    with torch.inference_mode():pue=core(torch.tensor(np.stack(mult_all),dtype=torch.float64),torch.tensor(np.stack(util_all),dtype=torch.float64))['pue'].numpy().astype(np.float32).reshape(-1,4)
    full_index=pd.DatetimeIndex(time_all);loc=full_index.get_indexer(index);assert (loc>=0).all()
    eagle=np.concatenate(work_all)[loc,None]*np.ones((1,4))/4
    qda=np.concatenate(qda_all)[loc];pue=pue[loc]
    da=prices['da_lbmp'];rt=prices['rt_integrated'];K=.25*((da-rt)*qda).sum(1)
    assert N==35136 and len(index.unique())==N and (np.diff(index.asi8)==900*10**9).all()
    np.savez_compressed(DATA/'forcing_2024.npz',times=index.tz_localize(None).to_numpy(),da=da,rt=rt,pv=pv_array,pue=pue,eagle_arrivals=eagle,q_da=qda,
        common_da_component=K,historical_rt=historical,k=k,idle=idle,grid_limit=gridlimit)
    pd.DataFrame(records).to_csv(DATA/'physical_day_provenance_2024.csv',index=False)
    local=pd.Series(index.tz_convert('America/New_York').date)
    daycounts=local.value_counts().sort_index()
    audit=dict(market_year=2024,quarter_hours=N,local_dates=len(daycounts),utc_start=str(index[0]),utc_stop=str(t1),
        leap_day_intervals=int(daycounts.loc[date(2024,2,29)]),short_DST_day=str(daycounts[daycounts.eq(92)].index[0]),long_DST_day=str(daycounts[daycounts.eq(100)].index[0]),
        local_day_length_counts={str(k):int(v) for k,v in daycounts.value_counts().items()},unique_keys=True,no_missing_prices=True,no_missing_PV=True,
        PV_year='2024 dated NASA-derived series, with 1 Jan 2025 UTC boundary needed to finish local 2024',PUE_template_pool='strictly pre-2024 same-weekday seasonal template; frozen physical twin',
        eagle_source_dates=len(sources),eagle_annual_variable_it_mwh=float(.25*np.sum(eagle*k)),PUE_min=float(pue.min()),PUE_max=float(pue.max()),
        common_da_component_usd=float(K.sum()),common_da_schedule_role='issue-time empirical pool schedule, identical for all execution policies and workload sources within year',
        R2_independent_test_claim=False,model_training=False)
    save(DATA/'calendar_and_physical_audit_2024.json',audit)
    raw_sources=[]
    for month in range(1,13):
        for folder,product in [(REPO/'data/raw/nyiso/damlbmp','damlbmp'),(PAPER/'experiments/raw_integrated_rt','rtlbmp')]:
            p=folder/f'2024{month:02}01{product}_zone_csv.zip'
            assert p.exists()
            raw_sources.append(dict(path=str(p.resolve()),sha256=sha(p),url=f'https://mis.nyiso.com/public/csv/{product}/{p.name}'))
    originals=[source2025,panel_path,pv_path,constraints_path,REPO/'src/dcvpp/e2e_moe_r4/training_data.py',REPO/'src/dcvpp/vpp_world_moe/r_ecmoe_final_2025_inputs.py']+[art/n for n in ['frozen_physical_artifact.json','frozen_piecewise_surrogate.json','physical_calibration_manifest.json']]
    save(DATA/'forcing_source_manifest.json',raw_sources+[dict(path=str(p.resolve()),sha256=sha(p)) for p in originals])
    print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
