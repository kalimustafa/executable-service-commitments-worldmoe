"""Original 48 h observed-price settlement tail on the frozen new-R2 books."""
from src.evaluation.paper_economics import *
from src.runtime import PAPER,REPO
from dcvpp.e2e_moe_r4.physical_twin import load_frozen_physical_twin_core
import zipfile,torch

arr,ref,shared=configure(2025);count=192
times=pd.date_range(core.times[-1]+pd.Timedelta(minutes=15),periods=count,freq='15min')
end=times[-1]+pd.Timedelta(minutes=15)
filters=[('timestamp_utc','>=',times[0]),('timestamp_utc','<',end)]
b1=pd.read_parquet(PAPER/'revision_20260918/annual_continuous_2025/data/boundary_pv_20260101.parquet',filters=filters)
b2=pd.read_parquet(PAPER/'revision_20260922_completion/data/pv_2026_tail.parquet',filters=filters)
f=pd.concat([b1,b2],ignore_index=True);f['zone']=f.zone.replace({'NYC':'N.Y.C.'})
pv=f.pivot_table(index='timestamp_utc',columns='zone',values='pv_dc_availability_mw_per_mwdc').reindex(index=times,columns=['WEST','CAPITL','N.Y.C.','LONGIL']).to_numpy()
assert np.isfinite(pv).all()
raw=[]
archive=PAPER/'revision_20260922_completion/data/20260101rtlbmp_zone_csv.zip'
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if name.startswith(('20260101','20260102')):
            f=pd.read_csv(z.open(name));f=f[f.Name.isin(['WEST','CAPITL','N.Y.C.','LONGIL'])].copy()
            f['timestamp_utc']=pd.to_datetime(f['Time Stamp'],format='%m/%d/%Y %H:%M').dt.tz_localize('America/New_York').dt.tz_convert('UTC');raw.append(f)
rt=pd.concat(raw).pivot(index='timestamp_utc',columns='Name',values='LBMP ($/MWHr)').reindex(index=times.floor('h'),columns=['WEST','CAPITL','N.Y.C.','LONGIL']).to_numpy()
assert np.isfinite(rt).all()
art=REPO/'artifacts/research/physical/run_id=paper-physical-v1'
twin=load_frozen_physical_twin_core(artifact_path=art/'frozen_physical_artifact.json',surrogate_path=art/'frozen_piecewise_surrogate.json',calibration_manifest_path=art/'physical_calibration_manifest.json')
with np.load(REPO/'artifacts/dart_value_select/counterfactual_inputs/v1/selection/daily_bundles/2022-12-28.npz',allow_pickle=False) as z:mult=z['physical_multiplier'].astype(np.float64)
capacity=twin.capacity_work_units.numpy().astype(float)
def settle(book):
    values=[]
    with torch.inference_mode():
        for start in range(0,count,96):
            util=np.clip(book[start:start+96]/capacity,0,1)
            values.append(twin(torch.tensor(mult[None]),torch.tensor(util[None]))['pue'][0].numpy())
    pue=np.concatenate(values);load=pue*(core.idle+core.k*book);used=np.minimum(pv,load);grid=load-used
    return float(core.DT*np.sum(rt*grid+pv-used)),grid
refcost,refgrid=settle(ref[core.N:core.N+count]);rows=[]
table=pd.read_csv(ROOT/'results/multiseed_economics_2025.csv')
for record in table.to_dict('records'):
    name=record['name']
    with np.load(ROOT/'results/economic_trajectories'/f'{name}_complete_book.npz',allow_pickle=False) as z:book=z['compute']
    assert not (book[core.N+count:]>1e-9).any()
    cost,grid=settle(book[core.N:core.N+count]);fee=float(core.FEE*core.DT*np.abs(grid-refgrid).sum())
    extended=record['cash_value_usd']+refcost-cost-fee
    event=pd.read_parquet(ROOT/'results/economic_trajectories'/f'{name}_assignments.parquet')
    tail=event[event.execution>=core.N];late=tail[tail.execution>tail.deadline]
    late_mwh=float(core.DT*np.sum(late.work_units*core.k[late.zone.to_numpy(dtype=int)]))
    rows.append(dict(seed=record['seed'],window_h=record['window_h'],policy=record['policy'],closing_work_mwh=record['final_queue_IT_mwh'],
        original_savings_usd=record['value_usd'],extended_savings_usd=extended,difference_usd=extended-record['value_usd'],
        tail_incremental_cost_usd=cost-refcost+fee,terminal_liability_replaced_usd=record['terminal_liability_difference_usd'],
        tail_late_mwh=late_mwh,observed_market_prices=True,physical_tail_modelled=True))
pd.DataFrame(rows).to_csv(ROOT/'results/settlement_tail_2026.csv',index=False)
(ROOT/'results/settlement_tail_complete.json').write_text(json.dumps(dict(status='complete',rows=len(rows),quarter_hours=count,
    first=str(times[0]),last=str(times[-1]),new_work_released=False,forecasts_updated=False,
    max_absolute_savings_change_usd=max(abs(r['difference_usd']) for r in rows),all_tail_deadlines_met=all(r['tail_late_mwh']<1e-8 for r in rows)),indent=2))
print('SETTLEMENT TAIL COMPLETE',len(rows),flush=True)
