"""48 h no-new-arrival settlement tail using the frozen observed 2025 prices."""
import sys,json
import numpy as np,pandas as pd,torch
from paths import *
sys.path.insert(0,str(REPO/'src'))
from dcvpp.e2e_moe_r4.physical_twin import load_frozen_physical_twin_core

def main():
    f=np.load(DATA/'forcing_2024.npz');nextyear=np.load(DATA/'forcing_2025.npz');N=len(f['times']);T=192
    k=f['k'];idle=f['idle'];rt=nextyear['rt'][:T];pv=nextyear['pv'][:T]
    assert pd.Timestamp(nextyear['times'][0]).tz_localize('UTC')==pd.Timestamp('2025-01-01',tz='America/New_York').tz_convert('UTC')
    cons=json.loads((PAPER/'revision_20260920_service_frontier/experiment/results/completed.json').read_text())['constraints']
    provenance=pd.read_csv(DATA/'physical_day_provenance_2024.csv')
    template=Path(provenance.loc[provenance.utc_day.eq('2024-12-31'),'issue_template'].iloc[0])
    multiplier=np.load(template)['physical_multiplier'].astype(np.float64)
    art=REPO/'artifacts/research/physical/run_id=paper-physical-v1'
    core=load_frozen_physical_twin_core(artifact_path=art/'frozen_physical_artifact.json',surrogate_path=art/'frozen_piecewise_surrogate.json',calibration_manifest_path=art/'physical_calibration_manifest.json')
    capacity=core.capacity_work_units.numpy().astype(np.float64)
    def settle(b):
        util=np.clip(b/capacity[None],0.,1.)
        with torch.inference_mode():pue=core(torch.tensor(np.broadcast_to(multiplier,(2,4)).copy()),torch.tensor(util.reshape(2,96,4)))['pue'].numpy().reshape(T,4)
        load=pue*(idle+k*b);used=np.minimum(pv,load);grid=load-used
        return float(.25*(rt*grid+pv-used).sum()),grid
    runs=pd.read_csv(RESULTS/'all_mechanism_runs.csv');rows=[]
    for trace,proxy in [('Eagle','registered'),('Frontier','requested_nodes')]:
        ref=np.load(BASE/'runs'/f'{trace}_2024_{proxy}_12_earliest_trajectory.npz')['compute']
        rcost,rg=settle(ref[N:N+T]);refliab=float(.25*np.sum(ref[N:]*k*np.max(cons['pue_max'])*f['historical_rt']))
        for w in [12,24]:
            for policy in ['myopic_da','persistent_da']:
                row=runs[runs.trace.eq(trace)&runs.market_year.eq(2024)&runs.proxy.eq(proxy)&runs.service_window_h.eq(w)&runs.policy.eq(policy)].iloc[0]
                book=np.load(BASE/'runs'/f'{trace}_2024_{proxy}_{w}_{policy}_trajectory.npz')['compute']
                remaining=float(.25*np.sum(book[N+T:]*k))
                assert remaining<1e-7,'Tail must not silently discard remaining work'
                cost,grid=settle(book[N:N+T]);fee=float(.25*np.abs(grid-rg).sum())
                extended=float(row.saving_usd+row.terminal_liability_usd-refliab+rcost-cost-fee)
                rows.append(dict(trace=trace,market_year=2024,service_window_h=w,policy=policy,closing_work_mwh=float(row.closing_work_mwh),tail_completed_mwh=float(.25*np.sum(book[N:N+T]*k)),
                    tail_remaining_mwh=remaining,original_saving_usd=float(row.saving_usd),extended_saving_usd=extended,difference_usd=extended-row.saving_usd,
                    observed_rt=True,physical_tail_modelled=True,PUE_template=str(template),PUE_template_sha256=sha(template)))
    pd.DataFrame(rows).to_csv(RESULTS/'nyiso_2024_observed_price_tail.csv',index=False)
    save(RESULTS/'tail_2024_completed.json',dict(status='complete',cases=len(rows),quarter_hours=T,observed_RT_source=str(PAPER/'revision_20260918/annual_continuous_2025/data/annual_forcing_2025.npz'),
        PV_source='dated NASA-derived 1-2 January 2025 forcing',PUE_method='frozen physical twin at tail utilisation, holding final 2024 local-day template multiplier',max_abs_change_usd=max(abs(r['difference_usd']) for r in rows)))
    print(pd.DataFrame(rows)[['trace','service_window_h','policy','difference_usd','tail_completed_mwh']].to_string(index=False))
if __name__=='__main__':main()
