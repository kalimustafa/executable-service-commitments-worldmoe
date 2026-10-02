"""Original grid/PV/recovery indicators from completed physical trajectories."""
from src.evaluation.paper_economics import *
configure(2025)
primary=pd.read_csv(ROOT/'results/multiseed_economics_2025.csv')
base=pd.read_csv(ROOT/'results/final_baseline_window_sweep.csv')
base=base[base.window_h.isin([7,12,24])]
specs=pd.concat([primary,base],ignore_index=True).to_dict('records')
for policy in ['point','interval']:
    specs.append(json.loads((ROOT/'results/paper_runs'/f'Eagle_2025_registered_seed20260920_7h_{policy}_particle_trailing4_normal.json').read_text()))
for p in (ROOT/'results/paper_runs').glob('Eagle_2025_R2_seed20260920_mix25_50_25_*.json'):specs.append(json.loads(p.read_text()))
top5=core.rt>=np.quantile(core.rt,.95,axis=0);top10=core.rt>=np.quantile(core.rt,.90,axis=0)
rows=[];deciles=[];month=[]
for r in specs:
    name=r['name']
    with np.load(ROOT/'results/economic_trajectories'/f'{name}_complete_book.npz',allow_pickle=False) as z:
        book=z['compute'][:core.N];reference=z['reference'][:core.N]
    load=core.pue*(core.idle+core.k*book);refload=core.pue*(core.idle+core.k*reference)
    used=np.minimum(core.pv,load);used_ref=np.minimum(core.pv,refload)
    grid=load-used;refgrid=refload-used_ref;delta=grid-refgrid;recovery=np.maximum(delta,0.);fleet=recovery.sum(1)
    ramp=np.abs(np.diff(grid.sum(1))-np.diff(refgrid.sum(1)))
    load_top10=refload>=np.quantile(refload,.9,axis=0)
    row=dict(name=name,model=r['model'],seed=r.get('seed',0),window_h=r['window_h'],policy=r['policy'],recourse=r.get('recourse'),
        grid_displacement_mwh=float(core.DT*np.abs(delta).sum()),max_fleet_recovery_mw=float(fleet.max()),
        p95_fleet_recovery_mw=float(np.quantile(fleet,.95)),max_import_ramp_change_mw_per_interval=float(ramp.max()),
        p95_import_ramp_change_mw_per_interval=float(np.quantile(ramp,.95)),max_abs_policy_import_ramp_mw=float(np.abs(np.diff(grid.sum(1))).max()),
        import_change_top5pct_rt_mwh=float(core.DT*np.sum(delta*top5)),import_change_top10pct_rt_mwh=float(core.DT*np.sum(delta*top10)),
        positive_recovery_energy_mwh=float(core.DT*recovery.sum()),recovery_energy_fraction_in_top10pct_rt=float(np.sum(recovery*top10)/max(recovery.sum(),1e-12)),
        import_change_top10pct_ref_load_mwh=float(core.DT*np.sum(delta*load_top10)),pv_used_mwh=float(core.DT*used.sum()),
        pv_curtailed_mwh=float(core.DT*np.sum(core.pv-used)),pv_curtailment_change_vs_reference_mwh=float(core.DT*np.sum(used_ref-used)),
        shifted_it_mwh=float(core.DT*np.sum(np.maximum(reference-book,0)*core.k)),recovery_it_mwh=float(core.DT*np.sum(np.maximum(book-reference,0)*core.k)),
        variable_it_capacity_mw=float(core.k.sum()),peak_recovery_over_variable_it_capacity=float(fleet.max()/core.k.sum()))
    rows.append(row)
    bins=pd.qcut(core.rt.mean(1),10,labels=False,duplicates='drop')
    for group in sorted(set(bins)):
        take=bins==group;deciles.append(dict(name=name,price_decile=int(group)+1,intervals=int(take.sum()),
            grid_change_mwh=float(core.DT*delta[take].sum()),positive_recovery_mwh=float(core.DT*recovery[take].sum())))
    months=core.times.tz_convert('America/New_York').month
    for m in range(1,13):
        take=months==m;month.append(dict(name=name,month=m,shifted_IT_mwh=float(core.DT*np.sum(np.maximum(reference[take]-book[take],0)*core.k)),
            pv_use_change_mwh=float(core.DT*np.sum(used[take]-used_ref[take])),grid_change_mwh=float(core.DT*delta[take].sum())))
pd.DataFrame(rows).to_csv(ROOT/'results/grid_operational_metrics_2025.csv',index=False)
pd.DataFrame(deciles).to_csv(ROOT/'results/grid_price_deciles_2025.csv',index=False)
pd.DataFrame(month).to_csv(ROOT/'results/grid_monthly_effects_2025.csv',index=False)
print('GRID / PV / RECOVERY INDICATORS COMPLETE',len(rows),'policies',flush=True)
