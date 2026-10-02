"""Recalculate comparators and physical metrics solely from archived outputs."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;D=ROOT/'source_data'
QLEVELS=[10,5,1];SEEDS=[20260920,20260921,20260922]
BOUND=132605.4208413913;REFCOST=555168.2050556969

def ci(v,block=7):
    v=np.asarray(v,float);n=len(v);rng=np.random.default_rng(20260928)
    starts=rng.integers(0,n,(2000,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
    return np.quantile(v[ix].sum(1),[.025,.975])

def main():
    # Copy additional complete comparison matrices used across the figure set.
    extra={
      'historical_frontier.csv':WORK/'NatureEnergy_Revised_Comparisons_20260927/source_data/discrete_model_window_comparison.csv',
      'G3_SUMMARY.csv':WORK/'g3_realistic_utilization_20260929/outputs/G3_SUMMARY.csv',
      'G3_RESULTS.csv':WORK/'g3_realistic_utilization_20260929/outputs/G3_REALISTIC_UTILIZATION_RESULTS.csv',
    }
    for dest,src in extra.items():shutil.copy2(src,D/dest)
    pkg=WORK/'paper_figures_20260929/source_data'
    for name,sub in [('Fig7_source.csv','fig7/Fig7_source_data.csv'),('Fig8_source.csv','fig8/Fig8_source_data.csv'),
                     ('ED1_source.csv','ed_fig1/ED_Fig1_source_data.csv'),('ED2_source.csv','ed_fig2/ED_Fig2_source_data.csv'),('ED3_source.csv','ed_fig3/ED_Fig3_source_data.csv')]:
        shutil.copy2(pkg/sub,D/name)
    wm=pd.read_csv(D/'current/worldmoe_daily_by_seed.csv').set_index('date')
    wmmean=wm[[str(x) for x in SEEDS]].mean(axis=1)
    hist=pd.read_csv(D/'history/fluid_model_window_summary.csv');prov=pd.read_csv(D/'history/fluid_raw_provenance.csv')
    native=hist[hist.window_h.eq(24)].copy()
    if len(native)!=26:raise RuntimeError('Unexpected historical 24h matrix')
    z=np.load(D/'grid_metrics_inputs_2025.npz',allow_pickle=False)
    ref=z['reference_grid_mw'];N=len(ref);ts=pd.DatetimeIndex(pd.to_datetime(z['timestamp_local'],utc=True));H=N//4
    rt_h=z['rt_lmp_usd_mwh'].reshape(H,4,4).mean(1);ph=rt_h.mean(1)
    refh=ref.reshape(H,4,4).mean(1);refp=refh.sum(1);rampref=np.diff(ref.sum(1))
    top={p:np.argsort(ph,kind='stable')[-int(np.ceil(H*p/100)):] for p in QLEVELS}
    rows=[];paired=[];daily_long=[];gm=[];profiles=[];sources=[]
    def meter_metrics(g,model,policy,seed,closing):
        gh=g.reshape(H,4,4).mean(1);gp=gh.sum(1);rr=np.diff(g.sum(1))
        row=dict(model=model,policy=policy,seed=seed,closing_it_mwh=closing,
          grid_import_mwh=float(g.sum()*.25),portfolio_peak_hourly_mw=float(gp.max()),
          site_peak_hourly_mw=float(gh.max()),portfolio_peak_15min_mw=float(g.sum(1).max()),
          max_upward_ramp_mw=float(rr.max()),max_downward_ramp_mw=float(-rr.min()),
          p95_abs_ramp_mw=float(np.quantile(abs(rr),.95)),ramp_sd_mw=float(rr.std()),
          top10_reference_peak_import_mw=float(gp[np.argsort(refp,kind='stable')[-10:]].mean()))
        for pct,ix in top.items():
            delta=refp[ix]-gp[ix]
            row.update({f'top{pct}_net_mwh':float(delta.sum()),f'top{pct}_mean_mw':float(delta.mean()),
                        f'top{pct}_pct':float(100*delta.sum()/refp[ix].sum())})
        gm.append(row)
    meter_metrics(ref,'No-flex','earliest',0,0.)
    curr=pd.read_csv(D/'current/worldmoe_three_seed_results.csv')
    for seed in SEEDS:meter_metrics(z[f'grid_seed_{seed}'],'WorldMoE','final',seed,0.)
    for r in native.itertuples():
        raw=prov[prov.model.eq(r.model)&prov.policy.eq(r.policy)&prov.window_h.eq(24)]
        if len(raw)!=r.n:raise RuntimeError('Missing baseline seed')
        series=[]
        for sr in raw.itertuples():
            daily=pd.read_csv(sr.daily).set_index('date')
            # Use the same service-adjusted (MTM) account as final WorldMoE.
            series.append(daily['mtm_value_usd'].reindex(wm.index))
            book=np.load(sr.book,allow_pickle=False)
            g=np.maximum(0,z['pue']*(z['idle_mw'][None,:]+z['work_power_mw'][None,:]*book['compute'][:N])-z['pv_mw'])
            meter_metrics(g,r.model,r.policy,int(sr.seed),float(sr.closing))
            sources.append(dict(model=r.model,policy=r.policy,seed=int(sr.seed),book=sr.book,daily=sr.daily,result=sr.source,
                                book_sha256=hashlib.sha256(Path(sr.book).read_bytes()).hexdigest()))
        mean=pd.concat(series,axis=1).mean(axis=1)
        if mean.isna().any() or abs(mean.sum()-r.saving)>.01:raise RuntimeError('Daily/annual mismatch')
        delta=(wmmean-mean).to_numpy();lo,hi=ci(delta)
        row=dict(model=r.model,policy=r.policy,saving_usd=r.saving,cost_usd=r.cost,saving_pct=100*r.saving/REFCOST,
                 late_it_mwh=r.late,closing_it_mwh=r.closing,n=r.n,cvar95_usd=r.cvar,
                 comparison_label=f'{r.model} / {r.policy}',worldmoe_minus_baseline_usd=float(delta.sum()),
                 bootstrap_lower95_usd=float(lo),bootstrap_upper95_usd=float(hi),paired_median_daily_usd=float(np.median(delta)),
                 paired_q25_daily_usd=float(np.quantile(delta,.25)),paired_q75_daily_usd=float(np.quantile(delta,.75)),
                 paired_positive_days=int((delta>0).sum()),paired_days=len(delta),block_days=7,resamples=2000,daily_account='mtm_value_usd')
        paired.append(row)
        for date,w,b,dd in zip(wm.index,wmmean,mean,delta):
            daily_long.append(dict(date=date,model=r.model,policy=r.policy,worldmoe_mean_usd=w,baseline_mean_usd=b,difference_usd=dd))
    paired=pd.DataFrame(paired).sort_values('saving_usd',ascending=False)
    paired.to_csv(D/'paired_all_ontime_comparators_24h.csv',index=False)
    long=pd.DataFrame(daily_long);long.to_csv(D/'paired_daily_all_comparators.csv',index=False)
    best=paired[~paired.model.eq('World-ME')].iloc[0]
    sel=long[long.model.eq(best.model)&long.policy.eq(best.policy)].rename(columns={'baseline_mean_usd':'lgb_recent_point_usd'})
    sel.to_csv(D/'paired_daily_worldmoe_vs_LGB_Recent_point.csv',index=False)
    grid=pd.DataFrame(gm);grid.to_csv(D/'grid_metrics_all_models_by_seed.csv',index=False)
    grid.groupby(['model','policy'],sort=False).mean(numeric_only=True).reset_index().to_csv(D/'grid_metrics_all_models.csv',index=False)
    pd.DataFrame(sources).to_csv(D/'comparison_raw_source_manifest.csv',index=False)
    blocks=[]
    for block in [3,7,14,28]:
        lo,hi=ci(sel.difference_usd,block);blocks.append(dict(block_days=block,difference_usd=sel.difference_usd.sum(),lower95=lo,upper95=hi))
    pd.DataFrame(blocks).to_csv(D/'strongest_comparator_block_sensitivity.csv',index=False)
    protocol=dict(quarter_hours=.25,calendar_hours=H,sites=4,time_column='timestamp_utc; local date is America/New_York',
      high_price='Top ceil(p*8760) hours of arithmetic mean across four zonal RT LMPs; stable chronological tie-break; ex-post diagnostic',
      top_price_cutoffs={p:float(ph[ix].min()) for p,ix in top.items()},
      baseline='Same-input earliest service',grid_import_equation='max(0, PUE*(idle+k*compute)-PV)',
      same_grid_formula_recovered_from_archived_books=True,deadline_late_tolerance_mwh=1e-8,
      financial_daily_account='mtm_value_usd, including changes in valued service debt; both compared pathways use this field',
      forecast_and_control_models_refitted=False)
    (D/'DERIVATION_PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Strongest historical independent comparator:',best.to_dict(),flush=True)
    print('Derived grid rows:',len(grid),'for',len(grid.groupby(['model','policy'])),'pipelines',flush=True)

if __name__=='__main__':main()
