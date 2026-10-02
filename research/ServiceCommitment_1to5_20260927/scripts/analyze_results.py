import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
OUT=ROOT/'source_data';OUT.mkdir(exist_ok=True)
SOURCE=WORKSPACE/'NatureEnergy_Revised_Comparisons_20260927/source_data'
provenance=pd.read_csv(SOURCE/'fluid_raw_provenance.csv')
raw=[]
for p in RESULTS.glob('I[1-4]*.json'):
    d=json.loads(p.read_text(encoding='utf-8'))
    if isinstance(d,dict) and 'experiment' in d:raw.append(d)
all_results=pd.DataFrame(raw);all_results.to_csv(OUT/'all_new_experiments.csv',index=False)

def daily(path):
    frame=pd.read_csv(path);frame['date']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    return frame.set_index('date').mtm_value_usd.sort_index()

def bootstrap_indices(n):
    rng=np.random.default_rng(20260927)
    starts=rng.integers(0,n,(2000,int(np.ceil(n/7))))
    return ((starts[:,:,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]

def interval(values,idx=None):
    x=np.asarray(values,float);idx=bootstrap_indices(len(x)) if idx is None else idx
    draws=x[idx].sum(axis=1)
    return np.quantile(draws,[.025,.975]).tolist()

def old_daily(model,window):
    row=provenance[(provenance.model==model)&(provenance.policy.eq('interval') if model not in ['DA-persistent'] else provenance.policy.eq('point'))&provenance.window_h.eq(window)].iloc[0]
    return daily(row.daily)

def new_daily(year,seed,window,info):return daily(TRAJECTORIES/f'I2_Eagle_{year}_s{seed}_w{window}_{info}_daily.csv')

information=[];paired=[];mechanism=[]
for window in [12,16,24]:
    full=pd.concat([new_daily(2025,s,window,'full') for s in [20260920,20260921,20260922]],axis=1).mean(axis=1)
    post=pd.concat([new_daily(2025,s,window,'post_only') for s in [20260920,20260921,20260922]],axis=1).mean(axis=1)
    rasc=old_daily('RASC-ME',window).reindex(full.index);seasonal=old_daily('Seasonal-ME',window).reindex(full.index)
    if rasc.isna().any() or seasonal.isna().any():raise RuntimeError('Daily comparison calendar mismatch')
    d=pd.DataFrame(dict(full=full,post_only=post,rasc=rasc,seasonal=seasonal));d.index.name='date'
    d.to_csv(OUT/f'information_daily_{window}h.csv')
    delta=full-post;lo,hi=interval(delta)
    row=dict(window_h=window,full_saving_usd=full.sum(),post_only_saving_usd=post.sum(),rasc_saving_usd=rasc.sum(),seasonal_saving_usd=seasonal.sum(),early_path_value_usd=delta.sum(),early_path_lower95=lo,early_path_upper95=hi,post_minus_rasc_usd=(post-rasc).sum(),full_minus_rasc_usd=(full-rasc).sum(),post_minus_seasonal_usd=(post-seasonal).sum())
    row['early_path_share_of_full_rasc_gap']=row['early_path_value_usd']/row['full_minus_rasc_usd'] if abs(row['full_minus_rasc_usd'])>1e-9 else np.nan
    information.append(row)
    for label,x in [('full_minus_rasc',full-rasc),('post_minus_rasc',post-rasc),('full_minus_seasonal',full-seasonal),('post_minus_seasonal',post-seasonal),('early_path',delta)]:
        lo,hi=interval(x);paired.append(dict(window_h=window,comparison=label,difference_usd=x.sum(),lower95=lo,upper95=hi,days=len(x),block_days=7,repeats=2000,seeds=3))
    arr,_,_=configure(2025)
    base=ForecastProvider(c.times,c.da,c.train_rt,kind='published_da');h=int(window*4)
    ids=np.arange(c.N)[:,None]+np.arange(h);pre=np.r_[base.pre_issue,np.full(h,np.iinfo(np.int64).max)];posttime=np.r_[base.post_issue,np.full(h,np.iinfo(np.int64).max)]
    early=(pre[ids]<=c.times.asi8[:,None])&(posttime[ids]>c.times.asi8[:,None])
    mechanism.append(dict(window_h=window,decision_target_pairs=int(early.size),prepublication_access_pairs=int(early.sum()),decision_fraction_with_early_target=float(early.any(axis=1).mean()),arrival_work_fraction_exposed=float(arr[early.any(axis=1)].sum()/arr.sum())))
pd.DataFrame(information).to_csv(OUT/'information_value_decomposition.csv',index=False)
pd.DataFrame(paired).to_csv(OUT/'paired_information_intervals.csv',index=False)
pd.DataFrame(mechanism).to_csv(OUT/'information_calendar_access.csv',index=False)

bounds=pd.read_csv(RESULTS/'cost_bounds.csv');bound_rows=[]
for _,row in bounds.iterrows():
    r=row.to_dict();match=all_results[(all_results.experiment=='I2')&all_results.year.eq(row.year)&all_results.trace.eq(row.trace)&all_results.window_h.eq(row.window_h)&all_results.information.eq('full')]
    if match.empty:
        match=all_results[(all_results.experiment=='I3')&all_results.year.eq(row.year)&all_results.trace.eq(row.trace)&all_results.window_h.eq(row.window_h)&all_results.scenario.eq('unconstrained')]
    if len(match):
        value=float(match.saving_usd.mean());r.update(policy_saving_usd=value,policy_cost_usd=float(match.adjusted_cost_usd.mean()),value_capture_lower_fraction=value/row.maximum_saving_upper_usd,cost_gap_upper_usd=float(match.adjusted_cost_usd.mean()-row.lower_bound_usd),seeds=len(match))
        if r['cost_gap_upper_usd']<-1e-5:raise RuntimeError('Policy beats claimed lower bound')
    bound_rows.append(r)
pd.DataFrame(bound_rows).to_csv(OUT/'certified_value_bounds.csv',index=False)

labels=[]
for year in [2024,2025]:
    configure(year)
    for day in np.unique(c.day_index):
        ix=c.day_index==day;spread=c.da[ix]-c.rt[ix]
        labels.append(dict(year=year,date=str(c.days[day]),spread_range_usd_mwh=float(spread.max()-spread.min()),intervals=int(ix.sum())))
labels=pd.DataFrame(labels);labels.to_csv(OUT/'market_state_daily.csv',index=False)
extreme=[];state_rows=[]
for q in [.9,.95,.975]:
    threshold=float(labels.loc[labels.year.eq(2024),'spread_range_usd_mwh'].quantile(q))
    mask=labels[labels.year.eq(2025)].set_index('date').spread_range_usd_mwh.gt(threshold)
    for window in [12,16,24]:
        full=pd.concat([new_daily(2025,s,window,'full') for s in [20260920,20260921,20260922]],axis=1).mean(axis=1)
        idx=bootstrap_indices(len(full));rare=mask.reindex(full.index).to_numpy(dtype=bool)
        for comparator in ['RASC-ME','Seasonal-ME','DA-persistent']:
            base=old_daily(comparator,window).reindex(full.index)
            difference=(full-base).to_numpy();rv=difference*rare;ov=difference*(~rare)
            rlo,rhi=interval(rv,idx);tlo,thi=interval(difference,idx)
            total=float(difference.sum());num=float(rv.sum());resamp_total=difference[idx].sum(1);resamp_rare=rv[idx].sum(1)
            valid=np.abs(resamp_total)>1e-8;share_draw=resamp_rare[valid]/resamp_total[valid]
            slo,shi=np.quantile(share_draw,[.025,.975])
            extreme.append(dict(quantile=q,threshold_usd_mwh=threshold,window_h=window,comparator=comparator,extreme_days=int(rare.sum()),days=len(rare),extreme_day_fraction=float(rare.mean()),extreme_increment_usd=num,ordinary_increment_usd=float(ov.sum()),total_increment_usd=total,extreme_share=num/total if abs(total)>1e-8 else np.nan,extreme_lower95=rlo,extreme_upper95=rhi,total_lower95=tlo,total_upper95=thi,share_lower95=slo,share_upper95=shi,rare_mean_increment_usd=float(difference[rare].mean()),ordinary_mean_increment_usd=float(difference[~rare].mean()),rare_win_fraction=float((difference[rare]>0).mean())))
            if q==.95:
                state_rows.extend(dict(date=date,window_h=window,comparator=comparator,extreme=bool(israre),increment_usd=float(delta)) for date,israre,delta in zip(full.index,rare,difference))
pd.DataFrame(extreme).to_csv(OUT/'extreme_state_results.csv',index=False)
pd.DataFrame(state_rows).to_csv(OUT/'extreme_state_paired_daily.csv',index=False)

grid=all_results[all_results.experiment.isin(['I3','I3G'])].copy()
grid.to_csv(OUT/'grid_execution_raw.csv',index=False)
gs=grid.groupby(['experiment','year','trace','window_h','scenario'],dropna=False).agg(saving_usd=('saving_usd','mean'),saving_min=('saving_usd','min'),saving_max=('saving_usd','max'),late_it_mwh=('late_IT_mwh','mean'),late_max=('late_IT_mwh','max'),reference_late_it_mwh=('reference_late_it_mwh','max'),closing_it_mwh=('closing_it_mwh','max'),recovery_mw=('max_fleet_recovery_mw','max'),upward_ramp_mw=('max_upward_site_ramp_mw','max'),cost_usd=('adjusted_cost_usd','mean'),n=('saving_usd','size')).reset_index()
gs.to_csv(OUT/'grid_execution_summary.csv',index=False)
adaptive_rows=all_results[all_results.experiment.eq('I4')].copy()
adaptive_rows.to_csv(OUT/'adaptive_execution_raw.csv',index=False)
if len(adaptive_rows):
    adaptive_rows.groupby(['year','trace','window_h','policy']).agg(saving_usd=('saving_usd','mean'),saving_min=('saving_usd','min'),saving_max=('saving_usd','max'),late_it_mwh=('late_IT_mwh','max'),closing_it_mwh=('closing_it_mwh','max'),cvar95_usd=('cvar95_usd','mean'),reassign_it_mwh=('reassigned_it_mwh','mean'),replan_calls=('replan_calls','mean'),n=('saving_usd','size')).reset_index().to_csv(OUT/'adaptive_execution_summary.csv',index=False)
print('INFORMATION',json.dumps(information,indent=2),flush=True)
print('EXTREME PRIMARY',pd.DataFrame(extreme).query('quantile==0.95 and window_h==24').to_string(index=False),flush=True)
print('EXPERIMENT COUNTS',all_results.groupby('experiment').size().to_dict(),flush=True)
