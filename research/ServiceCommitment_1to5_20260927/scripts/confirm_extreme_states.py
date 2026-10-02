"""Influence and definition diagnostics for the descriptive state concentration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
OUT=ROOT/'source_data';SRC=WORKSPACE/'NatureEnergy_Revised_Comparisons_20260927/source_data'
provenance=pd.read_csv(SRC/'fluid_raw_provenance.csv')
def series(path,column='mtm_value_usd'):
    x=pd.read_csv(path);x['date']=pd.to_datetime(x.date).dt.strftime('%Y-%m-%d')
    return x.set_index('date')[column].sort_index()
def base(model,column='mtm_value_usd'):
    r=provenance[provenance.model.eq(model)&provenance.window_h.eq(24)&provenance.policy.eq('point' if model=='DA-persistent' else 'interval')].iloc[0]
    return series(r.daily,column)

states=[]
for year in [2024,2025]:
    configure(year)
    for day in np.unique(c.day_index):
        ix=c.day_index==day;spread=c.da[ix]-c.rt[ix]
        states.append(dict(year=year,date=str(c.days[day]),spread_range=float(np.ptp(spread)),max_abs_spread=float(np.abs(spread).max()),max_rt=float(c.rt[ix].max()),mean_rt=float(c.rt[ix].mean())))
states=pd.DataFrame(states);states.to_csv(OUT/'extreme_state_definition_inputs.csv',index=False)
full=pd.concat([series(TRAJECTORIES/f'I2_Eagle_2025_s{s}_w24_full_daily.csv') for s in [20260920,20260921,20260922]],axis=1);full.columns=[20260920,20260921,20260922]
native=full.mean(axis=1);rasc=base('RASC-ME').reindex(native.index)
threshold=states.loc[states.year.eq(2024),'spread_range'].quantile(.95)
rare=states[states.year.eq(2025)].set_index('date').spread_range.gt(threshold).reindex(native.index).to_numpy()
delta=(native-rasc).to_numpy();dates=native.index.to_numpy();n=len(delta)
influence=[]
for block in [1,7]:
    for start in range(n-block+1):
        keep=np.ones(n,dtype=bool);keep[start:start+block]=False
        extreme=float(delta[keep&rare].sum());ordinary=float(delta[keep&~rare].sum());total=extreme+ordinary
        influence.append(dict(excluded_days=block,start_date=dates[start],end_date=dates[start+block-1],excluded_increment_usd=float(delta[~keep].sum()),remaining_extreme_usd=extreme,remaining_ordinary_usd=ordinary,remaining_total_usd=total,remaining_extreme_share=extreme/total if abs(total)>1e-9 else np.nan))
influence=pd.DataFrame(influence);influence.to_csv(OUT/'extreme_leave_window_out.csv',index=False)
top=int(np.argmax(delta));top_date=dates[top]
seedrows=[]
for seed in full:
    d=(full[seed]-rasc).to_numpy();keep=np.arange(n)!=top
    seedrows.append(dict(seed=seed,total_increment_usd=float(d.sum()),extreme_increment_usd=float(d[rare].sum()),top_date=top_date,top_date_increment_usd=float(d[top]),top_date_total_fraction=float(d[top]/d.sum()),remaining_extreme_usd=float(d[rare&keep].sum()),remaining_total_usd=float(d[keep].sum()),extreme_win_days=int((d[rare]>0).sum()),extreme_days=int(rare.sum())))
pd.DataFrame(seedrows).to_csv(OUT/'extreme_seed_consistency.csv',index=False)

robust=[]
for definition in ['spread_range','max_abs_spread','max_rt']:
    for q in [.90,.95,.975]:
        cutoff=float(states.loc[states.year.eq(2024),definition].quantile(q))
        mask=states[states.year.eq(2025)].set_index('date')[definition].gt(cutoff).reindex(native.index).to_numpy()
        for model in ['RASC-ME','Seasonal-ME','DA-persistent']:
            d=(native-base(model).reindex(native.index)).to_numpy();keep=np.arange(n)!=top
            robust.append(dict(definition=definition,quantile=q,threshold=cutoff,comparator=model,extreme_days=int(mask.sum()),extreme_increment_usd=float(d[mask].sum()),ordinary_increment_usd=float(d[~mask].sum()),total_increment_usd=float(d.sum()),extreme_share=float(d[mask].sum()/d.sum()),top_date_is_extreme=bool(mask[top]),without_top_date_extreme_usd=float(d[mask&keep].sum())))
pd.DataFrame(robust).to_csv(OUT/'extreme_definition_sensitivity.csv',index=False)

bootstrap=[]
for model in ['RASC-ME','Seasonal-ME','DA-persistent']:
    d=(native-base(model).reindex(native.index)).to_numpy()
    for days in [3,7,14,28]:
        rng=np.random.default_rng(20260927)
        starts=rng.integers(0,n,(2000,int(np.ceil(n/days))))
        idx=((starts[:,:,None]+np.arange(days))%n).reshape(2000,-1)[:,:n]
        rr=rare[idx];dd=d[idx]
        sums=(dd*rr).sum(1);mean_gap=np.divide(sums,rr.sum(1),out=np.full(2000,np.nan),where=rr.sum(1)>0)-np.divide((dd*(~rr)).sum(1),(~rr).sum(1),out=np.full(2000,np.nan),where=(~rr).sum(1)>0)
        elo,ehi=np.quantile(sums,[.025,.975]);mlo,mhi=np.nanquantile(mean_gap,[.025,.975])
        bootstrap.append(dict(comparator=model,block_days=days,extreme_increment_usd=float(d[rare].sum()),extreme_lower95=elo,extreme_upper95=ehi,mean_extreme_minus_ordinary_usd=float(d[rare].mean()-d[~rare].mean()),mean_gap_lower95=mlo,mean_gap_upper95=mhi,extreme_median_increment_usd=float(np.median(d[rare])),ordinary_median_increment_usd=float(np.median(d[~rare])),extreme_win_days=int((d[rare]>0).sum())))
pd.DataFrame(bootstrap).to_csv(OUT/'extreme_block_sensitivity.csv',index=False)

daypanel=pd.DataFrame({'native':native,'rasc':rasc,'delta':delta,'extreme':rare})
cash=pd.concat([series(TRAJECTORIES/f'I2_Eagle_2025_s{s}_w24_full_daily.csv','value_usd') for s in full],axis=1).mean(axis=1)-base('RASC-ME','value_usd').reindex(native.index)
daypanel['cash_delta']=cash;daypanel['early_path_delta']=native-pd.concat([series(TRAJECTORIES/f'I2_Eagle_2025_s{s}_w24_post_only_daily.csv') for s in full],axis=1).mean(axis=1)
daypanel.index.name='date';daypanel.to_csv(OUT/'extreme_event_daily_audit.csv')
configure(2025)
ix=np.asarray(c.times.tz_convert(TZ).strftime('%Y-%m-%d')==top_date)
row=provenance[provenance.model.eq('RASC-ME')&provenance.window_h.eq(24)&provenance.policy.eq('interval')].iloc[0]
with np.load(row.book,allow_pickle=False) as z:rasc_book=z['compute']
native_books=[]
for seed in full:
    with np.load(TRAJECTORIES/f'I2_Eagle_2025_s{seed}_w24_full_complete_book.npz',allow_pickle=False) as z:native_books.append(z['compute'].copy())
native_grid=np.mean([c.cash(b)[1] for b in native_books],axis=0);rasc_grid=c.cash(rasc_book)[1]
event=[]
for t in np.flatnonzero(ix):
    for z in range(4):event.append(dict(timestamp_utc=str(c.times[t]),zone=['WEST','CAPITL','N.Y.C.','LONGIL'][z],da_usd_mwh=float(c.da[t,z]),rt_usd_mwh=float(c.rt[t,z]),world_grid_mw=float(native_grid[t,z]),rasc_grid_mw=float(rasc_grid[t,z]),grid_bill_increment_usd=float(c.DT*c.rt[t,z]*(rasc_grid[t,z]-native_grid[t,z]))))
pd.DataFrame(event).to_csv(OUT/'extreme_top_event_intervals.csv',index=False)
without=influence[influence.excluded_days.eq(1)&influence.start_date.eq(top_date)].iloc[0]
worstweek=influence[influence.excluded_days.eq(7)].sort_values('remaining_extreme_usd').iloc[0]
summary=dict(status='concentration_is_dominated_by_one_event_not_a_general_extreme_state_advantage',threshold=float(threshold),extreme_days=int(rare.sum()),days=n,top_date=top_date,top_date_increment_usd=float(delta[top]),top_date_fraction_of_total=float(delta[top]/delta.sum()),top_date_fraction_of_extreme_sum=float(delta[top]/delta[rare].sum()),total_increment_usd=float(delta.sum()),extreme_increment_usd=float(delta[rare].sum()),remaining_extreme_without_top_date_usd=float(without.remaining_extreme_usd),remaining_total_without_top_date_usd=float(without.remaining_total_usd),extreme_win_days=int((delta[rare]>0).sum()),worst_week=worstweek.to_dict(),top_day_cash_increment_usd=float(cash.loc[top_date]),top_day_early_path_increment_usd=float(daypanel.loc[top_date,'early_path_delta']),max_rt_on_top_date=float(c.rt[ix].max()),max_da_on_top_date=float(c.da[ix].max()),scope='Financial influence diagnostics only; original continuous schedules retained. Outcome-selected removal is diagnostic, not a replacement primary test.')
(RESULTS/'extreme_confirmation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));print(pd.DataFrame(seedrows).to_string(index=False));print(pd.DataFrame(bootstrap).query("comparator=='RASC-ME'").to_string(index=False))
