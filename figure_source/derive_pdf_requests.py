"""Descriptive calculations from retained daily accounts; no model fit or replay."""
from pathlib import Path
import json, shutil, hashlib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WS=ROOT.parent
D=ROOT/'source_data'
src=WS/'zero_shot_isone_20260928/outputs/zero_shot_isone_daily.csv'
shutil.copy2(src,D/'zero_shot_isone_daily.csv')
daily=pd.read_csv(src)
selected=daily[(daily.market=='ISO-NE')&(daily.embedding_rule=='neutral_mean')&(daily.policy=='Frozen WorldMoE zero-shot')].copy()
selected=selected[['date','portfolio','seed','service_adjusted_mtm_saving_usd']].rename(columns={'service_adjusted_mtm_saving_usd':'saving_usd'})
ny=pd.read_csv(D/'current/worldmoe_daily_by_seed.csv').melt(id_vars='date',var_name='seed',value_name='saving_usd')
ny['portfolio']='NYISO';ny['seed']=ny.seed.astype(int)
all_daily=pd.concat([ny,selected],ignore_index=True)
all_daily.to_csv(D/'cross_market_mtm_daily.csv',index=False)

# Retained price archives supply descriptive, within-market 2025 labels.
price_path=WS/'zero_shot_isone_20260928/data_processed/isone_hourly_canonical.csv'
price=pd.read_csv(price_path)
price=price[price.market_date_local.str.startswith('2025')].copy()
price['spread']=price.da_lmp_usd_mwh-price.rt_lmp_final_usd_mwh
iso_range=price.groupby('market_date_local').spread.agg(lambda x: x.max()-x.min())
with np.load(WS/'external_validation/data/forcing_2025.npz',allow_pickle=False) as z:
    ts=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True)).tz_convert('America/New_York')
    spread=z['da'].astype(float)-z['rt'].astype(float)
frame=pd.DataFrame({'date':ts.strftime('%Y-%m-%d'),'max':spread.max(axis=1),'min':spread.min(axis=1)})
frame=frame[frame.date.str.startswith('2025')]
ng=frame.groupby('date').agg({'max':'max','min':'min'});ny_range=ng['max']-ng['min']
event=[];rows=[]
for market,g in all_daily.groupby('portfolio'):
    series=g.groupby('date').saving_usd.mean().sort_index()
    ranges=ny_range if market=='NYISO' else iso_range
    cut=float(ranges.quantile(.95));flags=ranges.ge(cut).reindex(series.index)
    if flags.isna().any():raise ValueError('Missing market event labels')
    values=series.to_numpy();total=values.sum();topday=values.max();weekly=np.convolve(values,np.ones(7),mode='valid');topweek=weekly.max()
    rows.append(dict(portfolio=market,n_seeds=g.seed.nunique(),days=len(series),annual_saving_usd=total,
      largest_day_usd=topday,largest_day_share_pct=100*topday/total,after_largest_day_usd=total-topday,
      largest_7day_usd=topweek,largest_7day_share_pct=100*topweek/total,after_largest_7day_usd=total-topweek,
      positive_days=int((values>0).sum()),extreme_threshold_usd_mwh=cut,extreme_days=int(flags.sum()),
      extreme_saving_usd=float(series[flags].sum()),ordinary_saving_usd=float(series[~flags].sum()),
      extreme_share_pct=float(100*series[flags].sum()/total),reference='Immediate execution / earliest feasible',
      account='Daily change in service-adjusted mark-to-market value',
      event_definition='Daily zonal DA-minus-RT range >= within-market 2025 p95; retrospective descriptive label'))
    event.extend([dict(portfolio=market,date=d,spread_range_usd_mwh=float(ranges.loc[d]),extreme=bool(flags.loc[d]),saving_usd=float(v)) for d,v in series.items()])
pd.DataFrame(rows).to_csv(D/'cross_market_event_concentration.csv',index=False)
pd.DataFrame(event).to_csv(D/'cross_market_event_days.csv',index=False)
all_daily['month']=all_daily.date.str[:7]
monthly=all_daily.groupby(['portfolio','seed','month']).saving_usd.sum().reset_index()
monthly.to_csv(D/'cross_market_monthly_mtm.csv',index=False)
manifest={str(p.relative_to(WS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,price_path,WS/'external_validation/data/forcing_2025.npz']}
(D/'PDF_REQUEST_DERIVATION.json').write_text(json.dumps({'inputs_sha256':manifest,'no_fitting_or_replay':True,'annual_seed_aggregation':'mean, never sum','daily_account':'MTM for every portfolio','event_labels':'within-market 2025 p95 descriptive; not predeclared 2024 labels'},indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
